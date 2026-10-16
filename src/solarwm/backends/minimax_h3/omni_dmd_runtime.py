"""Structure-guided DMD2 from a released four-step LoRA (DMAD in the release): student, teacher and critic on
three model copies. ``train.dmd.init_lora`` names the starting LoRA (the older name ``plug`` is still read).

``train.dmd.mode``:

``adapt_plug``   the released recipe. The control LoRA is frozen and merged into the student's base; the
                 student's own adapter starts as the released four-step LoRA (seeded into the leading
                 rank) and is distilled towards the teacher, base + control at many steps.
``self_distill`` an ablation: the released four-step LoRA is frozen and merged into the student's base; the
                 student's own adapter is the control LoRA. The teacher is the starting control LoRA
                 without the four-step LoRA, fixed, and the control LoRA also takes the ordinary
                 flow-matching loss on the sample's data through the teacher copy (lent the student's
                 control for that pass, then given back its own).
``joint``        an ablation: nothing is merged; the student's own adapter is the control LoRA and the
                 four-step LoRA stacked into one rank-512 LoRA (control in slots 0-383, four-step LoRA in
                 384-511: their sum, unchanged), and both train: the DMD gradient reaches all of it, the
                 flow-matching gradient (through the teacher copy, lent the control slots) only the
                 control slots; the teacher's own answer stays the starting control. Without a control
                 checkpoint the control slots start fresh (a zero update): joint training from the base.

The critic (fake score) starts from the teacher. Copies share the sample's conditions and every
random draw (the exit evaluation is drawn once for all ranks: FSDP needs the same forwards on each).
``train.dmd.teacher`` "follow" restores the first design, a teacher kept equal to the student's control
after every update (it fed the student's drift back as the answer; see _teacher_follows).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from solarwm.training import JsonlEventSink, StepPolicy
from solarwm.training.sgf import (
    compute_sgf_kl_gradient,
    project_out_residual,
    sgf_student_loss,
    should_update_student,
)

from .config import dmd_settings
from .distributed import get_sp_size, sync_lora_gradients
from .runtime import H3TrainingRuntime, _base_model_load_receipt
from .sgf import h3_sgf_critic_loss

MODES = ("adapt_plug", "self_distill", "joint", "causal")
TEACHED = ("self_distill", "joint")  # modes whose teacher follows the student's control LoRA


def decoupled_causal(dmd: Any) -> bool:
    """The block-causal recipe: the causal control LoRA (``student_init``) merged into the student's base and frozen,
    the student's own adapter a separate acceleration LoRA that starts as the released four-step LoRA of
    ``accel_init``; the teacher and critic keep the bidirectional control LoRA (``control``)."""

    return str(dmd.get("mode")) == "causal" and bool(dmd.get("decoupled", False))


def seeds_released_plug(dmd: Any) -> bool:
    """Whether an adapt_plug student starts as the released four-step LoRA (the default) or from a zero update
    (``seed_init_lora: false``)."""

    return bool(dmd.get("seed_plug", True))


class H3OmniDMDRuntime(H3TrainingRuntime):
    def __init__(self, config: Any) -> None:
        dmd = dmd_settings(config["train"]["dmd"])
        self.mode = str(dmd["mode"])
        if self.mode not in MODES:
            raise ValueError(f"train.dmd.mode must be one of {MODES}, got {self.mode!r}")
        self.dmd = dmd
        super().__init__(config, defer_resume=True)
        # No student EMA (LongLive-Plug's DMD runs without one): the adapter is replicated on every
        # rank, so its FP32 shadow would hold 8 GB a card that the three copies need.
        self.ema = None
        torch = self.torch
        torch.cuda.empty_cache()
        from .lora import released_lora_state, seed_lora_from_released

        if decoupled_causal(dmd):
            if dmd.get("accel_init"):
                # The acceleration LoRA starts as the released four-step LoRA, its update unchanged (the scale its
                # header names; alpha = rank when it names none).
                from safetensors import safe_open

                with safe_open(str(dmd["accel_init"]), framework="pt") as handle:
                    header = handle.metadata() or {}
                scale = float(header["alpha"]) / float(header["rank"]) if "alpha" in header and "rank" in header else 1.0
                seeded = seed_lora_from_released(
                    self.lora.parameter_by_key,
                    released_lora_state(str(dmd["accel_init"])),
                    live_scale=self.lora.alpha / self.lora.rank,
                    released_scale=scale,
                )
                if self.is_main:
                    print(f"[h3-dmd-init] causal acceleration LoRA from {dmd['accel_init']} in {seeded} modules over "
                          "the frozen causal control", flush=True)
            elif self.is_main:
                print("[h3-dmd-init] causal acceleration LoRA from a zero update over the frozen causal control",
                      flush=True)
        elif self.mode == "causal":
            raise ValueError("a causal DMD student is a separate acceleration LoRA: set train.dmd.decoupled")
        elif self.mode == "adapt_plug" and not seeds_released_plug(dmd):
            if self.is_main:
                print("[h3-dmd-init] student starts from a zero update (seed_init_lora false), not from the released LoRA", flush=True)
        elif self.mode == "adapt_plug":
            # The student's adapter starts as the released four-step LoRA: its update unchanged, spare rank fresh.
            seeded = seed_lora_from_released(
                self.lora.parameter_by_key,
                released_lora_state(str(dmd["plug"])),
                live_scale=self.lora.alpha / self.lora.rank,
                released_scale=float(dmd.get("plug_alpha", 128)) / float(dmd.get("plug_rank", 128)),
            )
            if self.is_main:
                print(f"[h3-dmd-init] student seeded from the released four-step LoRA in {seeded} modules", flush=True)
        elif self.mode == "joint":
            from .lora import adapter_state_as_released

            width = int(self.model_cfg["adapter"]["rank"])
            live = self.lora.alpha / self.lora.rank
            if dmd.get("control"):
                seed_lora_from_released(
                    self.lora.parameter_by_key,
                    adapter_state_as_released(self._control_state()),
                    live_scale=live,
                    released_scale=float(self.model_cfg["adapter"]["alpha"]) / width,
                )
            seeded = seed_lora_from_released(
                self.lora.parameter_by_key,
                released_lora_state(str(dmd["plug"])),
                live_scale=live,
                released_scale=float(dmd.get("plug_alpha", 128)) / float(dmd.get("plug_rank", 128)),
                offset=width,
            )
            if self.is_main:
                print(
                    f"[h3-dmd-init] student stacked from {'control' if dmd.get('control') else 'fresh control'}"
                    f" + the four-step LoRA in {seeded} modules",
                    flush=True,
                )
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        try:
            self.teacher, self.teacher_lora = self._build_score_role(trainable=False)
            critic, critic_lora = self._build_score_role(trainable=True)
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        from .optimizer import build_adapter_optimizer

        optimizer = build_adapter_optimizer(critic_lora, self.train_cfg["critic_optimizer"], None)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        self.extra_roles["critic"] = {
            "model": critic,
            "lora": critic_lora,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        self.contract = replace(
            self.contract,
            extras={
                **self.contract.extras,
                "dmd": {
                    "mode": self.mode,
                    "plug": str(dmd.get("plug") or ""),
                    "seed_plug": seeds_released_plug(dmd),
                    "control": str(dmd.get("control") or ""),
                    "forwards": int(dmd.get("forwards", 4)),
                },
            },
        )
        # The teacher as built holds the starting control: the fixed DMD answer (dmd.teacher "fixed",
        # the default) - kept on the CPU so flow matching can borrow the teacher copy and give it back.
        self._remember_fixed_teacher()
        resume = str(self.checkpoint_cfg.get("resume_from") or "")
        if resume:
            self.load_checkpoint(resume)
            if self.mode in TEACHED and self._teacher_follows():
                self._sync_teacher()

    # --- construction -------------------------------------------------------------------------

    def _control_state(self) -> dict[str, Any]:
        if not self.dmd.get("control"):
            # A joint run from the base: the control LoRA is the student's own fresh control
            # slots (a zero update), which the teacher then follows and the critic starts from.
            return {
                key: self._control_slots(key).detach().cpu().clone()
                for key in self.lora.parameter_by_key
            }
        root = Path(str(self.dmd["control"]))
        adapter = self.torch.load(root / "adapter.pt", map_location="cpu", weights_only=True)
        return adapter["state"]

    def _student_adapter(self) -> tuple[Any, tuple[int, ...]]:
        if self.mode != "joint":
            return super()._student_adapter()
        # Control (rank 384, alpha 384) and the four-step LoRA (rank 128, alpha 128) both update at scale 1,
        # so one rank-512 LoRA at alpha 512 holds their sum exactly.
        width = int(self.model_cfg["adapter"]["rank"]) + int(self.dmd.get("plug_rank", 128))
        return {"rank": width, "alpha": width, "dropout": 0.0}, (width,)

    def _before_adapter(self, transformer: Any) -> None:
        from .lora import adapter_state_as_released, merge_lora_delta, released_lora_state

        if self.mode == "joint":
            return  # control and four-step LoRA both live in the student's own (trained) adapter
        if self.mode == "adapt_plug" or decoupled_causal(self.dmd):
            # The trained control LoRA becomes part of the student's base, frozen: for a causal student its own causal
            # control LoRA (student_init), not the teacher's.
            control = (
                self.torch.load(Path(str(self.dmd["student_init"])) / "adapter.pt", map_location="cpu",
                                weights_only=True)["state"]
                if decoupled_causal(self.dmd) else self._control_state()
            )
            merged = merge_lora_delta(
                transformer,
                adapter_state_as_released(control),
                alpha=float(self.model_cfg["adapter"]["alpha"]),
                rank=int(self.model_cfg["adapter"]["rank"]),
            )
        else:
            # The four-step LoRA becomes part of the student's base, frozen, as inference merges it.
            merged = merge_lora_delta(
                transformer,
                released_lora_state(str(self.dmd["plug"])),
                alpha=float(self.dmd.get("plug_alpha", 128)),
                rank=int(self.dmd.get("plug_rank", 128)),
            )
        if self.rank == 0:
            print(
                f"[h3-dmd-init] {self.mode}: merged a LoRA into {merged} student weights",
                flush=True,
            )

    def _make_core(self) -> Any:
        from .observations import observation_profile_from_train_config
        from .omni_dmd import H3OmniDMDCore

        if self.mode == "causal":
            from .omni_causal import H3OmniCausalDMDCore

            return H3OmniCausalDMDCore(
                self.model,
                self.silence,
                self.device,
                self.dmd,
                dict(self.dmd["causal_stream"]),
                geometry=self.geometry,
                observations=observation_profile_from_train_config(self.train_cfg),
            )

        return H3OmniDMDCore(
            self.model,
            self.silence,
            self.device,
            self.dmd,
            geometry=self.geometry,
            observations=observation_profile_from_train_config(self.train_cfg),
        )

    def _build_score_role(self, *, trainable: bool) -> tuple[Any, Any]:
        """Base + control LoRA, no four-step LoRA.

        The teacher carries the control LoRA itself (a self-distillation keeps it equal to the
        student's and passes the flow-matching gradient through it, so it keeps a graph). The
        critic starts as the teacher at a fraction of its memory: the control LoRA merged into
        its base and a fresh rank-``critic_rank`` adapter (zero update) on top, as LongLive-Plug
        trains its fake score at rank 128.
        """

        from .fsdp import wrap_h3_fsdp
        from .lora import adapter_state_as_released, inject_h3_lora, merge_lora_delta
        from .optional import load_transformer

        seed = 0x48330000 + int(self.config["data"]["seed"])
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        modules = load_transformer(self.model_cfg, device=self.device)
        modules.transformer.requires_grad_(False)
        graph = trainable or self.mode in TEACHED
        control = (
            {key: value.detach().cpu() for key, value in self.lora.state_dict().items()}
            if self.mode == "self_distill"
            else self._control_state()
        )
        adapter_cfg = self.model_cfg["adapter"]
        ranks: tuple[int, ...] = (384,)
        if trainable:
            merge_lora_delta(
                modules.transformer,
                adapter_state_as_released(control),
                alpha=float(adapter_cfg["alpha"]),
                rank=int(adapter_cfg["rank"]),
            )
            rank = int(self.dmd.get("critic_rank", 128))
            adapter_cfg = {"rank": rank, "alpha": rank, "dropout": 0.0}
            ranks = (rank,)
        model, lora = inject_h3_lora(
            modules.transformer,
            adapter_cfg,
            base_identity=_base_model_load_receipt(self.model_cfg, modules.transformer),
            ranks=ranks,
        )
        model = wrap_h3_fsdp(
            model,
            local_rank=self.topology.local_rank,
            transformer_block_cls=modules.transformer_block_cls,
            fp32_units=modules.fp32_fsdp_units,
            ignored_parameters=lora.parameters,
            activation_checkpointing=graph,
            frozen_base_shard_size=self.train_cfg["fsdp"].get("frozen_base_shard_size"),
            saved_attention_blocks=int(self.train_cfg["fsdp"].get("saved_attention_blocks", 0)),
        )
        if not trainable:
            lora.load_state_dict(control, broadcast=False)
        if not graph:
            model.requires_grad_(False)
        return model.eval(), lora

    def _control_slots(self, key: str) -> Any:
        """The student's control LoRA under the teacher's key (joint: its first rank slots)."""

        live = self.lora.parameter_by_key[key]
        if self.mode != "joint":
            return live
        width = int(self.model_cfg["adapter"]["rank"])
        return live[:width] if key.endswith("lora_A.weight") else live[:, :width]

    def _sync_teacher(self) -> None:
        """The teacher is the student's control LoRA, without the four-step LoRA."""

        with self.torch.no_grad():
            for key, parameter in self.teacher_lora.parameter_by_key.items():
                parameter.copy_(self._control_slots(key))

    def _teacher_follows(self) -> bool:
        """dmd.teacher "follow": the teacher is the student's control after every update (the first
        design). "fixed": the teacher stays the starting
        control - a teacher that follows the student feeds the student's drift back as its own
        answer; self-distillation lost structure (0.644 engine-depth delta1 against 0.876) and
        joint swung between 0.61 and 0.87, while adapt_plug, whose teacher never moved, kept it."""

        return str(self.dmd.get("teacher", "fixed")) == "follow"

    def _remember_fixed_teacher(self) -> None:
        self.teacher_fixed = {
            key: parameter.detach().to("cpu", copy=True)
            for key, parameter in self.teacher_lora.parameter_by_key.items()
        }

    def _restore_fixed_teacher(self) -> None:
        """The teacher back to the starting control, after flow matching borrowed it."""

        with self.torch.no_grad():
            for key, parameter in self.teacher_lora.parameter_by_key.items():
                parameter.copy_(self.teacher_fixed[key].to(parameter.device))

    def _teacher_after_update(self) -> None:
        if self.mode in TEACHED and self._teacher_follows():
            self._sync_teacher()

    def _add_teacher_gradients(self) -> None:
        """Move the teacher's flow-matching gradient onto the student's control slots."""

        with self.torch.no_grad():
            for key, parameter in self.teacher_lora.parameter_by_key.items():
                extra = parameter.grad
                if extra is None:
                    continue
                live = self.lora.parameter_by_key[key]
                if live.grad is None:
                    live.grad = self.torch.zeros_like(live)
                if key.endswith("lora_A.weight"):
                    live.grad[: extra.shape[0]].add_(extra)
                else:
                    live.grad[:, : extra.shape[1]].add_(extra)
                parameter.grad = None

    # --- one outer step -----------------------------------------------------------------------

    def _shared_exit(self) -> int:
        """One exit evaluation for every rank: FSDP gathers per forward, so all ranks run alike."""

        torch = self.torch
        value = torch.randint(0, int(self.dmd.get("forwards", 4)), (), device=self.device)
        if self.dist.is_initialized():
            self.dist.broadcast(value, src=0)
        return int(value.item())

    def _shared_choice(self, count: int) -> int:
        value = self.torch.randint(0, int(count), (), device=self.device)
        if self.dist.is_initialized():
            self.dist.broadcast(value, src=0)
        return int(value.item())

    def _stream_student(self, batch: Any, *, graph: bool) -> Any:
        """The block-causal student's clip: streamed chunk by chunk through its key/value cache without a graph, every
        chunk exiting at one shared evaluation; with ``graph`` replayed as one teacher-forced document from the
        recorded inputs - the same numbers, now differentiable, the gradient reaching how each chunk was written as
        context for the chunks after it too (self-gradient forcing)."""

        from .distributed import get_dp_rank
        from .flow import make_shifted_schedule
        from .layout import unpatchify_video

        torch, core = self.torch, self.core
        forwards = int(self.dmd.get("forwards", 4))
        seed = self._shared_choice(2**30) + 7919 * int(get_dp_rank())
        schedule = make_shifted_schedule(forwards + 1, shift=12.0)
        exit_index = self._shared_exit()
        with torch.no_grad():
            clip, inputs, times, jitters = core.stream(
                batch, generator=torch.Generator(device=self.device).manual_seed(seed), schedule=schedule,
                exit_index=exit_index, model=self.model,
            )
        if not graph:
            return clip
        replay = torch.Generator(device=self.device).manual_seed(seed)
        document = core._document(batch, replay)
        rows = core.training_prediction(
            document, inputs, times, clip, generator=replay, jitter=jitters, model=self.model
        )
        frames, grid = document[4]
        velocity = unpatchify_video(rows, frames, *grid, channels=24).float()
        level = torch.ones((1, 1, frames, 1, 1), device=self.device)
        for (a, b), t in zip(document[3].frames, times, strict=True):
            level[:, :, a:b] = 1.0 - float(t)
        return inputs + level * velocity

    def _finish_update(self, model: Any, lora: Any, optimizer: Any, scheduler: Any) -> float:
        sync_lora_gradients(lora.trainable_parameters)
        finite, norm = self._finite_clip_norm(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            float(self.train_cfg["optimizer"].get("gradient_clip", 10.0)),
        )
        if not finite:
            raise FloatingPointError(f"non-finite H3 DMD gradient norm={norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return norm

    def _phase(self, name: str) -> None:
        if self.is_main:
            print(
                f"[h3-dmd-phase] outer_step={self.global_step + 1} {name} "
                f"elapsed_s={time.monotonic() - self._step_started:.1f}",
                flush=True,
            )

    @property
    def structure_guidance(self) -> float:
        """dmd.structure_guidance: the teacher's answer pushed along the structure direction; 1 = plain.
        dmd.structure_guidance_warmup: over that many student steps w climbs linearly from 1 - a
        self-distilling student (the control LoRA itself) lost its structure within 50 steps at w = 2 from the first."""

        guidance = float(self.dmd.get("structure_guidance", 1.0))
        warmup = int(self.dmd.get("structure_guidance_warmup", 0))
        if warmup <= 0:
            return guidance
        return 1.0 + (guidance - 1.0) * min(1.0, self.student_step / warmup)

    @property
    def structure_guidance_sigma_min(self) -> float:
        """dmd.structure_guidance_sigma_min: the guidance acts only at re-noise levels at least this."""

        return float(self.dmd.get("structure_guidance_sigma_min", 0.0))

    @property
    def quality_guidance(self) -> float:
        """dmd.quality_guidance: the teacher's answer pushed away from its answer under a fixed
        low-quality prompt; 0 = off."""

        return float(self.dmd.get("quality_guidance", 0.0))

    @property
    def quality_guidance_sigma_max(self) -> float:
        """dmd.quality_guidance_sigma_max: the quality guidance acts only below this re-noise level."""

        return float(self.dmd.get("quality_guidance_sigma_max", 1.0))

    @property
    def quality_negative(self) -> Any:
        """dmd.quality_negative: the low-quality prompt's (embeds, tags), read once from its safetensors."""

        path = self.dmd.get("quality_negative")
        if not path:
            return None
        cached = getattr(self, "_quality_negative", None)
        if cached is None:
            from safetensors import safe_open

            from .omni_data import _local

            with _local(str(path)) as local, safe_open(local, framework="pt") as handle:
                cached = (handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"))
            self._quality_negative = cached
        return cached

    def negative_for(self, batch: Any) -> Any:
        """The low-quality prompt for ``batch``: the one file for every sample, or - when dmd.quality_negative names a
        directory - the sample's own ``<sample_id>.safetensors``. The ref2va prompt carries the sample's references,
        so its low-quality twin is encoded with them, per sample."""

        path = str(self.dmd.get("quality_negative") or "")
        if path.endswith(".safetensors"):
            return self.quality_negative
        from safetensors import safe_open

        from .omni_data import _local

        with _local(f"{path.rstrip('/')}/{batch.sample_id}.safetensors") as local, safe_open(local, framework="pt") as handle:
            return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags")

    def _dmd_gradient(self, fake: Any, real: Any, output: Any) -> Any:
        """The DMD update on the student's endpoint. With ``dmd.pdmd`` the teacher-critic difference first loses its
        component along the critic-student endpoint residual (PDMD, 2609.35768: that residual estimates the critic's
        error), then is normalized as usual."""

        if bool(self.dmd.get("pdmd", False)):
            fake = real.float() + project_out_residual(fake - real, fake - output)
        return compute_sgf_kl_gradient(fake_x0=fake, real_x0=real, student_output=output)

    def _anchor(self, stats: dict[str, float], batch: Any, inputs: Any, forwards: int) -> None:
        """The real-data anchor on the few-step path (control + four-step LoRA): the sample's real target noised to one of the
        student's levels and regressed to, weighted by what the sample was given. Every rank runs the
        forward whatever its sample's weight: the model is sharded across ranks, and a rank that skipped it while
        another ran it left the two in different collectives (the 1-node smoke test hung)."""

        from .omni_dmd import anchor_condition_weight

        given = anchor_condition_weight(batch, dict(self.dmd.get("anchor_condition_weights", ANCHOR_WEIGHTS)))
        self._phase("anchor")
        term, sigma = self.core.anchor_loss(
            inputs, model=self.model, forwards=forwards, index=self._shared_exit()
        )
        if not bool(self.torch.isfinite(term)):
            raise FloatingPointError("non-finite H3 DMD anchor loss")
        (float(self.dmd["anchor_weight"]) * given * term / get_sp_size()).backward()
        stats["loss_anchor"] = float(term.detach())
        stats["anchor_sigma"] = float(sigma)
        stats["anchor_given"] = float(given)

    def _student_update(self, stats: dict[str, float]) -> None:
        torch, core = self.torch, self.core
        critic = self.extra_roles["critic"]
        forwards = int(self.dmd.get("forwards", 4))
        batch = self._next_batch()
        inputs = core.prepare_inputs(batch)
        self._phase("student_rollout")
        if self.mode == "causal":
            output = self._stream_student(batch, graph=True)
        else:
            output = core.rollout(
                inputs, model=self.model, forwards=forwards, exit_index=self._shared_exit()
            )
        with torch.no_grad():
            self._phase("student_scores")
            noisy, _noise, video_t, audio_t = core.score_inputs(output.detach())
            audio_noise = core._random(tuple(inputs.audio_clean.shape), None)
            sigma = 1.0 - video_t
            fake = noisy + sigma * core.score_forward(
                critic["model"], inputs, noisy, video_t, audio_t, audio_noise
            )
            real = core.teacher_answer(
                self.teacher, inputs, batch, noisy, video_t, audio_t, audio_noise,
                guidance=self.structure_guidance, sigma_min=self.structure_guidance_sigma_min,
                quality=self.quality_guidance, quality_sigma_max=self.quality_guidance_sigma_max,
                negative=self.negative_for(batch) if self.quality_guidance > 0 else None,
            )
            if not bool(torch.isfinite(fake).all() and torch.isfinite(real).all()):
                raise FloatingPointError("non-finite H3 DMD score output")
            gradient = self._dmd_gradient(fake, real, output.detach())
        loss = sgf_student_loss(output, gradient)
        self._phase("student_backward")
        (loss / get_sp_size()).backward()
        stats["loss_student"] = float(loss.detach())
        del output, noisy, fake, real, gradient, loss
        if float(self.dmd.get("anchor_weight", 0.0)) > 0:
            self._anchor(stats, batch, inputs, forwards)
        weight = float(self.dmd.get("flow_matching_weight", 1.0))
        if self.mode in TEACHED and weight > 0:
            # The ordinary loss on the sample's own data, through the teacher copy (control
            # LoRA, no four-step LoRA); its gradient joins the student's control LoRA gradient.
            self._phase("flow_matching")
            if not self._teacher_follows():
                self._sync_teacher()  # borrow the teacher copy: the student's control, no four-step LoRA
            fm = core.flow_matching_loss(inputs, self.teacher)
            if not bool(torch.isfinite(fm)):
                raise FloatingPointError("non-finite H3 DMD flow-matching loss")
            (weight * fm / get_sp_size()).backward()
            self._add_teacher_gradients()
            if not self._teacher_follows():
                self._restore_fixed_teacher()
            stats["loss_flow_matching"] = float(fm.detach())
        stats["student_grad_norm"] = self._finish_update(
            self.model, self.lora, self.optimizer, self.scheduler
        )
        self._teacher_after_update()
        self.student_step += 1
        if self.ema is not None:
            self.ema_update(self.student_step)

    def _critic_update(self, stats: dict[str, float]) -> None:
        torch, core = self.torch, self.core
        critic = self.extra_roles["critic"]
        batch = self._next_batch()
        inputs = core.prepare_inputs(batch)
        self._phase("critic_rollout")
        with torch.no_grad():
            if self.mode == "causal":
                clean = self._stream_student(batch, graph=False)
            else:
                clean = core.rollout(
                    inputs,
                    model=self.model,
                    forwards=int(self.dmd.get("forwards", 4)),
                    exit_index=self._shared_exit(),
                )
            noisy, noise, video_t, audio_t = core.score_inputs(clean)
        self._phase("critic_forward")
        velocity = core.score_forward(critic["model"], inputs, noisy, video_t, audio_t)
        loss = h3_sgf_critic_loss(velocity, noise=noise, clean=clean)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite H3 DMD critic loss")
        self._phase("critic_backward")
        (loss / get_sp_size()).backward()
        stats["loss_critic"] = float(loss.detach())
        stats["critic_grad_norm"] = self._finish_update(
            critic["model"], critic["lora"], critic["optimizer"], critic["scheduler"]
        )

    def train_outer_step(self) -> dict[str, float]:
        self._step_started = time.monotonic()
        ratio = int(self.dmd.get("critic_updates_per_student", 5))
        update_student = should_update_student(self.global_step, ratio)
        stats: dict[str, float] = {"student_updated": float(update_student), "loss_student": 0.0}
        if update_student:
            self._student_update(stats)
        self._critic_update(stats)
        self._global_step += 1
        stats.update(
            student_step=float(self.student_step),
            loss=stats["loss_student"] if update_student else stats["loss_critic"],
            lr=float(self.optimizer.param_groups[0]["lr"]),
            critic_lr=float(self.extra_roles["critic"]["optimizer"].param_groups[0]["lr"]),
            compute_time_s=time.monotonic() - self._step_started,
        )
        if self.is_main:
            extra = (
                f" fm_loss={stats['loss_flow_matching']:.6f}"
                if "loss_flow_matching" in stats
                else ""
            ) + (
                f" anchor_loss={stats['loss_anchor']:.6f} anchor_sigma={stats['anchor_sigma']:.3f}"
                if "loss_anchor" in stats
                else ""
            )
            print(
                f"[h3-dmd] step={self.global_step} student_step={self.student_step} "
                f"student_loss={stats['loss_student']:.6f} critic_loss={stats['loss_critic']:.6f}"
                f"{extra} compute_time_s={stats['compute_time_s']:.2f}",
                flush=True,
            )
        return stats


# How much the real-data anchor counts by what a sample was given: structure + first frame fully, structure alone a
# little (its appearance is open), first frame or text alone not at all (many right answers; regressing blurs).
ANCHOR_WEIGHTS = {"full": 1.0, "structure": 0.3, "first": 0.0, "text": 0.0}


def run_omni_dmd_training(config: Any) -> int:
    runtime = H3OmniDMDRuntime(config)
    policy = StepPolicy(
        max_steps=int(config["train"]["max_steps"]),
        save_every=int(config["checkpoint"]["save_every_steps"]),
        save_steps=tuple(config["checkpoint"].get("save_steps", ())),
        validate_every=int(config["validation"]["validate_every_steps"]),
        validation_steps=(),
    )
    sink = (
        JsonlEventSink(Path(str(config["runtime"]["output_dir"])) / "training-events.jsonl")
        if runtime.is_main
        else None
    )
    try:
        while runtime.global_step < policy.max_steps:
            stats = runtime.train_outer_step()
            step = runtime.global_step
            if sink is not None:
                sink({"event": "train_step", "step": step, **stats})
            if policy.should_save(step):
                runtime.save_checkpoint(step)
    finally:
        runtime.reader.close()
    return 0


__all__ = ["H3OmniDMDRuntime", "run_omni_dmd_training"]
