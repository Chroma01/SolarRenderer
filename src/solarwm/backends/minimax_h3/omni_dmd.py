"""Structure-guided DMD2 few-step distillation on omni samples, from a released four-step LoRA.

The sample is prepared exactly as the ordinary omni recipe prepares it (structure packed beside the
target, first frame, prompt); every role - student, teacher (real score), critic (fake score) - sees
the same conditions and differs only in its weights, so each runs on its own model copy through
``_forward(model=...)``. The student samples as the public four-step LoRAs (LongLive-Plug, DMAD) were
distilled: a fresh-noise (re-noised) Euler rollout over ``forwards`` evaluations of the 12-shifted grid, with one
randomly chosen exit evaluation carrying the graph (DMD2 backward simulation, Self-Forcing / the
LongLive-Plug reference). The score models see that sample re-noised at a shifted, clamped level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .flow import make_shifted_schedule
from .layout import patchify_video, unpatchify_video
from .stage0p5 import H3Stage0p5Core, _Prepared
from .torch_flow import shift_sigma


@dataclass
class H3DMDInputs:
    prepared: _Prepared
    audio_clean: Any
    frames: int
    grid: tuple[int, int]


def _is_structure(batch: Any, item: Any) -> bool:
    """Whether an observation is the structure video: laid over every target frame. Kept here rather than imported, so
    the distillation code reads the same with the code directory its runs were launched from."""

    frames = int(batch.target_latents.shape[1])
    return (
        item.frames is not None
        and int(item.latents.shape[1]) == len(item.frames)
        and set(range(frames)) <= {int(f) for f in item.frames}
    )


def structure_guided(real: Any, free: Any, guidance: float) -> Any:
    """The teacher's clean answer pushed along the structure direction: ``real + (w - 1) * (real - free)``,
    ``free`` the teacher's answer with the structure video taken out. w = 1 is the plain teacher. The 50-step teacher is
    softer and less faithful to the structure than its own few-step student; matching a teacher guided along what the
    structure adds asks the student for the structure-faithful distribution instead of the teacher's blur."""

    return real + (float(guidance) - 1.0) * (real - free)


def anchor_condition_weight(batch: Any, weights: dict[str, float]) -> float:
    """How much the real-data anchor counts for a sample, by what it was given: structure and first frame
    (``full``), structure only, first frame only, or neither (``text``). A sample that names the target closely gives a
    single right answer to regress to; one with little given has many right answers, and regressing to its mean blurs."""

    aligned = [o for o in getattr(batch, "visual_observations", None) or () if o.frames is not None]
    structure = any(len(o.frames) > 1 for o in aligned)
    first = any(tuple(int(f) for f in o.frames) == (0,) for o in aligned)
    kind = "full" if structure and first else "structure" if structure else "first" if first else "text"
    return float(weights.get(kind, 0.0))


class H3OmniDMDCore(H3Stage0p5Core):
    """Stage0.5 omni conditions with a few-step student rollout and score evaluations."""

    def __init__(
        self,
        model: Any,
        silence_latents: Any,
        device: Any,
        train: Any,
        *,
        geometry: Any = None,
        observations: Any = None,
    ) -> None:
        super().__init__(
            model, silence_latents, device, geometry=geometry, observations=observations
        )
        self.train = train

    def prepare_inputs(self, batch: Any, generator: Any = None) -> H3DMDInputs:
        prepared = self.prepare(
            batch,
            generator,
            lambda draw: self.shifted_timestep(draw, shift=12.0, device=self.device),
        )
        plan = prepared.plan
        # Every target row moves along one schedule; rows held at their own degree have no place.
        if plan is not None and plan.pattern != "shared":
            raise ValueError("DMD distillation needs the shared observation pattern alone")
        frames = self.latent_frames(batch)
        return H3DMDInputs(prepared, self.audio_rows(frames), frames, self.latent_grid(batch))

    def _random(self, shape: tuple[int, ...], generator: Any) -> Any:
        import torch

        value = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        self._broadcast_logical_sample(value)
        return value

    def _video_forward(
        self,
        model: Any,
        inputs: H3DMDInputs,
        sample: Any,
        video_t: Any,
        audio_t: Any,
        audio_noise: Any,
    ) -> Any:
        import torch

        p = inputs.prepared
        rows = self._forward(
            video_rows=torch.cat((*p.condition_rows, patchify_video(sample)), dim=1),
            audio_rows=audio_t * inputs.audio_clean + (1.0 - audio_t) * audio_noise,
            prompt=p.prompt,
            layout=p.layout,
            video_t=video_t,
            audio_t=audio_t,
            viewmats=p.viewmats,
            K=p.K,
            frame_ids=p.frame_ids,
            geometry_control=p.geometry,
            observation_degrees=p.observation_degrees,
            model=model,
        )
        return unpatchify_video(rows, inputs.frames, *inputs.grid, channels=24).float()

    def rollout(
        self,
        inputs: H3DMDInputs,
        *,
        model: Any,
        forwards: int,
        exit_index: int,
        generator: Any = None,
    ) -> Any:
        """The student's clean estimate at evaluation ``exit_index`` (0-based), with its graph.

        Evaluations before the exit run without a graph; between evaluations the clean estimate is
        noised afresh to the next level, as the four-step LoRA samples (``generate(renoise=True)``).
        """

        import torch

        if not 0 <= int(exit_index) < int(forwards):
            raise ValueError(f"exit_index {exit_index} outside {forwards} evaluations")
        video = make_shifted_schedule(int(forwards) + 1, shift=12.0)
        audio = make_shifted_schedule(int(forwards) + 1, shift=3.0)
        if len(video.timesteps) != int(forwards) or len(audio.timesteps) != int(forwards):
            raise RuntimeError("the shifted grid collapsed points; too many evaluations")
        shape = (1, 24, inputs.frames, *inputs.grid)
        audio_noise = self._random(tuple(inputs.audio_clean.shape), generator)
        current = self._random(shape, generator)
        for index in range(int(exit_index) + 1):
            video_t = torch.tensor(float(video.timesteps[index]), device=self.device)
            audio_t = torch.tensor(float(audio.timesteps[index]), device=self.device)
            sigma = float(video.sigmas[index])
            with torch.set_grad_enabled(index == int(exit_index) and torch.is_grad_enabled()):
                velocity = self._video_forward(
                    model, inputs, current, video_t, audio_t, audio_noise
                )
                denoised = current + sigma * velocity
            if index == int(exit_index):
                return denoised
            sigma_next = float(video.sigmas[index + 1])
            current = (1.0 - sigma_next) * denoised.detach() + sigma_next * self._random(
                shape, generator
            )
        raise AssertionError("unreachable")

    def score_inputs(self, clean: Any, generator: Any = None) -> tuple[Any, Any, Any, Any]:
        """``clean`` noised at one level: uniform, H3-shifted (video 12 / audio 3), clamped."""

        import torch

        raw = torch.rand((), generator=generator, device=self.device, dtype=torch.float32)
        self._broadcast_logical_sample(raw)
        shift = float(self.train.get("score_shift", 12.0))
        sigma = shift_sigma(raw, shift).clamp(0.02, 0.98)
        audio_t = 1.0 - shift_sigma(raw, 3.0).clamp(0.02, 0.98)
        noise = self._random(tuple(clean.shape), generator)
        noisy = (1.0 - sigma) * clean + sigma * noise
        return noisy, noise, 1.0 - sigma, audio_t

    def score_forward(
        self,
        model: Any,
        inputs: H3DMDInputs,
        noisy: Any,
        video_t: Any,
        audio_t: Any,
        audio_noise: Any = None,
    ) -> Any:
        """A score model's data-ward velocity at ``noisy`` (the H3 sign: clean - noise)."""

        if audio_noise is None:
            audio_noise = self._random(tuple(inputs.audio_clean.shape), None)
        return self._video_forward(
            model, inputs, noisy, video_t.reshape(()), audio_t.reshape(()), audio_noise
        )

    def anchor_loss(
        self,
        inputs: H3DMDInputs,
        *,
        model: Any,
        forwards: int,
        index: int,
        generator: Any = None,
    ) -> tuple[Any, float]:
        """The real-data anchor on the few-step path: the sample's real target noised to the student's
        level ``index`` of its ``forwards``-evaluation grid, the student's clean estimate from there, and its squared
        error to the real target, weighted by (1 - sigma) - nearly nothing at the noisiest level, where the only
        regression answer is a blurred mean. Returns (loss, sigma). The DMD student sees no real data otherwise."""

        import torch

        video = make_shifted_schedule(int(forwards) + 1, shift=12.0)
        audio = make_shifted_schedule(int(forwards) + 1, shift=3.0)
        clean = inputs.prepared.clean.float()
        sigma = float(video.sigmas[int(index)])
        video_t = torch.tensor(float(video.timesteps[int(index)]), device=self.device)
        audio_t = torch.tensor(float(audio.timesteps[int(index)]), device=self.device)
        noise = self._random(tuple(clean.shape), generator)
        audio_noise = self._random(tuple(inputs.audio_clean.shape), generator)
        noisy = (1.0 - sigma) * clean + sigma * noise
        velocity = self._video_forward(model, inputs, noisy, video_t, audio_t, audio_noise)
        estimate = noisy + sigma * velocity
        return (1.0 - sigma) * (estimate - clean).square().mean(), sigma

    def structure_free(self, inputs: H3DMDInputs, batch: Any) -> H3DMDInputs:
        """The sample's conditions with its structure video taken out - every other observation (the first frame) and
        the prompt kept, observations its plan dropped still dropped: training's own observation dropout, so the
        teacher has met these conditions. A run reading the structure through a control trunk turns the trunk off."""

        import dataclasses

        p = inputs.prepared
        if self.reads_observation:
            geometry = None if p.geometry is None else dataclasses.replace(p.geometry, gain=0.0)
            return dataclasses.replace(inputs, prepared=dataclasses.replace(p, geometry=geometry))
        items = tuple(batch.visual_observations or ())
        planned = None if p.plan is None else p.plan.keeps_observation

        def keep(index: int) -> bool:
            return (planned is None or planned(index)) and not _is_structure(batch, items[index])

        use_anchor = batch.anchor_latents is not None and (p.plan is None or p.plan.use_anchor)
        _prompt, tags = self.prompt(batch, use_anchor=use_anchor)
        specs, degrees, rows = self.observation_rows(batch, None, keep)
        layout = self.layout(tags, anchors=("first",) if use_anchor else (), observations=specs,
                             grid=inputs.grid, frames=inputs.frames)
        parts = list(p.condition_rows[:1]) if use_anchor else []
        if rows is not None:
            parts.append(rows)
        return dataclasses.replace(
            inputs, prepared=dataclasses.replace(p, condition_rows=parts, layout=layout, observation_degrees=degrees)
        )

    def teacher_answer(
        self,
        teacher: Any,
        inputs: H3DMDInputs,
        batch: Any,
        noisy: Any,
        video_t: Any,
        audio_t: Any,
        audio_noise: Any,
        *,
        guidance: float = 1.0,
        sigma_min: float = 0.0,
        quality: float = 0.0,
        quality_sigma_max: float = 1.0,
        negative: Any = None,
    ) -> Any:
        """The teacher's clean answer at ``noisy`` - guided along the structure when ``guidance`` > 1 and the noise
        level is at least ``sigma_min`` (the structure is laid down at high noise). The structure-free
        pass runs whenever ``guidance`` > 1, below the cutoff too: the level is drawn per sequence-parallel group, and
        a group that skipped a forward of the sharded teacher would leave the others waiting in a collective.

        ``quality`` > 0 adds ``quality * (real - low)``, ``low`` the teacher's answer under the fixed
        low-quality prompt ``negative`` (embeds, tags), where the level is below ``quality_sigma_max`` - the details
        are laid down at low noise. That pass too runs at every level."""

        if float(quality) > 0.0 and negative is None:
            raise ValueError("quality guidance needs the low-quality prompt")
        sigma = 1.0 - video_t
        real = noisy + sigma * self.score_forward(teacher, inputs, noisy, video_t, audio_t, audio_noise)
        answer = real
        if float(guidance) != 1.0:
            free_inputs = self.structure_free(inputs, batch)
            free = noisy + sigma * self.score_forward(teacher, free_inputs, noisy, video_t, audio_t, audio_noise)
            answer = structure_guided(real, free, guidance if float(sigma) >= float(sigma_min) else 1.0)
        if float(quality) > 0.0:
            low_inputs = self.low_quality(inputs, batch, negative)
            low = noisy + sigma * self.score_forward(teacher, low_inputs, noisy, video_t, audio_t, audio_noise)
            if float(sigma) < float(quality_sigma_max):
                answer = answer + float(quality) * (real - low)
        return answer

    def low_quality(self, inputs: H3DMDInputs, batch: Any, negative: Any) -> H3DMDInputs:
        """The sample's own conditions under the fixed low-quality prompt ``negative`` (embeds [N,5120], tags [N]):
        the observations and their rows exactly as the sample has them, the layout rebuilt for the other prompt's
        tokens. The H3 ref2va prompt is text alone, so one encoding serves every sample."""

        import dataclasses

        import torch

        from .stage0p5 import trunk_keep

        embeds, tags = negative
        embeds = embeds.to(self.device, dtype=torch.bfloat16).unsqueeze(0)
        tags = tags.to(self.device, dtype=torch.long)
        self._broadcast_logical_sample(tags, embeds)
        p = inputs.prepared
        use_anchor = batch.anchor_latents is not None and (p.plan is None or p.plan.use_anchor)
        keep = None if p.plan is None else p.plan.keeps_observation
        if self.reads_observation:
            keep = trunk_keep(batch)
        specs, _degrees, _rows = self.observation_rows(batch, None, keep)
        layout = self.layout(tags, anchors=("first",) if use_anchor else (), observations=specs,
                             grid=inputs.grid, frames=inputs.frames)
        viewmats, K, frame_ids = p.viewmats, p.K, p.frame_ids
        if frame_ids is not None:
            viewmats, K, frame_ids = self._camera(batch, layout)
            self._broadcast_logical_sample(viewmats, K)
        return dataclasses.replace(
            inputs, prepared=dataclasses.replace(p, prompt=embeds, layout=layout, viewmats=viewmats, K=K,
                                                 frame_ids=frame_ids)
        )

    def flow_matching_loss(self, inputs: H3DMDInputs, model: Any) -> Any:
        """The ordinary omni loss on the sample's own data, on ``model`` (self-distillation)."""

        return self.loss_from_inputs(inputs.prepared, model=model)


__all__ = ["H3DMDInputs", "H3OmniDMDCore", "anchor_condition_weight", "structure_guided"]
