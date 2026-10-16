"""Strict validation for the released H3 Stage0.5, AnyFlow and SGF profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.data import resolve_index_path
from solarwm.errors import ConfigurationError

from .camera import h3_fused_prope_contract
from .codec import H3_PREENCODE_VERSION
from .controlnet_contract import (
    geometry_profile_from_model_config,
    geometry_renders_from_preencode_config,
    validate_geometry_parallelism,
)
from .geometry import validate_stage0p5_geometry


@dataclass(frozen=True)
class H3RunContract:
    """Resolved fields that identify the supported H3 configuration."""

    action: str
    stage: str
    pixel_frames: int
    encoded_latents: int
    sequence_parallel_size: int
    adapter_rank: int
    camera_translation_transform: str
    data_input_mode: str


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{path}.{key} is required")
    return mapping[key]


def _equal(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    observed = _required(mapping, key, path)
    if isinstance(expected, bool):
        matches = observed is expected
    elif isinstance(expected, int):
        matches = (
            isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        )
    elif isinstance(expected, float):
        matches = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) == expected
        )
    else:
        matches = observed == expected
    if not matches:
        raise ConfigurationError(f"{path}.{key} must be {expected!r}, got {observed!r}")


def _positive_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{path}.{key} must be a positive integer")
    return value


def _number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be numeric")
    return float(value)


def _close(mapping: Mapping[str, Any], key: str, expected: float, path: str) -> None:
    observed = _number(mapping, key, path)
    if abs(observed - expected) > 1e-12:
        raise ConfigurationError(f"{path}.{key} must be {expected}, got {observed}")


def _nonempty_path(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty path")
    return value


def _validate_model(
    model: Mapping[str, Any], *, action: str, input_mode: str, stage: str = "stage0p5"
) -> None:
    if "checkpoint_digest" in model:
        raise ConfigurationError(
            "model does not support removed content-digest fields: ['checkpoint_digest']"
        )
    family = str(_required(model, "family", "model")).strip().lower()
    if family not in {"minimax_h3", "minimax-h3"}:
        raise ConfigurationError(f"model.family must select MiniMax-H3, got {family!r}")
    _nonempty_path(model, "checkpoint_path", "model")
    if action == "preencode":
        codec_identity = str(_required(model, "codec_identity", "model")).strip()
        if not codec_identity:
            raise ConfigurationError("model.codec_identity must be non-empty")
    for key, expected in (
        ("architecture", "minimax-h3-33b"),
        ("torch_dtype", "bfloat16"),
        ("camera_attention_mode", "fused_prope"),
        ("camera_translation_transform", "logd4"),
    ):
        _equal(model, key, expected, "model")
    intrinsics_mode = str(model.get("camera_intrinsics_mode", "wan_fixed"))
    if intrinsics_mode not in {"wan_fixed", "sample"}:
        raise ConfigurationError("model.camera_intrinsics_mode must be wan_fixed or sample")
    for key, expected in (
        ("attention_head_dim", 128),
        ("camera_prope_head_dim_start", 96),
        ("camera_prope_head_dim_end", 128),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("rows_per_latent", 1008),
        ("num_frames_per_block", 5),
        ("max_prior_clean_chunks", 5),
    ):
        _equal(model, key, expected, "model")
    if action == "preencode":
        return
    for key, expected in (
        ("training_mode", "lora"),
        ("transformer_subfolder", "transformer"),
        ("transformer_device_map", None),
        ("load_conditioners", False),
    ):
        _equal(model, key, expected, "model")
    # Stage1/2 use the compiled block-mask path. Stage0.5's unmasked attention may also run on cuDNN,
    # which is the faster kernel on recent GPUs at these sequence lengths (same maths, bf16 rounding apart).
    backends = ("flex",) if stage in {"stage1", "stage2"} else ("flash", "_native_cudnn")
    if model.get("attention_backend") not in backends:
        raise ConfigurationError(
            f"model.attention_backend must be one of {backends!r}, got {model.get('attention_backend')!r}"
        )
    adapter = _mapping(model, "adapter")
    for key, expected in (
        ("type", "lora"),
        ("target", "block_qkvo_ffn"),
        ("rank", 384),
        ("alpha", 384),
        ("dropout", 0.0),
        ("bias", "none"),
        ("dtype", "bfloat16"),
        ("expected_target_linear_modules", 312),
        ("expected_trainable_parameters", 2_075_394_048),
    ):
        _equal(adapter, key, expected, "model.adapter")


def _validate_data(data: Mapping[str, Any], *, action: str, stage: str = "stage0p5") -> str:
    input_mode = str(_required(data, "input_mode", "data")).strip().lower()
    allowed = {"raw"} if action == "preencode" else {"preencoded"}
    if input_mode not in allowed:
        raise ConfigurationError(
            f"MiniMax-H3 {action} data.input_mode must be one of {sorted(allowed)!r}; "
            f"got {input_mode!r}"
        )
    transport = _mapping(data, "transport")
    kind = str(_required(transport, "kind", "data.transport")).strip().lower()
    if kind != "local":
        raise ConfigurationError("data.transport.kind must be local")
    root = _nonempty_path(transport, "root", "data.transport")
    if not root.startswith("/"):
        raise ConfigurationError("local data.transport.root must be absolute")
    if data.get("index_root") is not None:
        index_root = _nonempty_path(data, "index_root", "data")
        if not index_root.startswith("/"):
            raise ConfigurationError("data.index_root must be absolute")
    index_fields = ("index",) if action == "preencode" else ("train_index", "test_index")
    for field in index_fields:
        _nonempty_path(data, field, "data")
        try:
            resolve_index_path(data, field)
        except Exception as exc:
            raise ConfigurationError(f"cannot resolve data.{field}: {exc}") from exc
    for key, expected in (
        ("pixel_frames", 158),
        ("encoded_latents", 47),
        ("train_target_latents", 45 if stage == "stage1" else 47),
        ("height", 768),
        ("width", 1344),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("frame_sampling", "contiguous"),
        ("source_fps_policy", "audit_only"),
        ("random_start", False),
        ("resolution_label", "default"),
    ):
        _equal(data, key, expected, "data")
    _close(data, "max_relative_translation", 20.0, "data")
    _close(data, "max_camera_absolute_value", 20.0, "data")
    validate_stage0p5_geometry(
        pixel_frames=int(data["pixel_frames"]),
        encoded_latents=int(data["encoded_latents"]),
        height=int(data["height"]),
        width=int(data["width"]),
        latent_channels=int(data["latent_channels"]),
        latent_height=int(data["latent_height"]),
        latent_width=int(data["latent_width"]),
    )
    if action == "train":
        workers = data.get("num_workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigurationError("data.num_workers must be a positive integer")
        if data.get("prefetch_factor") is not None:
            prefetch = data["prefetch_factor"]
            if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch < 1:
                raise ConfigurationError("data.prefetch_factor must be a positive integer")
    if "omni_index" in data or "omni_root" in data:
        # Omni samples (tools/omni_preencode.py): any 5n+2 clip length, untyped observations.
        if action != "train" or stage != "stage0p5":
            raise ConfigurationError("data.omni_index applies to Stage0.5 training")
        for key in ("omni_index", "omni_root"):
            location = _nonempty_path(data, key, "data")
            if not (location.startswith("/") or "://" in location):
                raise ConfigurationError(f"data.{key} must be absolute or a remote storage location")
        if not isinstance(data.get("omni_length_buckets", False), bool):
            raise ConfigurationError("data.omni_length_buckets must be true or false")
    if input_mode == "preencoded":
        _equal(data, "preencode_version", H3_PREENCODE_VERSION, "data")
        _equal(data, "dataset_name", "h3_preencoded_wds", "data")
        _nonempty_path(data, "silence_latents_path", "data")
        _nonempty_path(data, "encoder_contract_path", "data")
    else:
        _equal(data, "fixed_start_from_index", True, "data")
    return input_mode


def _validate_geometry(config: Mapping[str, Any], *, action: str, stage: str) -> None:
    """Validate the optional geometry ControlNet block and everything it implies."""

    if action == "preencode":
        if _mapping(config, "model").get("geometry_controlnet") is not None:
            raise ConfigurationError(
                "model.geometry_controlnet does not apply to preencode; select renders with "
                "preencode.geometry"
            )
        geometry_renders_from_preencode_config(_mapping(config, "preencode"))
        return
    profile = geometry_profile_from_model_config(_mapping(config, "model"))
    if profile is not None and profile.reads_observation:
        if not _mapping(config, "data").get("omni_index"):
            raise ConfigurationError(
                "model.geometry_controlnet.modalities [observation] reads each sample's aligned "
                "observation and needs omni samples (data.omni_index)"
            )
    initialization = config.get("checkpoint", {}).get("initialization") or {}
    if isinstance(initialization, Mapping) and "geometry_controlnet_weights" in initialization:
        if profile is None or stage != "stage0p5":
            raise ConfigurationError(
                "checkpoint.initialization.geometry_controlnet_weights seeds a Stage0.5 run "
                "with model.geometry_controlnet.enabled"
            )
        _nonempty_path(initialization, "geometry_controlnet_weights", "checkpoint.initialization")
    if isinstance(initialization, Mapping):
        for role, item in initialization.items():
            if isinstance(item, Mapping) and "geometry_controlnet" in item:
                if profile is None:
                    raise ConfigurationError(
                        f"checkpoint.initialization.{role}.geometry_controlnet requires "
                        "model.geometry_controlnet.enabled"
                    )
                if item["geometry_controlnet"] not in {"required", "fresh"}:
                    raise ConfigurationError(
                        f"checkpoint.initialization.{role}.geometry_controlnet must be "
                        "required or fresh"
                    )
    if profile is None:
        for key in ("geometry_controlnet_optimizer", "critic_geometry_controlnet_optimizer"):
            if key in config.get("train", {}):
                raise ConfigurationError(f"train.{key} requires model.geometry_controlnet.enabled")
        return
    validate_geometry_parallelism(
        profile, int(_mapping(config, "distributed")["sequence_parallel_size"])
    )
    if action != "train":
        return
    train = _mapping(config, "train")
    for key in (
        ("geometry_controlnet_optimizer", "critic_geometry_controlnet_optimizer")
        if stage == "stage2"
        else ("geometry_controlnet_optimizer",)
    ):
        optimizer = _mapping(train, key)
        unknown = sorted(set(optimizer) - {"learning_rate", "weight_decay"})
        if unknown:
            raise ConfigurationError(f"train.{key} has unsupported keys {unknown}")
        if _number(optimizer, "learning_rate", f"train.{key}") <= 0:
            raise ConfigurationError(f"train.{key}.learning_rate must be positive")
        if _number(optimizer, "weight_decay", f"train.{key}") < 0:
            raise ConfigurationError(f"train.{key}.weight_decay must be non-negative")
    if stage == "stage0p5" and isinstance(initialization, Mapping):
        item = initialization.get("student")
        if item is not None:
            if not isinstance(item, Mapping):
                raise ConfigurationError("checkpoint.initialization.student must be a mapping")
            _nonempty_path(item, "path", "checkpoint.initialization.student")
            _equal(item, "stage", "stage0p5", "checkpoint.initialization.student")
            if item.get("weight_source") not in {"live", "ema"}:
                raise ConfigurationError(
                    "checkpoint.initialization.student.weight_source must be live or ema"
                )


def _validate_observations(train: Mapping[str, Any], *, action: str, stage: str) -> None:
    """The optional known-degree training distribution; only Stage0.5 trains with it."""

    if "observations" not in train:
        return
    if action != "train" or stage != "stage0p5":
        raise ConfigurationError("train.observations applies to Stage0.5 training")
    from .observations import observation_profile_from_train_config

    observation_profile_from_train_config(train)


def _validate_flow_map(config: Mapping[str, Any]) -> None:
    """A Stage0.5 flow-map run distils a trained omni run: it starts from that run's weights."""

    item = (config.get("checkpoint", {}).get("initialization") or {}).get("student")
    if not isinstance(item, Mapping):
        raise ConfigurationError(
            "a Stage0.5 flow-map run starts from a trained omni checkpoint "
            "(checkpoint.initialization.student)"
        )
    _nonempty_path(item, "path", "checkpoint.initialization.student")
    _equal(item, "stage", "stage0p5", "checkpoint.initialization.student")
    if item.get("weight_source") not in {"live", "ema"}:
        raise ConfigurationError("checkpoint.initialization.student.weight_source must be live or ema")
    if not _mapping(config, "data").get("omni_index"):
        raise ConfigurationError("a Stage0.5 flow-map run distils omni samples (data.omni_index)")


def _validate_route(train: Mapping[str, Any]) -> None:
    from solarwm.config.routes import Route, supported_routes

    route = Route(
        "minimax_h3",
        str(train.get("stage", "")),
        str(train.get("causal_mode", "")),
        str(train.get("objective", "")),
        str(train.get("objective_variant", "")),
    )
    if route not in supported_routes():
        raise ConfigurationError(f"unsupported H3 training route {route.key}")
    if route.objective == "anyflow_forward_map":
        for key, expected in (
            ("anyflow_gate", 0.25),
            ("deltatime_type", "r"),
            ("finite_difference_epsilon", 5.0),
            ("diffusion_ratio", 0.5),
            ("consistency_ratio", 0.25),
            ("weight_type", "gaussian"),
        ):
            _equal(train, key, expected, "train")
    elif route.stage == "stage2":
        for key, expected in (
            ("critic_updates_per_student", 5),
            ("num_denoising_steps", 4),
            ("per_rank_exit_step", True),
            ("last_step_only", False),
            ("match_context", True),
            ("cache_mode", "exit"),
            ("tail_camera_policy", "repeat_last_latent"),
            ("audio_condition_policy", "fixed_noised_silence_per_rollout"),
            ("score_min_sigma", 0.02),
            ("score_max_sigma", 0.98),
        ):
            _equal(train, key, expected, "train")


_DMD_RECIPES = {
    # The LongLive-Plug paper's Wan2.2 DMD optimisation: the critic at 2e-6 with the student's betas.
    "wan22": (
        {"learning_rate": 1e-5, "betas": [0.0, 0.999], "weight_decay": 0.0, "gradient_clip": 10.0},
        {"learning_rate": 2e-6, "betas": [0.0, 0.999], "weight_decay": 0.0, "gradient_clip": 10.0},
    ),
    # What LongLive-Plug's own H3 release trained with (its training_config.json): a 25x faster critic
    # with its own betas, weight decay on the student, clip 1.
    "plug_h3": (
        {"learning_rate": 2e-5, "betas": [0.0, 0.9], "weight_decay": 0.01, "gradient_clip": 1.0},
        {"learning_rate": 5e-5, "betas": [0.9, 0.999], "weight_decay": 0.0, "gradient_clip": 1.0},
    ),
}


# train.dmd keys of the released four-step LoRA the student starts from (DMAD in the release); the older
# plug* names are still read.
_DMD_ALIASES = (("init_lora", "plug"), ("init_lora_rank", "plug_rank"), ("init_lora_alpha", "plug_alpha"),
                ("seed_init_lora", "seed_plug"))


def dmd_settings(dmd: Mapping[str, Any]) -> dict[str, Any]:
    """train.dmd with each init_lora* key also under its plug* name, which the runtime reads."""

    out = dict(dmd)
    for new, old in _DMD_ALIASES:
        if new in out:
            if old in out and out[old] != out[new]:
                raise ConfigurationError(f"train.dmd.{new} and train.dmd.{old} name the same setting; give one")
            out[old] = out[new]
    return out


def _validate_dmd(train: Mapping[str, Any], optimizer: Mapping[str, Any]) -> None:
    """Structure-guided DMD2 from a released four-step LoRA; train.dmd.recipe names its optimisation (wan22 by
    default)."""

    recipe = str(_mapping(train, "dmd").get("recipe", "wan22"))
    if recipe not in _DMD_RECIPES:
        raise ConfigurationError(f"train.dmd.recipe must be one of {sorted(_DMD_RECIPES)}")
    student, critic_values = _DMD_RECIPES[recipe]
    shared = (("name", "fp32_master_adamw"), ("warmup_steps", 0), ("min_lr_ratio", 1.0))
    for key, expected in (*shared, *student.items()):
        _equal(optimizer, key, expected, "train.optimizer")
    critic = _mapping(train, "critic_optimizer")
    for key, expected in (*shared, *critic_values.items()):
        _equal(critic, key, expected, "train.critic_optimizer")
    if critic.get("epsilon") != optimizer.get("epsilon"):
        raise ConfigurationError("train.critic_optimizer.epsilon must equal train.optimizer.epsilon")
    dmd = dmd_settings(_mapping(train, "dmd"))
    if dmd.get("mode") not in {"adapt_plug", "self_distill", "joint", "causal"}:
        raise ConfigurationError("train.dmd.mode must be adapt_plug, self_distill, joint or causal")
    if dmd["mode"] == "causal":
        # A block-causal student, streamed chunk by chunk and replayed: the causal control LoRA (student_init) frozen in
        # its base under a separate acceleration LoRA (decoupled) that starts as the released four-step LoRA of
        # accel_init; the bidirectional control LoRA (control) teaches.
        if not dmd.get("student_init") or not dmd.get("control"):
            raise ConfigurationError("a causal DMD names its causal control (student_init) and its teacher (control)")
        if dmd.get("decoupled") is not True:
            raise ConfigurationError("a causal DMD trains a separate acceleration LoRA: train.dmd.decoupled must be true")
        stream = dmd.get("causal_stream")
        if not isinstance(stream, Mapping) or set(stream) - {"chunk", "window"}:
            raise ConfigurationError("train.dmd.causal_stream holds chunk and window")
        from .omni_causal import validate_causal_omni

        validate_causal_omni(dict(stream))
    elif not dmd.get("plug"):
        raise ConfigurationError("train.dmd.init_lora names the released four-step LoRA the student starts from")
    if str(dmd.get("teacher", "fixed")) not in {"fixed", "follow"}:
        raise ConfigurationError("train.dmd.teacher must be fixed or follow")
    # seed_init_lora false: an adapt_plug student from a zero update instead of the released LoRA
    if not isinstance(dmd.get("seed_plug", True), bool):
        raise ConfigurationError("train.dmd.seed_init_lora must be true or false")
    # PDMD projection and the real-data anchor
    if not isinstance(dmd.get("pdmd", False), bool):
        raise ConfigurationError("train.dmd.pdmd must be true or false")
    if float(dmd.get("anchor_weight", 0.0)) < 0:
        raise ConfigurationError("train.dmd.anchor_weight must not be negative")
    for kind, value in dict(dmd.get("anchor_condition_weights", {})).items():
        if kind not in {"full", "structure", "first", "text"} or float(value) < 0:
            raise ConfigurationError(
                "train.dmd.anchor_condition_weights maps full / structure / first / text to weights >= 0"
            )
    # structure-guided distribution matching: the teacher's answer pushed along the structure direction
    guidance = dmd.get("structure_guidance", 1.0)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or float(guidance) < 1.0:
        raise ConfigurationError("train.dmd.structure_guidance must be a number >= 1 (1 = the plain teacher)")
    warmup = dmd.get("structure_guidance_warmup", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigurationError("train.dmd.structure_guidance_warmup must be a whole number of student steps >= 0")
    # guidance only where the re-noise level is at least this - the structure is laid down at high noise
    cutoff = dmd.get("structure_guidance_sigma_min", 0.0)
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not 0.0 <= float(cutoff) < 1.0:
        raise ConfigurationError("train.dmd.structure_guidance_sigma_min must be a noise level in [0, 1)")
    # quality-guided distribution matching: away from the teacher's answer under a low-quality prompt
    quality = dmd.get("quality_guidance", 0.0)
    if isinstance(quality, bool) or not isinstance(quality, (int, float)) or float(quality) < 0.0:
        raise ConfigurationError("train.dmd.quality_guidance must be a number >= 0 (0 = off)")
    if float(quality) > 0.0 and not isinstance(dmd.get("quality_negative"), str):
        raise ConfigurationError("train.dmd.quality_negative names the low-quality prompt's safetensors")
    ceiling = dmd.get("quality_guidance_sigma_max", 1.0)
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or not 0.0 < float(ceiling) <= 1.0:
        raise ConfigurationError("train.dmd.quality_guidance_sigma_max must be a noise level in (0, 1]")
    # joint may start from the base: its control slots then train from zero.
    if dmd["mode"] == "adapt_plug" and not dmd.get("control"):
        raise ConfigurationError("train.dmd.control names the frozen control checkpoint")


def _validate_training(
    train: Mapping[str, Any],
    distributed: Mapping[str, Any],
) -> None:
    stage = str(train["stage"])
    sgf = stage == "stage2"
    for key, expected in (
        ("precision", "bfloat16"),
        ("micro_batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("video_timestep_shift", 12.0),
        ("audio_timestep_shift", 3.0),
        ("keyframe_noise_augmentation", 0.999),
        ("audio_loss_weight", 0.0),
    ):
        _equal(train, key, expected, "train")
    optimizer = _mapping(train, "optimizer")
    # A flow map trains at the flow map's step size, whichever stage its inputs come from.
    schedule = "stage1" if str(train.get("objective")) == "anyflow_forward_map" else stage
    if str(train.get("objective")) == "dmd":
        _validate_dmd(train, optimizer)
    elif train.get("causal_omni") is not None:
        # block-causal omni: Stage1's teacher-forcing step size, or the 1e-4 the control LoRA itself was
        # trained at, after a short warmup from that LoRA
        for key, expected in (("name", "fp32_master_adamw"), ("warmup_steps", 50)):
            _equal(optimizer, key, expected, "train.optimizer")
        if _required(optimizer, "learning_rate", "train.optimizer") not in (3e-5, 1e-4):
            raise ConfigurationError("train.optimizer.learning_rate of a block-causal run must be 3e-05 or 0.0001")
    else:
        for key, expected in (
            ("name", "fp32_master_adamw"),
            ("learning_rate", {"stage0p5": 1e-4, "stage1": 3e-5, "stage2": 2e-6}[schedule]),
        ):
            _equal(optimizer, key, expected, "train.optimizer")
        # The warmup is counted in optimizer steps, so it scales with the global batch (500 at the paper's batch of
        # 128 is 16,000 at a batch of 4); any non-negative count is accepted.
        warmup = _required(optimizer, "warmup_steps", "train.optimizer")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ConfigurationError(f"train.optimizer.warmup_steps must be a non-negative integer, got {warmup!r}")
    if stage != "stage0p5":
        for key, expected in (
            ("betas", [0.0, 0.999] if sgf else [0.9, 0.95]),
            ("epsilon", 1e-8),
            ("weight_decay", 0.0 if sgf else 0.01),
            ("gradient_clip", 10.0 if sgf else 1.0),
            ("min_lr_ratio", 1.0 if sgf else 0.1),
        ):
            _equal(optimizer, key, expected, "train.optimizer")
    if sgf:
        critic = _mapping(train, "critic_optimizer")
        for key, value in optimizer.items():
            _equal(critic, key, 4e-7 if key == "learning_rate" else value, "train.critic_optimizer")
    fsdp = _mapping(train, "fsdp")
    for key, expected in (
        ("sharding_strategy", "HYBRID_SHARD" if sgf else "FULL_SHARD"),
        ("activation_checkpointing", True),
        ("preserve_checkpoint_dtype", True),
        ("param_dtype", None),
        ("reduce_dtype", "float32"),
        ("buffer_dtype", None),
    ):
        _equal(fsdp, key, expected, "train.fsdp")
    saved = fsdp.get("saved_attention_blocks", 0)
    if isinstance(saved, bool) or not isinstance(saved, int) or not 0 <= saved <= 50:
        raise ConfigurationError(
            f"train.fsdp.saved_attention_blocks must be an integer in [0, 50], got {saved!r}"
        )

    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(distributed, "sequence_parallel_size", "distributed")
    _equal(distributed, "rank_partition", "node_shard", "distributed")
    _equal(distributed, "context_parallel_size", 1, "distributed")
    _equal(distributed, "sp_peers_share_sample", True, "distributed")
    _equal(distributed, "sp_peers_share_rng", True, "distributed")
    if "observations" in train:
        # Visual observations lengthen the packed document; the run sizes SP for its data.
        if sequence_parallel not in (2, 4, 8):
            raise ConfigurationError("H3 observation runs need sequence_parallel_size 2, 4 or 8")
    elif sequence_parallel != (4 if sgf else 2):
        raise ConfigurationError(f"H3 {stage} requires sequence_parallel_size={4 if sgf else 2}")
    if sgf:
        _equal(fsdp, "frozen_base_shard_size", 8, "train.fsdp")
    if world_size % sequence_parallel:
        raise ConfigurationError("distributed.world_size must be divisible by SP size")
    # Each stage fixes SP, micro batch and accumulation. Changing the world size
    # also changes the global batch and optimization trajectory.
    computed_global_batch = (
        world_size
        // sequence_parallel
        * int(train["micro_batch_size"])
        * int(train["gradient_accumulation_steps"])
    )
    if computed_global_batch != int(train["global_batch_size"]):
        raise ConfigurationError(
            "global batch mismatch: (world_size / SP) * micro_batch * grad_accum "
            f"is {computed_global_batch}, configured {train['global_batch_size']}"
        )


def _validate_validation(
    validation: Mapping[str, Any], *, action: str, stage: str = "stage0p5"
) -> None:
    del action
    _positive_int(validation, "sample_count", "validation")
    for name in ("selection_seed", "noise_seed"):
        value = _required(validation, name, "validation")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"validation.{name} must be a non-negative integer")
    for key, expected in (
        ("pixel_frames", 158),
        ("latent_frames", 47),
        ("fps", 24),
        ("num_inference_steps", 30 if stage == "stage0p5" else 4),
        ("passes", ["live", "ema"]),
    ):
        _equal(validation, key, expected, "validation")
    smoke_step = _required(validation, "smoke_step", "validation")
    if isinstance(smoke_step, bool) or not isinstance(smoke_step, int) or smoke_step < 0:
        raise ConfigurationError("validation.smoke_step must be a non-negative integer")
    _close(validation, "max_relative_translation", 20.0, "validation")
    _close(validation, "max_camera_absolute_value", 20.0, "validation")


def _validate_checkpoint(checkpoint: Mapping[str, Any], *, stage: str = "stage0p5") -> None:
    _equal(checkpoint, "save_optimizer", True, "checkpoint")
    ema = _mapping(checkpoint, "ema")
    for key, expected in (
        ("enabled", True),
        ("dtype", "float32"),
        ("sharded", True),
        ("decay", {"stage0p5": 0.9999, "stage1": 0.999, "stage2": 0.99}[stage]),
        ("start_step", 39 if stage == "stage2" else 0),
        ("update_every_steps", 1),
    ):
        _equal(ema, key, expected, "checkpoint.ema")


def _validate_inference_distributed(
    distributed: Mapping[str, Any], *, stage: str = "stage0p5", full_length: bool = False
) -> None:
    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(
        distributed,
        "sequence_parallel_size",
        "distributed",
    )
    expected_sp = 8 if full_length else (4 if stage == "stage2" else 2)
    if full_length and world_size != 8:
        raise ConfigurationError("Source-length H3 inference requires one eight-GPU SP8 worker")
    if sequence_parallel != expected_sp or world_size % sequence_parallel:
        raise ConfigurationError(f"H3 inference requires a world divisible by SP{expected_sp}")
    for key, expected in (
        ("context_parallel_size", 1),
        ("sp_peers_share_sample", True),
        ("sp_peers_share_rng", True),
    ):
        _equal(distributed, key, expected, "distributed")


def validate_h3_config(config: Mapping[str, Any]) -> H3RunContract:
    """Validate one config against the supported H3 profile."""

    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    action = str(config.get("action", "")).strip().lower()
    if action not in {"train", "infer", "preencode"}:
        raise ConfigurationError("action must be train, infer, or preencode")
    inference = config.get("inference", {})
    if not isinstance(inference, Mapping):
        raise ConfigurationError("inference must be a mapping")
    length_policy = inference.get("length_policy", "fixed")
    if length_policy not in {"fixed", "source"}:
        raise ConfigurationError("inference.length_policy must be fixed or source")
    full_length = length_policy == "source"
    if full_length:
        if action != "infer" or config.get("train", {}).get("stage") != "stage2":
            raise ConfigurationError("Source-length inference is only supported for H3 Stage2 SGF")
        for key in ("plan", "work_dir", "dataset_root"):
            _nonempty_path(inference, key, "inference")
        if inference.get("fps_policy") != "source":
            raise ConfigurationError("Source-length inference requires fps_policy=source")
        if inference.get("phase", "all") not in {"all", "prepare", "generate"}:
            raise ConfigurationError("inference.phase must be all, prepare or generate")
        source = config.get("checkpoint", {}).get("weight_source")
        if source not in {"live", "ema"}:
            raise ConfigurationError(
                "Source-length inference requires explicit live or ema weights"
            )
        _positive_int(inference, "expected_step", "inference")
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    metadata = config.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("metadata must be a mapping")
    stage = "preencode" if action == "preencode" else str(_mapping(config, "train")["stage"])
    if stage not in {"preencode", "stage0p5", "stage1", "stage2"}:
        raise ConfigurationError(f"unsupported H3 stage {stage!r}")
    declared_input_mode = str(data.get("input_mode", "")).strip().lower()
    _validate_model(model, action=action, input_mode=declared_input_mode, stage=stage)
    input_mode = _validate_data(data, action=action, stage=stage)
    _validate_geometry(config, action=action, stage=stage)

    if action != "preencode":
        train = _mapping(config, "train")
        _validate_route(train)
        if stage == "stage0p5" and str(train.get("objective")) == "anyflow_forward_map":
            _validate_flow_map(config)
        _validate_observations(train, action=action, stage=stage)
        if train.get("causal_omni") is not None:
            # block-causal omni rendering: an omni Stage0.5 flow-matching run trained chunk by chunk
            if stage != "stage0p5" or str(train.get("objective")) != "flow_matching" or not data.get("omni_index"):
                raise ConfigurationError("train.causal_omni applies to omni Stage0.5 flow-matching training")
            from .omni_causal import validate_causal_omni

            validate_causal_omni(train["causal_omni"])
        if stage in {"stage1", "stage2"}:
            validation = _mapping(config, "validation")
            if validation.get("prepared_plan") is not None:
                _nonempty_path(validation, "prepared_plan", "validation")
            elif stage == "stage1":
                _nonempty_path(data, "raw_test_index", "data")
            for key, expected in (
                ("rollout_latents", 50),
                ("decode_latents", 47),
                ("camera_frames", 170 if stage == "stage1" else 158),
            ):
                _equal(validation, key, expected, "validation")
            if stage == "stage2":
                _equal(model, "student_rope_mode", "sliding_local", "model")
                _equal(model, "score_rope_mode", "native_absolute", "model")
            if action == "train":
                initialization = _mapping(_mapping(config, "checkpoint"), "initialization")
                for role in ("student", "teacher", "critic") if stage == "stage2" else ("student",):
                    item = _mapping(initialization, role)
                    _nonempty_path(item, "path", f"checkpoint.initialization.{role}")
                    _equal(item, "weight_source", "ema", f"checkpoint.initialization.{role}")
                    _equal(
                        item,
                        "stage",
                        "stage1" if stage == "stage2" and role == "student" else "stage0p5",
                        f"checkpoint.initialization.{role}",
                    )

        _validate_validation(_mapping(config, "validation"), action=action, stage=stage)
        if action == "train":
            _validate_training(train, _mapping(config, "distributed"))
            _validate_checkpoint(_mapping(config, "checkpoint"), stage=stage)
        else:
            _validate_inference_distributed(
                _mapping(config, "distributed"), stage=stage, full_length=full_length
            )
            _nonempty_path(_mapping(config, "checkpoint"), "resume_from", "checkpoint")
    else:
        preencode = _mapping(config, "preencode")
        _equal(preencode, "codec_protocol", "solarwm.minimax_h3.codec.v1", "preencode")
        _nonempty_path(preencode, "output_root", "preencode")

    runtime = _mapping(config, "runtime")
    _nonempty_path(runtime, "output_dir", "runtime")
    contract = h3_fused_prope_contract(
        str(_mapping(config, "model").get("camera_intrinsics_mode", "wan_fixed"))
    )
    if contract["camera_prope_head_slice"] != [96, 128]:
        raise ConfigurationError("internal H3 camera suffix contract is inconsistent")
    return H3RunContract(
        action=action,
        stage=stage,
        pixel_frames=158,
        encoded_latents=47,
        sequence_parallel_size=(
            0
            if action == "preencode"
            else int(_mapping(config, "distributed")["sequence_parallel_size"])
        ),
        adapter_rank=0 if action == "preencode" else 384,
        camera_translation_transform="logd4",
        data_input_mode=input_mode,
    )


__all__ = ["H3RunContract", "validate_h3_config"]
