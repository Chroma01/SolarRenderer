"""Weights-only transitions between published H3 stages.

Supports SolarWM checkpoints and PEFT safetensors sidecars with a model.pt
configuration. Only wrapper/named-adapter prefixes are translated; optimizer,
RNG, scheduler, step and EMA history are never inherited here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError


def canonical_lora_key(key: str) -> str:
    parts = (
        part
        for part in key.split(".")
        if part
        not in {
            "_fsdp_wrapped_module",
            "_checkpoint_wrapped_module",
        }
    )
    return (
        ".".join(parts)
        .replace(".lora_A.default.weight", ".lora_A.weight")
        .replace(".lora_B.default.weight", ".lora_B.weight")
    )


def load_initial_weights(spec: Mapping[str, Any], lora: Any) -> str:
    """Validate the stage/camera/geometry contract before copying any tensor."""
    import torch

    root = Path(str(spec["path"]))
    if root.is_file():
        root = root.parent
    stage = str(spec["stage"])
    source = str(spec["weight_source"])
    if stage not in {"stage0p5", "stage1", "stage2"} or source not in {"live", "ema"}:
        raise BackendContractError("unsupported H3 initialization stage or weight source")
    if not (root / "COMPLETE.json").is_file():
        raise BackendContractError(f"H3 initialization checkpoint is incomplete: {root}")
    geometry_policy = str(spec.get("geometry_controlnet", "required"))
    if geometry_policy not in {"required", "fresh"}:
        raise BackendContractError(
            "H3 initialization geometry_controlnet must be required or fresh"
        )
    source_geometry = None
    manifest_path = root / "checkpoint-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("contract", {})
        required = {
            "family": "minimax_h3",
            "stage": stage,
            "camera_translation_transform": "logd4",
            "parameterization": "peft-lora-r384-alpha384",
        }
        profile = contract.get("extras", {}).get("encoder_profile", {})
        if any(contract.get(key) != value for key, value in required.items()) or (
            profile.get("pixel_frames"),
            profile.get("height"),
            profile.get("width"),
        ) != (158, 768, 1344):
            raise BackendContractError("H3 initialization stage/camera/encoder contract differs")
        if stage in {"stage1", "stage2"}:
            expected = {
                "chunk_latents": 5,
                "window_chunks": 6,
                "target_latents": 45 if stage == "stage1" else 47,
                "rollout_latents": 50,
                "student_rope_mode": "native_absolute" if stage == "stage1" else "sliding_local",
            }
            if any(contract.get("extras", {}).get(key) != value for key, value in expected.items()):
                raise BackendContractError("H3 initialization causal window/RoPE contract differs")
        source_geometry = contract.get("extras", {}).get("geometry_controlnet")
        component = "ema.pt" if source == "ema" else "adapter.pt"
        payload = torch.load(root / component, map_location="cpu", mmap=True, weights_only=True)
        if source == "ema":
            if payload is None or payload.get("schema") != "solarwm.minimax-h3-ema.v1":
                raise BackendContractError("H3 source checkpoint has no compatible EMA")
            values = payload["shadow"]
            frozen_lora = (
                isinstance(source_geometry, Mapping)
                and source_geometry.get("lora_trainable") is False
            )
            if frozen_lora:
                # A frozen-LoRA run's EMA tracked only the trained ControlNet; its
                # LoRA never moved, so the live adapter value is the EMA value.
                adapter = torch.load(
                    root / "adapter.pt", map_location="cpu", mmap=True, weights_only=True
                )
                shadowed = {canonical_lora_key(key) for key in values}
                values = {
                    **{
                        key: value
                        for key, value in adapter["state"].items()
                        if canonical_lora_key(key) not in shadowed
                    },
                    **values,
                }
        else:
            values = payload["state"]
        step = int(manifest["step"])
    else:
        # PEFT sidecars use model.pt for the configuration contract.
        payload = torch.load(root / "model.pt", map_location="cpu", mmap=True, weights_only=True)
        contract = payload.get("config", {}).get("h3_runtime_contract", {})
        expected_stage = {"stage0p5": "stage0p5", "stage1": "stage1_tf", "stage2": "stage2_sgf"}[
            stage
        ]
        required = {
            "model_family": "minimax_h3",
            "stage": expected_stage,
            "pixel_frames": 158,
            "encoded_latents": 47,
            "target_latents": 45 if stage == "stage1" else 47,
            "target_height": 768,
            "target_width": 1344,
            "video_timestep_shift": 12.0,
            "audio_timestep_shift": 3.0,
            "keyframe_noise_aug": 0.999,
        }
        if any(contract.get(key) != value for key, value in required.items()):
            raise BackendContractError("H3 PEFT checkpoint contract differs")
        camera = contract.get("camera_prope", {})
        if camera.get("relative_translation_transform") != "logd4":
            raise BackendContractError("H3 PEFT checkpoint must use logd4 camera conditioning")
        if stage == "stage1" and any(
            contract.get(key) != value
            for key, value in {
                "chunk_latents": 5,
                "window_chunks": 6,
                "flow_objective": "anyflow_forward_map",
            }.items()
        ):
            raise BackendContractError("H3 Stage1 initialization requires AnyFlow W6")
        step = int(payload.get("global_step", payload.get("step", -1)))
        del payload
        from safetensors.torch import load_file

        component = (
            "adapter_model_ema.safetensors" if source == "ema" else "adapter_model.safetensors"
        )
        values = load_file(str(root / component), device="cpu")
    translated = {canonical_lora_key(key): value for key, value in values.items()}
    expected = {
        key: parameter
        for key, parameter in lora.parameter_by_key.items()
        if key in _required_initialization_keys(lora, source_geometry, geometry_policy)
    }
    if len(translated) != len(values) or set(translated) != set(expected):
        raise BackendContractError("H3 initialization LoRA tensor keys differ")
    for key, parameter in expected.items():
        value = translated[key]
        if value.shape != parameter.shape or value.dtype not in {torch.float32, torch.bfloat16}:
            raise BackendContractError(f"H3 initialization tensor descriptor differs: {key}")
    with torch.no_grad():
        for key, parameter in expected.items():
            parameter.copy_(translated[key].to(device=parameter.device, dtype=parameter.dtype))
    return f"{stage}:{source}:step={step}"


def load_geometry_controlnet_weights(path: str, lora: Any) -> str:
    """Overlay a converted ``geometry_controlnet.*`` file onto the adapter set.

    Used to start a run from a ControlNet trained elsewhere (FastVideo); the key
    set and every shape must match the run's trunk exactly.
    """

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from .controlnet_contract import H3_GEOMETRY_MODULE, H3_GEOMETRY_STATE_PREFIX
    from .fastvideo_import import H3_GEOMETRY_WEIGHTS_SCHEMA

    if getattr(lora, "geometry_profile", None) is None:
        raise BackendContractError("geometry ControlNet weights need a run that builds the trunk")
    with safe_open(str(path), framework="pt") as handle:
        schema = (handle.metadata() or {}).get("schema")
    if schema != H3_GEOMETRY_WEIGHTS_SCHEMA:
        raise BackendContractError(f"{path} is not a SolarWM geometry ControlNet weights file")
    prefix = f"{H3_GEOMETRY_MODULE}."
    values = {
        f"{H3_GEOMETRY_STATE_PREFIX}{key[len(prefix) :]}": value
        for key, value in load_file(str(path), device="cpu").items()
        if key.startswith(prefix)
    }
    expected = set(lora.geometry_keys)
    if set(values) != expected:
        raise BackendContractError(
            "geometry ControlNet weights differ from the run's trunk: "
            f"missing={sorted(expected - set(values))[:8]} "
            f"extra={sorted(set(values) - expected)[:8]}"
        )
    for key, value in values.items():
        if tuple(value.shape) != tuple(lora.parameter_by_key[key].shape):
            raise BackendContractError(f"geometry ControlNet weight shape differs for {key!r}")
    with torch.no_grad():
        for key, value in values.items():
            parameter = lora.parameter_by_key[key]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return f"geometry_controlnet:{Path(path).name}"


def _required_initialization_keys(lora: Any, source_geometry: Any, policy: str) -> set[str]:
    """Keys a source must provide, after matching geometry ControlNet identities.

    ``fresh`` is the one way to seed a ControlNet run from a checkpoint that has
    none: the trunk keeps its zero-initialised (exact no-op) state.
    """

    keys = set(lora.parameter_by_key)
    profile = getattr(lora, "geometry_profile", None)
    if profile is None:
        if source_geometry is not None:
            raise BackendContractError(
                "H3 initialization carries a geometry ControlNet this run does not build"
            )
        return keys
    if source_geometry is None:
        if policy != "fresh":
            raise BackendContractError(
                "H3 initialization source has no geometry ControlNet; set "
                "geometry_controlnet: fresh to start the trunk from its no-op initialisation"
            )
        return keys - set(lora.geometry_keys)
    if not isinstance(source_geometry, Mapping) or (
        source_geometry.get("architecture") != profile.architecture()
    ):
        raise BackendContractError("H3 initialization geometry ControlNet architecture differs")
    return keys
