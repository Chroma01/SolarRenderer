"""Torch-free identity and configuration contract for the H3 geometry ControlNet.

The ControlNet reads VAE-encoded renders of the target clip (canonical coordinate
maps, optionally depth/normal and a wider-FOV render) and adds zero-initialised
residuals to the target video rows. Its architecture and parameter names follow
the FastVideo spatial-AI trunk so trained weights transfer between the two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.errors import ConfigurationError

# Fixed concatenation order into ``fusion_proj``; part of the state-dict contract.
# ``observation`` is not a kind of signal. It is the single input of a trunk that reads whatever
# the sample lays over the target frame for frame - a render, a pose or edge video, the source of
# an edit - without being told which, so it never stands next to a named render.
H3_GEOMETRY_MODALITIES = ("depth", "normal", "ccm", "observation")
H3_OBSERVATION_INPUT = "observation"
H3_GEOMETRY_MODULE = "geometry_controlnet"
H3_GEOMETRY_STATE_PREFIX = f"base_model.model.{H3_GEOMETRY_MODULE}."
H3_GEOMETRY_SCHEMA = "solarwm.minimax-h3-geometry-controlnet.v1"
# MiniMax-H3 rotates the leading 2 * 3 * rope_freq_dim channels of every head.
H3_ROTARY_CHANNELS = 96
H3_BACKBONE_LAYERS = 50

_ARCHITECTURE_KEYS = ("layer_stride", "dim", "ffn_dim", "num_heads", "temb_dim")
_ALLOWED_KEYS = frozenset({"enabled", "modalities", "wide", "lora_trainable", *_ARCHITECTURE_KEYS})
_ALLOWED_WIDE_KEYS = frozenset({"enabled", "modalities", "fov_scale"})
# Raw preencode defines the pixel encoding of canonical coordinate maps only.
H3_RAW_GEOMETRY_MODALITIES = ("ccm",)


@dataclass(frozen=True)
class H3GeometryRenders:
    """Which rendered videos a latent release carries, independent of the trunk."""

    modalities: tuple[str, ...]
    wide_modalities: tuple[str, ...] = ()
    wide_fov_scale: float | None = None

    @property
    def wide_enabled(self) -> bool:
        return bool(self.wide_modalities)

    def tensor_keys(self) -> tuple[str, ...]:
        """Artifact tensors, narrow before wide."""

        return tuple(geometry_tensor_key(name) for name in self.modalities) + tuple(
            geometry_tensor_key(name, wide=True) for name in self.wide_modalities
        )


@dataclass(frozen=True)
class H3GeometryProfile:
    """Resolved ControlNet architecture plus the one training-surface flag."""

    modalities: tuple[str, ...]
    layer_stride: int
    dim: int
    ffn_dim: int
    num_heads: int
    temb_dim: int
    wide_modalities: tuple[str, ...] = ()
    wide_fov_scale: float | None = None
    lora_trainable: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def wide_enabled(self) -> bool:
        return bool(self.wide_modalities)

    @property
    def injected_blocks(self) -> tuple[int, ...]:
        return tuple(range(0, H3_BACKBONE_LAYERS, self.layer_stride))

    def architecture(self) -> dict[str, Any]:
        """Portable identity of everything that changes the ControlNet's computation."""

        return {
            "schema": H3_GEOMETRY_SCHEMA,
            "parameter_names": "fastvideo-minimax-h3-geometry-controlnet",
            "modalities": list(self.modalities),
            "layer_stride": self.layer_stride,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "temb_dim": self.temb_dim,
            "wide": (
                {"modalities": list(self.wide_modalities), "fov_scale": self.wide_fov_scale}
                if self.wide_enabled
                else None
            ),
        }

    @property
    def renders(self) -> H3GeometryRenders:
        return H3GeometryRenders(self.modalities, self.wide_modalities, self.wide_fov_scale)

    @property
    def reads_observation(self) -> bool:
        """Whether the trunk reads the sample's aligned observation instead of named renders."""

        return self.modalities == (H3_OBSERVATION_INPUT,)

    def tensor_keys(self) -> tuple[str, ...]:
        """Artifact tensors this profile reads, narrow before wide; none for the observation input."""

        return () if self.reads_observation else self.renders.tensor_keys()


def geometry_tensor_key(modality: str, *, wide: bool = False) -> str:
    if modality == H3_OBSERVATION_INPUT:
        raise ValueError(
            "the observation input is read from the sample's observations, not an artifact tensor"
        )
    if modality not in H3_GEOMETRY_MODALITIES:
        raise ValueError(f"unsupported H3 geometry modality {modality!r}")
    return f"geometry_{modality}_{'wide_' if wide else ''}latents"


def _canonical_modalities(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigurationError(f"{path} must be a non-empty list of {H3_GEOMETRY_MODALITIES}")
    names = [str(item) for item in value]
    if len(set(names)) != len(names):
        raise ConfigurationError(f"{path} contains duplicate modalities")
    unknown = sorted(set(names) - set(H3_GEOMETRY_MODALITIES))
    if unknown:
        raise ConfigurationError(f"{path} has unsupported modalities {unknown}")
    if H3_OBSERVATION_INPUT in names and len(names) > 1:
        raise ConfigurationError(
            f"{path}: {H3_OBSERVATION_INPUT!r} is the trunk's only input - it reads the sample's "
            "aligned observation, whatever that is - and cannot be combined with named renders"
        )
    return tuple(name for name in H3_GEOMETRY_MODALITIES if name in names)


def _positive(block: Mapping[str, Any], key: str, path: str) -> int:
    if key not in block:
        raise ConfigurationError(f"{path}.{key} is required")
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{path}.{key} must be a positive integer")
    return value


def _wide(
    wide: Any, modalities: tuple[str, ...], path: str
) -> tuple[tuple[str, ...], float | None]:
    if wide is None:
        return (), None
    if not isinstance(wide, Mapping):
        raise ConfigurationError(f"{path}.wide must be a mapping")
    unknown = sorted(set(wide) - _ALLOWED_WIDE_KEYS)
    if unknown:
        raise ConfigurationError(f"{path}.wide has unsupported keys {unknown}")
    if not isinstance(wide.get("enabled"), bool):
        raise ConfigurationError(f"{path}.wide.enabled must be true or false")
    if not wide["enabled"]:
        return (), None
    wide_modalities = _canonical_modalities(wide.get("modalities"), f"{path}.wide.modalities")
    if H3_OBSERVATION_INPUT in wide_modalities:
        raise ConfigurationError(f"{path}.wide: the observation input has no wide render")
    missing = sorted(set(wide_modalities) - set(modalities))
    if missing:
        raise ConfigurationError(
            f"{path}.wide.modalities {missing} must be a subset of {path}.modalities: "
            "the wide render is peripheral context for a modality the trunk reads"
        )
    scale = wide.get("fov_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ConfigurationError(f"{path}.wide.fov_scale must be a positive number")
    return wide_modalities, float(scale)


def geometry_renders_from_preencode_config(
    preencode: Mapping[str, Any],
) -> H3GeometryRenders | None:
    """Renders a raw preencode should encode next to the RGB target, if any."""

    path = "preencode.geometry"
    block = preencode.get("geometry")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    unknown = sorted(set(block) - {"modalities", "wide"})
    if unknown:
        raise ConfigurationError(f"{path} has unsupported keys {unknown}")
    modalities = _canonical_modalities(block.get("modalities"), f"{path}.modalities")
    unsupported = sorted(set(modalities) - set(H3_RAW_GEOMETRY_MODALITIES))
    if unsupported:
        raise ConfigurationError(
            f"{path}.modalities {unsupported}: raw preencode defines the pixel encoding of "
            f"{list(H3_RAW_GEOMETRY_MODALITIES)} renders only"
        )
    wide_modalities, wide_fov_scale = _wide(block.get("wide"), modalities, path)
    return H3GeometryRenders(modalities, wide_modalities, wide_fov_scale)


def geometry_member_field(tensor_key: str) -> str:
    """Raw index field naming the render member for one artifact tensor."""

    if not tensor_key.startswith("geometry_") or not tensor_key.endswith("_latents"):
        raise ValueError(f"not an H3 geometry tensor key: {tensor_key!r}")
    return f"{tensor_key.removesuffix('_latents')}_member"


def geometry_profile_from_model_config(model: Mapping[str, Any]) -> H3GeometryProfile | None:
    """Return the enabled geometry profile, ``None`` when the block is absent or off."""

    path = "model.geometry_controlnet"
    block = model.get("geometry_controlnet")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    unknown = sorted(set(block) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigurationError(f"{path} has unsupported keys {unknown}")
    enabled = block.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigurationError(f"{path}.enabled must be true or false")
    if not enabled:
        return None
    modalities = _canonical_modalities(block.get("modalities"), f"{path}.modalities")
    layer_stride = _positive(block, "layer_stride", path)
    if layer_stride > H3_BACKBONE_LAYERS:
        raise ConfigurationError(f"{path}.layer_stride must be at most {H3_BACKBONE_LAYERS}")
    dim = _positive(block, "dim", path)
    ffn_dim = _positive(block, "ffn_dim", path)
    num_heads = _positive(block, "num_heads", path)
    temb_dim = _positive(block, "temb_dim", path)
    if dim % num_heads:
        raise ConfigurationError(f"{path}.dim={dim} must be divisible by num_heads={num_heads}")
    if dim // num_heads < H3_ROTARY_CHANNELS:
        raise ConfigurationError(
            f"{path}: dim / num_heads = {dim // num_heads} is below the "
            f"{H3_ROTARY_CHANNELS} channels H3's rotary embedding rotates per head"
        )
    lora_trainable = block.get("lora_trainable")
    if not isinstance(lora_trainable, bool):
        raise ConfigurationError(f"{path}.lora_trainable must be true or false")
    wide_modalities, wide_fov_scale = _wide(block.get("wide"), modalities, path)
    return H3GeometryProfile(
        modalities=modalities,
        layer_stride=layer_stride,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        temb_dim=temb_dim,
        wide_modalities=wide_modalities,
        wide_fov_scale=wide_fov_scale,
        lora_trainable=lora_trainable,
    )


def validate_geometry_parallelism(profile: H3GeometryProfile, sequence_parallel_size: int) -> None:
    """The Ulysses exchange scatters ControlNet heads just like backbone heads."""

    if profile.num_heads % int(sequence_parallel_size):
        raise ConfigurationError(
            f"model.geometry_controlnet.num_heads={profile.num_heads} must be divisible by "
            f"distributed.sequence_parallel_size={sequence_parallel_size}"
        )


def geometry_checkpoint_extras(profile: H3GeometryProfile | None) -> dict[str, Any]:
    """Checkpoint-contract extras; empty for runs without the ControlNet."""

    if profile is None:
        return {}
    return {
        "geometry_controlnet": {
            "architecture": profile.architecture(),
            "lora_trainable": profile.lora_trainable,
        }
    }


__all__ = [
    "H3_GEOMETRY_MODALITIES",
    "H3_GEOMETRY_MODULE",
    "H3_GEOMETRY_SCHEMA",
    "H3_GEOMETRY_STATE_PREFIX",
    "H3_OBSERVATION_INPUT",
    "H3_RAW_GEOMETRY_MODALITIES",
    "H3_ROTARY_CHANNELS",
    "H3GeometryProfile",
    "H3GeometryRenders",
    "geometry_checkpoint_extras",
    "geometry_member_field",
    "geometry_profile_from_model_config",
    "geometry_renders_from_preencode_config",
    "geometry_tensor_key",
    "validate_geometry_parallelism",
]
