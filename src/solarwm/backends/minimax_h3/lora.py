"""Exact PEFT LoRA-384 target and checkpoint contract for MiniMax-H3."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.errors import BackendContractError

H3_LORA_TARGET_COUNT = 312
H3_LORA_TRAINABLE_PARAMETERS = 2_075_394_048
H3_LORA_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _indexed_prefixes(
    modules: Mapping[str, Any], pattern: str, expected: int, label: str
) -> tuple[str, ...]:
    expression = re.compile(pattern)
    found: dict[int, str] = {}
    for name in modules:
        match = expression.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index in found:
                raise BackendContractError(f"duplicate {label} block index {index}")
            found[index] = name
    required = set(range(expected))
    if set(found) != required:
        raise BackendContractError(
            f"{label} topology differs: missing={sorted(required - set(found))} "
            f"extra={sorted(set(found) - required)}"
        )
    return tuple(found[index] for index in range(expected))


def discover_h3_lora_targets(model: Any) -> tuple[str, ...]:
    """Discover and audit all 50 main + 2 refiner block QKVO/FFN linears."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise BackendContractError("PyTorch is required for H3 LoRA discovery") from exc
    modules = dict(model.named_modules())
    main = _indexed_prefixes(
        modules,
        r"(?:.*\.)?transformer_blocks\.(\d+)",
        50,
        "MiniMax-H3 main transformer",
    )
    refiner = _indexed_prefixes(
        modules,
        r"(?:.*\.)?token_refiner\.refiner_blocks\.(\d+)",
        2,
        "MiniMax-H3 token refiner",
    )
    blocks = main + refiner
    for block in blocks:
        attention = modules.get(f"{block}.attn")
        if attention is None or getattr(attention, "fused_projections", None) is not False:
            raise BackendContractError(f"H3 LoRA requires split Q/K/V projections at {block!r}")
    targets = tuple(sorted(f"{block}.{suffix}" for block in blocks for suffix in H3_LORA_SUFFIXES))
    if len(targets) != H3_LORA_TARGET_COUNT or len(set(targets)) != len(targets):
        raise AssertionError("internal H3 LoRA target-count error")
    invalid = [
        name
        for name in targets
        if name not in modules or not isinstance(modules[name], torch.nn.Linear)
    ]
    if invalid:
        raise BackendContractError(
            f"H3 LoRA expected nn.Linear targets; first invalid={invalid[:8]}"
        )
    return targets


@dataclass
class H3LoRARuntime:
    """Replicated adapter state: the LoRA pairs plus an optional geometry ControlNet.

    ``parameter_by_key`` is the complete saved/FSDP-ignored set. The LoRA fields
    of :meth:`metadata` and :attr:`parameter_count` describe the LoRA alone, so a
    run without the ControlNet keeps its released identity unchanged.
    """

    model: Any
    targets: tuple[str, ...]
    parameter_by_key: OrderedDict[str, Any]
    peft_config: Any
    peft_module: Any
    base_identity: Mapping[str, Any]
    rank: int
    alpha: int
    geometry_keys: tuple[str, ...] = ()
    geometry_profile: Any = None

    @property
    def parameters(self) -> tuple[Any, ...]:
        return tuple(self.parameter_by_key.values())

    @property
    def lora_keys(self) -> tuple[str, ...]:
        geometry = set(self.geometry_keys)
        return tuple(key for key in self.parameter_by_key if key not in geometry)

    @property
    def lora_parameters(self) -> tuple[Any, ...]:
        return tuple(self.parameter_by_key[key] for key in self.lora_keys)

    @property
    def geometry_parameters(self) -> tuple[Any, ...]:
        return tuple(self.parameter_by_key[key] for key in self.geometry_keys)

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        """What the optimizer, gradient sync and EMA own."""

        return tuple(parameter for parameter in self.parameters if parameter.requires_grad)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.lora_parameters)

    def state_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict(
            (key, parameter.detach()) for key, parameter in self.parameter_by_key.items()
        )

    def load_state_dict(self, values: Mapping[str, Any], *, broadcast: bool = True) -> None:
        import torch
        import torch.distributed as dist

        if set(values) != set(self.parameter_by_key):
            raise BackendContractError(
                "H3 LoRA checkpoint keys differ: "
                f"missing={sorted(set(self.parameter_by_key) - set(values))[:8]} "
                f"extra={sorted(set(values) - set(self.parameter_by_key))[:8]}"
            )
        with torch.no_grad():
            for key, parameter in self.parameter_by_key.items():
                value = values[key]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise BackendContractError(f"H3 LoRA shape differs for {key!r}")
                if value.dtype != parameter.dtype:
                    raise BackendContractError(f"H3 LoRA dtype differs for {key!r}")
                parameter.copy_(value.to(device=parameter.device))
                if broadcast and dist.is_available() and dist.is_initialized():
                    dist.broadcast(parameter, src=0)

    @staticmethod
    def _descriptors(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "state_keys": list(values),
            "state_shapes": {key: list(parameter.shape) for key, parameter in values.items()},
            "state_dtypes": {
                key: str(parameter.dtype).removeprefix("torch.")
                for key, parameter in values.items()
            },
        }

    def metadata(self) -> dict[str, Any]:
        lora = OrderedDict((key, self.parameter_by_key[key]) for key in self.lora_keys)
        values = {
            "schema": "solarwm.minimax-h3-lora.v1",
            "peft_version": str(self.peft_module.__version__),
            "rank": self.rank,
            "alpha": self.alpha,
            "target_count": len(self.targets),
            "target_modules": list(self.targets),
            **self._descriptors(lora),
            "trainable_parameters": self.parameter_count,
            "base_identity": json.loads(json.dumps(dict(self.base_identity), sort_keys=True)),
        }
        if self.geometry_profile is not None:
            geometry = OrderedDict((key, self.parameter_by_key[key]) for key in self.geometry_keys)
            values["geometry_controlnet"] = {
                "architecture": self.geometry_profile.architecture(),
                "lora_trainable": bool(self.geometry_profile.lora_trainable),
                "parameters": sum(parameter.numel() for parameter in geometry.values()),
                **self._descriptors(geometry),
            }
        return values


def inject_h3_lora(
    model: Any,
    adapter_cfg: Mapping[str, Any],
    *,
    base_identity: Mapping[str, Any],
    ranks: tuple[int, ...] = (384,),
) -> tuple[Any, H3LoRARuntime]:
    """Inject standard BF16 PEFT LoRA and verify its realized topology/size.

    The run's adapter is rank 384; ``ranks`` admits another for an auxiliary role (a DMD critic
    at rank 128, as LongLive-Plug trains it), whose size is checked at that rank.
    """

    try:
        import peft
        import torch
        import torch.distributed.tensor
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise BackendContractError("H3 LoRA requires peft==0.20.0") from exc

    rank = int(adapter_cfg.get("rank", 0))
    alpha = int(adapter_cfg.get("alpha", 0))
    if rank not in ranks or alpha != rank:
        raise BackendContractError(f"H3 adapter requires rank=alpha in {list(ranks)}")
    targets = discover_h3_lora_targets(model)
    configuration = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(adapter_cfg.get("dropout", 0.0)),
        target_modules=list(targets),
        bias="none",
        init_lora_weights=True,
    )
    wrapped = peft.get_peft_model(
        model,
        configuration,
        adapter_name="default",
        autocast_adapter_dtype=False,
    )
    realized = tuple(sorted(str(name) for name in wrapped.base_model.targeted_module_names))
    if realized != targets:
        raise BackendContractError(
            "PEFT realized a different H3 topology: "
            f"missing={sorted(set(targets) - set(realized))[:8]} "
            f"extra={sorted(set(realized) - set(targets))[:8]}"
        )
    wrapped.peft_config["default"].target_modules = set(targets)
    for parameter in wrapped.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    keep_vars = wrapped.state_dict(keep_vars=True)
    standard = peft.get_peft_model_state_dict(
        wrapped,
        state_dict=keep_vars,
        adapter_name="default",
        save_embedding_layers=False,
    )
    parameter_by_key: OrderedDict[str, Any] = OrderedDict()
    for key in sorted(standard):
        value = standard[key]
        if not isinstance(value, torch.nn.Parameter):
            raise BackendContractError(f"PEFT state {key!r} is not a Parameter")
        parameter_by_key[key] = value
    trainable = tuple(parameter for parameter in wrapped.parameters() if parameter.requires_grad)
    if {id(value) for value in trainable} != {id(value) for value in parameter_by_key.values()}:
        raise BackendContractError("all and only H3 LoRA parameters must be trainable")
    if len(parameter_by_key) != 2 * H3_LORA_TARGET_COUNT:
        raise BackendContractError("H3 LoRA must expose one A/B tensor pair per target")
    runtime = H3LoRARuntime(
        model=wrapped,
        targets=targets,
        parameter_by_key=parameter_by_key,
        peft_config=wrapped.peft_config["default"],
        peft_module=peft,
        base_identity=dict(base_identity),
        rank=rank,
        alpha=alpha,
    )
    expected = int(
        adapter_cfg.get("expected_trainable_parameters", H3_LORA_TRAINABLE_PARAMETERS * rank // 384)
    )
    if runtime.parameter_count != expected:
        raise BackendContractError(
            f"H3 LoRA trainable parameters={runtime.parameter_count:,}, expected={expected:,}"
        )
    if any(parameter.dtype != torch.bfloat16 for parameter in runtime.parameters):
        raise BackendContractError("all H3 LoRA parameters must remain BF16")
    _attach_geometry_controlnet(wrapped, runtime)
    return wrapped, runtime


def _attach_geometry_controlnet(wrapped: Any, runtime: H3LoRARuntime) -> None:
    """Add the ControlNet to the adapter set after the LoRA-only audits passed.

    PEFT froze the trunk with the rest of the base model. It trains in full,
    shares LoRA's replicated/ignored-by-FSDP ownership, and is keyed exactly as
    its EMA shadow names canonicalize, so every stage transition treats it like
    one more adapter tensor.
    """

    from .controlnet_contract import H3_GEOMETRY_MODULE, H3_GEOMETRY_STATE_PREFIX

    controlnet = getattr(wrapped.base_model.model, H3_GEOMETRY_MODULE, None)
    if controlnet is None:
        return
    profile = controlnet.profile
    parameters = OrderedDict(
        (f"{H3_GEOMETRY_STATE_PREFIX}{name}", parameter)
        for name, parameter in sorted(controlnet.named_parameters())
    )
    if not parameters:
        raise BackendContractError("H3 geometry ControlNet exposes no parameters")
    if set(parameters) & set(runtime.parameter_by_key):
        raise BackendContractError("H3 geometry ControlNet keys collide with LoRA keys")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise BackendContractError("H3 geometry ControlNet requires PyTorch") from exc
    for key, parameter in parameters.items():
        if parameter.dtype != torch.bfloat16:
            raise BackendContractError(f"H3 geometry ControlNet parameter {key!r} must be BF16")
        parameter.requires_grad_(True)
    if not profile.lora_trainable:
        for parameter in runtime.parameters:
            parameter.requires_grad_(False)
    runtime.parameter_by_key.update(parameters)
    runtime.geometry_keys = tuple(parameters)
    runtime.geometry_profile = profile


__all__ = [
    "H3_LORA_SUFFIXES",
    "H3_LORA_TARGET_COUNT",
    "H3_LORA_TRAINABLE_PARAMETERS",
    "H3LoRARuntime",
    "discover_h3_lora_targets",
    "inject_h3_lora",
]


def merge_lora_delta(
    model: Any, state: Mapping[str, Any], *, alpha: float | None, rank: int | None, scale: float = 1.0
) -> int:
    """Add a released LoRA's ``scale * alpha/rank * B @ A`` into the weights it names.

    ``alpha=None``: a release that names no alpha means alpha = rank on every layer, whatever each layer's rank
    (InstantX's H3 Turbo has rank 64 and 16 layers) - every pair adds ``scale * B @ A``.

    For a LoRA trained on the released H3 and carried onto a model derived from it (
    LongLive-Plug's few-step and guidance LoRAs). It is merged before this run's own adapter is
    injected, so the adapter sits on top of it. Returns the number of weights changed.
    """

    import torch

    modules = dict(model.named_modules())
    pairs: dict[str, dict[str, Any]] = {}
    # "<module>.diff" / "<module>.diff_b": whole-tensor deltas added to that module's weight / bias after the pairs
    # (FastVideo's FastH3 LoRA also carries its embedders and output projections this way)
    deltas = {key: value for key, value in state.items() if re.fullmatch(r".+\.diff(_b)?", key)}
    state = {key: value for key, value in state.items() if key not in deltas}
    for key, value in state.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", key)
        if match is None:
            raise BackendContractError(f"LoRA key {key!r} is not <module>.lora_A/B.weight")
        pairs.setdefault(match.group(1), {})[match.group(2)] = value
    factor = float(scale) if alpha is None else float(scale) * float(alpha) / float(rank)
    with torch.no_grad():
        for name, pair in sorted(pairs.items()):
            target = modules.get(name)
            if target is None or not hasattr(target, "weight"):
                raise BackendContractError(f"LoRA names {name}, which the model lacks")
            if set(pair) != {"A", "B"}:
                raise BackendContractError(f"LoRA for {name} lacks half of its pair")
            weight = target.weight
            down, up = pair["A"], pair["B"]
            pair_rank = down.shape[0] if alpha is None else rank
            if (
                down.shape[0] != pair_rank
                or up.shape[1] != pair_rank
                or (up.shape[0], down.shape[1]) != tuple(weight.shape)
            ):
                raise BackendContractError(
                    f"LoRA for {name} has shape {tuple(up.shape)} @ {tuple(down.shape)}, "
                    f"not the weight's {tuple(weight.shape)}"
                )
            delta = up.to(weight.device, torch.float32) @ down.to(weight.device, torch.float32)
            weight.copy_((weight.float() + factor * delta).to(weight.dtype))
        for key, value in sorted(deltas.items()):
            name, attr = (key[: -len(".diff_b")], "bias") if key.endswith(".diff_b") else (key[: -len(".diff")], "weight")
            target = getattr(modules.get(name), attr, None)
            if target is None:
                raise BackendContractError(f"delta {key!r} names {name}.{attr}, which the model lacks")
            if tuple(value.shape) != tuple(target.shape):
                raise BackendContractError(
                    f"delta {key!r} has shape {tuple(value.shape)}, not the {attr}'s {tuple(target.shape)}"
                )
            target.copy_((target.float() + float(scale) * value.to(target.device, torch.float32)).to(target.dtype))
    return len(pairs) + len({key.rsplit(".", 1)[0] for key in deltas})


def _released_pairs(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", key)
        if match is None:
            raise BackendContractError(f"LoRA key {key!r} is not <module>.lora_A/B.weight")
        pairs.setdefault(match.group(1), {})[match.group(2)] = value
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise BackendContractError(f"LoRA for {name} lacks half of its pair")
    return pairs


def released_lora_state(path: str) -> dict[str, Any]:
    """A released LoRA's ``<module>.lora_A/B.weight`` tensors, from a .safetensors file or from a
    training export that keeps them under ``student_lora`` (LongLive-Plug's few-step H3 LoRA)."""

    import torch

    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return state.get("student_lora", state)


def adapter_state_as_released(state: Mapping[str, Any]) -> dict[str, Any]:
    """A saved run adapter (PEFT keys, maybe with the control trunk) as ``<module>.lora_A/B``.

    So a trained adapter can be merged into a base the way a released LoRA is
    (:func:`merge_lora_delta`), e.g. to make a frozen control LoRA part of a student's base.
    """

    prefix = "base_model.model."
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if re.fullmatch(r".+\.lora_[AB]\.weight", key)
    }


def seed_lora_from_released(
    live: Mapping[str, Any],
    released: Mapping[str, Any],
    *,
    live_scale: float,
    released_scale: float,
    offset: int = 0,
) -> int:
    """Start a live (wider) LoRA from a released one, its update unchanged (DMD).

    ``offset`` places the released pair at rank slots ``[offset, offset + rank)``, keeping the slots
    before it, so two LoRAs stack into one whose update is their sum (control + four-step LoRA, trained
    jointly).

    The released pair fills the leading rank slots - ``B`` rescaled by
    ``released_scale / live_scale`` so ``live_scale * B @ A`` equals the released update - and the
    spare slots keep their fresh ``A`` with a zero ``B``: the same update, with room to grow.
    ``live`` is keyed like the run's adapter (PEFT ``base_model.model.`` prefix allowed).
    Returns the number of modules seeded.
    """

    import torch

    by_module: dict[str, dict[str, Any]] = {}
    for key, value in live.items():
        match = re.fullmatch(r"(?:base_model\.model\.)?(.+)\.lora_([AB])\.weight", key)
        if match is not None:
            by_module.setdefault(match.group(1), {})[match.group(2)] = value
    factor = float(released_scale) / float(live_scale)
    pairs = _released_pairs(released)
    with torch.no_grad():
        for name, pair in sorted(pairs.items()):
            target = by_module.get(name)
            if target is None or set(target) != {"A", "B"}:
                raise BackendContractError(
                    f"released LoRA names {name}, which the live LoRA lacks"
                )
            down, up = pair["A"], pair["B"]
            rank = int(down.shape[0])
            stop = int(offset) + rank
            if stop > int(target["A"].shape[0]):
                raise BackendContractError(
                    f"released rank {rank} at offset {offset} exceeds the live rank "
                    f"{int(target['A'].shape[0])} at {name}"
                )
            if (
                tuple(down.shape[1:]) != tuple(target["A"].shape[1:])
                or up.shape[0] != target["B"].shape[0]
            ):
                raise BackendContractError(f"released LoRA shape differs at {name}")
            a, b = target["A"], target["B"]
            a[offset:stop].copy_(down.to(a.device, a.dtype))
            b[:, offset:].zero_()
            b[:, offset:stop].copy_((factor * up.float()).to(b.device, b.dtype))
    return len(pairs)
