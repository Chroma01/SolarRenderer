"""Backend protocol and lazy builtin discovery."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from solarwm.errors import BackendContractError


@dataclass(frozen=True)
class BackendSpec:
    family: str
    display_name: str
    module: str
    factory: str = "create_backend"


@runtime_checkable
class Backend(Protocol):
    family: str

    def validate_config(self, config: Mapping[str, Any]) -> None: ...

    def train(self, config: Mapping[str, Any]) -> int: ...

    def infer(self, config: Mapping[str, Any]) -> int: ...

    def preencode(self, config: Mapping[str, Any]) -> int: ...


_BUILTINS = {
    "minimax_h3": BackendSpec("minimax_h3", "MiniMax-H3", "solarwm.backends.minimax_h3"),
}


def backend_spec(family: str) -> BackendSpec:
    try:
        return _BUILTINS[family]
    except KeyError as exc:
        raise BackendContractError(f"unknown backend family {family!r}") from exc


def load_backend(family: str) -> Backend:
    spec = backend_spec(family)
    try:
        module = importlib.import_module(spec.module)
        factory = getattr(module, spec.factory)
        backend = factory(family=family)
    except Exception as exc:
        raise BackendContractError(
            f"failed to load {spec.display_name} backend from {spec.module}: {exc}"
        ) from exc
    if not isinstance(backend, Backend):
        raise BackendContractError(f"{spec.module}.{spec.factory} returned an invalid backend")
    return backend
