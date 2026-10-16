"""Physical shard resolution below the canonical data plan.

The transport receives an :class:`~solarwm.data.index.IndexRow` only after
sample ownership, shuffle order, and frame starts have been decided, so where a
shard is read from cannot perturb the logical RNG stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from solarwm.errors import DataContractError

from .index import IndexRow, ShardIdentity, shard_identity, validate_relative_key


def join_root(root: str, relative_key: str) -> str:
    """Join a runtime root and portable index key without changing semantics."""

    key = validate_relative_key(relative_key)
    return str(Path(root).expanduser() / key)


class ShardResolver(Protocol):
    """Resolve one immutable index row to a readable local file."""

    def resolve(self, row: IndexRow) -> Path: ...


@dataclass(frozen=True)
class LocalResolver:
    """Resolve relative shard keys beneath a mounted directory."""

    root: Path

    def resolve(self, row: IndexRow) -> Path:
        target = (self.root / row.shard).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise DataContractError(f"shard escapes local root: {row.shard}")
        if not target.is_file():
            raise DataContractError(f"local shard is missing: {target}")
        identity = shard_identity(row)
        if identity is not None:
            _verify_size(target, identity, label="local shard")
        return target


def _verify_size(path: Path, identity: ShardIdentity, *, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataContractError(f"cannot stat {label} {path}: {exc}") from exc
    if size != identity.size:
        raise DataContractError(f"{label} size mismatch for {path}: {size} != {identity.size}")


def resolver_from_config(root: str, *, cache_dir: str | None = None, max_gib: float = 256.0) -> ShardResolver:
    """The shard resolver for a run: shards are read from a local (or mounted) directory."""

    if cache_dir:
        raise DataContractError("data.cache_dir is not used: shards are read from a local directory")
    return LocalResolver(Path(root).expanduser())
