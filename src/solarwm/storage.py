"""Files that may live in object storage.

A path with a ``<scheme>://`` prefix is remote and is read and written through fsspec, which picks the
filesystem from the scheme; install the fsspec backend for your storage and configure its credentials the
way that backend reads them. Any other path is a local file. Each call retries a few times: a busy store
turns a request away now and then, and a member thrown away over that is encoding time spent for nothing.
"""

from __future__ import annotations

import datetime as _dt
import os
import posixpath
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def is_remote(path: str) -> bool:
    return "://" in str(path)


def _fs(path: str):
    from fsspec.core import url_to_fs

    return url_to_fs(path)


def _retry(call: Callable[[], T], tries: int = 4) -> T:
    for attempt in range(tries):
        try:
            return call()
        except FileNotFoundError:
            raise
        except Exception:
            if attempt + 1 == tries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def get(source: str, destination: str) -> None:
    """Copy one remote object to a local file."""

    fs, path = _fs(source)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    _retry(lambda: fs.get_file(path, destination))


def get_matching(pattern: str, directory: str) -> list[str]:
    """Copy every object matching a glob (``.../*.json``) into a local directory; returns the local files."""

    fs, path = _fs(pattern)
    os.makedirs(directory, exist_ok=True)
    local = []
    for name in _retry(lambda: fs.glob(path)):
        target = os.path.join(directory, posixpath.basename(name))
        _retry(lambda name=name, target=target: fs.get_file(name, target))
        local.append(target)
    return local


def put(source: str, destination: str) -> None:
    """Copy one local file to a remote object."""

    fs, path = _fs(destination)
    _retry(lambda: fs.put_file(source, path))


def write_bytes(destination: str, data: bytes) -> None:
    fs, path = _fs(destination)
    _retry(lambda: fs.pipe_file(path, data))


def move(source: str, destination: str) -> None:
    """A local file to a remote object; the local file is removed once the copy succeeded."""

    put(source, destination)
    os.remove(source)


def _mtime(info: dict) -> float | None:
    for key in ("mtime", "LastModified", "last_modified", "updated", "modified", "created", "timeCreated"):
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, _dt.datetime):
            return (value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)).timestamp()
        try:
            return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def stat(path: str) -> tuple[int, float | None] | None:
    """(size, modification time) of one object, or None when it is not there."""

    fs, key = _fs(path)
    try:
        info = _retry(lambda: fs.info(key))
    except FileNotFoundError:
        return None
    if info.get("type") == "directory":
        return None
    return int(info.get("size") or 0), _mtime(info)


def listing(prefix: str) -> dict[str, float | None] | None:
    """Base name -> modification time of the objects directly under a prefix; an empty dict when the prefix
    holds nothing, None when the listing failed (it then says nothing about what is there)."""

    fs, key = _fs(prefix.rstrip("/"))
    try:
        entries = _retry(lambda: fs.ls(key, detail=True, refresh=True) if _accepts_refresh(fs) else fs.ls(key, detail=True))
    except FileNotFoundError:
        return {}
    except Exception:
        return None
    return {
        posixpath.basename(entry["name"].rstrip("/")): _mtime(entry)
        for entry in entries
        if entry.get("type") != "directory"
    }


def _accepts_refresh(fs) -> bool:
    import inspect

    try:
        return "refresh" in inspect.signature(fs.ls).parameters
    except (TypeError, ValueError):
        return False
