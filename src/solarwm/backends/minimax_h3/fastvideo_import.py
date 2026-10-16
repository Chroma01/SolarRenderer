"""Import FastVideo spatial-AI renders and geometry ControlNet weights into SolarWM.

FastVideo's renderer writes canonical coordinate maps as 16-bit RGB PNG frames
(``coord/NNNNNN.png`` or ``coord.zip``). SolarWM's raw render member is the
same bytes as one uint16 ``[F,H,W,3]`` array (see ``raw_data``). FastVideo
training saves the ControlNet under ``geometry_controlnet.*`` inside a DCP
checkpoint; the names are SolarWM's too.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.errors import DataContractError

from .controlnet_contract import H3_GEOMETRY_MODULE
from .raw_data import H3_GEOMETRY_RENDER_KEY

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {2: 3, 6: 4}
FASTVIDEO_STUDENT_STATE_KEY = "roles.student.transformer"
H3_GEOMETRY_WEIGHTS_SCHEMA = "solarwm.minimax-h3-geometry-controlnet-weights.v1"
_WRAPPER_PARTS = frozenset({"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"})


def _unfilter(raw: bytes, height: int, stride: int, pixel_bytes: int) -> np.ndarray:
    """Undo PNG scanline filters 0-4 (None, Sub, Up, Average, Paeth).

    None/Sub/Up are vectorized; Average and Paeth depend on the reconstructed
    left pixel and advance one pixel (all its bytes at once) at a time.
    """

    rows = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.int32)
    offset = 0
    for row in range(height):
        kind = raw[offset]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset + 1).astype(np.int32)
        offset += stride + 1
        if kind == 0:
            current = line
        elif kind == 1:
            current = np.cumsum(line.reshape(-1, pixel_bytes), axis=0).reshape(-1) & 0xFF
        elif kind == 2:
            current = (line + previous) & 0xFF
        elif kind in (3, 4):
            current = np.zeros(stride, dtype=np.int32)
            left = np.zeros(pixel_bytes, dtype=np.int32)
            upper_left = np.zeros(pixel_bytes, dtype=np.int32)
            for start in range(0, stride, pixel_bytes):
                pixel = slice(start, start + pixel_bytes)
                up = previous[pixel]
                if kind == 3:
                    predictor = (left + up) // 2
                else:
                    estimate = left + up - upper_left
                    distance_left = np.abs(estimate - left)
                    distance_up = np.abs(estimate - up)
                    distance_corner = np.abs(estimate - upper_left)
                    predictor = np.where(
                        (distance_left <= distance_up) & (distance_left <= distance_corner),
                        left,
                        np.where(distance_up <= distance_corner, up, upper_left),
                    )
                current[pixel] = (line[pixel] + predictor) & 0xFF
                left = current[pixel]
                upper_left = up
        else:
            raise DataContractError(f"PNG scanline uses unknown filter {kind}")
        rows[row] = current
        previous = current
    return rows


def _decode_with_opencv(data: bytes) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise DataContractError("coordinate PNG must be RGB(A)")
    rgb = image[..., 2::-1]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.uint16) * np.uint16(257)
    if rgb.dtype != np.uint16:
        raise DataContractError(f"coordinate PNG has unsupported sample type {rgb.dtype}")
    return np.ascontiguousarray(rgb)


def decode_png_rgb16(data: bytes, *, prefer_opencv: bool = True) -> np.ndarray:
    """Decode a non-interlaced RGB/RGBA PNG to uint16 ``[H,W,3]`` without resampling.

    16-bit samples are returned as stored; 8-bit samples are widened by ``*257``
    so that full scale stays full scale. Alpha is dropped. OpenCV (the decoder
    FastVideo reads these files with) is used when installed; the built-in
    decoder is exact but slow on Average/Paeth-filtered scanlines.
    """

    if prefer_opencv:
        decoded = _decode_with_opencv(data)
        if decoded is not None:
            return decoded
    if not data.startswith(_PNG_SIGNATURE):
        raise DataContractError("not a PNG file")
    offset = len(_PNG_SIGNATURE)
    header: tuple[int, ...] | None = None
    compressed = bytearray()
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset : offset + 8])
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
    if header is None:
        raise DataContractError("PNG lacks IHDR")
    width, height, depth, color, _, _, interlace = header
    if color not in _CHANNELS or depth not in (8, 16) or interlace:
        raise DataContractError(
            f"coordinate PNG must be non-interlaced 8/16-bit RGB(A); got color={color} "
            f"depth={depth} interlace={interlace}"
        )
    channels = _CHANNELS[color]
    pixel_bytes = channels * depth // 8
    rows = _unfilter(zlib.decompress(bytes(compressed)), height, width * pixel_bytes, pixel_bytes)
    if depth == 16:
        samples = rows.reshape(height, width * channels, 2)
        values = (samples[..., 0].astype(np.uint16) << 8) | samples[..., 1]
    else:
        values = rows.astype(np.uint16) * np.uint16(257)
    return values.reshape(height, width, channels)[..., :3].copy()


def read_fastvideo_coordinates(source: str | Path) -> np.ndarray:
    """Read a FastVideo ``coord/`` directory or ``coord.zip`` as uint16 ``[F,H,W,3]``.

    Frames are ordered by file name, as the FastVideo renderer numbers them.
    """

    path = Path(source)
    frames: list[np.ndarray] = []
    if path.is_dir():
        names = sorted(item for item in path.iterdir() if item.suffix.lower() == ".png")
        frames = [decode_png_rgb16(item.read_bytes()) for item in names]
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".png"))
            frames = [decode_png_rgb16(archive.read(name)) for name in names]
    else:
        raise DataContractError(f"{path} is neither a coordinate directory nor a zip")
    if not frames:
        raise DataContractError(f"{path} contains no coordinate PNG frames")
    if len({frame.shape for frame in frames}) != 1:
        raise DataContractError(f"{path} mixes coordinate frame sizes")
    return np.stack(frames)


def render_member_bytes(coords: np.ndarray) -> bytes:
    """Serialize a render as SolarWM's raw member: a stored (uncompressed) npz.

    Stored members let preencode slice one window by byte offset instead of
    inflating a whole source clip.
    """

    array = np.ascontiguousarray(coords)
    if array.dtype != np.uint16 or array.ndim != 4 or array.shape[-1] != 3:
        raise DataContractError(f"render must be uint16 [F,H,W,3], got {array.dtype} {array.shape}")
    buffer = io.BytesIO()
    np.savez(buffer, **{H3_GEOMETRY_RENDER_KEY: array.astype("<u2", copy=False)})
    return buffer.getvalue()


def _canonical_controlnet_name(key: str, state_key: str) -> str | None:
    if not key.startswith(f"{state_key}."):
        return None
    parts = [part for part in key[len(state_key) + 1 :].split(".") if part not in _WRAPPER_PARTS]
    if H3_GEOMETRY_MODULE not in parts:
        return None
    return ".".join(parts[parts.index(H3_GEOMETRY_MODULE) :])


def read_fastvideo_controlnet(
    checkpoint: str | Path, *, state_key: str = FASTVIDEO_STUDENT_STATE_KEY
) -> dict[str, Any]:
    """Load only the ``geometry_controlnet.*`` tensors of a FastVideo DCP checkpoint."""

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata()
    selected: dict[str, str] = {}
    for key, value in metadata.state_dict_metadata.items():
        name = _canonical_controlnet_name(key, state_key)
        if name is None or not isinstance(value, TensorStorageMetadata):
            continue
        if name in selected.values():
            raise DataContractError(f"FastVideo checkpoint has duplicate tensors for {name!r}")
        selected[key] = name
    if not selected:
        raise DataContractError(
            f"{checkpoint} holds no {H3_GEOMETRY_MODULE}.* tensors under {state_key!r}"
        )
    buffers = {
        key: torch.empty(
            tuple(metadata.state_dict_metadata[key].size),
            dtype=metadata.state_dict_metadata[key].properties.dtype,
        )
        for key in selected
    }
    dcp.load(buffers, storage_reader=reader)
    return {selected[key]: value for key, value in buffers.items()}


def save_controlnet_weights(tensors: Mapping[str, Any], path: str | Path) -> None:
    """Write ``geometry_controlnet.*`` tensors as a SolarWM initialization file."""

    import torch
    from safetensors.torch import save_file

    if not tensors or any(not key.startswith(f"{H3_GEOMETRY_MODULE}.") for key in tensors):
        raise DataContractError(f"ControlNet weights must all be {H3_GEOMETRY_MODULE}.* tensors")
    values = {key: value.detach().to(torch.bfloat16).contiguous() for key, value in tensors.items()}
    save_file(values, str(path), metadata={"schema": H3_GEOMETRY_WEIGHTS_SCHEMA})


__all__ = [
    "FASTVIDEO_STUDENT_STATE_KEY",
    "H3_GEOMETRY_WEIGHTS_SCHEMA",
    "decode_png_rgb16",
    "read_fastvideo_controlnet",
    "read_fastvideo_coordinates",
    "render_member_bytes",
    "save_controlnet_weights",
]
