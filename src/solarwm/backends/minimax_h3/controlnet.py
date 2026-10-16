"""Geometry ControlNet trunk for MiniMax-H3 video-to-video conditioning.

A narrow transformer reads VAE-encoded renders of the target clip and adds a
zero-initialised residual into the backbone every ``layer_stride`` blocks.
Structure, parameter names and initialisation follow FastVideo's spatial-AI
trunk; only the attention plumbing is SolarWM's.

The control stream lies over the same packed rows as the backbone stream: zero
everywhere except the rows that receive control. Both streams are therefore
sharded identically by sequence parallelism, and the residual add is local.
Every row attends, but only the control rows receive a residual.

Two attention patterns cover every stage. ``dense`` is full bidirectional
attention (Stage0.5, bidirectional sampling, SGF score models). ``segments``
is chunk-local: condition rows see condition rows, audio rows see the prefix,
and each (clean|noisy, chunk) group sees the prefix plus itself. A chunk's
control never reads another chunk, so autoregressive rollouts need no control
KV cache, and teacher-forced windows, rollouts and replays compute identical
control for a chunk.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controlnet_contract import (
    H3_GEOMETRY_MODALITIES,
    H3_GEOMETRY_MODULE,
    H3_ROTARY_CHANNELS,
    H3GeometryProfile,
)
from .layout import AUDIO_ROLE, CONDITION_ROLE

Segment = tuple[int, int, tuple[tuple[int, int], ...]]


def apply_rotary(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """H3's rotate-half RoPE over the leading channels of ``[B,S,H,D]`` heads."""

    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


@dataclass(frozen=True)
class H3GeometryControl:
    """Per-forward ControlNet inputs, laid out against one packed document.

    ``rows`` holds one ``[B,N,96]`` patchified latent per modality, in the order
    of ``row_indices`` (the packed rows that receive control). ``position_ids``
    is the ``[S,3]`` rotary grid the control trunk uses for every packed row.
    ``segments`` is ``None`` for dense attention.
    """

    rows: Mapping[str, Any]
    row_indices: Any
    position_ids: Any
    segments: tuple[Segment, ...] | None = None
    wide_segments: tuple[Segment, ...] | None = None
    wide_rows: Mapping[str, Any] | None = None
    wide_position_ids: Any = None
    # Scales the residual. Zero reproduces the backbone exactly while every trunk
    # parameter still receives a gradient, so replicated ranks that drop the control
    # keep the same gradient pattern as ranks that use it.
    gain: float = 1.0


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _runs(roles: Any, chunks: Any) -> list[tuple[int, int, tuple[int, int]]]:
    """Contiguous ``(start, stop, (role, chunk))`` runs over the packed rows."""

    import numpy as np

    if roles.shape != chunks.shape or roles.ndim != 1 or not roles.size:
        raise ValueError("H3 control segments need one role and chunk ID per packed row")
    changed = (roles[1:] != roles[:-1]) | (chunks[1:] != chunks[:-1])
    bounds = [0, *(np.flatnonzero(changed) + 1).tolist(), int(roles.size)]
    return [
        (start, stop, (int(roles[start]), int(chunks[start]))) for start, stop in pairwise(bounds)
    ]


def control_segments(layout: Any) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """Chunk-local self-attention and wide cross-attention segments for ``layout``.

    Derived from the layout's row roles and chunk IDs, so Stage1 windows,
    sampler windows and SGF rollout/replay documents share one rule.
    """

    roles = _as_numpy(layout.row_roles).astype("int64")
    chunks = _as_numpy(layout.target_video_chunk_ids).astype("int64")
    runs = _runs(roles, chunks)
    if len(runs) < 3 or runs[0][2][0] != CONDITION_ROLE or runs[1][2][0] != AUDIO_ROLE:
        raise ValueError("H3 control segments require [condition | audio | video groups]")
    condition = (runs[0][0], runs[0][1])
    audio = (runs[1][0], runs[1][1])
    prefix = (condition, audio)
    seen: set[tuple[int, int]] = set()
    segments: list[Segment] = [
        (condition[0], condition[1], (condition,)),
        (audio[0], audio[1], prefix),
    ]
    wide: list[Segment] = []
    for start, stop, label in runs[2:]:
        if label[0] in (CONDITION_ROLE, AUDIO_ROLE) or label in seen:
            raise ValueError("H3 control segments require one contiguous run per video group")
        seen.add(label)
        segments.append((start, stop, (*prefix, (start, stop))))
        wide.append((start, stop, ((start, stop),)))
    return tuple(segments), tuple(wide)


def wide_position_ids(position_ids: Any, row_indices: Any, fov_scale: float) -> Any:
    """Place wide-render rows on the narrow render's ray grid.

    Wide and narrow renders share pose, resolution and principal point; only
    the focal length differs by ``k``. A ray ``d`` from the centre of the narrow
    image lands at ``k*d`` in the wide one, so the wide rows sit at
    ``pivot + (pos - pivot) / k``. The pivot is the midpoint of the actual
    (endpoint-excluded) grid. Time is unchanged.
    """

    if not fov_scale > 0.0:
        raise ValueError("wide fov_scale must be positive")
    result = position_ids.clone()
    if float(fov_scale) == 1.0:
        return result
    rows = row_indices.to(device=position_ids.device, dtype=torch.long)
    spatial = result.index_select(0, rows)[:, 1:]
    pivot = 0.5 * (spatial.amin(dim=0) + spatial.amax(dim=0))
    result[rows, 1:] = pivot + (spatial - pivot) / float(fov_scale)
    return result


def pad_control_latents(latents: Any, frames: int) -> Any:
    """Extend ``[B,C,F,H,W]`` latents to ``frames`` by repeating the last frame.

    Matches the SGF camera tail policy (``repeat_last_latent``) for rollouts
    longer than the encoded clip.
    """

    available = int(latents.shape[2])
    if frames < available:
        return latents[:, :, :frames]
    if frames == available:
        return latents
    tail = latents[:, :, -1:].expand(-1, -1, frames - available, -1, -1)
    return torch.cat((latents, tail), dim=2)


def geometry_control_for_layout(
    layout: Any,
    rows: Mapping[str, Any],
    *,
    profile: H3GeometryProfile,
    wide_rows: Mapping[str, Any] | None = None,
    position_ids: Any = None,
    row_indices: Any = None,
    attention: str = "dense",
) -> H3GeometryControl:
    """Bind patchified geometry rows to every non-anchor video row of ``layout``.

    ``row_indices``/``position_ids`` default to the layout's own arrays; callers
    holding device copies pass them to avoid a host round trip.
    """

    if attention not in {"dense", "segments"}:
        raise ValueError("H3 control attention must be dense or segments")
    if row_indices is None:
        video_indices = layout.video_indices
        if not isinstance(video_indices, torch.Tensor):
            video_indices = torch.as_tensor(video_indices)
        row_indices = video_indices[int(layout.num_condition_video_rows) :]
    row_indices = row_indices.long()
    positions = layout.position_ids if position_ids is None else position_ids
    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions)
    segments = wide_segments = None
    if attention == "segments":
        segments, wide_segments = control_segments(layout)
    wide_positions = None
    if profile.wide_enabled:
        if wide_rows is None:
            raise ValueError("H3 wide geometry profile requires wide rows")
        wide_positions = wide_position_ids(positions, row_indices, float(profile.wide_fov_scale))
    return H3GeometryControl(
        rows=dict(rows),
        row_indices=row_indices,
        position_ids=positions,
        segments=segments,
        wide_segments=wide_segments,
        wide_rows=None if wide_rows is None else dict(wide_rows),
        wide_position_ids=wide_positions,
    )


def teacher_forced_geometry_control(
    layout: Any,
    latents: Mapping[str, Any] | None,
    *,
    profile: H3GeometryProfile,
    frames: slice,
) -> H3GeometryControl | None:
    """Chunk-local control for a ``[condition | audio | clean | noisy]`` document.

    Both copies of the window carry the geometry of their own frames: a history
    frame's control reaches the target through the backbone's causal residual
    stream, exactly as it does when a rollout commits that frame.
    """

    if latents is None:
        return None
    from .layout import patchify_video

    def rows(name: str) -> Any:
        window = patchify_video(latents[name][:, :, frames])
        return torch.cat((window, window), dim=1)

    return geometry_control_for_layout(
        layout,
        {name: rows(name) for name in profile.modalities},
        profile=profile,
        wide_rows=(
            {name: rows(f"{name}_wide") for name in profile.wide_modalities}
            if profile.wide_enabled
            else None
        ),
        attention="segments",
    )


class _SequenceExchange(torch.autograd.Function):
    """Ulysses all-to-all that restores its own document layout in both passes.

    The shared SP state holds the most recent document's per-rank lengths. A
    Stage2 student backward can run after score forwards over other documents,
    so the control exchange carries its lengths rather than reading them late.
    """

    @staticmethod
    def forward(ctx: Any, value: Any, scatter_dim: int, gather_dim: int, lengths: Any) -> Any:
        ctx.scatter_dim, ctx.gather_dim, ctx.lengths = scatter_dim, gather_dim, lengths
        return _exchange(value, scatter_dim, gather_dim, lengths)

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None, None]:
        return _exchange(gradient, ctx.gather_dim, ctx.scatter_dim, ctx.lengths), None, None, None


def _exchange(value: Any, scatter_dim: int, gather_dim: int, lengths: Any) -> Any:
    from .distributed import _all_to_all_raw, state

    current = state()
    previous = current.sequence_lengths
    current.sequence_lengths = tuple(lengths)
    try:
        return _all_to_all_raw(value.contiguous(), scatter_dim, gather_dim)
    finally:
        current.sequence_lengths = previous


def _attend(query: Any, key: Any, value: Any, backend: Any) -> Any:
    """Unmasked attention over ``[B,S,H,D]`` heads."""

    if backend is None:
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return output.transpose(1, 2)
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    return dispatch_attention_fn(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend
    )


def _segment_attention(
    query: Any, key: Any, value: Any, segments: Sequence[Segment], backend: Any
) -> Any:
    outputs = []
    for start, stop, key_ranges in segments:
        if not key_ranges:
            outputs.append(query.new_zeros(query[:, start:stop].shape))
            continue
        keys = torch.cat([key[:, first:last] for first, last in key_ranges], dim=1)
        values = torch.cat([value[:, first:last] for first, last in key_ranges], dim=1)
        outputs.append(_attend(query[:, start:stop], keys, values, backend))
    return torch.cat(outputs, dim=1)


@dataclass(frozen=True)
class _AttentionContext:
    segments: tuple[Segment, ...] | None
    wide_segments: tuple[Segment, ...] | None
    sequence_lengths: tuple[int, ...]
    backend: Any


class H3ControlAttention(nn.Module):
    """Bias-free Q/K/V/O self-attention with per-head RMS query/key norms."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        inner = num_heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def _heads(self, projection: nn.Module, states: Any) -> Any:
        return projection(states).unflatten(-1, (self.num_heads, -1))

    def _exchange(self, value: Any, context: _AttentionContext, *, to_full: bool) -> Any:
        if len(context.sequence_lengths) <= 1:
            return value
        if to_full:
            return _SequenceExchange.apply(value, 2, 1, context.sequence_lengths)
        return _SequenceExchange.apply(value, 1, 2, context.sequence_lengths)

    def _finish(self, attended: Any, context: _AttentionContext, dtype: Any) -> Any:
        attended = self._exchange(attended, context, to_full=False)
        return self.to_out(attended.flatten(2, 3).to(dtype))

    def forward(self, states: Any, rotary: tuple[Any, Any], context: _AttentionContext) -> Any:
        query = apply_rotary(self.norm_q(self._heads(self.to_q, states)), *rotary)
        key = apply_rotary(self.norm_k(self._heads(self.to_k, states)), *rotary)
        value = self._heads(self.to_v, states)
        query = self._exchange(query, context, to_full=True)
        key = self._exchange(key, context, to_full=True)
        value = self._exchange(value, context, to_full=True)
        if context.segments is None:
            attended = _attend(query, key, value, context.backend)
        else:
            attended = _segment_attention(query, key, value, context.segments, context.backend)
        return self._finish(attended, context, states.dtype)


class H3ControlWideCrossAttention(H3ControlAttention):
    """Cross-attention from the control stream onto the wide-FOV render.

    ``to_k``/``to_v`` carry no bias: packed rows without wide content stay
    exactly zero and act as a fixed attention sink under dense attention.
    """

    def forward(  # type: ignore[override]
        self,
        states: Any,
        wide_states: Any,
        rotary: tuple[Any, Any],
        wide_rotary: tuple[Any, Any],
        context: _AttentionContext,
    ) -> Any:
        query = apply_rotary(self.norm_q(self._heads(self.to_q, states)), *rotary)
        key = apply_rotary(self.norm_k(self._heads(self.to_k, wide_states)), *wide_rotary)
        value = self._heads(self.to_v, wide_states)
        query = self._exchange(query, context, to_full=True)
        key = self._exchange(key, context, to_full=True)
        value = self._exchange(value, context, to_full=True)
        if context.wide_segments is None:
            attended = _attend(query, key, value, context.backend)
        else:
            segments = list(context.wide_segments)
            covered = sorted((start, stop) for start, stop, _ in segments)
            # Rows outside the video groups carry no wide content; their output is zero.
            cursor = 0
            for start, stop in covered:
                if cursor < start:
                    segments.append((cursor, start, ()))
                cursor = stop
            if cursor < query.shape[1]:
                segments.append((cursor, int(query.shape[1]), ()))
            segments.sort(key=lambda item: item[0])
            attended = _segment_attention(query, key, value, segments, context.backend)
        return self._finish(attended, context, states.dtype)


class H3ControlSwiGLU(nn.Module):
    """``fc_in`` emits ``[value; gate]``."""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.fc_out = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, states: Any) -> Any:
        value, gate = self.fc_in(states).chunk(2, dim=-1)
        return self.fc_out(value * F.silu(gate))


class H3GeometryControlBlock(nn.Module):
    """One control block plus its zero-initialised projection into the backbone."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        temb_dim: int,
        out_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        wide: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attn = H3ControlAttention(dim, num_heads, head_dim, qk_norm_eps)
        self.wca = (
            H3ControlWideCrossAttention(dim, num_heads, head_dim, qk_norm_eps) if wide else None
        )
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ff = H3ControlSwiGLU(dim, ffn_dim)
        self.adaln_proj = nn.Linear(temb_dim, 6 * dim, bias=True)
        self.proj_out = nn.Linear(dim, out_dim, bias=True)

    def forward(
        self,
        states: Any,
        temb: Any,
        timestep_indices: Any,
        rotary: tuple[Any, Any],
        context: _AttentionContext,
        wide_states: Any = None,
        wide_rotary: tuple[Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        modulation = self.adaln_proj(F.silu(temb).to(self.adaln_proj.weight.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, -1)

        def pick(table: Any) -> Any:
            return table.index_select(0, timestep_indices)

        normalized = self.norm1(states) * (1.0 + pick(scale_msa)) + pick(shift_msa)
        states = states + pick(gate_msa) * self.attn(normalized, rotary, context)
        if self.wca is not None and wide_states is not None:
            if wide_rotary is None:
                raise ValueError("H3 wide cross-attention requires the wide rotary table")
            # Same query and modulation as the self-attention, ungated: the trunk's
            # single gate is the zero-initialised proj_out.
            states = states + self.wca(normalized, wide_states, rotary, wide_rotary, context)
        normalized = self.norm2(states) * (1.0 + pick(scale_mlp)) + pick(shift_mlp)
        states = states + pick(gate_mlp) * self.ff(normalized)
        return states, self.proj_out(states)


@dataclass
class H3ControlStream:
    """The trunk's per-forward state on this rank's packed rows."""

    states: Any
    mask: Any
    rotary: tuple[Any, Any]
    temb: Any
    timestep_indices: Any
    context: _AttentionContext
    wide_states: Any = None
    wide_rotary: tuple[Any, Any] | None = None


class H3GeometryControlNet(nn.Module):
    """Geometry embedding, modality fusion and one block per injection point."""

    def __init__(
        self,
        profile: H3GeometryProfile,
        *,
        video_patch_dim: int = 96,
        hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        num_layers: int = 50,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        rotary_channels: int = H3_ROTARY_CHANNELS,
    ) -> None:
        super().__init__()
        if profile.head_dim < rotary_channels:
            raise ValueError(
                f"H3 control head_dim={profile.head_dim} is below the {rotary_channels} "
                "rotary channels of the backbone table it reuses"
            )
        self.profile = profile
        self.modalities = tuple(
            name for name in H3_GEOMETRY_MODALITIES if name in profile.modalities
        )
        self.wide_modalities = tuple(
            name for name in H3_GEOMETRY_MODALITIES if name in profile.wide_modalities
        )
        dim = profile.dim
        self.embeddings = nn.ModuleDict(
            {name: nn.Linear(video_patch_dim, dim, bias=True) for name in self.modalities}
        )
        self.fusion_proj = (
            nn.Linear(dim * len(self.modalities), dim, bias=True)
            if len(self.modalities) > 1
            else None
        )
        self.wide_embeddings = None
        self.wide_fusion_proj = None
        self.norm_wide = None
        if self.wide_modalities:
            self.wide_embeddings = nn.ModuleDict(
                {name: nn.Linear(video_patch_dim, dim, bias=True) for name in self.wide_modalities}
            )
            if len(self.wide_modalities) > 1:
                self.wide_fusion_proj = nn.Linear(dim * len(self.wide_modalities), dim, bias=True)
            self.norm_wide = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.temb_proj = nn.Linear(time_embed_dim, profile.temb_dim, bias=True)
        self.control_blocks = nn.ModuleList()
        self.block_to_control: dict[int, int] = {}
        for block_index in range(num_layers):
            if block_index % profile.layer_stride:
                continue
            self.block_to_control[block_index] = len(self.control_blocks)
            self.control_blocks.append(
                H3GeometryControlBlock(
                    dim=dim,
                    ffn_dim=profile.ffn_dim,
                    num_heads=profile.num_heads,
                    head_dim=profile.head_dim,
                    temb_dim=profile.temb_dim,
                    out_dim=hidden_size,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    wide=bool(self.wide_modalities),
                )
            )
        self.attention_backend: Any = None
        self.activation_checkpointing = False

    def _fuse(
        self,
        rows: Mapping[str, Any],
        names: tuple[str, ...],
        embeddings: nn.ModuleDict,
        fusion: nn.Module | None,
    ) -> Any:
        missing = [name for name in names if name not in rows]
        if missing:
            raise ValueError(f"H3 geometry control lacks modalities {missing}")
        embedded = []
        for name in names:
            projection = embeddings[name]
            embedded.append(projection(rows[name].to(projection.weight.dtype)))
        fused = embedded[0] if len(embedded) == 1 else torch.cat(embedded, dim=-1)
        return fused if fusion is None else fusion(fused)

    @staticmethod
    def _scatter(fused: Any, row_indices: Any, sequence_length: int) -> Any:
        if int(fused.shape[1]) != int(row_indices.numel()):
            raise ValueError(
                f"H3 geometry has {int(fused.shape[1])} rows for "
                f"{int(row_indices.numel())} control rows"
            )
        stream = fused.new_zeros((fused.shape[0], sequence_length, fused.shape[-1]))
        return stream.index_copy(1, row_indices, fused)

    def prepare(
        self,
        control: H3GeometryControl,
        *,
        sequence_length: int,
        temb: Any,
        timestep_indices: Any,
        rope: Any,
        start: int = 0,
        stop: int | None = None,
        sequence_lengths: tuple[int, ...] = (),
    ) -> H3ControlStream:
        """Embed, scatter and shard the control stream for ``[start, stop)``."""

        stop = sequence_length if stop is None else int(stop)
        device = temb.device
        row_indices = control.row_indices.to(device=device, dtype=torch.long)
        positions = control.position_ids.to(device)
        if tuple(positions.shape) != (sequence_length, 3):
            raise ValueError("H3 control position_ids must be [sequence, 3]")
        fused = self._fuse(control.rows, self.modalities, self.embeddings, self.fusion_proj)
        states = self._scatter(fused, row_indices, sequence_length)
        mask = torch.zeros((1, sequence_length, 1), device=device, dtype=states.dtype)
        mask = mask.index_fill(1, row_indices, float(control.gain))
        wide_states = wide_rotary = None
        if self.wide_modalities:
            if control.wide_rows is None or control.wide_position_ids is None:
                raise ValueError("H3 wide geometry profile requires wide rows and positions")
            wide = self._fuse(
                control.wide_rows,
                self.wide_modalities,
                self.wide_embeddings,
                self.wide_fusion_proj,
            )
            wide_states = self._scatter(self.norm_wide(wide), row_indices, sequence_length)
            wide_states = wide_states[:, start:stop]
            wide_positions = control.wide_position_ids.to(device)[start:stop]
            wide_rotary = rope(wide_positions)
        context = _AttentionContext(
            segments=control.segments,
            wide_segments=control.wide_segments,
            sequence_lengths=tuple(int(value) for value in sequence_lengths),
            backend=self.attention_backend,
        )
        projected = self.temb_proj(F.silu(temb).to(self.temb_proj.weight.dtype))
        return H3ControlStream(
            states=states[:, start:stop],
            mask=mask[:, start:stop],
            rotary=rope(positions[start:stop]),
            temb=projected,
            timestep_indices=timestep_indices,
            context=context,
            wide_states=wide_states,
            wide_rotary=wide_rotary,
        )

    def inject(self, block_index: int, hidden_states: Any, stream: H3ControlStream) -> Any:
        """Advance the trunk past backbone block ``block_index`` and add its residual."""

        control_index = self.block_to_control.get(block_index)
        if control_index is None:
            return hidden_states
        block = self.control_blocks[control_index]
        arguments = (
            stream.states,
            stream.temb,
            stream.timestep_indices,
            stream.rotary,
            stream.context,
            stream.wide_states,
            stream.wide_rotary,
        )
        if self.activation_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            stream.states, residual = checkpoint(block, *arguments, use_reentrant=False)
        else:
            stream.states, residual = block(*arguments)
        return hidden_states + residual.to(hidden_states.dtype) * stream.mask.to(
            hidden_states.dtype
        )


def _parameter_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@torch.no_grad()
def initialize_geometry_controlnet(module: H3GeometryControlNet) -> None:
    """FastVideo's rule for parameters no checkpoint provides.

    Biases and every ``proj_out`` are zero (an untrained trunk is an exact
    no-op), norm weights are one, and each remaining matrix is kaiming-uniform
    from a generator seeded by its stable FastVideo name. Global RNG streams
    are untouched, so LoRA initialisation does not depend on the ControlNet.
    """

    for local_name, parameter in module.named_parameters():
        name = f"{H3_GEOMETRY_MODULE}.{local_name}"
        if name.endswith("bias") or "proj_out" in name:
            parameter.zero_()
        elif "norm_" in name and name.endswith("weight"):
            parameter.fill_(1.0)
        else:
            generator = torch.Generator(device=parameter.device).manual_seed(_parameter_seed(name))
            value = torch.empty(parameter.shape, device=parameter.device, dtype=torch.float32)
            if value.ndim >= 2:
                fan_in = value.shape[1] * math.prod(value.shape[2:])
                gain = math.sqrt(2.0 / (1.0 + 5.0))
                bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
                value.uniform_(-bound, bound, generator=generator)
            else:
                value.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.copy_(value.to(parameter.dtype))


def build_geometry_controlnet(
    profile: H3GeometryProfile,
    *,
    device: Any,
    dtype: Any = torch.bfloat16,
    hidden_size: int = 5376,
    time_embed_dim: int = 2688,
    num_layers: int = 50,
    video_patch_dim: int = 96,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
    rotary_channels: int = H3_ROTARY_CHANNELS,
) -> H3GeometryControlNet:
    """Construct on ``meta`` (no RNG use), materialize on ``device``, then initialise."""

    with torch.device("meta"):
        module = H3GeometryControlNet(
            profile,
            video_patch_dim=video_patch_dim,
            hidden_size=hidden_size,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            rotary_channels=rotary_channels,
        )
    module = module.to_empty(device=device).to(dtype)
    initialize_geometry_controlnet(module)
    return module


__all__ = [
    "H3ControlAttention",
    "H3ControlStream",
    "H3ControlSwiGLU",
    "H3ControlWideCrossAttention",
    "H3GeometryControl",
    "H3GeometryControlBlock",
    "H3GeometryControlNet",
    "apply_rotary",
    "build_geometry_controlnet",
    "control_segments",
    "geometry_control_for_layout",
    "initialize_geometry_controlnet",
    "pad_control_latents",
    "teacher_forced_geometry_control",
    "wide_position_ids",
]
