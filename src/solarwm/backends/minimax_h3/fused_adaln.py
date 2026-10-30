"""Indexed AdaLN kernels for H3's frozen affine weights and bf16 activations."""
from __future__ import annotations

import logging
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

COUNTS = {"norm": 0, "gate": 0, "norm_backward": 0, "gate_backward": 0, "fallback": 0}
_REASONS = set()
_GATE_CHECKED = False
_GATE_BACKWARD_CHECKED = False


@triton.jit
def _norm_forward_kernel(X, W, SCALE, SHIFT, IDS, Y, RSTD,
                         WIDTH: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
                         SS: tl.constexpr, HS: tl.constexpr, EPS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * XS + col, valid, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    w = tl.load(W + col * WS, valid, other=0).to(tl.float32)
    normalized = ((x * r) * w).to(tl.bfloat16).to(tl.float32)
    index = tl.load(IDS + row)
    scale = tl.load(SCALE + index * SS + col, valid, other=0).to(tl.float32)
    shift = tl.load(SHIFT + index * HS + col, valid, other=0).to(tl.float32)
    plus = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
    product = (normalized * plus).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * WIDTH + col, (product + shift).to(tl.bfloat16), valid)
    tl.store(RSTD + row, r)


@triton.jit
def _norm_backward_kernel(DY, X, W, SCALE, IDS, RSTD, DX,
                          WIDTH: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
                          SS: tl.constexpr, DYS: tl.constexpr, DYC: tl.constexpr,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * XS + col, valid, other=0).to(tl.float32)
    dy = tl.load(DY + row * DYS + col * DYC, valid, other=0).to(tl.float32)
    index = tl.load(IDS + row)
    scale = tl.load(SCALE + index * SS + col, valid, other=0).to(tl.float32)
    plus = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
    dn = (dy * plus).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + col * WS, valid, other=0).to(tl.float32)
    g = dn * w
    r = tl.load(RSTD + row)
    dot = tl.sum(g * x, 0)
    dmean = ((-0.5 * dot) * ((r * r) * r)) / WIDTH
    dx = g * r + (2.0 * x) * dmean
    tl.store(DX + row * WIDTH + col, dx.to(tl.bfloat16), valid)


@triton.jit
def _gate_forward_kernel(R, U, TABLE, IDS, Y, N,
                         WIDTH: tl.constexpr, RS: tl.constexpr, US: tl.constexpr,
                         TS: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    row, col = pos // WIDTH, pos % WIDTH
    index = tl.load(IDS + row, valid, other=0)
    r = tl.load(R + row * RS + col, valid, other=0).to(tl.float32)
    u = tl.load(U + row * US + col, valid, other=0).to(tl.float32)
    gate = tl.load(TABLE + index * TS + col, valid, other=0).to(tl.float32)
    product = (gate * u).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + pos, (r + product).to(tl.bfloat16), valid)


@triton.jit
def _gate_backward_kernel(DY, TABLE, IDS, DU, N,
                          WIDTH: tl.constexpr, DYS: tl.constexpr, DYC: tl.constexpr,
                          TS: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    row, col = pos // WIDTH, pos % WIDTH
    index = tl.load(IDS + row, valid, other=0)
    dy = tl.load(DY + row * DYS + col * DYC, valid, other=0).to(tl.float32)
    gate = tl.load(TABLE + index * TS + col, valid, other=0).to(tl.float32)
    tl.store(DU + pos, (dy * gate).to(tl.bfloat16), valid)


def _norm_impl(x, weight, scale, shift, indices, eps):
    y = F.rms_norm(x, (x.shape[-1],), weight, eps)
    return y * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _gate_impl(residual, update, table, indices):
    return residual + table.index_select(0, indices) * update


@torch.library.custom_op("solar_adaln::norm", mutates_args=(), device_types="cuda")
def _norm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
          shift: torch.Tensor, indices: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    r = torch.empty((x.shape[1],), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(x.shape[-1])
    _norm_forward_kernel[(x.shape[1],)](
        x, weight, scale, shift, indices, y, r, x.shape[-1], x.stride(1), weight.stride(0),
        scale.stride(0), shift.stride(0), eps, block,
        num_warps=min(8, max(1, block // 256)), enable_fp_fusion=False)
    COUNTS["norm"] += 1
    return y, r


@_norm.register_fake
def _norm_fake(x, weight, scale, shift, indices, eps):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype), x.new_empty((x.shape[1],), dtype=torch.float32)


@torch.library.custom_op("solar_adaln::norm_backward", mutates_args=(), device_types="cuda")
def _norm_backward(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                   scale: torch.Tensor, indices: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block = triton.next_power_of_2(x.shape[-1])
    _norm_backward_kernel[(x.shape[1],)](
        dy, x, weight, scale, indices, r, dx, x.shape[-1], x.stride(1), weight.stride(0),
        scale.stride(0), dy.stride(1), dy.stride(2), block,
        num_warps=min(8, max(1, block // 256)), enable_fp_fusion=False)
    COUNTS["norm_backward"] += 1
    return dx


@_norm_backward.register_fake
def _norm_backward_fake(dy, x, weight, scale, indices, r):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _norm_setup(ctx, inputs, output):
    x, weight, scale, shift, indices, eps = inputs
    ctx.save_for_backward(x, weight, scale, indices, output[1])
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _norm_autograd(ctx, dy, dr):
    dx = None if dy is None else _norm_backward(dy, *ctx.saved_tensors)
    return dx, None, None, None, None, None


torch.library.register_autograd("solar_adaln::norm", _norm_autograd, setup_context=_norm_setup)


@torch.library.custom_op("solar_adaln::gate", mutates_args=(), device_types="cuda")
def _gate(residual: torch.Tensor, update: torch.Tensor, table: torch.Tensor,
          indices: torch.Tensor) -> torch.Tensor:
    global _GATE_CHECKED
    y = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    _gate_forward_kernel[(triton.cdiv(y.numel(), 2048),)](
        residual, update, table, indices, y, y.numel(), y.shape[-1], residual.stride(1),
        update.stride(1), table.stride(0), 2048, num_warps=4, enable_fp_fusion=False)
    if not _GATE_CHECKED:
        if not torch.equal(y, _gate_impl(residual, update, table, indices)):
            raise RuntimeError("Indexed gate forward failed its live equality check")
        _GATE_CHECKED = True
    COUNTS["gate"] += 1
    return y


@_gate.register_fake
def _gate_fake(residual, update, table, indices):
    return torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)


@torch.library.custom_op("solar_adaln::gate_backward", mutates_args=(), device_types="cuda")
def _gate_backward(dy: torch.Tensor, table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    global _GATE_BACKWARD_CHECKED
    du = torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)
    _gate_backward_kernel[(triton.cdiv(du.numel(), 2048),)](
        dy, table, indices, du, du.numel(), du.shape[-1], dy.stride(1), dy.stride(2),
        table.stride(0), 2048, num_warps=4, enable_fp_fusion=False)
    if not _GATE_BACKWARD_CHECKED:
        if not torch.equal(du, dy * table.index_select(0, indices)):
            raise RuntimeError("Indexed gate backward failed its live equality check")
        _GATE_BACKWARD_CHECKED = True
    COUNTS["gate_backward"] += 1
    return du


@_gate_backward.register_fake
def _gate_backward_fake(dy, table, indices):
    return torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)


def _gate_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[2], inputs[3])


def _gate_autograd(ctx, dy):
    return dy, _gate_backward(dy, *ctx.saved_tensors), None, None


torch.library.register_autograd("solar_adaln::gate", _gate_autograd, setup_context=_gate_setup)


def _fallback(reason):
    COUNTS["fallback"] += 1
    if reason not in _REASONS:
        _REASONS.add(reason)
        logging.getLogger(__name__).warning("[solar-adaln] original formula: %s", reason)


def _layout(x):
    return (x.device.type == "cuda" and x.dtype == torch.bfloat16 and x.ndim == 3
            and x.shape[0] == 1 and x.shape[1] > 0 and x.stride(2) == 1)


def _table(t, x):
    return (t.device == x.device and t.dtype == x.dtype and t.ndim == 2
            and t.shape[1] == x.shape[2] and t.stride(1) == 1 and not t.requires_grad)


def norm(x, weight, scale, shift, indices, eps):
    if (not _layout(x) or not 128 <= x.shape[-1] <= 8192
            or weight.device != x.device or weight.ndim != 1 or weight.numel() != x.shape[-1]
            or weight.requires_grad or weight.dtype not in (torch.bfloat16, torch.float32)
            or not _table(scale, x) or not _table(shift, x) or indices.device != x.device
            or indices.dtype != torch.int64 or indices.ndim != 1 or indices.stride(0) != 1
            or indices.numel() != x.shape[1]):
        _fallback("norm needs B=1 CUDA bf16 contiguous channels, width 128..8192, frozen affine tables")
        return _norm_impl(x, weight, scale, shift, indices, eps)
    return _norm(x, weight, scale, shift, indices, float(eps))[0]


def gate(residual, update, table, indices):
    if (not _layout(residual) or not _layout(update) or update.shape != residual.shape
            or update.device != residual.device or not _table(table, residual)
            or indices.device != residual.device or indices.dtype != torch.int64
            or indices.ndim != 1 or indices.stride(0) != 1 or indices.numel() != residual.shape[1]):
        _fallback("gate needs B=1 CUDA bf16 contiguous channels and a frozen gate table")
        return _gate_impl(residual, update, table, indices)
    return _gate(residual, update, table, indices)


def _block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
    residual = hidden_states
    normalized = norm(hidden_states, self.norm1.weight, scale_msa, shift_msa, adaln_indices, self.norm1.eps)
    attention = self.attn(normalized, rotary_emb, attention_mask)
    hidden_states = gate(residual, attention, gate_msa, adaln_indices)
    residual = hidden_states
    normalized = norm(hidden_states, self.norm2.weight, scale_mlp, shift_mlp, adaln_indices, self.norm2.eps)
    update = self.ff(normalized)
    return gate(residual, update, gate_mlp, adaln_indices)


def install(model):
    for block in model.transformer_blocks:
        block.forward = types.MethodType(_block_forward, block)
