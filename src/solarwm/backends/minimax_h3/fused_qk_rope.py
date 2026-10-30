"""Frozen per-head RMSNorm and native H3 RoPE, with original bf16 boundaries."""
from collections import Counter
import math

import torch
import triton
import triton.language as tl

COUNTS = Counter()
_REASONS = set()


@triton.jit
def _forward_kernel(Q, K, WQ, WK, C, S, OQ, OK, RQ, RK,
                    N, H: tl.constexpr, QS, QH, KS, KH, CS, CC, SS, SC,
                    EQ: tl.constexpr, EK: tl.constexpr, ROWS: tl.constexpr):
    which = tl.program_id(1)
    X = tl.where(which == 0, Q, K)
    W = tl.where(which == 0, WQ, WK)
    O = tl.where(which == 0, OQ, OK)
    R = tl.where(which == 0, RQ, RK)
    xs = tl.where(which == 0, QS, KS)
    xh = tl.where(which == 0, QH, KH)
    eps = tl.where(which == 0, EQ, EK)
    row = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    token, head = row // H, row % H
    ch = tl.arange(0, 128)
    x = tl.load(X + token[:, None] * xs + head[:, None] * xh + ch[None, :],
                token[:, None] < N, other=0).to(tl.float32)
    w = tl.load(W + ch).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / 128.0 + eps)
    norm = ((x * r[:, None]) * w[None, :]).to(tl.bfloat16).to(tl.float32)
    partner = tl.where(ch < 96, (ch + 48) % 96, ch)
    rotated = tl.gather(norm, tl.broadcast_to(partner[None, :], (ROWS, 128)), 1)
    rotated = tl.where(ch[None, :] < 48, -rotated, rotated)
    co = tl.load(C + token[:, None] * CS + ch[None, :] * CC,
                 (token[:, None] < N) & (ch[None, :] < 96), other=0).to(tl.bfloat16).to(tl.float32)
    si = tl.load(S + token[:, None] * SS + ch[None, :] * SC,
                 (token[:, None] < N) & (ch[None, :] < 96), other=0).to(tl.bfloat16).to(tl.float32)
    a = (norm * co).to(tl.bfloat16).to(tl.float32)
    b = (rotated * si).to(tl.bfloat16).to(tl.float32)
    out = tl.where(ch[None, :] < 96, a + b, norm)
    tl.store(O + row[:, None] * 128 + ch[None, :], out, token[:, None] < N)
    tl.store(R + row, r, token < N)


@triton.jit
def _backward_kernel(DQ, DK, Q, K, WQ, WK, C, S, RQ, RK, XQ, XK,
                     N, H: tl.constexpr, QS, QH, KS, KH,
                     DQS, DQH, DQC, DKS, DKH, DKC, CS, CC, SS, SC,
                     ROWS: tl.constexpr):
    which = tl.program_id(1)
    X = tl.where(which == 0, Q, K)
    W = tl.where(which == 0, WQ, WK)
    DY = tl.where(which == 0, DQ, DK)
    DX = tl.where(which == 0, XQ, XK)
    R = tl.where(which == 0, RQ, RK)
    xs, xh = tl.where(which == 0, QS, KS), tl.where(which == 0, QH, KH)
    ds = tl.where(which == 0, DQS, DKS)
    dh = tl.where(which == 0, DQH, DKH)
    dc = tl.where(which == 0, DQC, DKC)
    row = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    token, head = row // H, row % H
    ch = tl.arange(0, 128)
    dy = tl.load(DY + token[:, None] * ds + head[:, None] * dh + ch[None, :] * dc,
                 token[:, None] < N, other=0).to(tl.float32)
    partner = tl.where(ch < 96, (ch + 48) % 96, ch)
    other_dy = tl.gather(dy, tl.broadcast_to(partner[None, :], (ROWS, 128)), 1)
    co = tl.load(C + token[:, None] * CS + ch[None, :] * CC,
                 (token[:, None] < N) & (ch[None, :] < 96), other=0).to(tl.bfloat16).to(tl.float32)
    si = tl.load(S + token[:, None] * SS + partner[None, :] * SC,
                 (token[:, None] < N) & (ch[None, :] < 96), other=0).to(tl.bfloat16).to(tl.float32)
    a = (dy * co).to(tl.bfloat16).to(tl.float32)
    b = (other_dy * si).to(tl.bfloat16).to(tl.float32)
    b = tl.where(ch[None, :] < 48, b, -b)
    dn = tl.where(ch[None, :] < 96, a + b, dy).to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + token[:, None] * xs + head[:, None] * xh + ch[None, :],
                token[:, None] < N, other=0).to(tl.float32)
    w = tl.load(W + ch).to(tl.float32)
    r = tl.load(R + row, token < N, other=0)
    g = dn * w[None, :]
    dot = tl.sum(g * x, 1)
    dmean = ((-0.5 * dot) * ((r * r) * r)) / 128.0
    dx = g * r[:, None] + (2.0 * x) * dmean[:, None]
    tl.store(DX + row[:, None] * 128 + ch[None, :], dx, token[:, None] < N)


@torch.library.custom_op('solar_qk_rope::forward', mutates_args=(), device_types='cuda')
def _forward(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
             c: torch.Tensor, s: torch.Tensor, eq: float, ek: float
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, h = q.shape[1:3]
    oq, ok = (torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (q, k))
    rq, rk = (torch.empty((n, h), device=q.device, dtype=torch.float32) for _ in range(2))
    _forward_kernel[(triton.cdiv(n * h, 8), 2)](
        q, k, wq, wk, c, s, oq, ok, rq, rk, n, h,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), *c.stride(), *s.stride(),
        eq, ek, 8, num_warps=4, enable_fp_fusion=False)
    COUNTS['forward'] += 1
    if COUNTS['forward'] == 1:
        print('[solar-qk-rope] fused bf16 norm/RoPE forward active', flush=True)
    return oq, ok, rq, rk


@_forward.register_fake
def _forward_fake(q, k, wq, wk, c, s, eq, ek):
    return (torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(q.shape[1:3], device=q.device, dtype=torch.float32),
            torch.empty(q.shape[1:3], device=q.device, dtype=torch.float32))


@torch.library.custom_op('solar_qk_rope::backward', mutates_args=(), device_types='cuda')
def _backward(dq: torch.Tensor, dk: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
              wq: torch.Tensor, wk: torch.Tensor, c: torch.Tensor, s: torch.Tensor,
              rq: torch.Tensor, rk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, h = q.shape[1:3]
    xq, xk = (torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (q, k))
    _backward_kernel[(triton.cdiv(n * h, 8), 2)](
        dq, dk, q, k, wq, wk, c, s, rq, rk, xq, xk, n, h,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        *dq.stride()[1:], *dk.stride()[1:], *c.stride(), *s.stride(),
        8, num_warps=4, enable_fp_fusion=False)
    COUNTS['backward'] += 1
    return xq, xk


@_backward.register_fake
def _backward_fake(dq, dk, q, k, wq, wk, c, s, rq, rk):
    return tuple(torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (q, k))


def _setup(ctx, inputs, output):
    q, k, wq, wk, c, s, eq, ek = inputs
    oq, ok, rq, rk = output
    ctx.save_for_backward(q, k, wq, wk, c, s, rq, rk)
    ctx.mark_non_differentiable(rq, rk)
    ctx.set_materialize_grads(False)


def _autograd(ctx, dq, dk, drq, drk):
    q, k, wq, wk, c, s, rq, rk = ctx.saved_tensors
    used_q, used_k = dq is not None, dk is not None
    if not used_q and not used_k:
        return (None,) * 8
    if dq is None:
        dq = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.zeros(k.shape, device=k.device, dtype=k.dtype)
    xq, xk = _backward(dq, dk, q, k, wq, wk, c, s, rq, rk)
    return xq if used_q else None, xk if used_k else None, None, None, None, None, None, None


torch.library.register_autograd('solar_qk_rope::forward', _autograd, setup_context=_setup)


def apply(q, k, norm_q, norm_k, c, s):
    """Use the fused path only for frozen native H3 head norms and angles."""
    wq, wk = norm_q.weight, norm_k.weight
    valid = (q.is_cuda and q.dtype == torch.bfloat16 and q.ndim == 4
             and q.shape[0] == 1 and q.shape[-1] == 128 and q.shape[1] > 0
             and q.shape == k.shape and q.dtype == k.dtype and q.device == k.device
             and q.stride(-1) == k.stride(-1) == 1
             and tuple(norm_q.normalized_shape) == tuple(norm_k.normalized_shape) == (128,)
             and wq is not None and wk is not None and wq.shape == wk.shape == (128,)
             and wq.dtype == wk.dtype and wq.dtype in (torch.bfloat16, torch.float32)
             and wq.device == wk.device == q.device and not wq.requires_grad and not wk.requires_grad
             and wq.stride(0) == wk.stride(0) == 1
             and c.ndim == s.ndim == 2 and c.shape == s.shape == (q.shape[1], 96)
             and c.device == s.device == q.device and not c.requires_grad and not s.requires_grad
             and c.dtype in (torch.bfloat16, torch.float32) and s.dtype in (torch.bfloat16, torch.float32)
             and isinstance(norm_q.eps, (float, int)) and isinstance(norm_k.eps, (float, int))
             and math.isfinite(norm_q.eps) and math.isfinite(norm_k.eps)
             and min(norm_q.eps, norm_k.eps) >= 0)
    if valid:
        oq, ok, _, _ = _forward(q, k, wq, wk, c, s, float(norm_q.eps), float(norm_k.eps))
        return oq, ok
    COUNTS['fallback'] += 1
    reason = (tuple(q.shape), str(q.dtype), str(q.device),
              None if wq is None else (str(wq.dtype), wq.requires_grad),
              None if wk is None else (str(wk.dtype), wk.requires_grad), tuple(c.shape))
    if reason not in _REASONS:
        _REASONS.add(reason)
        print(f'[solar-qk-rope] original formula fallback: {reason}', flush=True)
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    return _apply_rotary_emb(norm_q(q), c, s), _apply_rotary_emb(norm_k(k), c, s)


def install(model):
    if len(model.transformer_blocks) != 50:
        raise RuntimeError('H3 Q/K norm/RoPE expects 50 blocks')
    for block in model.transformer_blocks:
        block.attn._solar_fused_qk_rope = True
