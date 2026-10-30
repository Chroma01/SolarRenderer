"""Ulysses packing exactness probe (SOLAR_SP_PACK_B1, eight GPUs); not collected by pytest.

From the repository root:
SOLAR_PROBE_OUT=<dir> PYTHONPATH=src torchrun --standalone --nproc-per-node 8 tests/gpu/probe_sp_pack.py

Compares the packed and the original exchange bit for bit, forward and backward; writes profiler traces to
SOLAR_PROBE_OUT.
"""
import contextlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from solarwm.backends.minimax_h3.distributed import (
    _PACK_COUNTS,
    register_sequence_length,
    sequence_all_to_all,
)
from solarwm.backends.minimax_h3.fsdp import initialize_distributed


def rng():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def unchanged(before):
    assert all(torch.equal(a, b) for a, b in zip(before, rng()))


def exchange(x, scatter, gather, enabled, dy=None):
    os.environ["SOLAR_SP_PACK_B1"] = str(int(enabled))
    before = rng()
    y = sequence_all_to_all(x, scatter, gather)
    unchanged(before)
    if dy is None:
        dy = torch.randn_like(y)
    before = rng()
    grad, = torch.autograd.grad(y, x, dy)
    unchanged(before)
    return y.detach(), grad.detach(), dy


def run_case(rank, dtype, layout, length=129, heads=8, width=128, batch=1):
    torch.manual_seed(1009 + rank)
    n = length + (rank % 2 if layout == "unequal" else 0)
    register_sequence_length(n)
    if layout == "head_major":
        x = torch.randn(batch, heads, n, width, device="cuda", dtype=dtype).transpose(1, 2)
    elif layout == "channel_stride":
        x = torch.randn(batch, n, heads, width * 2, device="cuda", dtype=dtype)[..., ::2]
    elif layout == "offset":
        x = torch.randn(batch, n + 1, heads, width, device="cuda", dtype=dtype)[:, 1:]
    else:
        x = torch.randn(batch, n, heads, width, device="cuda", dtype=dtype)
    x.requires_grad_()
    ref, grad_ref, dy = exchange(x, 2, 1, False)
    got, grad, _ = exchange(x, 2, 1, True, dy)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    torch.testing.assert_close(grad, grad_ref, rtol=0, atol=0)
    assert got.is_contiguous()
    z = got.detach().requires_grad_()
    ref_back, grad_ref_back, dy_back = exchange(z, 1, 2, False)
    back, grad_back, _ = exchange(z, 1, 2, True, dy_back)
    torch.testing.assert_close(back, ref_back, rtol=0, atol=0)
    torch.testing.assert_close(grad_back, grad_ref_back, rtol=0, atol=0)
    torch.testing.assert_close(back, x, rtol=0, atol=0)
    assert back.is_contiguous()
    if rank == 0:
        print("PASS exact output/gradient/roundtrip/RNG", dtype, layout, n, heads, batch, flush=True)


def trace_layout(rank, unequal):
    n = 4097 + (rank % 2 if unequal else 0)
    register_sequence_length(n)
    x = torch.randn(1, n, 56, 128, device="cuda", dtype=torch.bfloat16)
    counts = []
    for enabled in (False, True):
        os.environ["SOLAR_SP_PACK_B1"] = str(int(enabled))
        ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if rank == 0 else contextlib.nullcontext()
        with ctx as p:
            y = sequence_all_to_all(x, 2, 1)
            z = sequence_all_to_all(y, 1, 2)
            torch.cuda.synchronize()
        torch.testing.assert_close(z, x, rtol=0, atol=0)
        if rank == 0:
            p.export_chrome_trace(str(Path(os.environ["SOLAR_PROBE_OUT"]) / f"pack_u{int(unequal)}_{int(enabled)}.trace.json"))
            counts.append({e.key: e.count for e in p.key_averages()})
    if rank == 0:
        assert counts[0].get("aten::cat", 0) > 0
        assert counts[1].get("aten::cat", 0) == 0
        assert counts[1].get("aten::clone", 0) < counts[0].get("aten::clone", 0)
        print("PASS actual cat removal/clone reduction", "unequal", unequal, [{k: c.get(k, 0) for k in ("aten::cat", "aten::clone")} for c in counts], flush=True)


def sac_probe(rank):
    n = 31 + rank % 2
    register_sequence_length(n)
    x = torch.randn(1, n, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def fn(value):
        y = sequence_all_to_all(value, 2, 1)
        return sequence_all_to_all(y, 1, 2).sin()

    def contexts():
        return create_selective_checkpoint_contexts(lambda *_a, **_kw: CheckpointPolicy.PREFER_RECOMPUTE)

    results = []
    for enabled in (False, True):
        os.environ["SOLAR_SP_PACK_B1"] = str(int(enabled))
        before = rng()
        out = checkpoint(fn, x, use_reentrant=False, context_fn=contexts)
        grad, = torch.autograd.grad(out, x, dy)
        unchanged(before)
        results.append((out.detach(), grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if rank == 0:
        print("PASS checkpoint/SAC unequal output/gradient/RNG", flush=True)


if __name__ == "__main__":
    rank, _ = initialize_distributed(sp_size=2, local_rank=int(os.environ["LOCAL_RANK"]))
    torch.use_deterministic_algorithms(True)
    for dtype in (torch.bfloat16, torch.float32):
        for layout in ("contiguous", "head_major", "channel_stride", "offset", "unequal"):
            run_case(rank, dtype, layout)
        run_case(rank, dtype, "contiguous", length=7, batch=2)
        run_case(rank, dtype, "contiguous", length=0)
        run_case(rank, dtype, "unequal", length=0)
        for n in (1, 127, 128, 257):
            run_case(rank, dtype, "contiguous", length=n)
    for layout in ("contiguous", "head_major", "unequal"):
        run_case(rank, torch.bfloat16, layout, length=52340, heads=56)
    sac_probe(rank)
    for unequal in (False, True):
        trace_layout(rank, unequal)
    assert all(_PACK_COUNTS.get(k, 0) for k in ("sequence_to_heads", "heads_to_sequence", "sequence_to_heads_unequal", "heads_to_sequence_unequal", "fallback_batch", "fallback_empty_peer"))
    dist.barrier()
    if rank == 0:
        print("PASS all eight ranks; both directions, all gradients and fallback paths", flush=True)
    dist.destroy_process_group()
