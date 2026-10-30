"""Dense bf16 H3 attention using the locally shipped FlashAttention Hopper kernels."""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path

import torch


@functools.cache
def initialize():
    """Compile only the supported bf16/128 specialization, once per process group."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("hopper_src")
    # CUTLASS headers: the NGC PyTorch image ships them with its FlashAttention sources.
    cutlass = Path(os.environ.get("SOLAR_CUTLASS_INCLUDE")
                   or "/opt/pytorch/pytorch/third_party/flash-attention/csrc/cutlass/include")
    if not cutlass.is_dir():
        raise RuntimeError(f"Hopper attention needs the CUTLASS headers (not found at {cutlass}); "
                           "point SOLAR_CUTLASS_INCLUDE at a CUTLASS include directory")
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR")
                or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or tempfile.gettempdir())
    build = root / "solar_hopper128"
    build.mkdir(parents=True, exist_ok=True)
    disabled = ("SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP", "PACKGQA",
                "FP16", "FP8", "VARLEN", "HDIM64", "HDIM96", "HDIM192", "HDIM256", "SM8x")
    defines = [f"-DFLASHATTENTION_DISABLE_{name}" for name in disabled]
    if os.environ.get("SOLAR_HOPPER_FWD192") == "1":
        defines.append("-DSOLAR_HOPPER_FWD192")
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name="solar_hopper128_cuda",
        sources=[str(source / name) for name in (
            "flash_api.cpp", "flash_fwd_hdim128_bf16_sm90.cu", "flash_bwd_hdim128_bf16_sm90.cu")],
        extra_include_paths=[str(source), str(cutlass)],
        extra_cflags=["-O3", "-std=c++17", *defines],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "--threads=2",
                          "-gencode=arch=compute_90a,code=sm_90a",
                          "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DNDEBUG", *defines],
        build_directory=str(build),
    )


@torch.library.custom_op("solar_hopper::forward", mutates_args=(), device_types="cuda")
def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = (x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v))
    out, lse, *_ = initialize().fwd(
        q, k, v, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        q.shape[-1] ** -0.5, False, -1, -1, 0, 0.0, True, 1, False, 0,
    )
    return out, lse


@_forward.register_fake
def _forward_fake(q, k, v):
    return (torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32))


@torch.library.custom_op("solar_hopper::backward", mutates_args=(), device_types="cuda")
def _backward(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              out: torch.Tensor, lse: torch.Tensor, deterministic: bool
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out = (x.contiguous() if x.stride(-1) != 1 else x for x in (dout, q, k, v, out))
    dq, dk, dv = (torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k, v))
    initialize().bwd(dout, q, k, v, out, lse, dq, dk, dv,
                     None, None, None, None, None, None,
                     q.shape[-1] ** -0.5, False, -1, -1, 0, 0.0, deterministic, 0)
    return dq, dk, dv


@_backward.register_fake
def _backward_fake(dout, q, k, v, out, lse, deterministic):
    return tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.mark_non_differentiable(lse)
    ctx.deterministic = torch.are_deterministic_algorithms_enabled()


def _autograd_backward(ctx, dout, dlse):
    return _backward(dout, *ctx.saved_tensors, ctx.deterministic)


torch.library.register_autograd("solar_hopper::forward", _autograd_backward, setup_context=_setup_context)


def attention(q, k, v, *, attn_mask=None):
    """Full attention with no dropout; bf16 I/O, fp32 softmax and accumulation."""
    if attn_mask is not None:
        raise ValueError("Hopper H3 backend supports dense unmasked attention only")
    if any(x.dtype != torch.bfloat16 or x.ndim != 4 or x.shape[-1] != 128 for x in (q, k, v)):
        raise ValueError("Hopper H3 backend requires bf16 BSHD inputs with head dimension 128")
    if torch.are_deterministic_algorithms_enabled():
        # The image's Hopper revision has a non-repeatable dQ even with its
        # deterministic flag. FA2 keeps the same bf16 I/O and fp32 math.
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v, dropout_p=0.0, deterministic=True)
    return _forward(q, k, v)[0]
