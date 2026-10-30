"""Iteration-local selective attention checkpoints with exact host offload.

Uses torch's selective-checkpoint modes (torch 2.8). Attention outputs must be
immutable, as required by the original selective-checkpoint policy as well.
"""
from __future__ import annotations

import atexit
import json
import weakref
from collections import Counter, defaultdict

import torch
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import _CachingTorchDispatchMode, _CachedTorchDispatchMode

STATS = Counter()
_STREAMS = {}


def _stream(device):
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device=device)
    return _STREAMS[device]


class _HostTensor:
    def __init__(self, tensor):
        self.source = weakref.ref(tensor)
        self.version = tensor._version
        self.shape, self.stride = tensor.shape, tensor.stride()
        self.device, self.dtype = tensor.device, tensor.dtype
        # Rounded pinned allocations are reusable across the reader's lengths.
        extent = 1 + sum((n - 1) * s for n, s in zip(self.shape, self.stride))
        capacity = 1 << (extent - 1).bit_length()
        self.host = torch.empty(capacity, dtype=self.dtype, pin_memory=True).as_strided(
            self.shape, self.stride
        )
        stream = _stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.host.copy_(tensor.detach(), non_blocking=True)
            tensor.record_stream(stream)
            self.saved = torch.cuda.Event()
            self.saved.record(stream)
        self.value = None
        STATS['d2h_bytes'] += tensor.numel() * tensor.element_size()

    def prefetch(self):
        if self.value is not None:
            return
        compute = torch.cuda.current_stream(self.device)
        # Allocate in the compute stream's allocator pool, not the copy pool.
        self.value = torch.empty_strided(
            self.shape, self.stride, dtype=self.dtype, device=self.device
        )
        stream = _stream(self.device)
        stream.wait_stream(compute)
        stream.wait_event(self.saved)
        with torch.cuda.stream(stream):
            self.value.copy_(self.host, non_blocking=True)
            self.value.record_stream(stream)
            self.ready = torch.cuda.Event()
            self.ready.record(stream)
        STATS['h2d_bytes'] += self.value.numel() * self.value.element_size()

    def get_val(self, allow_cache_entry_mutation):
        source = self.source()
        if not allow_cache_entry_mutation and source is not None and source._version != self.version:
            raise RuntimeError('Attention output mutated after checkpoint caching')
        self.prefetch()
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(self.ready)
        self.value.record_stream(compute)
        return self.value


class _Save(_CachingTorchDispatchMode):
    def __init__(self, policy, storage, ops, offload):
        super().__init__(policy, storage)
        self.ops, self.offload = ops, offload

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        if func in self.ops:
            STATS['saved'] += 1
            if self.offload:
                def transfer(wrapper, original):
                    if isinstance(original, torch.Tensor) and original.is_cuda and original.numel():
                        return _HostTensor(original)
                    return wrapper
                self.storage[func][-1] = tree_map(transfer, self.storage[func][-1], out)
                STATS['cpu_saved'] += 1
        return out


class _Replay(_CachedTorchDispatchMode):
    def __init__(self, policy, storage, ops, offload):
        super().__init__(policy, storage, False)
        self.ops, self.offload = ops, offload

    def __enter__(self):
        def prefetch(value):
            if isinstance(value, _HostTensor):
                value.prefetch()
        # Only this block's entries. Start H2D before Q/K/V recomputation.
        tree_map(prefetch, dict(self.storage))
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        if func in self.ops:
            STATS['replayed'] += 1
            if self.offload:
                STATS['cpu_replayed'] += 1
        return out


def contexts(*, offload):
    from .fsdp import keep_attention, _attention_ops
    storage = defaultdict(list)
    ops = _attention_ops()
    return _Save(keep_attention, storage, ops, offload), _Replay(keep_attention, storage, ops, offload)


@atexit.register
def _report():
    if STATS:
        print('[solar-attention-cache] ' + json.dumps(dict(STATS)), flush=True)
