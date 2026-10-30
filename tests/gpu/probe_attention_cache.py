"""Attention output cache regression probe (one GPU); not collected by pytest.

From the repository root:
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python tests/gpu/probe_attention_cache.py

Checks that cached and offloaded attention outputs give the same gradients as recomputation; correctness diagnostics
only.
"""

if __name__ == "__main__":
    import copy, functools
    import torch
    from flash_attn import flash_attn_func
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
    from solarwm.backends.minimax_h3.fsdp import keep_attention
    from solarwm.backends.minimax_h3.attention_cache import contexts, STATS

    torch.use_deterministic_algorithms(True)
    class Block(torch.nn.Module):
     def __init__(self):
      super().__init__(); self.qkv=torch.nn.Linear(512,1536,device='cuda',dtype=torch.bfloat16);self.out=torch.nn.Linear(512,512,device='cuda',dtype=torch.bfloat16)
     def forward(self,x,drop):
      q,k,v=self.qkv(x).reshape(1,-1,3,4,128).unbind(2)
      a=flash_attn_func(q,k,v,dropout_p=drop,deterministic=True)
      return x+self.out(a.flatten(2)).tanh()

    def run(blocks,inp,seed,new,drop):
     torch.manual_seed(seed); x=inp.detach().clone().requires_grad_(); leaf=x
     for i,b in enumerate(blocks):
      kw={}
      if new: kw['context_fn']=functools.partial(contexts,offload=i<2)
      elif i<2: kw['context_fn']=functools.partial(create_selective_checkpoint_contexts,keep_attention)
      x=checkpoint(b,x,drop,use_reentrant=False,**kw)
     loss=x.float().square().mean();loss.backward()
     return x.detach(),[leaf.grad.detach().clone()]+[p.grad.detach().clone() for b in blocks for p in b.parameters()],torch.cuda.get_rng_state().clone()

    torch.manual_seed(13)
    a=torch.nn.ModuleList([Block() for _ in range(4)]);b=copy.deepcopy(a)
    x=torch.randn(1,257,512,device='cuda',dtype=torch.bfloat16)
    for step,drop in enumerate([0.,0.05,1e-30,0.]):
     for m in [a,b]:m.zero_grad(set_to_none=True)
     ra=run(a,x,40+step,False,drop); rb=run(b,x,40+step,True,drop)
     assert torch.equal(ra[0],rb[0]),'output'
     assert torch.equal(ra[2],rb[2]),'rng'
     for i,(ga,gb) in enumerate(zip(ra[1],rb[1])):assert torch.equal(ga,gb),('grad',i,(ga-gb).abs().max())
     with torch.no_grad():
      for pa,pb in zip(a.parameters(),b.parameters()):pa.add_(pa.grad,alpha=-0.01);pb.add_(pb.grad,alpha=-0.01)
     torch.cuda.synchronize();print('PASS step',step,'dropout',drop,dict(STATS),flush=True)
    assert STATS['saved']==STATS['replayed']==16
    assert STATS['cpu_saved']==STATS['cpu_replayed']==8
    print('PASS all exact, RNG unchanged, no stale steps',flush=True)

    from solarwm.backends.minimax_h3.attention_cache import _HostTensor
    for dtype in [torch.bfloat16,torch.float32,torch.int64]:
     for shape in [(3,17,29),(1,104680,28,128)]:
      base=torch.arange(torch.tensor(shape).prod().item()+11,device='cuda',dtype=torch.int64).to(dtype)
      t=base[11:].view(shape).transpose(-1,-2)
      expected=t.cpu(); stride=t.stride()
      holder=_HostTensor(t); del t,base
      # A second stream must observe the completed D2H and producer dependencies.
      other=torch.cuda.Stream()
      with torch.cuda.stream(other):
       holder.prefetch(); result=holder.get_val(False)
       assert result.stride()==stride
       assert torch.equal(result.cpu(),expected)
      del holder,result,expected
      torch.cuda.synchronize()
      print('PASS strided transfer',dtype,shape,flush=True)
    t=torch.ones(13,device='cuda');h=_HostTensor(t);t.add_(1)
    try:h.get_val(False)
    except RuntimeError:pass
    else:raise AssertionError('missing version check')
    print('PASS strided/dtypes/lifetime/stream/version checks',flush=True)
