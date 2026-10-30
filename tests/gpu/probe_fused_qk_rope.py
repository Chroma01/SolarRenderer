"""Fused Q/K RMSNorm + RoPE regression probe (one GPU); not collected by pytest.

From the repository root:
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python tests/gpu/probe_fused_qk_rope.py

Checks the kernels against the eager path; reports correctness diagnostics only.
"""

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    probe_dir = Path(tempfile.mkdtemp(prefix="solar-qk-rope-probe-"))
    import functools,json,torch
    from torch.profiler import profile,ProfilerActivity
    from torch.utils.checkpoint import checkpoint,create_selective_checkpoint_contexts
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    from solarwm.backends.minimax_h3 import fused_qk_rope as f
    from solarwm.backends.minimax_h3.fsdp import keep_attention

    torch.use_deterministic_algorithms(True)
    def rel(a,b):return ((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-20)).item()
    def norms(dtype,eps):
     ns=[torch.nn.RMSNorm(128,eps=eps,device='cuda',dtype=dtype).requires_grad_(False)for _ in range(2)]
     for n in ns:
      w=torch.randn(150,device='cuda',dtype=dtype)[11:139]
      n.weight=torch.nn.Parameter(w,requires_grad=False)
     return ns

    def reference(q,k,nq,nk,c,s):return _apply_rotary_emb(nq(q),c,s),_apply_rotary_emb(nk(k),c,s)

    def case(n,h=4,wd=torch.bfloat16,strided=False,distribution='random',exact=False):
     torch.manual_seed(7);ns=norms(wd,0. if exact else 1e-6)
     sh=(1,n,h*2,128)if strided else(1,n,h,128)
     xs=[torch.randn(sh,device='cuda',dtype=torch.bfloat16)for _ in range(2)]
     xs=[(x[:,:,::2,:]if strided else x).detach().requires_grad_()for x in xs]
     with torch.no_grad():
      if exact:
       for x in xs:x.copy_(x.sign())
      elif distribution=='nearzero':
       for x in xs:x.mul_(1e-6).add_(1e-6)
      elif distribution=='offset':
       for x in xs:x.mul_(.1).add_(20)
     cs=[torch.randn(n,192 if strided else 96,device='cuda',dtype=torch.float32)for _ in range(2)]
     cs=[t[:,::2]if strided else t for t in cs]
     dys=[torch.randn(1,n,h,256,device='cuda',dtype=torch.bfloat16)[...,::2]for _ in range(2)]
     before=torch.cuda.get_rng_state().clone();ys=f.apply(*xs,*ns,*cs);gs=torch.autograd.grad(ys,xs,dys)
     rs=reference(*xs,*ns,*cs);rg=torch.autograd.grad(rs,xs,dys)
     errors=[rel(a,b)for a,b in zip((*ys,*gs),(*rs,*rg))]
     for a,b in zip((*ys,*gs),(*rs,*rg)):
      torch.testing.assert_close(a,b,atol=.03125,rtol=.03)
     assert max(errors)<4e-5,errors
     if exact:
      assert all(torch.equal(a,b)for a,b in zip(ys,rs)),'unit-RMS RoPE forward boundaries'
      assert all(torch.equal(a,b)for a,b in zip(gs,rg)),'unit-RMS RoPE backward boundaries'
     y2=f.apply(*xs,*ns,*cs);g2=torch.autograd.grad(y2,xs,dys)
     assert all(torch.equal(a,b)for a,b in zip((*ys,*gs),(*y2,*g2))),'repeatability'
     assert torch.equal(before,torch.cuda.get_rng_state()),'RNG'
     if n==17:
      for context in [None,functools.partial(create_selective_checkpoint_contexts,keep_attention)]:
       kw={}if context is None else{'context_fn':context}
       out=checkpoint(lambda q,k:f.apply(q,k,*ns,*cs),*xs,use_reentrant=False,**kw)
       gg=torch.autograd.grad(out,xs,dys)
       assert all(torch.equal(a,b)for a,b in zip((*ys,*gs),(*out,*gg))),'checkpoint'
      out=f.apply(*xs,*ns,*cs);only=torch.autograd.grad(out[1],xs,dys[1],allow_unused=True)
      assert only[0]is None and torch.equal(only[1],gs[1]),'unused Q gradient'
     print('PASS',n,h,wd,strided,distribution,'exact',exact,'relL2',errors,flush=True)

    for n in(1,7,17,129,193,513):case(n)
    case(17,strided=True);case(129,wd=torch.float32,strided=True)
    case(129,distribution='nearzero');case(129,distribution='offset');case(193,strided=True,exact=True)
    case(52340,h=56)
    ns=norms(torch.bfloat16,1e-6);q,k=[torch.randn(1,17,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)for _ in range(2)];c,s=[torch.randn(17,96,device='cuda')for _ in range(2)]
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA])as p:
     out=checkpoint(lambda q,k:f.apply(q,k,*ns,c,s),q,k,use_reentrant=False,context_fn=functools.partial(create_selective_checkpoint_contexts,keep_attention));(out[0].float().sum()+out[1].float().sum()).backward();torch.cuda.synchronize()
    p.export_chrome_trace(str(probe_dir/'trace.json'))
    es=json.load(open(str(probe_dir/'trace.json')))['traceEvents'];names=[e['name']for e in es if e.get('cat')=='kernel']
    assert any('_forward_kernel' in x for x in names)and any('_backward_kernel'in x for x in names),names
    ns[0].weight.requires_grad_(True);c.requires_grad_(True)
    out=f.apply(q,k,*ns,c,s);gg=torch.autograd.grad(sum(y.float().sum()for y in out),(q,k,ns[0].weight,c))
    ref=reference(q,k,*ns,c,s);rr=torch.autograd.grad(sum(y.float().sum()for y in ref),(q,k,ns[0].weight,c))
    assert all(torch.equal(a,b)for a,b in zip(gg,rr)),'fallback gradients'
    assert f.COUNTS['fallback']==1
    print('PASS actual CUDA kernels, fallback trainable affine/angles and counters',dict(f.COUNTS),flush=True)

    # Distinct Q/K strides; mixed angle dtypes/offsets; each unused branch.
    ns=norms(torch.float32,1e-6)
    q=torch.randn(1,37,10,128,device='cuda',dtype=torch.bfloat16)[:,:,::2,:].requires_grad_()
    k=torch.randn(1,74,5,128,device='cuda',dtype=torch.bfloat16)[:,::2,:,:].requires_grad_()
    c=torch.randn(39,193,device='cuda',dtype=torch.float32)[1:38,1:193:2]
    s=torch.randn(39,193,device='cuda',dtype=torch.bfloat16)[1:38,1:193:2]
    dys=[torch.randn(1,37,10,256,device='cuda',dtype=torch.bfloat16)[:,:,::2,::2]for _ in range(2)]
    y=f.apply(q,k,*ns,c,s);g=torch.autograd.grad(y,(q,k),dys)
    r=reference(q,k,*ns,c,s);rg=torch.autograd.grad(r,(q,k),dys)
    assert max(rel(a,b)for a,b in zip((*y,*g),(*r,*rg)))<4e-5
    for use in range(2):
     y=f.apply(q,k,*ns,c,s);gg=torch.autograd.grad(y[use],(q,k),dys[use],allow_unused=True)
     assert gg[1-use]is None and torch.equal(gg[use],g[use])
    print('PASS distinct Q/K and dy layouts, mixed/offset angles and both unused branches',flush=True)
