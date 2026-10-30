"""Fused AdaLN regression probe (one GPU); not collected by pytest.

From the repository root:
PYTHONPATH=src python tests/gpu/probe_fused_adaln.py

Checks the kernels against the eager path; reports correctness diagnostics only.
"""

if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    probe_dir=Path(tempfile.mkdtemp(prefix="solar-adaln-probe-"))
    import json, functools
    import torch
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
    from torch.profiler import profile, ProfilerActivity
    from solarwm.backends.minimax_h3 import fused_adaln as f
    
    def error(a,b):
     return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
    def close(a,b,label):
     e=error(a,b)
     print(label,'relative_l2',e,'max_abs',(a.float()-b.float()).abs().max().item(),'equal_fraction',(a==b).float().mean().item(),flush=True)
     assert e<0.003,(label,e)
    
    torch.manual_seed(321)
    for n,d in ((129,128),(65,511),(512,5376),(1025,5376),(33,4096),(129,7680)):
     x=torch.randn(1,n,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
     w=torch.randn(d,device='cuda',dtype=torch.bfloat16)
     scale,shift,table=[torch.randn(6,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
     ids=torch.randint(6,(n,),device='cuda')
     before=torch.cuda.get_rng_state().clone()
     y=f.norm(x,w,scale,shift,ids,1e-5)
     assert torch.equal(before,torch.cuda.get_rng_state()),'compile/forward advanced RNG'
     r=f._norm_impl(x,w,scale,shift,ids,1e-5)
     close(y,r,f'norm {n}/{d}')
     dy=torch.randn_like(y)
     gy=torch.autograd.grad(y,x,dy)[0]; gr=torch.autograd.grad(r,x,dy)[0]
     close(gy,gr,f'norm gradient {n}/{d}')
     u=torch.randn_like(x,requires_grad=True)
     z=f.gate(x,u,table,ids); rz=f._gate_impl(x,u,table,ids)
     close(z,rz,f'gate {n}/{d}')
     gs=torch.autograd.grad(z,(x,u),dy); rs=torch.autograd.grad(rz,(x,u),dy)
     for a,b in zip(gs,rs):close(a,b,f'gate gradient {n}/{d}')
     print('PASS shape',n,d,flush=True)
    
    def fn(a):return f.gate(a,f.norm(a,w,scale,shift,ids,1e-5),table,ids)
    ctx=functools.partial(create_selective_checkpoint_contexts,lambda *a,**k:CheckpointPolicy.PREFER_RECOMPUTE)
    # Warm up the checkpoint path, then capture only that path to prove compile survives SAC.
    y=checkpoint(fn,x,use_reentrant=False,context_fn=ctx)
    torch.autograd.grad(y,x,dy)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
     y=checkpoint(fn,x,use_reentrant=False,context_fn=ctx)
     g=torch.autograd.grad(y,x,dy)[0]
    r=fn(x); rg=torch.autograd.grad(r,x,dy)[0]
    close(g,rg,'checkpoint gradient')
    p.export_chrome_trace(str(probe_dir / 'trace.json'))
    compiled=[e for e in p.events() if any(n in e.name for n in ('_norm_forward_kernel','_gate_forward_kernel','_norm_backward_kernel','_gate_backward_kernel'))]
    print('SAC compiled events',len(compiled),'COUNTS',f.COUNTS,flush=True)
    assert compiled, 'no compiled kernels on selective-checkpoint path'
    assert f.COUNTS['fallback']==0
    # The training-ineligible case must retain gradients for affine tables.
    torch.use_deterministic_algorithms(True)
    s=scale.detach().requires_grad_()
    y=f.norm(x,w,s,shift,ids,1e-5); r=f._norm_impl(x,w,s,shift,ids,1e-5)
    for a,b in zip(torch.autograd.grad(y,(x,s),dy),torch.autograd.grad(r,(x,s),dy)):
     assert torch.equal(a,b)
    print('PASS frozen-table guard and original-formula gradients; final COUNTS',f.COUNTS,flush=True)
    
    trace=json.load(open(probe_dir / 'trace.json'))['traceEvents']
    names=[z['name'] for z in trace if z.get('cat')=='kernel']
    print('SAC CUDA kernels',names,flush=True)
    for kind in ('_norm_forward_kernel','_gate_forward_kernel','_norm_backward_kernel','_gate_backward_kernel'):
     assert any(kind in n for n in names),kind
    # Production-like noncontiguous affine views and strided activation/gradient rows.
    torch.use_deterministic_algorithms(True)
    n,d=52340,5376
    x=torch.randn(1,2*n,d,device='cuda',dtype=torch.bfloat16)[:,::2,:].requires_grad_()
    pool=torch.randn(6,6*d,device='cuda',dtype=torch.bfloat16)
    shift,scale,table=pool[:,:d],pool[:,d:2*d],pool[:,2*d:3*d]
    w=torch.nn.Parameter(torch.randn(d+7,device='cuda',dtype=torch.bfloat16)[7:],requires_grad=False)
    ids=torch.randint(6,(n,),device='cuda')
    dy=torch.randn(1,2*n,d,device='cuda',dtype=torch.bfloat16)[:,::2,:]
    y=f.norm(x,w,scale,shift,ids,1e-5);r=f._norm_impl(x,w,scale,shift,ids,1e-5)
    close(y,r,'strided norm')
    close(torch.autograd.grad(y,x,dy)[0],torch.autograd.grad(r,x,dy)[0],'strided norm gradient')
    u=torch.randn(1,2*n,d,device='cuda',dtype=torch.bfloat16)[:,::2,:].requires_grad_()
    y=f.gate(x,u,table,ids);r=f._gate_impl(x,u,table,ids)
    assert torch.equal(y,r)
    for a,b in zip(torch.autograd.grad(y,(x,u),dy),torch.autograd.grad(r,(x,u),dy)):assert torch.equal(a,b)
    print('PASS production-like affine/storage offsets and strided rows',f.COUNTS,flush=True)
    
    rng=torch.cuda.get_rng_state().clone()
    ys=[];gs=[]
    for _ in range(2):
     y=f.norm(x,w,scale,shift,ids,1e-5)
     ys.append(y.detach())
     gs.append(torch.autograd.grad(y,x,dy)[0])
    assert torch.equal(ys[0],ys[1]) and torch.equal(gs[0],gs[1])
    assert torch.equal(rng,torch.cuda.get_rng_state())
    wf=w.detach().float()
    y=f.norm(x,wf,scale,shift,ids,1e-5);r=f._norm_impl(x,wf,scale,shift,ids,1e-5)
    close(y,r,'fp32 frozen norm weight')
    close(torch.autograd.grad(y,x,dy)[0],torch.autograd.grad(r,x,dy)[0],'fp32 frozen norm weight gradient')
    print('PASS production length, deterministic repeatability, RNG and fp32 frozen weight',f.COUNTS,flush=True)
    
    for d in (1,2,3,5,7,15,16,17,31,32,33,63,64,65,127,128,129,130,131,132,133,134,135,255,256,257,511,512,513,5376,7680):
     n=3+d%5
     a=torch.randn(1,n,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
     b=torch.randn_like(a,requires_grad=True)
     pool=torch.randn(6,6*d,device='cuda',dtype=torch.bfloat16)
     table=pool[:,2*d:3*d]
     ids=torch.randint(6,(n,),device='cuda')
     dy=torch.randn(1,d,n,device='cuda',dtype=torch.bfloat16).transpose(1,2)
     y=f.gate(a,b,table,ids);r=f._gate_impl(a,b,table,ids)
     assert torch.equal(y,r),('gate',d)
     for ga,gr in zip(torch.autograd.grad(y,(a,b),dy),torch.autograd.grad(r,(a,b),dy)):
      assert torch.equal(ga,gr),('gate backward',d)
    print('PASS 31 additional gate widths/tails and noncontiguous output gradients',flush=True)
    for magnitude,offset in ((1e-4,0.0),(1.0,1000.0)):
     n,d=17,5376
     x=(torch.randn(1,n,d,device='cuda')*magnitude+offset).to(torch.bfloat16).requires_grad_()
     w=torch.randn(d,device='cuda',dtype=torch.bfloat16)
     scale,shift=[torch.randn(6,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
     ids=torch.randint(6,(n,),device='cuda');dy=torch.randn_like(x)
     y=f.norm(x,w,scale,shift,ids,1e-5);r=f._norm_impl(x,w,scale,shift,ids,1e-5)
     close(y,r,'offset/scale norm')
     close(torch.autograd.grad(y,x,dy)[0],torch.autograd.grad(r,x,dy)[0],'offset/scale norm gradient')
    print('PASS complete regression probe; trace:',probe_dir/'trace.json',flush=True)
