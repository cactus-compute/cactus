#!/bin/bash
# Origin-vs-PR comparison for PR #706. Run from the repo root with this branch
# checked out. Swaps a single file to origin/main, runs the probe, then restores.
# The cactus package is editable-installed against this tree, so a file swap +
# fresh interpreter is the way to compare; worktrees would all import this tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BR=audit/transpiler-lowering
F=python/cactus/transpile

restore() { git checkout -q "$BR" -- "$1"; }

echo "########## aten_ops: ORIGIN routing (expect add/min/max/slice/index) ##########"
git checkout -q origin/main -- "$F/aten_ops.py"
python -c "
from cactus.transpile.normalize import normalize_target
import torch
for op in ['addmm','minimum','maximum','slice_scatter','select_scatter']:
    print('  ORIGIN', op, '->', normalize_target(getattr(torch.ops.aten, op).default))"
restore "$F/aten_ops.py"

echo "########## rms_norm: WITH guard, non-last must NOT fuse / last must fuse ##########"
# this branch ships the guard; show it explicitly
python -c "
import torch, torch.nn as nn
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.optimize_graph import fuse_rms_norm
def rms(x,d,w,e=1e-6): return x*torch.rsqrt(x.pow(2).mean(d,keepdim=True)+e)*w
class Chan(nn.Module):
    def __init__(s): super().__init__(); s.w=nn.Parameter(torch.ones(4,1,1,dtype=torch.float16))
    def forward(s,x): return rms(x,1,s.w)
class Last(nn.Module):
    def __init__(s): super().__init__(); s.w=nn.Parameter(torch.ones(8,dtype=torch.float16))
    def forward(s,x): return rms(x,-1,s.w)
for name,m,ex in [('non-last',Chan(),torch.randn(2,4,5,5,dtype=torch.float16)),('last',Last(),torch.randn(2,3,8,dtype=torch.float16))]:
    g=capture_model(m.half().eval(),(ex,)).ir_graph
    ch=fuse_rms_norm(g); rn=sum(1 for n in g.order if g.nodes[n].op=='rms_norm')
    print('  WITH-GUARD',name,'fused=',ch,'rms_norm=',rn)"

echo "########## capture_jax: ORIGIN aliases strided slice (expect ops=[] shape=(6,)) ##########"
git checkout -q origin/main -- "$F/capture_jax.py"
python python/tests/transpile/tools/repro_706/repro_jax.py 2>/dev/null || echo "  (needs jax[cpu])"
restore "$F/capture_jax.py"

echo "########## batch_norm: run_decompositions turns it into the no_training op ##########"
python -c "
import torch, torch.nn as nn
m=nn.BatchNorm2d(4).eval()
ep=torch.export.export(m,(torch.randn(1,4,8,8),))
print('  default export  :', [str(n.target) for n in ep.graph.nodes if 'batch_norm' in str(n.target)])
ep2=ep.run_decompositions()
print('  after decompose :', [str(n.target) for n in ep2.graph.nodes if 'batch_norm' in str(n.target)])
print('  (decompose only runs on the NPU->CoreML path, never into import_captured_to_ir)')"
