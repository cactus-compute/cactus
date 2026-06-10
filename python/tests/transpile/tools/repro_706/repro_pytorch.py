"""Reachability repros for PR #706 (PyTorch capture/import path).

Run from the repo root with the branch checked out:

    python python/tests/transpile/tools/repro_706/repro_pytorch.py

Each section reaches a code path the PR touches (or shows it is unreachable),
on the *current* checkout. Pair with origin_vs_pr.sh to see the origin behaviour.
"""
from __future__ import annotations
import traceback
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F

from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.optimize_graph import fuse_rms_norm
from cactus.transpile.normalize import normalize_target
from cactus.transpile.import_semantics import apply_import_semantics


def opc(g):
    return Counter(g.nodes[n].op for n in g.order)


def line(t):
    print(f"\n========== {t} ==========")


# ---- 1. aten_ops longest-prefix routing (KEEP: reached by natural ops) ----
line("1. aten_ops routing (this branch)")
for op in ["addmm", "minimum", "maximum", "slice_scatter", "select_scatter"]:
    t = getattr(torch.ops.aten, op).default
    print(f"  normalize_target(aten.{op}) -> {normalize_target(t)}")

class MinMax(nn.Module):
    def forward(self, a, b):
        return torch.minimum(a, b) + torch.maximum(a, b)
c = capture_model(MinMax(), (torch.randn(3), torch.randn(3)))
print("  MinMax model IR ops:", dict(opc(c.ir_graph)))


# ---- 2. BatchNorm: which aten op does the IR import path actually see? ----
# REMOVED from PR: the no_training (7-arg) branch is unreachable here because
# the capture path never runs run_decompositions (that only happens on the
# NPU->CoreML path). torch.export emits the 8-arg aten.batch_norm.default.
line("2. eval-mode BatchNorm export form")
class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
    def forward(self, x):
        return self.bn(self.conv(x))
ep = torch.export.export(ConvBN().eval(), (torch.randn(1, 3, 8, 8),))
print("  exported BN target(s):", [str(n.target) for n in ep.graph.nodes if "batch_norm" in str(n.target)])
print("  (run_decompositions would instead emit _native_batch_norm_legit_no_training -> only feeds CoreML)")


# ---- 3. scaled addmm fail-closed (KEEP: reached) ----
line("3. scaled addmm (beta=2.0)")
class ScaledAddmm(nn.Module):
    def forward(self, bias, a, b):
        return torch.addmm(bias, a, b, beta=2.0)
try:
    capture_model(ScaledAddmm(), (torch.randn(4), torch.randn(4, 5), torch.randn(5, 4)))
    print("  imported WITHOUT failing (origin behaviour: silently drops beta)")
except Exception as e:
    print("  fail-closed OK:", type(e).__name__, "->", str(e).split(":")[-1].strip()[:80])


# ---- 4. SDPA literal mask -> attrs['mask']? (REMOVED: never produced) ----
line("4. SDPA mask -> attrs['mask']? (3 natural forms)")
def check(name, mod, args):
    g = capture_model(mod, args).ir_graph
    apply_import_semantics(g)
    for n in g.order:
        nd = g.nodes[n]
        if nd.op == "attention":
            print(f"  [{name}] attrs={sorted(nd.attrs)} mask_in_attrs={'mask' in nd.attrs} n_inputs={len(nd.inputs)}")
            return
    print(f"  [{name}] no attention node")

class MaskTensorArg(nn.Module):
    def forward(self, q, k, v, m):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)
check("tensor-mask-as-input", MaskTensorArg(), (torch.randn(1, 1, 4, 8),) * 3 + (torch.zeros(1, 1, 4, 4),))

class MaskInlineConst(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(4, 4))
check("inline-const-mask", MaskInlineConst(), (torch.randn(1, 1, 4, 8),) * 3)

class CausalFlag(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
check("is_causal-flag", CausalFlag(), (torch.randn(1, 1, 4, 8),) * 3)


# ---- 6. rms_norm guard (KEEP: non-last-dim mis-fuses without it) ----
# NOTE: fuse_rms_norm skips FP32 inputs (kept unfused on purpose), so the inputs
# must be FP16 to reach the fusion -- as they are in real transpilation.
def rms(x, dim, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + eps) * w

line("6a. LAST-dim RMSNorm, FP16")
class RMSLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(8, dtype=torch.float16))
    def forward(self, x):
        return rms(x, -1, self.w)
g = capture_model(RMSLast().half().eval(), (torch.randn(2, 3, 8, dtype=torch.float16),)).ir_graph
changed = fuse_rms_norm(g)
print("  fuse_rms_norm changed?", changed, "| rms_norm nodes:", opc(g).get("rms_norm", 0))

line("6b. NON-last-dim RMSNorm (channel dim=1), FP16")
class RMSChan(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4, 1, 1, dtype=torch.float16))
    def forward(self, x):
        return rms(x, 1, self.w)
g = capture_model(RMSChan().half().eval(), (torch.randn(2, 4, 5, 5, dtype=torch.float16),)).ir_graph
print("  mean axes:", [(n, g.nodes[n].attrs.get("axis")) for n in g.order if g.nodes[n].op == "mean"])
changed = fuse_rms_norm(g)
for n in g.order:
    if g.nodes[n].op == "rms_norm":
        wv = g.values.get(g.nodes[n].inputs[1])
        print("  fused rms_norm weight shape:", getattr(wv, "shape", None), "(channel=4, last dim=5)")
print("  fuse_rms_norm changed?", changed,
      "| this branch ships the guard so changed=False (correct);",
      "checkout origin/main rms_norm.py to see changed=True with weight (5,)")

print("\n[done]")
