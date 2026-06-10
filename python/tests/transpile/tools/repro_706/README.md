# PR #706 reachability repros

Scripts that reach (or show as unreachable) each code path touched by #706, on the
real capture/import pipeline. This commit is intentionally reverted in the next
commit so the repros are referenceable by SHA but are **not** part of the squash/merge.

## Run

```bash
# PyTorch paths (routing, batchnorm export form, scaled-addmm, SDPA mask, rms_norm)
python python/tests/transpile/tools/repro_706/repro_pytorch.py

# JAX path (strided-slice aliasing)
pip install "jax[cpu]"
python python/tests/transpile/tools/repro_706/repro_jax.py

# origin/main vs this branch, one file swapped at a time
bash python/tests/transpile/tools/repro_706/origin_vs_pr.sh
```

## What each shows

| Path | Kept? | Repro result |
|---|---|---|
| `aten_ops` longest-prefix | keep | origin mis-routes `addmm→add`, `minimum→min`, `maximum→max`, `slice_scatter→slice`, `select_scatter→index`; a `torch.minimum/maximum` model imports to `{minimum,maximum,add}` here |
| `importers` scaled-addmm | keep | `torch.addmm(b,a,c, beta=2.0)` fails closed |
| `fusion/rms_norm` guard | keep | FP16 channel-wise (non-last) RMS fuses into a last-dim kernel with weight `(5,)` for a channel dim of 4 without the guard; guard skips it, last-dim still fuses |
| `capture_jax` strided slice | keep | `x[::2]` → origin aliases full length-6 input (`ops=[]`), branch gives length-3 `slice` |
| `lower.py` | drop | only the non-FP16 branch changes; FP16 engine never executes it |
| `import_semantics` dropout | drop | `dropout_p` always 0 after `model.eval()` |
| `import_semantics` mask | drop | `attrs["mask"]` only set for a non-tensor literal; SDPA `attn_mask` is always Tensor/None (3 forms tested, none set it) |
| `importers` batchnorm no-training | drop | `_native_batch_norm_legit_no_training` only appears after `run_decompositions`, which only feeds CoreML; the IR importer always sees 8-arg `aten.batch_norm.default` |
