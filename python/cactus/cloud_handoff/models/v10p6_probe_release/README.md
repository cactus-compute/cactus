# v10 p6_winner_full Wrongness Probe — Standalone Release

`global_attn_probe_v10p6.pt` is a **64,833-parameter** classifier head that
reads Gemma 4 E2B-it **layer-28 hidden states** for one rollout and returns
a single scalar `p(Gemma got this answer wrong)`. Val AUROC = **0.884**,
nearly tied with the older v9.4 probe (0.887) at ~60% of the parameter count.

## Files

- `probe.py` — `GlobalAttnPoolProbeV10` module + `load_probe()` helper.
  No external repo dependencies; only `torch` is required.
- `global_attn_probe_v10p6.pt` — trained checkpoint.
- `demo.py` — smoke test: loads the checkpoint, runs forward on dummy
  inputs at T=12 / 180 / 1024, asserts determinism.
- `config.json` — exact architecture config used for training.
- `training_summary.json` — full train-time summary (shard mix,
  hyperparameters, val metrics).

## Run the demo

```bash
pip install torch
python3 demo.py
```

## Architecture (1536 → scalar)

p6 differs from v9.4 in two places: a smaller pooling working-dim
(`t_h=32` vs 64) and a deeper MLP head (`[128, 64]` vs `[64]`).

```
input          (T, 1536)              # T = generated tokens, hidden = layer 28
  LayerNorm(1536)
  Dropout(0.20)                       # eval-mode dropout is identity
  Linear(1536, 32) + ReLU             # project to working dim t_h=32
  attention pool with a single learned 32-dim query:
      scores = u @ attn_query / sqrt(32)
      alpha  = softmax(scores, dim=0)
      z      = sum_t alpha_t * u_t    # (32,)
  Linear(32, 128) + ReLU
  Linear(128, 64) + ReLU
  Linear(64, 1)
output         scalar logit           # sigmoid -> p(wrong) in [0, 1]
```

`T` is the number of generated tokens in the rollout — variable per example,
no padding. For batched inference, loop over rollouts.

## How to use in practice

1. Run Gemma 4 E2B-it on your prompt with a generation loop, hooking
   `model.model.layers[28]` to capture the **hidden state at each generated
   token**.
2. Stack the captured per-token hidden states into a `(T, 1536)` tensor.
3. `logit = probe(x); p_wrong = torch.sigmoid(logit)`. Near 1 ⇒ Gemma is
   probably wrong; near 0 ⇒ Gemma is probably right.

## Loading other v10 sweep configs

`GlobalAttnPoolProbeV10` is the full configurable class used across the
v10 sweep family (sm1-sm6, p1-p8). To load a different sweep checkpoint
with the same code, pass its config as kwargs:

```python
# sm6_t2_lin (6,151 params, val AUROC 0.879)
probe = load_probe("sm6_t2_lin.pt", t_h=2, mlp_hidden_dims=[])

# p7_th32_full (54,465 params, val AUROC 0.885)
probe = load_probe("p7_th32_full.pt", t_h=32, mlp_hidden_dims=[64])
```

## License / provenance

Internal research artifact, same training pool as v9.4 (56,937 samples
from `text_train_shards*` + `mcq_shards*` + `vision_train_shards*`).
