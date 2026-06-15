# Benchmarking

Reproducible CPU inference benchmarks via `cactus bench`. Runs locally by
default; `--android` cross-builds for arm64-v8a and runs on a connected
device over adb. The bundle is downloaded or converted automatically if
missing.

```
cactus bench                                   # gemma-4-e2b-it, local
cactus bench --android                         # same workload on a device
cactus bench Qwen/Qwen3-0.6B --android
cactus bench --prompt "Explain relativity:"    # chat-template path
```

Two workloads:

- **Token mode** (default) — prefill exactly `--prefill-tokens` raw token
  IDs (default 512 = 4 full 128-token prefill chunks, zero padding) +
  `--decode-tokens` decode (default 32), via `cactus_benchmark_tokens`.
  No tokenizer, template, or padding variance: quote these numbers.
  Each round runs in a fresh process; the first (warmup) round is
  discarded. Token IDs are synthetic (1000, 1001, …); keep
  `--prefill-tokens` well below the model's vocabulary size.
- **Prompt mode** (`--prompt`) — a full `cactus_complete` through the
  chat-template path, greedy decoding, `--max-tokens` decode budget.
  Each round runs a discarded in-process warmup generation first.

Both runs always set `CACTUS_DISABLE_CLOUD_HANDOFF=1` (no routing probe or
network fallback mid-generation) and `CACTUS_NO_CLOUD_TELE=1` (no telemetry
uploads during rounds). Results are not valid without them.

Reference numbers, token mode 512+32, default configuration
(verified 2026-06-09):

| Device | gemma-4-e2b-it (decode / prefill tps) |
|---|---|
| Pixel 10a (Tensor G4) | 10.8 / 75 |
| Samsung S25 (8 Elite) | 22–24 / 200+ |

## Device preparation (Android)

1. **Charging.** `adb shell dumpsys battery` must show `AC/USB powered:
   true` (the CLI warns if not). A discharging pack current-limits the
   SoC and silently costs 15–45% decode. On Pixels, also check the
   "Charge connected device" setting: it puts the USB port in source
   role, so the phone discharges even with a cable attached.
2. **Cool device, short batches.** Run 1 warmup + 3 measured rounds from
   rest; back-to-back hot batches read 20–30% lower. Rest a few minutes
   between batches.
3. **Quiet device.** One benchmark per device at a time; no other adb
   workloads during a run. This applies equally to local runs: a
   concurrent compile costs ~4× on compute-bound prefill while leaving
   memory-bound decode plausible-looking.

## Core configuration

Cactus's multi-core scheduling is deliberately **balanced, not
maximum-throughput**: workers are pinned to performance cores and work is
distributed so the device stays responsive and inside its sustained power
envelope. Do not expect (or force) all-core saturation — on phones that
trades thermals and UX for little gain, and on asymmetric SoCs it can lose
outright.

- **Default (mixed-core)** — no extra setup; this is the configuration
  behind the reference table. Best for ~2B-class models (gemma-4-e2b).
- **Single-core (prime only)** — `--cpu-mask <hex>` (Android only), e.g.
  `--cpu-mask 80` for cpu7 on a 1-prime + 3-mid + 4-little SoC. Small
  models whose decode is bound by single-stream memory bandwidth can run
  *faster* this way: on Pixel 10a at the 512+32 spec, qwen3-0.6b decodes
  ~26 tps single-prime vs ~18 mixed-core, while gemma-4-e2b is better
  mixed (10.8 vs ~5.6). Report which configuration a number came from.

## Interpreting prefill numbers

The transpiled prefill graph processes fixed 128-token chunks; prompts are
padded up to a chunk multiple. Prompt-mode runs with short prompts (e.g. a
~30-token prompt) therefore understate true prefill throughput by ≈
chunk/prompt (~4.3×). Token mode uses exact multiples of full chunks with
zero padding — quote its prefill numbers.

## If your numbers come out low: check bundle vintage

Bundles converted with the **current** converter need none of this. If you
are benchmarking artifacts produced by an **older release tag or converter
snapshot**, two known issues silently depress results:

- **Row-major LM head.** Older converters stored orthogonal CQ4 output
  heads row-major (header flags=2) instead of 4-row-interleaved (flags=6);
  that path costs ~2.4× more per byte — measured −35% decode on Pixel,
  −30% on Samsung. Check the flags word (bytes 4–7 of the `.weights`
  file); fix without reconversion:
  `cd python && python3 -m cactus.convert.interleave_orthogonal_cq4 IN.weights OUT.weights`
  Interleaved artifacts require an engine from 2026-05-26 or later.
- **Pre-chunked-prefill text bundles.** Text-model bundles converted
  before chunked prefill became the default show low long-prompt prefill
  despite healthy short-prompt rates (seen with a June-2026 qwen3 bundle:
  ~21–38 tps at 512 tokens). Re-convert with the current converter.
