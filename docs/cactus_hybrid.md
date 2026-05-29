---
title: "Cactus Hybrid Inference"
description: "Serve most queries locally, route the hard ones to the cloud. A lightweight probe on the local model decides when to ask for help."
keywords: ["hybrid inference", "cloud handoff", "routing", "on-device AI", "confidence", "edge-cloud"]
---

# Cactus Hybrid Inference

Even a well-quantized local model trails a frontier cloud model on the hardest queries.
Cactus Hybrid Inference closes the gap: serve most queries locally, and route only the
ones the local model is likely to get wrong to the cloud.

A lightweight probe (~65k parameters) attached to the local model's hidden states emits
a single confidence score per generation. Queries above a threshold are sent to the
cloud. The result: frontier-class accuracy at a fraction of the cloud cost, with most
data never leaving the device.

## Why Hybrid?

A 2B-parameter model running locally handles the majority of user queries well. But
there's a long tail of hard questions — complex reasoning, domain-specific knowledge,
ambiguous multimodal inputs — where a frontier cloud model (Gemini Pro, Claude, GPT-4)
would get it right and the local model won't.

The naive solutions don't work:

- **Always local**: misses the hard queries. Users notice.
- **Always cloud**: expensive, slow, and sends all data off-device.
- **Token entropy as confidence**: anti-predictive on transcription and vision MCQ.
  High entropy often means the model is considering multiple valid candidates, not
  that it's wrong.

The probe solves this by reading the model's internal state directly — a much richer
signal than output token probabilities.

## What the Probe Does

The probe reads the local model's hidden states after generation and outputs a single
score: the probability the model got this query wrong. Above a threshold → route to
cloud. Below → serve locally.

The probe:

- **Adds negligible latency** — 65k parameters, runs after generation is already complete
- **Handles all modalities** — text, vision, and audio through the same architecture, since
  it reads post-fusion hidden states where modalities are already merged
- **Transfers to audio with zero audio training** — trained on text + vision only, yet
  achieves 0.79–0.88 AUROC on audio benchmarks
- **Works across quantization levels** — same probe checkpoint at FP16, 4-bit, 3-bit;
  no per-quantization retraining needed

## Results

Evaluated on 12 hold-out benchmarks spanning text, vision, and audio — none seen during
training. Cloud target: Gemini 3.1 Pro. All results averaged over 3 seeds.

### Routing Quality (AUROC)

AUROC measures how well the probe separates wrong answers from right ones (higher = better).
0.5 is random, 1.0 is perfect:

| Hold-out | Modality | Probe | Token Entropy |
|---|---|---|---|
| MMLU | text MCQ | **0.770** | 0.697 |
| MMLU-Pro | text MCQ | **0.771** | 0.692 |
| ARC-Easy | text MCQ | **0.888** | 0.655 |
| ARC-Challenge | text MCQ | **0.834** | 0.646 |
| GSM8K (3-shot) | text gen | **0.782** | 0.731 |
| MMBench-EN-Dev | vision MCQ | **0.840** | 0.435 |
| ChartQA | vision QA | **0.779** | 0.615 |
| DocVQA | vision QA | **0.781** | 0.512 |
| MMAU | audio MCQ | **0.789** | 0.517 |
| GigaSpeech | audio | **0.876** | 0.343 |
| Earnings-22 | audio | **0.839** | 0.323 |
| LibriSpeech | audio | **0.822** | 0.427 |
| **Mean** | | **0.814** | **0.549** |

The probe outperforms token entropy on every benchmark. On audio and vision, token
entropy is actively harmful (AUROC < 0.5 means it routes the *right* answers to the
cloud and keeps the wrong ones local).

### Cross-Modality Transfer

The strongest result: the probe was trained on **zero audio data**, yet achieves
0.79–0.88 AUROC on four audio benchmarks (two transcription, one audio MCQ, one
out-of-domain transcription). This rules out surface-level explanations — the probe
is reading a modality-independent correctness signal from the hidden state, not
memorizing patterns from training data.

This means hybrid inference requires no per-task probe tuning and no re-training when
the model is re-quantized.

### How Much Handoff Is Needed?

At FP16, the hybrid system matches Gemini 3.1 Flash-Lite accuracy on most benchmarks
by routing only 15–35% of queries to the cloud:

| Benchmark | Handoff to match Flash-Lite (FP16) | At 4-bit | At 3-bit |
|---|---|---|---|
| ChartQA | 15–20% | 25–30% | 40–50% |
| MMBench | 30–35% | 40–45% | 50–55% |
| LibriSpeech | 25–30% | 35–40% | 55–65% |
| GigaSpeech | 30–35% | 40–45% | 50–55% |
| MMAU | 30–35% | 35–40% | 50–55% |
| MMLU-Pro | 45–55% | ~90% | n/a |

At 4-bit quantization, the deployment cost is modest: 5–10 extra percentage points of
handoff versus FP16 on most benchmarks. At 3-bit it grows to ~20–35 points. MMLU-Pro
is the hardest case — the local model's accuracy drops sharply under quantization,
leaving less signal for the probe to work with.

### Probe Architecture Ablations

How much does the attention pool matter vs. simpler alternatives?

| Head | Mean AUROC |
|---|---|
| **Learned-query attention pool + MLP** (ours) | **0.814** |
| Mean pool + MLP | 0.781 |
| Linear (logistic regression on mean pool) | 0.773 |
| Token entropy | 0.549 |

Mean pooling captures most of the value. The attention pool opens a gap on
long-generation modalities (audio: +6 points on MMAU) and loosely-structured tasks
(MMLU-Pro: +9 points), where selectively weighting tokens within a generation matters
more than pooling-of-any-kind.

## Where It's Weakest

1. **When the local model is broken.** On MMLU-Pro at 4-bit, local accuracy drops to
   0.28 and probe AUROC falls toward 0.5. Probe utility scales with local-model
   competence — when the model is broken, there's no correctness signal to read.

2. **When the cloud target is worse.** On DocVQA and GSM8K, Gemini Pro is worse than
   Flash-Lite, so routing to Pro can never help. A production system would select the
   cloud target per task.

3. **High handoff regime.** Above 50% handoff, the probe's marginal value over random
   routing narrows. The probe's value is concentrated in the low-handoff regime — which
   is the regime that matters for deployment cost.

## Using Hybrid Inference

Hybrid inference is built into the Cactus Engine. Set a confidence threshold in the
completion options:

```c
const char* options = R"({
    "confidence_threshold": 0.7,
    "auto_handoff": true,
    "cloud_timeout_ms": 15000
})";

char response[4096];
cactus_complete(model, messages, response, sizeof(response), options, NULL, NULL, NULL, NULL, 0);
```

When the model's confidence drops below the threshold, the request is automatically
routed to the cloud. The response JSON indicates whether handoff occurred:

```json
{
    "success": true,
    "cloud_handoff": true,
    "response": "Cloud-provided answer.",
    "confidence": 0.18
}
```

In Python:

```python
import json
from cactus import cactus_init, cactus_complete

model = cactus_init("weights/gemma-4-E2B-it", None, False)
options = json.dumps({
    "confidence_threshold": 0.7,
    "auto_handoff": True
})
messages = json.dumps([{"role": "user", "content": "Explain quantum entanglement"}])
result = json.loads(cactus_complete(model, messages, options, None, None))

if result["cloud_handoff"]:
    print("Answered by cloud")
else:
    print(f"Answered locally (confidence: {result['confidence']:.2f})")
```

Set your cloud API key with `cactus auth` to enable automatic handoff.

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — `confidence_threshold` and `auto_handoff` options
- [Cactus Quants](/docs/cactus_quants.md) — Quantization that the probe works across
- [Gemma 4 on Cactus](/blog/gemma4.md) — The base model used for hybrid inference
