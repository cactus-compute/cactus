---
title: "Runtime & Weights Compatibility"
description: "How Cactus runtime versions map to model weight versions on HuggingFace. Explains versioning, compatibility checks, and when to re-download weights."
keywords: ["versioning", "compatibility", "model weights", "HuggingFace", "Cactus runtime"]
---

# Runtime & Weights Compatibility

Some Cactus releases change the internal weight format. When this happens, cached weights from an older version will not load with a newer runtime and must be re-downloaded.

Breaking weight changes are called out in the [release notes](https://github.com/cactus-compute/cactus/releases).

## How Versioning Works

Weights are published to [Hugging Face](https://huggingface.co/Cactus-Compute) and **only re-tagged when they actually change**. If a release does not affect the weight format, the previous tag remains — no new upload.

```
Runtime v1.7  -> weights tagged v1.7 on HF
Runtime v1.8  -> no new tag (unchanged) - still use v1.7
...
Runtime v1.14 -> no new tag - still use v1.7
Runtime v1.15 -> new tag v1.15 (changed!) - must update
```

**The rule:** use the latest HF weight tag that is ≤ your runtime version.

## Checking Compatibility

1. Open your model on [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute)
2. Click **Files and versions → open branch dropdown from Main**
3. Find the latest tag that is ≤ your runtime version
4. If your local weights use an older tag, re-download them

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full inference API reference
- [Fine-tuning Guide](/docs/finetuning.md) — Convert and deploy custom fine-tunes
- [HuggingFace Weights](https://huggingface.co/Cactus-Compute) — Official Cactus model weights
