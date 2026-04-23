---
title: "Pruning Gemma 4: Turning MLP Compression into Staged Model Surgery"
description: "How 30% uniform MLP pruning on Gemma 4 E2B became a recovery pipeline built from teacher trajectories, activation caches, local layer optimization, and end-to-end distillation."
keywords: ["Gemma 4", "MLP pruning", "distillation", "on-device AI", "Cactus", "teacher trajectories", "local layer optimization", "model compression"]
author: "Noah Cylich"
date: 2026-04-23
tags: ["Gemma 4", "pruning", "distillation", "model compression", "on-device AI"]
---

# Pruning Gemma 4

*By Noah Cylich*

One realization led this research: dense LLMs do not really use their MLPs densely at inference time. Per token, the top 10% of FFN channels carry around 90% of the activation energy. The structure is already sparse; the question is how to expose it in a form that survives compression. The goal became making a calculated structural cut, then recovering the behavior that mattered.

Here was my research story over time:

1. **Matryoshka-style post-training.** Given the effectiveness of Matryoshka submodels, I thought this might be a viable post-training technique, following the nested sequence idea behind SMEC. In practice, this raw approach's PPL, even with saliency initialization, was about 2x that of the base model.
2. **Structured sparsity.** The 3BASiL technique could roughly compress models by 50% by representing weights as $W \approx S + L$, while only being somewhat lossy. However, what Cactus can efficiently exploit is structured per-group sparsity, and that requirement made the technique ineffective for this target.
3. **Other approximations.** In further toy experiments, I tried approximating $W \approx AB$ where $A$ and $B$ are both sparse. Since this still was not working, I tried approximating the whole MLP: $MLP(x) = Af(Bx)$ with $MLP(x) \approx A'f(B'x)$. Iterating on these designs, rather ironically, led back to per-column, per-hidden-dimension Matryoshka-style pruning.
4. **Enhanced Matryoshka.** Going full circle, I re-applied Matryoshka pruning, but this time with 3BASiL-style per-layer calibration on self-generated trajectories.

Thus, the pruning work gradually turned into a five-stage pipeline:

1. Generate teacher trajectories.
2. Build an activation cache.
3. Apply uniform pruning.
4. Run local layer optimization.
5. Run end-to-end distillation.

I used WildChat prompts with the uncompressed teacher, saved them as JSONL, and split them with `filter_trajectories.py` into two pools: the higher-quality `_final` pool for activation caching and local repair, and `_bulk` for the larger recovery passes later on.

The pruning itself is hidden-unit pruning, not generic sparse masking. Each GeGLU hidden unit is either kept or removed by zeroing full rows of `gate_proj` and `up_proj` and the matching columns of `down_proj`. That keeps the model structurally shrinkable. On the final Gemma 4 run, the schedule was uniform 30% MLP pruning across all 35 layers.

The most important stage was local layer optimization. After pruning, each MLP was briefly optimized against the teacher's cached activations before I asked the whole network to recover globally. That matters because pure one-shot pruning is too destructive, but pure end-to-end fine-tuning wastes time relearning obviously local structure. The local pass fixes the immediate damage first, serving as a much warmer init for the global objective.

Then we performed the end-to-end distillation stage. In code, that final phase has two passes: TM, a KL-only token-matching warm-up, and GD, a stronger global distillation stage with logit KL plus hidden-state MSE. In practice, I think of those as one operational stage, and partly as an artifact of adapting the pipeline on top of 3BASiL.

At 30% MLP sparsity, the Gemma 3 init-pruned student started at 2.44x teacher perplexity. Local optimization brought that to 1.36x, TM to 1.25x, and a mixed-data GD run to 1.04x. The biggest gain was not a better saliency score. It was better trajectories and better staged recovery.

Scaling the same pipeline to [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) made the result much more serious: 4.65B parameters, 35 layers, 30% MLP pruned, trained on 4x H100s in about an hour. The final model is [`ncylich/gemma4-e2b-it-uniform-tm-gd`](https://huggingface.co/ncylich/gemma4-e2b-it-uniform-tm-gd).

Its recovery curve is the cleanest summary of the work:

| Stage | PPL |
|---|---:|
| Teacher | 3.08 |
| Init-pruned student | 12.54 |
| After local layer optimization | 9.63 |
| After TM | 5.34 |
| After GD | 4.45 |

The downstream numbers are also revealing. The average benchmark score moved from 56.10% to 50.73%, a 5.37-point drop. ARC-Easy fell only 1.47 points, HellaSwag 2.19, and PIQA 2.62, while MMLU dropped 13.68. That pattern shows something foundational: pruning did not damage everything equally. Commonsense and multiple-choice reasoning held up reasonably well, but knowledge-heavy evaluation degraded much more. This matches the broader view that transformer MLPs act partly as key-value memories, storing a meaningful amount of model knowledge.

Ultimately, the hard part was not finding a magical mask. It was building a recovery pipeline that could survive a real structural cut: generate the right teacher data, cache the right activations, prune whole hidden units, repair each layer locally, then distill end to end. Rather than being a one-shot compression trick, pruning became staged model surgery.

I'm currently exploring transforming dense models into MoEs so we can keep the reduced compute and memory-bandwidth demands without cutting out as much of the core knowledge stored in the MLPs.
