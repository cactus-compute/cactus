"""Strategy 1 with multiple prompt/prefix variants.

Loads the model once, sweeps each variant over the full manifest. Each variant
writes its own output file: outputs/strategy1_<variant>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HARNESS))

from src.cactus import (                 # noqa: E402
    cactus_init, cactus_destroy, cactus_reset,
    cactus_tokenize, cactus_score_candidates,
)
from io_utils import pcm_bytes_from_wav  # noqa: E402

DEFAULT_WEIGHTS = str(ROOT / "weights" / "gemma-4-e2b-it")
MANIFEST = HARNESS / "manifest.jsonl"
OUT_DIR = HARNESS / "outputs"

OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_tokens": 1,
    "auto_handoff": False,
    "enable_thinking_if_supported": False,
    "telemetry_enabled": False,
}

# Each variant is (name, user_prompt_template, prefill_suffix, (pos_word, neg_word))
# where pos_word corresponds to "handoff NOT needed" and neg_word to "handoff needed".
# The user_prompt_template uses {POS}/{NEG} as placeholders for the binary words.

_BASE_TASK = (
    "Look at the image and listen to the spoken instruction. Assess "
    "whether you can complete the instruction reliably given these inputs. "
    "Fill in the blank with a single plain word, no markdown, no bold, no "
    "punctuation, no quotes: \"This task is ___\". Use \"{POS}\" if you can "
    "confidently carry out the instruction, or \"{NEG}\" if the inputs are "
    "unclear, the instruction is ambiguous, or you are unsure of the answer."
)
_BASE_PROBLEM = _BASE_TASK.replace("This task", "This problem")
_BASE_AVG = (
    "Look at the image and listen to the spoken instruction. Assess "
    "whether an average person could complete the instruction reliably "
    "given these inputs. Fill in the blank with a single plain word, no "
    "markdown, no bold, no punctuation, no quotes: \"For an average "
    "person, this task is ___\". Use \"{POS}\" if the average person could "
    "confidently carry out the instruction, or \"{NEG}\" if the inputs are "
    "unclear, the instruction is ambiguous, or you are unsure of the answer."
)

VARIANTS: list[tuple[str, str, str, tuple[str, str]]] = [
    ("task_easy_hard",        _BASE_TASK,    "This task is",                        ("easy", "hard")),
    ("task_simple_complex",   _BASE_TASK,    "This task is",                        ("simple", "complex")),
    ("problem_easy_hard",     _BASE_PROBLEM, "This problem is",                     ("easy", "hard")),
    ("problem_simple_complex",_BASE_PROBLEM, "This problem is",                     ("simple", "complex")),
    ("avg_easy_hard",         _BASE_AVG,     "For an average person, this task is", ("easy", "hard")),
    ("avg_simple_complex",    _BASE_AVG,     "For an average person, this task is", ("simple", "complex")),
]


def resolve_pair(model, pos_word: str, neg_word: str) -> tuple[list[int], list[int]]:
    """Return (pos_token_ids, neg_token_ids) after probing a few surface forms."""
    def probe(word: str) -> list[int]:
        seen: set[int] = set()
        ids: list[int] = []
        for form in [word, " " + word, word.capitalize(), " " + word.capitalize()]:
            toks = cactus_tokenize(model, form)
            if toks and toks[0] not in seen:
                seen.add(toks[0])
                ids.append(toks[0])
        return ids
    return probe(pos_word), probe(neg_word)


def build_messages(image_path: str, user_prompt: str) -> str:
    return json.dumps([
        {"role": "user", "content": user_prompt, "images": [image_path]},
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset of variant names to run (default: all)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    selected = VARIANTS if args.variants is None else \
        [v for v in VARIANTS if v[0] in set(args.variants)]

    print(f"loading model {args.weights}")
    model = cactus_init(args.weights, None, False)
    print(f"variants: {[v[0] for v in selected]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant_name, user_tmpl, suffix, (pos_word, neg_word) in selected:
        pos_ids, neg_ids = resolve_pair(model, pos_word, neg_word)
        all_ids = pos_ids + neg_ids
        split = len(pos_ids)
        user_prompt = user_tmpl.format(POS=pos_word, NEG=neg_word)

        out_path = OUT_DIR / f"strategy1_{variant_name}.jsonl"
        print(f"\n=== variant={variant_name!r}  suffix={suffix!r}  "
              f"pos={pos_word!r}→{pos_ids}  neg={neg_word!r}→{neg_ids} ===")
        t_start = time.time()
        with open(out_path, "w") as f:
            for i, row in enumerate(rows):
                cactus_reset(model)
                pcm = pcm_bytes_from_wav(Path(row["audio"]))
                messages = build_messages(row["image"], user_prompt)
                t0 = time.time()
                try:
                    probs = cactus_score_candidates(
                        model, messages, json.dumps(OPTIONS), None,
                        suffix, all_ids, pcm_data=pcm,
                    )
                    p_pos = sum(probs[:split])
                    p_neg = sum(probs[split:])
                    verdict = pos_word if p_pos >= p_neg else neg_word
                    out = {
                        **row,
                        "variant": variant_name,
                        "pos_word": pos_word,
                        "neg_word": neg_word,
                        "p_easy": p_pos,     # keep schema compatible with strategy1 analysis
                        "p_hard": p_neg,
                        "verdict": "easy" if verdict == pos_word else "hard",
                        "verdict_word": verdict,
                        "prob_margin": p_pos - p_neg,
                        "per_token_probs": probs,
                        "elapsed_s": round(time.time() - t0, 2),
                        "error": None,
                    }
                except Exception as e:
                    out = {**row, "variant": variant_name, "p_easy": None,
                           "p_hard": None, "verdict": None, "error": str(e),
                           "elapsed_s": round(time.time() - t0, 2)}
                f.write(json.dumps(out) + "\n")
                f.flush()
                if (i + 1) % 50 == 0 or i == len(rows) - 1:
                    print(f"  [{i+1}/{len(rows)}] {row['audio_tag']}/"
                          f"{row['noise_tag']}/snr={row['snr_db']} "
                          f"P({pos_word})={out.get('p_easy')} P({neg_word})={out.get('p_hard')}")
        print(f"  -> {out_path} in {time.time()-t_start:.1f}s")

    cactus_destroy(model)
    print("\ndone.")


if __name__ == "__main__":
    main()
