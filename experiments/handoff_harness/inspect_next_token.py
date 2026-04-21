"""Dump the top next-token distribution after the prefill suffix for one row."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HARNESS))

from src.cactus import cactus_init, cactus_destroy, cactus_score_candidates  # noqa: E402
from io_utils import pcm_bytes_from_wav                                      # noqa: E402
from run_strategy1_prefill import USER_PROMPT, PREFILL_SUFFIX, build_messages, OPTIONS  # noqa: E402

VOCAB = 262144
TOPN = 25

WEIGHTS = str(ROOT / "weights" / "gemma-4-e2b-it")
MANIFEST = HARNESS / "manifest.jsonl"


def decode_one(model, tid: int) -> str:
    # cheap way to get a string for a token id: call cactus_tokenize roundtrip won't work,
    # we use the vocab file if present
    return f"<id {tid}>"


def load_vocab_map() -> dict[int, str]:
    # Try common locations
    for name in ["tokenizer.json", "tokenizer_vocab.txt", "vocab.json"]:
        p = Path(WEIGHTS) / name
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict) and "model" in data and "vocab" in data["model"]:
                    return {v: k for k, v in data["model"]["vocab"].items()}
            except Exception:
                pass
    return {}


def main() -> None:
    row = json.loads(MANIFEST.read_text().splitlines()[0])
    print(f"row: {row['image']} + {row['audio']}")

    model = cactus_init(WEIGHTS, None, False)
    pcm = pcm_bytes_from_wav(Path(row["audio"]))
    messages = build_messages(row["image"])

    all_ids = list(range(VOCAB))
    probs = cactus_score_candidates(
        model, messages, json.dumps(OPTIONS), None,
        PREFILL_SUFFIX, all_ids, pcm_data=pcm,
    )
    cactus_destroy(model)

    ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:TOPN]
    vocab_map = load_vocab_map()
    print(f"\nTop {TOPN} next-token predictions after prefix {PREFILL_SUFFIX!r}:")
    for i, (tid, p) in enumerate(ranked, 1):
        surface = vocab_map.get(tid, f"<id {tid}>")
        print(f"  {i:2d}. id={tid:>6d}  p={p:.4f}  token={surface!r}")


if __name__ == "__main__":
    main()
