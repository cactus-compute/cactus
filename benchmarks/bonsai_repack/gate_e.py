"""GATE E: exact-answer battery + GSM8K slice on a full-27B bundle, greedy local-only.

Usage: gate_e.py <bundle_dir> <out.json> [--floor 0.7]
Score = GSM8K accuracy (last number in the reply == reference answer).
The fixed factual battery is reported alongside (not part of the floor).
"""

import json
import re
import sys
import time
from pathlib import Path

from cactus.bindings.cactus import cactus_complete, cactus_destroy, cactus_init

FACTUAL = [
    ("What is 17*23? Answer with just the number.", "391"),
    ("Name the capital of Australia. Answer with just the city name.", "Canberra"),
    ("What year did the Apollo 11 mission land on the moon? Just the year.", "1969"),
    ("What is 144 divided by 12? Just the number.", "12"),
]
SYSTEM = "You are a helpful assistant"
OPTIONS = {"temperature": 0.0, "max_tokens": 512, "auto_handoff": False}


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def ask(model, prompt):
    resp = cactus_complete(
        model,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        options=OPTIONS,
    )
    return resp.get("response", ""), resp.get("decode_tps", 0.0)


def main(bundle, out_path, floor):
    gsm = json.loads((Path(__file__).parent / "gsm8k_20.json").read_text())
    model = cactus_init(bundle)
    results = {"factual": [], "gsm8k": [], "bundle": bundle}
    tps = []
    try:
        for prompt, expected in FACTUAL:
            text, t = ask(model, prompt)
            ok = expected.lower() in text.lower()
            results["factual"].append({"prompt": prompt, "expected": expected, "text": text[-200:], "ok": ok})
            tps.append(t)
            print(f"factual {'OK ' if ok else 'MISS'} ({expected}): {text[-80:]!r}")
        for i, item in enumerate(gsm):
            start = time.time()
            text, t = ask(model, item["question"] + "\nGive the final numeric answer after '####'.")
            got = last_number(text)
            ok = got is not None and got == item["answer"]
            results["gsm8k"].append({"i": i, "expected": item["answer"], "got": got, "ok": ok})
            tps.append(t)
            print(f"gsm8k[{i}] {'OK ' if ok else 'MISS'} expected={item['answer']} got={got} ({time.time()-start:.0f}s, {t:.1f} tok/s)")
    finally:
        cactus_destroy(model)
    acc = sum(r["ok"] for r in results["gsm8k"]) / len(gsm)
    fact = sum(r["ok"] for r in results["factual"])
    results["gsm8k_accuracy"] = acc
    results["factual_correct"] = fact
    results["mean_decode_tps"] = sum(tps) / len(tps)
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"GSM8K {acc:.0%} | factual {fact}/{len(FACTUAL)} | mean decode {results['mean_decode_tps']:.1f} tok/s")
    print("GATE_E", "PASS" if acc >= floor else "FAIL")
    return 0 if acc >= floor else 1


if __name__ == "__main__":
    floor = 0.7 if "--floor" not in sys.argv else float(sys.argv[sys.argv.index("--floor") + 1])
    sys.exit(main(sys.argv[1], sys.argv[2], floor))
