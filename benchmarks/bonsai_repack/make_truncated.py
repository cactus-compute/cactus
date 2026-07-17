"""Build a 4-layer truncated Bonsai checkpoint for fast numerical-parity iteration.

Keeps original tensor names (model.language_model.*) so cactus convert routes
identically to the full model. Config keeps the nested VLM shape with
text_config.num_hidden_layers=4 and an explicit truncated layer_types list —
cactus does not derive layer_types from full_attention_interval, transformers
does, so omitting it would silently diverge the two stacks.
"""

import glob
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NUM_LAYERS = 4
KEEP_PREFIXES = tuple(
    [f"model.language_model.layers.{i}." for i in range(NUM_LAYERS)]
    + ["model.language_model.embed_tokens.", "model.language_model.norm.", "lm_head."]
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.jinja2",
)


def main(out_dir: str) -> None:
    snapshots = glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--prism-ml--Bonsai-27B-unpacked/snapshots/*")
    )
    if not snapshots:
        sys.exit("Bonsai-27B-unpacked not found in HF cache")
    src = Path(snapshots[0])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    wanted = {n: f for n, f in index["weight_map"].items() if n.startswith(KEEP_PREFIXES)}
    tensors = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(src / shard, framework="pt") as f:
            for name in f.keys():
                if name in wanted:
                    tensors[name] = f.get_tensor(name)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = NUM_LAYERS
    text["layer_types"] = text["layer_types"][:NUM_LAYERS]
    assert text["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    (out / "config.json").write_text(json.dumps(config, indent=2))

    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    print(f"tensors={len(tensors)} bytes={total/1e9:.2f}GB layer_types={text['layer_types']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights/bonsai-truncated-4l-src")
