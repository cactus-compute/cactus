"""GATE B: every CQ1-repacked tensor in a converted bundle must dequantize
bit-exactly to the source checkpoint tensor. Also spot-checks full-27B tensors
through quantize_binary_repack directly.

Usage: gate_b.py <bundle_dir> <src_checkpoint_dir> [--sample-27b]
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from cactus.convert.export.qdq import FLAG_NO_ROTATION, dequantize_cq_file, read_header
from cactus.convert.quantization.cq import quantize_binary_repack


def source_tensors(src_dir):
    out = {}
    for shard in sorted(Path(src_dir).glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                out[name] = shard
    return out


def check_bundle(bundle_dir, src_dir):
    manifest = json.loads((Path(bundle_dir) / "conversion_manifest.json").read_text())
    src_map = source_tensors(src_dir)
    checked = failed = 0
    for row in manifest:
        if row["precision"] != "CQ1" or row["status"] != "converted":
            continue
        path = Path(bundle_dir) / row["output_file"]
        header = read_header(path)
        if not header.flags & FLAG_NO_ROTATION:
            print(f"NOT-REPACKED (rotated CQ1): {row['output_file']}")
            failed += 1
            continue
        if row.get("scale_factor", 1.0) != 1.0:
            print(f"SKIP scale_factor={row['scale_factor']}: {row['output_file']}")
            continue
        name = row["source_name"]
        with safe_open(src_map[name], framework="pt") as f:
            w = f.get_tensor(name).float().numpy()
        deq = dequantize_cq_file(path, header, torch.float32, row_batch_size=1024).numpy()
        checked += 1
        if not np.array_equal(deq, w):
            bad = np.abs(deq - w)
            print(f"MISMATCH {row['output_file']}: max abs err {bad.max()}, {np.count_nonzero(bad)} elems")
            failed += 1
    return checked, failed


def sample_27b():
    snaps = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--prism-ml--Bonsai-27B-unpacked/snapshots/*"))
    src = Path(snaps[0])
    index = json.loads((src / "model.safetensors.index.json").read_text())
    names = [
        "model.language_model.layers.31.self_attn.q_proj.weight",
        "model.language_model.layers.40.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.60.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ]
    failed = 0
    for name in names:
        with safe_open(src / index["weight_map"][name], framework="pt") as f:
            w = f.get_tensor(name).float().numpy()
        cq = quantize_binary_repack(w)
        if cq is None:
            print(f"27B sample NOT binary: {name}")
            failed += 1
            continue
        recon = np.where(cq.indices.astype(bool), 1.0, -1.0).astype(np.float32) * np.repeat(
            cq.norms.astype(np.float32), cq.group_size, axis=1
        )
        ok = np.array_equal(recon, w)
        print(f"27B sample {'EXACT' if ok else 'MISMATCH'}: {name} {w.shape}")
        failed += 0 if ok else 1
    return failed


if __name__ == "__main__":
    bundle, src = sys.argv[1], sys.argv[2]
    checked, failed = check_bundle(bundle, src)
    print(f"bundle: {checked} CQ1 tensors checked, {failed} failures")
    if "--sample-27b" in sys.argv:
        failed += sample_27b()
    print("GATE_B", "PASS" if (failed == 0 and checked > 0) else "FAIL")
    sys.exit(0 if (failed == 0 and checked > 0) else 1)
