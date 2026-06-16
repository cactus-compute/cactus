#!/usr/bin/env python3
"""Convert, transpile, package, and upload a Cactus weight bundle to huggingface.co/Cactus-Compute.

    python scripts/upload_weights.py google/gemma-4-E2B-it --acceleration apple --token hf_xxx
    python scripts/upload_weights.py openai/whisper-base --acceleration cpu --token hf_xxx
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from cactus.cli.download import (  # noqa: E402
    get_bundle_dir,
    get_model_dir_name,
    resolve_platform,
)
from cactus.cli.model import TranspileOptions, ensure_bundle  # noqa: E402
from cactus.cli.utils import (  # noqa: E402
    suggested_cq_repo,
    variant_suffix,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert, transpile, and upload a Cactus weight bundle to HuggingFace.")
    p.add_argument("model", help="model id, e.g. google/gemma-4-E2B-it")
    p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4, help="CQ quantization bits (default: 4)")
    p.add_argument("--acceleration", default="cpu", help="'cpu' or 'apple' (default: cpu)")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HuggingFace token")
    args = p.parse_args()

    if not args.token:
        raise SystemExit("missing HuggingFace token: pass --token or set HF_TOKEN")

    platform = resolve_platform(args.acceleration)
    repo_id = suggested_cq_repo(args.model)
    stem = f"{get_model_dir_name(args.model)}-{variant_suffix(args.bits, platform)}"
    archive_name = f"{stem}.zip"
    tag = (REPO_ROOT / "CACTUS_VERSION").read_text(encoding="utf-8").strip()

    bundle_dir = get_bundle_dir(args.model, bits=args.bits, platform=platform)
    ensure_bundle(
        args.model,
        bits=args.bits,
        token=args.token,
        output_dir=bundle_dir,
        transpile=TranspileOptions(npu=platform is not None),
    )

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cactus-upload-") as tmp:
        archive_path = Path(tmp) / archive_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
                zf.write(path, f"{stem}/{path.relative_to(bundle_dir).as_posix()}")
        print(f"uploading {archive_name} ({archive_path.stat().st_size / 1e6:.1f} MB) -> {repo_id}")
        api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=archive_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {archive_name} ({tag})",
        )

    try:
        api.delete_tag(repo_id=repo_id, tag=tag, repo_type="model")
    except Exception:
        pass
    api.create_tag(repo_id=repo_id, tag=tag, repo_type="model", revision="main")
    print(f"done: {repo_id} @ {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
