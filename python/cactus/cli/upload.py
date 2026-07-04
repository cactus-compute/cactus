import os
import tempfile
import zipfile
from pathlib import Path

from .common import BLUE, GREEN, PROJECT_ROOT, print_color
from .download import get_model_dir_name, resolve_weights_variant
from .model import prepare_bundle
from .utils import suggested_cq_repo, variant_suffix


def cmd_upload(args):
    """Build a runnable bundle locally and upload it to Cactus-Compute on HuggingFace."""
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("missing HuggingFace token: pass --token or set HF_TOKEN")

    platform = resolve_weights_variant(getattr(args, "weights", "general"))
    repo_id = suggested_cq_repo(args.model_id)
    stem = f"{get_model_dir_name(args.model_id)}-{variant_suffix(args.bits, platform)}"
    archive_name = f"{stem}.zip"
    tag = (PROJECT_ROOT / "CACTUS_VERSION").read_text(encoding="utf-8").strip()

    bundle_dir = prepare_bundle(args, prebuilt=False, fail_prefix=f"Build failed for {args.model_id}")
    if bundle_dir is None:
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cactus-upload-") as tmp:
        archive_path = Path(tmp) / archive_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
                zf.write(path, f"{stem}/{path.relative_to(bundle_dir).as_posix()}")
        print_color(BLUE, f"Uploading {archive_name} ({archive_path.stat().st_size / 1e6:.1f} MB) -> {repo_id}")
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
    print_color(GREEN, f"Done: {repo_id} @ {tag}")
    return 0
