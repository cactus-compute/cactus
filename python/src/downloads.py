"""Download pre-converted model weights from Cactus-Compute HuggingFace.

Usage::

    from src.downloads import ensure_model
    weights_dir = ensure_model("openai/whisper-tiny")
"""
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_model_dir_name(model_id: str) -> str:
    """Convert HuggingFace model ID to local directory name."""
    return model_id.split("/")[-1].lower()


def get_weights_dir(model_id: str) -> Path:
    """Return ``<project>/weights/<model_name>``."""
    if "silero-vad" in model_id.lower():
        return _PROJECT_ROOT / "weights" / "silero-vad"
    return _PROJECT_ROOT / "weights" / get_model_dir_name(model_id)


def download_from_hf(model_id: str, weights_dir: Path, precision: str = "INT4") -> bool:
    """Download pre-converted weights from Cactus-Compute HuggingFace.

    Returns True on success, False if the model is unavailable or download fails.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import zipfile
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub")
        return False

    model_name = get_model_dir_name(model_id)
    repo_id = f"Cactus-Compute/{model_id.split('/')[-1]}"

    try:
        precision_lower = precision.lower()
        apple_zip = f"{model_name}-{precision_lower}-apple.zip"
        standard_zip = f"{model_name}-{precision_lower}.zip"

        repo_files = list_repo_files(repo_id, repo_type="model")

        zip_file = None
        if f"weights/{apple_zip}" in repo_files:
            zip_file = apple_zip
        elif f"weights/{standard_zip}" in repo_files:
            zip_file = standard_zip
        else:
            print(f"Pre-converted model not found in {repo_id}")
            return False

        print(f"Downloading {repo_id}/{zip_file} ...")

        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"weights/{zip_file}",
            repo_type="model",
        )

        weights_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting model weights...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(weights_dir)

        if not (weights_dir / "config.txt").exists():
            print(f"Error: downloaded model is missing config.txt")
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            return False

        config_path = weights_dir / "config.txt"
        config_text = config_path.read_text()
        if "quantization=" not in config_text:
            with open(config_path, "a") as f:
                f.write(f"quantization={precision}\n")

        print(f"Model ready at {weights_dir}")
        return True

    except Exception as exc:
        print(f"Download failed: {exc}")
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        return False


def ensure_model(model_id: str, precision: str = "INT4") -> Path:
    """Return the weights directory, downloading if necessary.

    Raises ``RuntimeError`` if the model cannot be obtained.
    """
    weights_dir = get_weights_dir(model_id)
    if (weights_dir / "config.txt").exists():
        return weights_dir
    if not download_from_hf(model_id, weights_dir, precision):
        raise RuntimeError(f"Could not download model {model_id}")
    return weights_dir
