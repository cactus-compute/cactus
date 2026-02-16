import argparse
import hashlib
import json
import os
import shutil
import subprocess
from huggingface_hub import HfApi, hf_hub_download
from .cli import cmd_convert, get_weights_dir, PROJECT_ROOT

STAGE_DIR = PROJECT_ROOT / "stage"

MODELS = [
    "google/gemma-3-270m-it",
    "google/functiongemma-270m-it",
    "LiquidAI/LFM2-350M",
    "Qwen/Qwen3-0.6B",
    "LiquidAI/LFM2-700M",
    "google/gemma-3-1b-it",
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "Qwen/Qwen3-1.7B",
    "LiquidAI/LFM2-2.6B",
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
    "UsefulSensors/moonshine-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "snakers4/silero-vad",
    "nomic-ai/nomic-embed-text-v2-moe",
    "Qwen/Qwen3-Embedding-0.6B",
]

PRO_MODELS = [
    "openai/whisper-small",
    "LiquidAI/LFM2-VL-450M",
    "openai/whisper-medium",
    "LiquidAI/LFM2.5-VL-1.6B",
]


def sha256(file):
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zip_dir(source_dir, output_path):
    subprocess.run(
        ["find", ".", "-exec", "touch", "-t", "200310131122", "{}", "+"],
        cwd=source_dir,
        check=True,
    )

    subprocess.run(
        ["zip", "-X", "-o", "-r", "-9", str(output_path), "."],
        cwd=source_dir,
        check=True,
        capture_output=True,
    )


def get_model_name(model_id):
    return model_id.split("/")[-1]


def export_model(model_id, token, precision):
    args = argparse.Namespace(
        model_name=model_id, output_dir=None, precision=precision, token=token
    )
    if cmd_convert(args) != 0:
        return None
    return get_weights_dir(model_id)


def export_pro_weights(model_id, bits):
    pro_repo = PROJECT_ROOT / "cactus-pro"
    if not pro_repo.exists():
        return None

    build_script = pro_repo / "apple" / "build.sh"
    if not build_script.exists():
        return None

    result = subprocess.run(
        ["bash", str(build_script), "--model", model_id, "--bits", bits],
        cwd=pro_repo,
        capture_output=True,
    )

    if result.returncode != 0:
        return None

    mlpackage = pro_repo / "apple" / "build" / "model.mlpackage"
    return mlpackage if mlpackage.exists() else None


def stage_model(model_id, weights_dir, precision, bits):
    model_name = get_model_name(model_id)
    stage = STAGE_DIR / model_name

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    model_name_lower = model_name.lower()
    weights_out = stage / "weights" / model_name_lower
    shutil.move(str(weights_dir), str(weights_out))

    model_zip = stage / "weights" / f"{model_name_lower}.zip"
    zip_dir(weights_out, model_zip)

    fingerprint = hashlib.sha256()
    fingerprint.update(sha256(model_zip).encode())

    config = {
        "model_type": model_name,
        "precision": precision,
    }

    if model_id in PRO_MODELS:
        try:
            mlpackage = export_pro_weights(model_id, bits)

            shutil.move(mlpackage, weights_out)

            model_pro_zip = stage / "weights" / f"{model_name_lower}-pro.zip"
            zip_dir(weights_out, model_pro_zip)

            fingerprint.update(sha256(model_pro_zip).encode())
            config["bits"] = bits
        except Exception:
            print("Failed to export pro weights")

    shutil.rmtree(weights_out)

    config["fingerprint"] = fingerprint.hexdigest()

    with open(stage / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return stage, config


def get_prev_config(api, repo, current):
    try:
        tags = api.list_repo_refs(repo_id=repo, repo_type="model").tags
        versions = sorted([t.name for t in tags], reverse=True)
        prev_ver = next((v for v in versions if v != current), None)
        if not prev_ver:
            return None
        local = hf_hub_download(
            repo_id=repo,
            filename="config.json",
            revision=prev_ver,
            repo_type="model",
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return None


def changed(curr, prev):
    if not prev:
        return True
    return curr.get("fingerprint") != prev.get("fingerprint")


def update_org_readme(api, org):
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        return

    try:
        api.create_repo(repo_id=f"{org}/README", repo_type="space", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=f"{org}/README",
            repo_type="space",
            commit_message="Update organization README",
        )
        print("Updated organization README")
    except Exception:
        print("Failed to update organization README")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--bits", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set")
        return 1

    api = HfApi(token=token)

    for model_id in MODELS:
        name = get_model_name(model_id)
        repo_id = f"{args.org}/{name}"

        stage_dir = None
        try:
            weights_dir = export_model(model_id, token, args.precision)
            if not weights_dir:
                print("Export failed")
                continue

            stage_dir, config = stage_model(
                model_id, weights_dir, args.precision, args.bits
            )
            prev = get_prev_config(api, repo_id, args.version)

            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

            if changed(config, prev):
                api.upload_folder(
                    folder_path=str(stage_dir),
                    path_in_repo=".",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {args.version}",
                )
                print("Uploaded")
            else:
                print("Unchanged")

            info = api.repo_info(repo_id=repo_id, repo_type="model")
            api.create_tag(
                repo_id=repo_id,
                tag=args.version,
                revision=info.sha,
                repo_type="model",
                tag_message=f"Release {args.version}",
            )
            print("Tagged release")

        except Exception:
            print("Model processing failed")
        finally:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir)
                print("Cleaned up stage directory")

    update_org_readme(api, args.org)


if __name__ == "__main__":
    main()
