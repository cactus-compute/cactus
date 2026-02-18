import argparse
import hashlib
import json
import os
import shutil
import subprocess
from huggingface_hub import HfApi, hf_hub_download
from .cli import cmd_convert, get_weights_dir, PROJECT_ROOT

STAGE_DIR = PROJECT_ROOT / "stage"


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
        print("README.md not found")
        return 1

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
        return 0
    except Exception:
        print("Failed to update organization README")
        return 1


def export_and_publish_model(args, api):
    model_name = get_model_name(args.model)
    model_name_lower = model_name.lower()
    repo_id = f"{args.org}/{model_name}"

    precisions = []
    if args.int4:
        precisions.append(("int4", "4"))
    if args.int8:
        precisions.append(("int8", "8"))
    if args.fp16:
        precisions.append(("fp16", "16"))

    stage = STAGE_DIR / model_name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    weights_dir = stage / "weights"
    weights_dir.mkdir()

    try:
        fingerprint = hashlib.sha256()
        precisions_list = []

        for precision, bits in precisions:
            print(f"Exporting {args.model} with {precision}...")

            exported = export_model(args.model, os.environ.get("HF_TOKEN"), precision.upper())
            if not exported:
                print(f"Failed to export {precision}")
                continue

            base_zip = weights_dir / f"{model_name_lower}-{precision}.zip"
            zip_dir(exported, base_zip)
            fingerprint.update(sha256(base_zip).encode())

            if args.apple:
                try:
                    mlpackage = export_pro_weights(args.model, bits)
                    if mlpackage:
                        shutil.copytree(str(mlpackage), str(exported / mlpackage.name))
                        apple_zip = weights_dir / f"{model_name_lower}-{precision}-apple.zip"
                        zip_dir(exported, apple_zip)
                        fingerprint.update(sha256(apple_zip).encode())
                except Exception:
                    print(f"Failed to export Apple weights for {precision}")

            shutil.rmtree(exported)
            precisions_list.append(precision)

        config = {"model_type": model_name, "precisions": precisions_list, "fingerprint": fingerprint.hexdigest()}
        with open(stage / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        if changed(config, get_prev_config(api, repo_id, args.version)):
            api.upload_folder(
                folder_path=str(stage),
                path_in_repo=".",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {args.version}",
                delete_patterns=["*"],
            )
            print("Uploaded")
        else:
            print("Unchanged")

        api.create_tag(
            repo_id=repo_id,
            tag=args.version,
            revision=api.repo_info(repo_id=repo_id, repo_type="model").sha,
            repo_type="model",
            tag_message=f"Release {args.version}",
        )
        print("Tagged release")
        return 0

    except Exception:
        print("Model processing failed")
        return 1
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["export_model", "update_org_readme"])
    parser.add_argument("--version")
    parser.add_argument("--org")
    parser.add_argument("--model")
    parser.add_argument("--int4", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--apple", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set")
        return 1
    api = HfApi(token=token)

    if args.task == "export_model":
        if not all([args.version, args.org, args.model]):
            print("Error: export_model requires --version, --org, and --model")
            return 1
        if not any([args.int4, args.int8, args.fp16]):
            print("Error: At least one precision flag must be set")
            return 1
        return export_and_publish_model(args, api)
    elif args.task == "update_org_readme":
        if not args.org:
            print("Error: update_org_readme requires --org")
            return 1
        return update_org_readme(api, args.org)


if __name__ == "__main__":
    main()
