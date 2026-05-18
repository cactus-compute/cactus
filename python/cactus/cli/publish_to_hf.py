import argparse
import os
import shutil
import subprocess
import sys

import yaml
from huggingface_hub import HfApi

from .convert import cmd_convert
from .common import PROJECT_ROOT
from cactus.transpile.component_plan import infer_component_plan_from_output


def _convert(model_id, bits, output_dir):
    if cmd_convert(argparse.Namespace(model_name=model_id, output_dir=str(output_dir), bits=bits)) != 0:
        return None
    return output_dir


def _archive_name(weights_dir, model_id, bits):
    plan = infer_component_plan_from_output(weights_dir, model_id=model_id)
    modalities = "L"
    if plan:
        if plan.needs_image:
            modalities += "V"
        if plan.needs_audio:
            modalities += "A"
    return "".join(f"{m}{bits}" for m in modalities)


def _zip_dir(source_dir, output_path):
    subprocess.run(["find", ".", "-exec", "touch", "-t", "200310131122", "{}", "+"],
                   cwd=source_dir, check=True)
    subprocess.run(["zip", "-X", "-o", "-r", "-9", str(output_path), "."],
                   cwd=source_dir, check=True, capture_output=True)


def _publish(args, api):
    model_name = args.model.split("/")[-1].replace("_", "-")
    cq_name = f"{model_name}-cq" if not model_name.lower().endswith("-cq") else model_name
    repo_id = f"{args.org}/{cq_name}"

    stage = PROJECT_ROOT / "stage" / model_name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "weights").mkdir()

    try:
        archives = []
        for bits in sorted(set(args.bits)):
            print(f"Converting {args.model} at {bits}-bit...")
            work = stage / f"work-{bits}"
            exported = _convert(args.model, bits, work)
            if not exported:
                print(f"Failed to convert at {bits}-bit")
                return 1

            name = _archive_name(exported, args.model, bits)
            path = stage / "weights" / f"{name}.zip"
            _zip_dir(exported, path)
            archives.append(name)
            shutil.rmtree(exported)

        # Model card
        try:
            info = api.model_info(args.model)
            license_ = getattr(info.card_data, "license", None) if info.card_data else None
        except Exception:
            license_ = None

        meta = {"base_model": args.model}
        if args.pipeline_tag:
            meta["pipeline_tag"] = args.pipeline_tag
        if args.tags:
            tags = [t.strip() for t in args.tags.split(",") if t.strip()]
            if tags:
                meta["tags"] = tags
        if license_:
            meta["license"] = license_
        if args.description:
            meta["description"] = args.description

        readme = f"---\n{yaml.safe_dump(meta, default_flow_style=False, allow_unicode=True).strip()}\n---\n"
        (stage / "README.md").write_text(readme, encoding="utf-8")

        # Upload everything in one commit
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(stage), path_in_repo=".",
                          repo_id=repo_id, repo_type="model",
                          commit_message=f"Upload {args.version}")
        api.create_tag(repo_id=repo_id, tag=args.version, repo_type="model",
                       revision=api.repo_info(repo_id=repo_id, repo_type="model").sha,
                       tag_message=f"Release {args.version}", exist_ok=True)

        print(f"Published {repo_id} [{', '.join(archives)}] tagged {args.version}")
        return 0

    except Exception as e:
        print(f"Failed: {e}")
        return 1
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    p = argparse.ArgumentParser(description="Publish CQ models to Hugging Face")
    p.add_argument("--version", required=True)
    p.add_argument("--org", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], nargs="+", default=[4])
    p.add_argument("--pipeline-tag", dest="pipeline_tag")
    p.add_argument("--tags")
    p.add_argument("--description")
    return _publish(p.parse_args(), HfApi(token=os.environ.get("HF_TOKEN") or sys.exit("Error: HF_TOKEN not set")))


if __name__ == "__main__":
    sys.exit(main())
