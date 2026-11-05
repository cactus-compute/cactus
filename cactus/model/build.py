import os
import shutil

from ..agent import fetch


def build_main(args):
    layers = getattr(args, "layers", None)
    if args.lora or layers:
        from .finetune import DEFAULT_BASE, merge_lora, rung
        import jax.numpy as jnp
        from .architecture import ConfidenceHead, effective_kv_window
        from .checkpoints import read_adapter
        from .export import read_layers, read_tokenizer_blob, write_export
        from .quantize import WEIGHT_BITS
        from .run import load_checkpoint

    base_archive = fetch.fetch_weights(3, force=True)
    print(f"  {'base':<9} {base_archive}  {os.path.getsize(base_archive) / 1e6:.2f} MB")
    platform = getattr(args, "platform", None)
    folder = None
    if platform:
        folder = os.path.abspath(args.out or platform)
        for path in fetch.download_platform(platform, os.path.dirname(folder), generation=3, dest=folder):
            print(f"  {'engine':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        out = os.path.join(folder, fetch.base_weights(3))
    else:
        out = args.out
    if not args.lora and (not layers or layers == read_layers(base_archive)):
        if not out:
            raise SystemExit("pass --out <archive.cact>, or --platform to build a runnable folder")
        shutil.copyfile(base_archive, out)
        print(f"  {'wrote':<9} {out}  {os.path.getsize(out) / 1e6:.2f} MB  the published base archive")
    else:
        adapter = read_adapter(args.lora) if args.lora else None
        if adapter and adapter.get("languages") is not None:
            raise SystemExit(f"{args.lora} is a Whistle adapter; build it with cactus whistle build")
        checkpoint = args.checkpoint
        if not checkpoint and adapter and adapter.get("base") and os.path.exists(adapter["base"]):
            checkpoint = adapter["base"]
        checkpoint = checkpoint or DEFAULT_BASE
        params, config, _ = load_checkpoint(checkpoint, return_run=True)
        if args.lora:
            lora = {tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                    for key, v in adapter["lora"].items()}
            params = merge_lora(params, lora, adapter["scale"])
            print(f"  {'merged':<9} {len(lora)} weight groups  {args.lora}")
            if ConfidenceHead.key in params:
                params = {k: v for k, v in params.items() if k != ConfidenceHead.key}
                print(f"  {'dropped':<9} the confidence head; it is not trained locally, so confidence reports None")
        params, config = rung(params, config, layers)
        print(f"  {'depth':<9} {config.num_layers} layers")
        out = out or (os.path.splitext(os.path.basename(checkpoint))[0] + ".cact")
        info = write_export(params, config, out, bits=WEIGHT_BITS,
                            tokenizer=read_tokenizer_blob(base_archive),
                            kv_window=effective_kv_window(config))
        print(f"  {'wrote':<9} {info['path']}  {info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors  W{WEIGHT_BITS}A8")
    if folder:
        runner = next((n for n in ("cactus", "cactus.exe") if os.path.exists(os.path.join(folder, n))), None)
        if runner:
            print(f"  {'next':<9} {os.path.join(folder, runner)} --model {os.path.basename(out)} --tools tools.json --serve")
    else:
        print(f"  {'next':<9} cactus.Needle(weights={out!r}, tools=[...])")

    if args.upload:
        upload_archive(out)


def upload_archive(out):
    repo = os.environ.get("CACTUS_HF_REPO")
    if not repo:
        raise SystemExit("set CACTUS_HF_REPO=<you>/<model> to upload")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=out, path_in_repo=os.path.basename(out),
                    repo_id=repo, repo_type="model")
    print(f"  {'uploaded':<9} {os.path.basename(out)}  {repo}")
