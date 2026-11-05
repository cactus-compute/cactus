import os

def _weights_spec(spec):
    parts = [p for p in spec.split("/") if p]
    if len(parts) < 2:
        raise SystemExit("pass <org>/<repo>/<file>.cact or <org>/<repo>")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def _download_target(spec):
    """Classify a `cactus download` spec: engine build, base weights, checkpoint or hub archive."""
    from ..agent import fetch

    if "/" in spec:
        return "hub", spec
    if spec.startswith("model-"):
        return "hosted", spec
    if spec in fetch.PLATFORMS:
        return "platform", spec
    name = spec[:-5] if spec.endswith(".cact") else spec
    if name == "needle3":
        return "base", int(name[-1])
    if name in (fetch.WHISTLE, fetch.MIRAGE):
        return "base", name
    if spec.endswith(".safetensors") and not spec.startswith("needle2"):
        return "checkpoint", spec
    raise SystemExit(
        f"unknown download {spec!r}: pass needle3 (base weights), needle3.safetensors "
        f"(the checkpoint to fine-tune), a platform ("
        + ", ".join(fetch.PLATFORMS) + "), model-<id> from cactuscompute.com, "
        "or <org>/<repo>[/<file>.cact]")


def download(args):
    import shutil
    from huggingface_hub import hf_hub_download, list_repo_files
    from ..agent import fetch
    kind, target = _download_target(args.spec)
    if kind == "platform":
        paths = fetch.download_platform(target, args.out)
        model = 3 if args.model == "needle3" else args.model
        if model is not None:
            paths.append(fetch.fetch_weights(model, os.path.join(args.out, target)))
        for path in paths:
            print(f"  {'file':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
    elif kind == "base":
        os.makedirs(args.out, exist_ok=True)
        path = fetch.fetch_weights(target, args.out)
        print(f"  {'weights':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        if target == fetch.WHISTLE:
            print(f"  {'next':<9} cactus.transcribe('clip.wav', weights={path!r})")
        elif target == fetch.MIRAGE:
            print(f"  {'next':<9} cactus.Mirage(weights={path!r}).describe('image.png')")
        else:
            print(f"  {'next':<9} cactus.Cactus(weights={path!r}, tools=[...])")
    elif kind == "hosted":
        from ..platform import Platform
        for path in Platform().download(target, args.out, depth=args.depth):
            print(f"  {'weights':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        print(f"  {'next':<9} cactus.Cactus(weights=<path>, tools=[...], auto_date=False)")
    elif kind == "checkpoint":
        generation = fetch.WHISTLE if target.startswith("whistle") else fetch.MIRAGE if target.startswith("mirage") else 3
        path = fetch.fetch_checkpoint(target, os.path.join(args.out, fetch.CHECKPOINT_PREFIX),
                                      generation=generation)
        print(f"  {'file':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        print(f"  {'next':<9} cactus finetune data.jsonl --checkpoint {path}")
    else:
        repo, filename = _weights_spec(args.spec)
        if not filename:
            cacts = [f for f in list_repo_files(repo) if f.endswith(".cact")]
            if len(cacts) != 1:
                raise SystemExit(f"{repo} holds {len(cacts)} .cact files, name one: "
                                 + ", ".join(cacts[:5]))
            filename = cacts[0]
        cached = hf_hub_download(repo_id=repo, filename=filename, repo_type="model")
        os.makedirs(args.out, exist_ok=True)
        dest = os.path.join(args.out, os.path.basename(filename))
        shutil.copyfile(cached, dest)
        print(f"  {'weights':<9} {dest}  {os.path.getsize(dest) / 1e6:.2f} MB")
        print(f"  {'next':<9} cactus.Cactus(weights={dest!r}, tools=[...])")

def engine(args):
    from ..agent import fetch
    version = fetch.engine_version()
    dest = args.out or fetch.engine_cache_dir()
    os.makedirs(dest, exist_ok=True)
    path = fetch.fetch_library(version, dest, tag=args.platform_tag)
    print(f"  {'engine':<9} {path}")
    print(f"  {'deploy':<9} copy to {fetch.engine_cache_dir()}/ "
          f"on the device, or point {fetch.lib_path_env()} at the file")
