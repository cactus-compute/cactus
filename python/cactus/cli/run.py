import os
import subprocess
from pathlib import Path

from .common import GREEN, RED, YELLOW, apply_cloud_api_key_env, print_color, resolve_binary


def cmd_run(args) -> int:
    from .download import resolve_platform

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
    apply_cloud_api_key_env()

    if args.image:
        args.image = str(Path(args.image).expanduser())
    if args.audio:
        args.audio = str(Path(args.audio).expanduser())
    if args.result_json:
        args.result_json = str(Path(args.result_json).expanduser())

    platform = resolve_platform(args.platform)
    bundle_dir = _resolve_or_fetch_bundle(
        args.model_id, bits=args.bits, platform=platform,
        token=args.token, reconvert=args.reconvert,
        image=args.image, audio=args.audio,
    )
    if bundle_dir is None:
        return 1

    binary = resolve_binary("run")
    if binary is None:
        return 1

    cmd = [str(binary), str(bundle_dir)]
    for flag, value in (
        ("--system", args.system),
        ("--prompt", args.prompt),
        ("--image", args.image),
        ("--audio", args.audio),
        ("--input-ids", args.input_ids),
        ("--input-ids-file", args.input_ids_file),
        ("--max-new-tokens", args.max_new_tokens),
        ("--result-json", args.result_json),
        ("--confidence-threshold", args.confidence_threshold),
        ("--cloud-timeout-ms", args.cloud_timeout_ms),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.thinking:
        cmd.append("--thinking")
    if args.no_cloud_handoff:
        cmd.append("--no-cloud-handoff")

    print_color(GREEN, f"Running: {bundle_dir}")
    print()
    return subprocess.run(cmd).returncode


def _resolve_or_fetch_bundle(model_id, *, bits, platform, token, reconvert, image, audio):
    from .download import download_bundle, get_bundle_dir
    from .model import TranspileOptions, ensure_bundle, resolve_bundle_dir

    local = resolve_bundle_dir(model_id)
    if local is not None:
        return local

    cached = get_bundle_dir(model_id, bits=bits, platform=platform)
    if (cached / "components" / "manifest.json").exists() and not reconvert:
        return cached

    if not reconvert:
        try:
            return download_bundle(model_id, bits=bits, platform=platform, token=token, reconvert=reconvert)
        except (RuntimeError, OSError) as exc:
            print_color(YELLOW, f"HF download unavailable ({exc}); building locally")

    try:
        return ensure_bundle(
            model_id, bits=bits, token=token, reconvert=reconvert,
            transpile=TranspileOptions(
                image_files=[image] if image else None,
                audio_file=audio,
            ),
        )
    except RuntimeError as exc:
        print_color(RED, f"Model setup failed: {exc}")
        return None
