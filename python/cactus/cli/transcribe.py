import os
import subprocess
from pathlib import Path

from .common import (
    DEFAULT_ASR_MODEL_ID,
    print_color,
    RED, GREEN,
)


def cmd_transcribe(args):
    from .model import resolve_model_id, ensure_bundle
    from .config_utils import CactusConfig
    from .common import prompt_for_api_key

    config = CactusConfig()
    api_key = prompt_for_api_key(config)

    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

    model_id = resolve_model_id(args.model_id)
    audio_file = args.audio_file

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    if args.force_handoff:
        os.environ["CACTUS_FORCE_HANDOFF"] = "1"
    else:
        os.environ.pop("CACTUS_FORCE_HANDOFF", None)

    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    if model_id and model_id.lower().endswith(audio_extensions):
        audio_file = model_id
        model_id = DEFAULT_ASR_MODEL_ID

    local_path = Path(model_id)
    if local_path.exists() and (local_path / "components" / "manifest.json").exists():
        bundle_dir = local_path
        print_color(GREEN, f"Using local model: {bundle_dir}")
    else:
        try:
            bundle_dir = ensure_bundle(
                model_id,
                token=args.token,
                cache_dir=args.cache_dir,
                reconvert=args.reconvert,
            )
        except RuntimeError as e:
            print_color(RED, f"Model setup failed: {e}")
            return 1

    asr = Path(__file__).resolve().parent.parent / "bin" / "asr"
    if not asr.exists():
        print_color(RED, "ASR binary not found. Run `cactus build` first.")
        return 1

    if not audio_file:
        print_color(RED, "No audio file specified. Use --file <audio.wav>")
        return 1

    cmd = [str(asr), str(bundle_dir), audio_file]
    if args.language:
        cmd.extend(["--language", args.language])

    print_color(GREEN, f"Starting Cactus ASR with model: {model_id}")
    print()

    return subprocess.run(cmd).returncode
