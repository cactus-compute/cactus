import sys
import argparse

from .. import __version__
from .common import (
    DEFAULT_MODEL_ID,
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    DEFAULT_TEST_MODEL_ID,
    DEFAULT_TEST_TRANSCRIPTION_MODEL_ID,
    SUPPORTED_PLATFORMS,
)
from .download import cmd_download

_PLATFORM_CHOICES = ("auto", "cpu", *SUPPORTED_PLATFORMS)
_PLATFORM_HELP = (
    f"target platform: auto = best for this host, e.g. apple on macOS (default); "
    f"cpu = generic ARM; "
    f"or one of: {', '.join(SUPPORTED_PLATFORMS) if SUPPORTED_PLATFORMS else '(none yet)'}"
)
_PLATFORM_PIPE = "|".join(_PLATFORM_CHOICES)
from .compile import cmd_build
from .serve import cmd_serve
from .transcribe import cmd_transcribe
from .test import cmd_test, COMPONENTS
from .convert import cmd_convert, cmd_transpile
from .upload import cmd_upload
from .run import cmd_run
from .list import cmd_list

from .auth import cmd_auth
from .clean import cmd_clean


def _telemetry_parent():
    """Args shared by commands that support telemetry toggle."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-cloud-tele", action="store_true",
                   help="Disable cloud telemetry (write to cache only)")
    return p


def _build_parent():
    """Bundle-build flags shared by every command that prepares a model."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                   help="CQ quantization (default: 4)")
    p.add_argument("--platform", choices=_PLATFORM_CHOICES, default="auto",
                   help=_PLATFORM_HELP)
    p.add_argument("--token", help="HuggingFace token")
    p.add_argument("--reconvert", action="store_true",
                   help="Force local rebuild from source")
    return p


def _positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {n}")
    return n


def _non_negative_int(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _port_int(value):
    n = int(value)
    if n < 1 or n > 65535:
        raise argparse.ArgumentTypeError(f"port must be in 1..65535, got {n}")
    return n


def _unit_float(value):
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0.0, 1.0], got {f}")
    return f


def _hf_id_or_path(value):
    v = (value or "").strip()
    if "/" not in v:
        raise argparse.ArgumentTypeError(
            f"invalid model {value!r}. Use a HuggingFace id like 'openai/whisper-base' "
            f"or a path like '/abs/path' or './bundle'."
        )
    return v




def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=argparse.SUPPRESS,
        description=f"""

  -----------------------------------------------------------------

  Cactus CLI:

  -----------------------------------------------------------------

  cactus auth                          manage cloud API key
    --status                           show key status
    --clear                            remove saved key

  cactus run [model|path]              run a model (default: {DEFAULT_MODEL_ID})
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --image <path>                     image file for VLM inference
    --audio <path>                     audio file for audio chat
    --system <prompt>                  system prompt
    --prompt <text>                    send prompt immediately
    --thinking                         enable thinking/reasoning mode
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source

  cactus transcribe [model]            live microphone transcription with a model
    --file <audio.wav>                 audio file to transcribe (WAV)
    --language <code>                  language code (default: en)
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source

  cactus download [model]              fetch a prebuilt bundle, else build locally (default: {DEFAULT_MODEL_ID})
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source

  cactus convert <model> [dir]         build a runnable bundle locally (skips prebuilt fetch)
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source
    --lora <path>                      merge a LoRA adapter before converting

  cactus serve [model]                 OpenAI-compatible local HTTP server
    --host <addr>                      bind address (default: 127.0.0.1)
    --port <port>                      port (default: 8080)
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source
    --no-cloud-handoff                 disable automatic cloud handoff
    --confidence-threshold <0..1>      handoff to cloud below this confidence
    --cloud-timeout-ms <n>             max wait for cloud handoff before local fallback

  cactus list                          list downloaded models

  cactus build                         build cactus libraries
    --apple                            Apple (iOS/macOS)
    --android                          Android
    --python                           shared lib for Python FFI

  cactus test                          run the test suite
    --component <name>                 kernels | graph | engine | all
                                       (default: all)
    --model <hf-id>                    default: {DEFAULT_TEST_MODEL_ID}
    --transcription-model <hf-id>      default: {DEFAULT_TEST_TRANSCRIPTION_MODEL_ID}
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --platform {_PLATFORM_PIPE:<22}  target platform (default: auto)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild of test models
    --suite <name>                     run a single test suite from any
                                       component (kernels, graph, or engine)
    --list                             list components and suites
    --ios                              run on connected iPhone
    --android                          run on connected Android
    --enable-telemetry                 send cloud telemetry (off by default)

  cactus clean                         delete build artifacts, weights, venv

  cactus --help                        show this help

  -----------------------------------------------------------------
"""
    )

    parser.add_argument("--version", action="version", version=f"cactus {__version__}")

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = False

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action.help = argparse.SUPPRESS

    parser._action_groups = []

    download_parser = subparsers.add_parser("download",
                                            help="Download a pre-built bundle from huggingface.co/Cactus-Compute",
                                            parents=[_build_parent()])
    download_parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                                 type=_hf_id_or_path,
                                 help=f"HuggingFace model id (default: {DEFAULT_MODEL_ID})")

    build_parser = subparsers.add_parser("build", help="Build cactus libraries")
    build_group = build_parser.add_mutually_exclusive_group()
    build_group.add_argument("--apple", action="store_true",
                             help="Build for Apple (iOS/macOS)")
    build_group.add_argument("--android", action="store_true",
                             help="Build for Android")
    build_group.add_argument("--python", action="store_true",
                             help="Build shared library for Python FFI")

    run_parser = subparsers.add_parser("run", help="Run a model (downloads bundle if needed)",
                                       parents=[_telemetry_parent(), _build_parent()])
    run_parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                            type=_hf_id_or_path,
                            help=f"HuggingFace model id or local bundle path (default: {DEFAULT_MODEL_ID})")
    run_parser.add_argument("--image",
                            help="Path to image file for VLM inference (attached to first message)")
    run_parser.add_argument("--audio",
                            help="Path to audio file (WAV) for audio chat (attached to first message)")
    run_parser.add_argument("--system",
                            help="System prompt to prepend to all messages")
    run_parser.add_argument("--prompt",
                            help="Initial prompt to send immediately")
    run_parser.add_argument("--input-ids", default=None,
                            help="Comma-separated token ids for causal-LM bundles")
    run_parser.add_argument("--input-ids-file", default=None,
                            help="File containing token ids for causal-LM bundles")
    run_parser.add_argument("--max-new-tokens", type=_positive_int, default=None,
                            help="Maximum tokens to generate for causal-LM bundles")
    run_parser.add_argument("--result-json", default=None,
                            help="Optional path to save bundle results as JSON")
    run_parser.add_argument("--thinking", action="store_true",
                            help="Enable thinking/reasoning for models that support it")
    run_parser.add_argument("--no-cloud-handoff", action="store_true",
                            help="Disable automatic cloud handoff for this run")
    run_parser.add_argument("--confidence-threshold", type=_unit_float, default=None,
                            help="Confidence threshold below which local completions may hand off to cloud")
    run_parser.add_argument("--cloud-timeout-ms", type=_non_negative_int, default=None,
                            help="Maximum time to wait for cloud handoff before falling back locally")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio with a model",
                                              parents=[_telemetry_parent(), _build_parent()])
    transcribe_parser.add_argument("model_id", nargs="?", default=DEFAULT_TRANSCRIPTION_MODEL_ID,
                                   type=_hf_id_or_path,
                                   help=f"HuggingFace model id (default: {DEFAULT_TRANSCRIPTION_MODEL_ID})")
    transcribe_parser.add_argument("--file", dest="audio_file", default=None,
                                   help="Audio file to transcribe (WAV)")
    transcribe_parser.add_argument("--language", default="en",
                                   help="Language code (default: en)")

    serve_parser = subparsers.add_parser("serve", help="OpenAI-compatible local HTTP server",
                                         parents=[_telemetry_parent(), _build_parent()])
    serve_parser.add_argument("model_id", nargs="?", default=None,
                              type=_hf_id_or_path,
                              help="HuggingFace model id (e.g. openai/whisper-base) or bundle path")
    serve_parser.add_argument("--host", default="127.0.0.1",
                              help="Bind address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=_port_int, default=8080,
                              help="Port (default: 8080)")
    serve_parser.add_argument("--no-cloud-handoff", action="store_true",
                              help="Disable automatic cloud handoff for all requests")
    serve_parser.add_argument("--confidence-threshold", type=_unit_float, default=None,
                              help="Confidence threshold below which completions hand off to cloud (1.0 forces cloud handoff)")
    serve_parser.add_argument("--cloud-timeout-ms", type=_non_negative_int, default=None,
                              help="Maximum time to wait for cloud handoff before falling back locally")

    test_parser = subparsers.add_parser("test", help="Run the test suite",
                                        parents=[_build_parent()])
    test_parser.add_argument("--component", choices=COMPONENTS, default="all",
                             help="Component to test (default: all)")
    test_parser.add_argument("--model", dest="model_id", default=None,
                             type=_hf_id_or_path,
                             help=f"HF model ID under test (default: {DEFAULT_TEST_MODEL_ID})")
    test_parser.add_argument("--transcription-model", dest="transcription_model_id", default=None,
                             type=_hf_id_or_path,
                             help=f"HF transcription model ID under test (default: {DEFAULT_TEST_TRANSCRIPTION_MODEL_ID})")
    test_parser.add_argument("--suite", default=None,
                             help="Run a single test suite by name; resolved across all components (e.g. llm → engine)")
    test_parser.add_argument("--list", action="store_true",
                             help="List available components and engine tests, then exit")
    test_parser.add_argument("--android", action="store_true", help="Run tests on Android")
    test_parser.add_argument("--ios", action="store_true", help="Run tests on iOS")
    test_parser.add_argument("--enable-telemetry", action="store_true",
                             help="Enable cloud telemetry (disabled by default in tests)")

    auth_parser = subparsers.add_parser("auth", help="Manage cloud API key")
    auth_parser.add_argument("--clear", action="store_true",
                             help="Remove the saved API key")
    auth_parser.add_argument("--status", action="store_true",
                             help="Show current key status")

    clean_parser = subparsers.add_parser("clean", help="Delete build artifacts, downloaded weights, and venv")
    clean_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    subparsers.add_parser("list", help="List downloaded models")

    convert_parser = subparsers.add_parser("convert",
                                           help="Build a runnable bundle locally from source (no prebuilt fetch)",
                                           parents=[_build_parent()])
    convert_parser.add_argument("model_id", type=_hf_id_or_path,
                                help="HuggingFace model id (e.g. openai/whisper-base)")
    convert_parser.add_argument("output_dir", nargs="?", default=None,
                                help="Output directory (default: weights/<model>)")
    convert_parser.add_argument("--lora",
                                help="Path or HF id of a LoRA adapter to merge before converting (requires `peft`)")

    upload_parser = subparsers.add_parser("upload", parents=[_build_parent()])
    upload_parser.add_argument("model_id", type=_hf_id_or_path)

    transpile_parser = subparsers.add_parser("transpile",
                                             help="Build a runnable bundle from CQ weights")
    transpile_parser.add_argument("model_id", type=_hf_id_or_path,
                                  help="HuggingFace model id or local checkpoint path")
    transpile_parser.add_argument("--weights-dir",
                                  help="CQ weights directory (default: weights/<model>)")
    transpile_parser.add_argument("--task", default="auto",
                                  choices=["auto", "causal_lm_logits", "multimodal_causal_lm_logits",
                                           "ctc_logits", "encoder_hidden_states",
                                           "seq2seq_transcription", "tdt_transcription"],
                                  help="Transpile task (default: auto, from model config)")
    transpile_parser.add_argument("--prompt",
                                  help="Prompt for causal/multimodal shape capture")
    transpile_parser.add_argument("--system-prompt", default=None,
                                  help="System prompt for multimodal chat formats")
    transpile_parser.add_argument("--enable-thinking", action="store_true",
                                  help="Enable thinking markers when the prompt supports them")
    transpile_parser.add_argument("--input-ids", default=None,
                                  help="Comma-separated token ids for causal-LM shape capture")
    transpile_parser.add_argument("--image-file", action="append", default=[],
                                  help="Image for multimodal shape capture (repeatable)")
    transpile_parser.add_argument("--audio-file",
                                  help="Audio file (WAV) for audio/multimodal shape capture")
    transpile_parser.add_argument("--max-new-tokens", type=_positive_int, default=None,
                                  help="Decode context to preallocate for causal LM (default: 32)")
    transpile_parser.add_argument("--component-pipeline", default="auto", choices=["auto", "on", "off"],
                                  help="Split-component transpilation when supported (default: auto)")
    transpile_parser.add_argument("--components",
                                  help="Comma-separated component subset (e.g. vision_encoder,decoder)")
    transpile_parser.add_argument("--torch-dtype", default=None,
                                  choices=["float16", "float32", "bfloat16"],
                                  help="Torch dtype for HF loading (default: float16)")
    transpile_parser.add_argument("--token", default=None,
                                  help="HuggingFace token for gated models (default: $HF_TOKEN)")
    transpile_parser.add_argument("--trust-remote-code", action="store_true",
                                  help="Pass trust_remote_code=True to HF loaders")
    transpile_parser.add_argument("--local-files-only", action="store_true",
                                  help="Require model/processor to already be local")
    transpile_parser.add_argument("--allow-unconverted-weights", action="store_true",
                                  help="Debug: transpile without CQ weights")
    transpile_parser.add_argument("--execute-after-transpile", action="store_true",
                                  help="Run a reference execution after lowering")
    transpile_parser.add_argument("--artifact-dir",
                                  help="Output directory (default: weights/<model>)")
    transpile_parser.add_argument("--graph-filename", default=None,
                                  help="Saved graph filename (default: graph.cactus)")
    transpile_parser.add_argument("--skip-reference-compare", action="store_true",
                                  help="Skip PyTorch comparison (requires --execute-after-transpile)")
    transpile_parser.add_argument("--no-fuse-rms-norm", action="store_true",
                                  help="Disable RMSNorm fusion")
    transpile_parser.add_argument("--no-fuse-rope", action="store_true",
                                  help="Disable RoPE fusion")
    transpile_parser.add_argument("--no-fuse-attention", action="store_true",
                                  help="Disable attention fusion")
    transpile_parser.add_argument("--no-fuse-attention-block", action="store_true",
                                  help="Disable attention-block fusion")
    transpile_parser.add_argument("--no-fuse-add-clipped", action="store_true",
                                  help="Disable add-clipped fusion")
    transpile_parser.add_argument("--no-fuse-gated-deltanet", action="store_true",
                                  help="Disable gated DeltaNet fusion")
    transpile_parser.add_argument("--npu", action="store_true",
                                  help="Also emit CoreML .mlpackage(s) for Apple NPU encoders")
    transpile_parser.add_argument("--npu-quantize", type=int, choices=[0, 4, 8], default=None,
                                  help="Legacy: force both NPU encoders to same quant (0=fp16, 4=int4, 8=int8)")
    transpile_parser.add_argument("--npu-audio-quantize", type=int, choices=[0, 4, 8], default=None,
                                  help="NPU audio encoder quant: 0=fp16, 4=int4, 8=int8 (default: 8)")
    transpile_parser.add_argument("--npu-vision-quantize", type=int, choices=[0, 4, 8], default=None,
                                  help="NPU vision encoder quant: 0=fp16, 4=int4, 8=int8 (default: 0; int4 degrades Gemma4 vision)")
    transpile_parser.add_argument("--cache-context-length", default=None,
                                  help="KV cache context length for cached decode graphs (default: model config)")

    return parser



_COMMANDS = {
    "download":   cmd_download,
    "build":      cmd_build,
    "run":        cmd_run,
    "serve":      cmd_serve,
    "transcribe": cmd_transcribe,
    "test":       cmd_test,
    "list":       cmd_list,

    "auth":           cmd_auth,
    "clean":          cmd_clean,
    "convert":        cmd_convert,
    "transpile":      cmd_transpile,
    "upload":         cmd_upload,
}


_REPO_ONLY = {"build", "test", "clean"}


def main():
    from .common import is_repo_checkout

    parser = create_parser()
    args = parser.parse_args()

    if args.command in _REPO_ONLY and not is_repo_checkout():
        print(f"Error: `cactus {args.command}` requires a git clone of the cactus repository.")
        print("See: https://github.com/cactus-compute/cactus")
        sys.exit(1)

    handler = _COMMANDS.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
