import sys
import argparse

from .. import __version__
from .common import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEST_MODEL_ID,
    DEFAULT_ASR_MODEL_ID,
)
from .download import cmd_download
from .compile import cmd_build
from .run import cmd_run
from .serve import cmd_serve
from .transcribe import cmd_transcribe
from .test import cmd_test
from .convert import cmd_convert

from .auth import cmd_auth
from .clean import cmd_clean


def _telemetry_parent():
    """Args shared by commands that support telemetry toggle."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-cloud-tele", action="store_true",
                   help="Disable cloud telemetry (write to cache only)")
    return p


# ── Parser setup ──────────────────────────────────────────────────────


def create_parser():
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=argparse.SUPPRESS,
        description="""

  -----------------------------------------------------------------

  Cactus CLI:

  -----------------------------------------------------------------

  cactus run <model>                   chat playground for any model
                                       auto-converts and transpiles if needed
                                       default model: LiquidAI/LFM2-VL-450M

    Optional flags:
    --image <path>                     image file for VLM inference
    --audio <path>                     audio file (WAV) for audio chat
    --system <prompt>                  system prompt
    --prompt <text>                    send prompt immediately
    --thinking                         enable reasoning
    --token <token>                    HF token (for gated models)
    --reconvert                        force reconversion from source

  -----------------------------------------------------------------

  cactus transcribe [model]            speech-to-text transcription
                                       default model: openai/whisper-base

    Optional flags:
    --file <audio.wav>                 audio file to transcribe
    --language <code>                  language code (default: en)
    --token <token>                    HF token (for gated models)
    --reconvert                        force reconversion from source

  -----------------------------------------------------------------

  cactus serve [model]                 OpenAI-compatible local HTTP server
                                       serves prepared v2 bundles only

    Optional flags:
    --host <addr>                      bind address (default: 127.0.0.1)
    --port <port>                      port (default: 8080)

  -----------------------------------------------------------------

  cactus download <model>              fetch pre-converted CQ from Cactus-Compute

    Optional flags:
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --token <token>                    HuggingFace API token

  -----------------------------------------------------------------

  cactus convert <model> [dir]         convert model to CQ format
                                       (pre-converted if available, else
                                       built from source)

    Optional flags:
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --token <token>                    HuggingFace API token
    --reconvert                        force build from source

  -----------------------------------------------------------------

  cactus build                         builds cactus for ARM chips
                                       output: build/libcactus.a

    Optional flags:
    --apple                            build for Apple (iOS/macOS)
    --android                          build for Android
    --python                           build shared lib for Python FFI

  -----------------------------------------------------------------

  cactus test                          runs unit tests and benchmarks
                                       all must pass for contributions

    Optional flags:
    --model <model>                    default: LiquidAI/LFM2-VL-450M
    --token <token>                    HuggingFace API token
    --suite <name>                     run a specific test suite
    --reconvert                        force reconversion from source
    --ios                              run on connected iPhone
    --android                          run on connected Android

  -----------------------------------------------------------------

  cactus auth                          manage Cactus Cloud API key
                                       shows status and prompts to set key

    Optional flags:
    --status                           show key status without prompting
    --clear                            remove the saved API key

  -----------------------------------------------------------------

  cactus clean                         removes all build artifacts

  -----------------------------------------------------------------

  cactus --help                        shows these instructions

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

    # ── download ──────────────────────────────────────────────────────
    download_parser = subparsers.add_parser("download",
                                            help="Fetch pre-converted CQ weights from huggingface.co/Cactus-Compute")
    download_parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                                 help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    download_parser.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                                 help="CQ quantization bits (default: 4)")
    download_parser.add_argument("--token", help="HuggingFace API token")

    # ── build ─────────────────────────────────────────────────────────
    build_parser = subparsers.add_parser("build", help="Build the chat application")
    build_group = build_parser.add_mutually_exclusive_group()
    build_group.add_argument("--apple", action="store_true",
                             help="Build for Apple platforms (iOS/macOS)")
    build_group.add_argument("--android", action="store_true",
                             help="Build for Android")
    build_group.add_argument("--python", action="store_true",
                             help="Build shared library for Python FFI")

    # ── run ───────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Build, download (if needed), and run chat",
                                       parents=[_telemetry_parent()])
    run_parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                            help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    run_parser.add_argument("--token", help="HuggingFace API token")
    run_parser.add_argument("--reconvert", action="store_true",
                            help="Force conversion from source")
    run_parser.add_argument("--image",
                            help="Path to image file for VLM inference (attached to first message)")
    run_parser.add_argument("--audio",
                            help="Path to audio file (WAV) for audio chat (attached to first message)")
    run_parser.add_argument("--system",
                            help="System prompt to prepend to all messages")
    run_parser.add_argument("--prompt",
                            help="Initial prompt to send immediately")
    run_parser.add_argument("--input-ids", default=None,
                            help="Comma-separated token ids for transpiled causal-LM bundles")
    run_parser.add_argument("--max-new-tokens", type=int, default=None,
                            help="Maximum tokens to generate for transpiled causal-LM bundles")
    run_parser.add_argument("--result-json", default=None,
                            help="Optional path to save transpiled bundle results as JSON")
    run_parser.add_argument("--thinking", action="store_true",
                            help="Enable thinking/reasoning for models that support it")
    run_parser.add_argument("--no-cloud-handoff", action="store_true",
                            help="Disable automatic cloud handoff for this run")
    run_parser.add_argument("--confidence-threshold", type=float, default=None,
                            help="Confidence threshold below which local completions may hand off to cloud")
    run_parser.add_argument("--cloud-timeout-ms", type=int, default=None,
                            help="Maximum time to wait for cloud handoff before falling back locally")

    # ── transcribe ────────────────────────────────────────────────────
    transcribe_parser = subparsers.add_parser("transcribe", help="Download ASR model and run transcription",
                                              parents=[_telemetry_parent()])
    transcribe_parser.add_argument("model_id", nargs="?", default=DEFAULT_ASR_MODEL_ID,
                                   help=f"HuggingFace model ID (default: {DEFAULT_ASR_MODEL_ID})")
    transcribe_parser.add_argument("--file", dest="audio_file", required=True,
                                   help="Audio file to transcribe (WAV format)")
    transcribe_parser.add_argument("--language", default="en",
                                   help="Language code for transcription (default: en). Examples: es, fr, de, zh, ja")
    transcribe_parser.add_argument("--token", help="HuggingFace API token")
    transcribe_parser.add_argument("--force-handoff", action="store_true",
                                   help=argparse.SUPPRESS)
    transcribe_parser.add_argument("--reconvert", action="store_true",
                                   help="Download original model and convert (instead of using pre-converted from Cactus-Compute)")

    # ── serve ─────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    serve_parser.add_argument("model", nargs="?", default=None,
                              help="Prepared v2 bundle path, local model dir name, or HF model ID")
    serve_parser.add_argument("--host", default="127.0.0.1",
                              help="Bind address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080,
                              help="Port (default: 8080)")

    # ── test ──────────────────────────────────────────────────────────
    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.add_argument("--model", dest="model_id", default=DEFAULT_TEST_MODEL_ID,
                             help="Model to use for tests (default: Gemma4)")
    test_parser.add_argument("--token", help="HuggingFace API token")
    test_parser.add_argument("--android", action="store_true",
                             help="Run tests on Android")
    test_parser.add_argument("--ios", action="store_true",
                             help="Run tests on iOS")
    from .test import discover_suites
    suites = discover_suites()
    test_parser.add_argument("--suite", choices=suites or None,
                             help="Run a specific test suite")
    test_parser.add_argument("--enable-telemetry", action="store_true",
                             help="Enable cloud telemetry (disabled by default in tests)")
    test_parser.add_argument("--reconvert", action="store_true",
                             help="Download original model and convert (instead of using pre-converted from Cactus-Compute)")

    # ── auth ──────────────────────────────────────────────────────────
    auth_parser = subparsers.add_parser("auth", help="Manage Cactus Cloud API key")
    auth_parser.add_argument("--clear", action="store_true",
                             help="Remove the saved API key")
    auth_parser.add_argument("--status", action="store_true",
                             help="Show current key status without prompting")

    # ── clean ─────────────────────────────────────────────────────────
    subparsers.add_parser("clean", help="Remove all build artifacts")

    # ── convert ───────────────────────────────────────────────────────
    convert_parser = subparsers.add_parser("convert", help="Convert HuggingFace model to CQ format")
    convert_parser.add_argument("model_id", help="HuggingFace model name")
    convert_parser.add_argument("output_dir", nargs="?", default=None,
                                help="Output directory (default: weights/<model_name>)")
    convert_parser.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                                help="CQ quantization bits (default: 4)")
    convert_parser.add_argument("--token", help="HuggingFace API token")
    convert_parser.add_argument("--task", default="auto",
                                choices=["auto", "causal_lm_logits", "multimodal_causal_lm_logits",
                                         "ctc_logits", "encoder_hidden_states",
                                         "seq2seq_transcription", "tdt_transcription"],
                                help="Transpile task after conversion (default: auto)")
    convert_parser.add_argument("--prompt",
                                help="Prompt used for causal or multimodal graph shape capture")
    convert_parser.add_argument("--system-prompt", default="",
                                help="Optional system prompt for multimodal prompt construction")
    convert_parser.add_argument("--image-file", action="append", default=[],
                                help="Representative image file for multimodal transpile")
    convert_parser.add_argument("--audio-file",
                                help="Representative audio file for audio/multimodal transpile")
    convert_parser.add_argument("--max-new-tokens", type=int, default=None,
                                help="Generation room to preallocate for causal decode graphs")
    convert_parser.add_argument("--component-pipeline", default="auto", choices=["auto", "on", "off"],
                                help="Use split component graph transpilation when supported")
    convert_parser.add_argument("--components",
                                help="Comma-separated component subset for component-pipeline models")
    convert_parser.add_argument("--trust-remote-code", action="store_true",
                                help="Allow HF remote code during the transpile phase")
    convert_parser.add_argument("--local-files-only", action="store_true",
                                help="Require HF model/processor files to already be local during transpile")
    convert_parser.add_argument("--reconvert", action="store_true",
                                help="Force conversion from source")
    convert_parser.add_argument("--npu", action="store_true",
                                help="Also emit CoreML .mlpackage(s) for NPU (Apple Neural Engine) audio + vision encoders")
    convert_parser.add_argument("--npu-quantize", type=int, choices=[0, 4, 8], default=None,
                                help="Legacy override that forces BOTH audio and vision .mlpackages to the same quant (0=fp16, 4=int4, 8=int8). When unset, per-component defaults apply: audio=int8, vision=fp16.")
    convert_parser.add_argument("--npu-audio-quantize", type=int, choices=[0, 4, 8], default=None,
                                help="Audio encoder weight quant (0=fp16, 4=int4, 8=int8). Default int8.")
    convert_parser.add_argument("--npu-vision-quantize", type=int, choices=[0, 4, 8], default=None,
                                help="Vision encoder weight quant (0=fp16, 4=int4, 8=int8). Default fp16 — int4 is known to degrade Gemma 4 vision output.")

    return parser


# ── Command dispatch ──────────────────────────────────────────────────

_COMMANDS = {
    "download":   cmd_download,
    "build":      cmd_build,
    "run":        cmd_run,
    "serve":      cmd_serve,
    "transcribe": cmd_transcribe,
    "test":       cmd_test,

    "auth":       cmd_auth,
    "clean":      cmd_clean,
    "convert":    cmd_convert,
}


_REPO_ONLY = {"build", "test", "clean"}


def main():
    """Main entry point for the Cactus CLI."""
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
