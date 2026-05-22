import sys
import argparse

from .common import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEST_MODEL_ID,
    DEFAULT_ASR_MODEL_ID,
)
from .download import cmd_download
from .compile import cmd_build
from .run import cmd_run
from .transcribe import cmd_transcribe
from .test import cmd_test
from .convert import cmd_convert
from .eval import cmd_eval
from .auth import cmd_auth
from .clean import cmd_clean


# ── Shared parent parsers ─────────────────────────────────────────────


def _model_parent():
    """Args shared by commands that take a model ID."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                   help="CQ quantization bits (default: 4)")
    p.add_argument("--cache-dir", help="Cache directory for HuggingFace models")
    p.add_argument("--token", help="HuggingFace API token")
    p.add_argument("--reconvert", action="store_true",
                   help="Force conversion from source")
    return p


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

  How to use the Cactus Repo/CLI:

  -----------------------------------------------------------------

  cactus auth                          manage Cactus Cloud API key
                                       shows status and prompts to set key

    Optional flags:
    --status                           show key status without prompting
    --clear                            remove the saved API key

  -----------------------------------------------------------------

  cactus run <model>                   opens playground for the model
                                       auto downloads and spins up

    Optional flags:
    --token <token>                    HF token (for gated models)
    --reconvert                        force model weights reconversion from source

  -----------------------------------------------------------------

  cactus transcribe [model]            live microphone transcription
                                       default model: parakeet-tdt-0.6b-v3

    Optional flags:
    --file <audio.wav>                 transcribe audio file instead of mic
    --token <token>                    HF token (for gated models)
    --reconvert                        force model weights reconversion from source

    Examples:
    cactus transcribe                  live microphone transcription
    cactus transcribe --file audio.wav transcribe single file
    cactus transcribe nvidia/parakeet-ctc-1.1b     use different model
    cactus transcribe nvidia/parakeet-tdt-0.6b-v3 --file audio.wav

   -----------------------------------------------------------------

  cactus download <model>              downloads model from HuggingFace

    Optional flags:
    --token <token>                    HuggingFace API token

  -----------------------------------------------------------------

  cactus convert <model> [output_dir]  converts HuggingFace model to CQ format
                                       downloads pre-converted from Cactus-Compute
                                       if available, otherwise converts locally

    Optional flags:
    --bits 1|2|3|4                     CQ quantization bits (default: 4)
    --token <token>                    HuggingFace API token
    --reconvert                        force local conversion

  -----------------------------------------------------------------

  cactus run <bundle_dir>              runs a transpiled Cactus bundle

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
    --model <model>                    default: LFM2-VL-450M
    --transcribe_model <model>         default: nvidia/parakeet-tdt-0.6b-v3
    --whisper_model <model>            default: openai/whisper-small (language detection)
    --benchmark                        use larger models (LFM2.5-VL-1.6B + nvidia/parakeet-ctc-1.1b)
    --reconvert                        force model weights reconversion from source
    --suite <name>                     run a specific test suite (auto-detected)
    --ios                              run on connected iPhone
    --android                          run on connected Android

  -----------------------------------------------------------------

  cactus clean                         removes all build artifacts

  -----------------------------------------------------------------

  cactus --help                        shows these instructions

  -----------------------------------------------------------------

  Python bindings:

  Cactus python package is auto installed for researchers and testing
  Please see python/example.py and run the following instructions.

  1. cactus build
  2. cactus download google/gemma-4-E2B-it
  3. python python/example.py

  Note: Use any supported model

  -----------------------------------------------------------------
"""
    )

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = False

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action.help = argparse.SUPPRESS

    parser._action_groups = []

    # ── download ──────────────────────────────────────────────────────
    download_parser = subparsers.add_parser("download", help="Download model from HuggingFace")
    download_parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                                 help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    download_parser.add_argument("--cache-dir", help="Cache directory for HuggingFace models")
    download_parser.add_argument("--token", help="HuggingFace API token")

    # ── build ─────────────────────────────────────────────────────────
    build_parser = subparsers.add_parser("build", help="Build the chat application")
    build_parser.add_argument("--apple", action="store_true",
                              help="Build for Apple platforms (iOS/macOS)")
    build_parser.add_argument("--android", action="store_true",
                              help="Build for Android")
    build_parser.add_argument("--python", action="store_true",
                              help="Build shared library for Python FFI")

    # ── run ───────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Build, download (if needed), and run chat",
                                       parents=[_model_parent(), _telemetry_parent()])
    run_parser.add_argument("--image",
                            help="Path to image file for VLM inference (attached to first message)")
    run_parser.add_argument("--audio",
                            help="Path to audio file (WAV) for audio chat (attached to first message)")
    run_parser.add_argument("--file", dest="audio_file", default=None,
                            help="Audio file for transpiled bundles when model_id points to a transpiled folder")
    run_parser.add_argument("--image-file", action="append", default=[],
                            help="Repeatable image input for transpiled bundles")
    run_parser.add_argument("--weights-dir", default=None,
                            help="Converted weights directory for transpiled bundles with bound weights")
    run_parser.add_argument("--system",
                            help="System prompt to prepend to all messages")
    run_parser.add_argument("--prompt",
                            help="Initial prompt to send immediately")
    run_parser.add_argument("--input-ids", default=None,
                            help="Comma-separated token ids for transpiled causal-LM bundles")
    run_parser.add_argument("--max-new-tokens", type=int, default=None,
                            help="Maximum tokens to generate for transpiled causal-LM bundles")
    run_parser.add_argument("--stop-sequence", action="append", default=[],
                            help="Optional stop sequence for transpiled causal-LM bundles. Repeatable.")
    run_parser.add_argument("--result-json", default=None,
                            help="Optional path to save transpiled bundle results as JSON")
    run_parser.add_argument("--thinking", action="store_true",
                            help="Enable thinking/reasoning for models that support it")

    # ── transcribe ────────────────────────────────────────────────────
    transcribe_parser = subparsers.add_parser("transcribe", help="Download ASR model and run transcription",
                                              parents=[_telemetry_parent()])
    transcribe_parser.add_argument("model_id", nargs="?", default=DEFAULT_ASR_MODEL_ID,
                                   help=f"HuggingFace model ID (default: {DEFAULT_ASR_MODEL_ID})")
    transcribe_parser.add_argument("--file", dest="audio_file", default=None,
                                   help="Audio file to transcribe (WAV format). Omit for live microphone.")
    transcribe_parser.add_argument("--language", default="en",
                                   help="Language code for transcription (default: en). Examples: es, fr, de, zh, ja")
    transcribe_parser.add_argument("--cache-dir", help="Cache directory for HuggingFace models")
    transcribe_parser.add_argument("--token", help="HuggingFace API token")
    transcribe_parser.add_argument("--force-handoff", action="store_true",
                                   help="Force cloud handoff by assuming low confidence")
    transcribe_parser.add_argument("--reconvert", action="store_true",
                                   help="Download original model and convert (instead of using pre-converted from Cactus-Compute)")
    transcribe_parser.add_argument("--android", action="store_true",
                                   help="Run transcription on a connected Android device (requires --file)")
    transcribe_parser.add_argument("--ios", action="store_true",
                                   help="Run transcription on a connected iOS device (requires --file)")
    transcribe_parser.add_argument("--device", default=None,
                                   help="ADB device ID to use with --android")

    # ── eval ──────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("eval", help="Run evaluation scripts outside the cactus submodule",
                                        parents=[_model_parent(), _telemetry_parent()])
    eval_parser.add_argument("--suite", choices=["tools", "llm", "vlm", "stt"],
                             default="tools", help="Eval suite to run (default: tools)")

    # ── test ──────────────────────────────────────────────────────────
    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.add_argument("--model", default=DEFAULT_TEST_MODEL_ID,
                             help="Model to use for tests (default: Gemma4)")
    test_parser.add_argument("--token", help="HuggingFace API token")
    test_parser.add_argument("--android", action="store_true",
                             help="Run tests on Android")
    test_parser.add_argument("--ios", action="store_true",
                             help="Run tests on iOS")
    from .test import discover_suites
    test_parser.add_argument("--suite", choices=discover_suites(),
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
    convert_parser.add_argument("model_name", help="HuggingFace model name")
    convert_parser.add_argument("output_dir", nargs="?", default=None,
                                help="Output directory (default: weights/<model_name>)")
    convert_parser.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                                help="CQ quantization bits (default: 4)")
    convert_parser.add_argument("--cache-dir", help="Cache directory for HuggingFace models")
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

    return parser


def preprocess_eval_args(parser, argv):
    args, unknown = parser.parse_known_args(argv)

    if getattr(args, 'command', None) == 'eval':
        setattr(args, 'extra_args', unknown)
        return args

    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args


# ── Command dispatch ──────────────────────────────────────────────────

_COMMANDS = {
    "download":   cmd_download,
    "build":      cmd_build,
    "run":        cmd_run,
    "transcribe": cmd_transcribe,
    "test":       cmd_test,
    "eval":       cmd_eval,
    "auth":       cmd_auth,
    "clean":      cmd_clean,
    "convert":    cmd_convert,
}


def main():
    """Main entry point for the Cactus CLI."""
    parser = create_parser()

    argv = sys.argv[1:]
    args = preprocess_eval_args(parser, argv)

    handler = _COMMANDS.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
