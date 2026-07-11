import sys
import argparse

from .. import __version__
from .common import (
    DEFAULT_MODEL_ID,
    DEFAULT_TRANSCRIPTION_MODEL_ID,
    DEFAULT_TEST_MODEL_ID,
    DEFAULT_TEST_TRANSCRIPTION_MODEL_ID,
)
from .download import cmd_download
from .compile import cmd_build
from .serve import cmd_serve
from .transcribe import cmd_transcribe
from .test import cmd_test, cmd_benchmark, COMPONENTS
from .convert import cmd_convert
from .upload import cmd_upload
from .run import cmd_run
from .list import cmd_list

from .auth import cmd_auth
from .clean import cmd_clean
from .code import cmd_code
from .utils import bits_arg, ALLOWED_BITS

_BITS_METAVAR = "{" + ",".join(str(b) for b in ALLOWED_BITS) + "}"


def _telemetry_parent():
    """Args shared by commands that support telemetry toggle."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-cloud-tele", action="store_true",
                   help="Disable cloud telemetry (write to cache only)")
    return p


def _build_parent(mixed: bool = True):
    """Bundle-build flags shared by model-preparing commands
    (mixed=False limits --bits to uniform 1-4 for build-only commands)."""
    p = argparse.ArgumentParser(add_help=False)
    if mixed:
        p.add_argument("--bits", type=bits_arg, default=4, metavar=_BITS_METAVAR,
                       help="CQ quantization: uniform 1-4 or gemma-4 mixed 3.26/2.54 (default: 4)")
    else:
        p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                       help="CQ quantization (default: 4)")
    p.add_argument("--token", help="HuggingFace token")
    p.add_argument("--reconvert", action="store_true",
                   help="Force local rebuild from source")
    return p


def _engine_test_parent():
    """Args shared by `test` and `benchmark`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--backend", choices=["cpu", "metal"], default=None,
                   help="Inference backend (default: auto)")
    p.add_argument("--model", dest="model_id", default=None,
                   type=_hf_id_or_path,
                   help=f"HF model ID under test (default: {DEFAULT_TEST_MODEL_ID})")
    p.add_argument("--transcription-model", dest="transcription_model_id", default=None,
                   type=_hf_id_or_path,
                   help=f"HF transcription model ID under test (default: {DEFAULT_TEST_TRANSCRIPTION_MODEL_ID})")
    p.add_argument("--android", action="store_true", help="Run tests on Android")
    p.add_argument("--ios", action="store_true", help="Run tests on iOS")
    p.add_argument("--enable-telemetry", action="store_true",
                   help="Enable cloud telemetry (disabled by default in tests)")
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
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
    --image <path>                     image file for VLM inference
    --audio <path>                     audio file for audio chat
    --system <prompt>                  system prompt
    --prompt <text>                    send prompt immediately
    --thinking                         enable thinking/reasoning mode
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local convert fallback

  cactus transcribe [model]            live microphone transcription with a model
    --file <audio.wav>                 audio file to transcribe (WAV)
    --language <code>                  language code (default: en)
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source

  cactus download [model]              fetch a prebuilt bundle, else build locally (default: {DEFAULT_MODEL_ID})
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source

  cactus convert <model> [dir]         build a runnable bundle locally (skips prebuilt fetch)
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild from source
    --lora <path>                      merge a LoRA adapter before converting

  cactus serve [model]                 OpenAI-compatible local HTTP server
    --host <addr>                      bind address (default: 127.0.0.1)
    --port <port>                      port (default: 8080)
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
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
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
    --token <token>                    HuggingFace token (gated models)
    --reconvert                        force local rebuild of test models
    --suite <name>                     run a single test suite from any
                                       component (kernels, graph, or engine)
    --list                             list components and suites
    --ios                              run on connected iPhone
    --android                          run on connected Android
    --enable-telemetry                 send cloud telemetry (off by default)

  cactus benchmark                     run the engine benchmark suite
    --model <hf-id>                    default: {DEFAULT_TEST_MODEL_ID}
    --transcription-model <hf-id>      default: {DEFAULT_TEST_TRANSCRIPTION_MODEL_ID}
    --bits 1|2|3|4|2.54|3.26           CQ quantization (default: 4)
    --backend cpu|metal                inference backend (default: auto)
    --ios                              run on connected iPhone
    --android                          run on connected Android

  cactus clean                         delete build artifacts, weights, venv

  cactus --help                        show this help

  -----------------------------------------------------------------

  Advanced / build pipeline — convert runs automatically inside run,
  serve, transcribe and download; reach for it only to control the
  build (custom flags, LoRA, debugging):

  cactus convert <model> [dir]         HuggingFace -> runnable cactus bundle
                                       (quantizes weights to CQ, then builds
                                       the runtime graph)
    --bits 1|2|3|4                     CQ quantization (default: 4)
    --token <token>                    HuggingFace token (defaults to $HF_TOKEN)
    --reconvert                        force weight conversion from source
    --lora <path>                      merge a LoRA adapter before converting
    --weights-only                     stop after CQ weights (skip the graph)
    --weights-dir <path>              path to CQ weights (default: weights/<model>)
    --task <auto|...>                  task (default: auto, inferred from config)
    --artifact-dir <path>              write bundle here (default: weights/<model>)
    --prompt <text>                    representative prompt for shape capture
    --system-prompt <text>             system prompt for multimodal chat
    --enable-thinking                  enable thinking markers when supported
    --input-ids <a,b,...>              token ids for causal-LM shape capture
    --image-file <path>                representative image (repeatable)
    --audio-file <path>                representative audio file (WAV)
    --max-new-tokens <n>               preallocate decode context for causal LM
    --component-pipeline auto|on|off   force component-pipeline graph build
    --components <a,b,...>             subset of components to build
    --torch-dtype <dtype>              float16 | float32 | bfloat16
    --trust-remote-code                allow HF remote code during the build
    --local-files-only                 require model/processor to be local
    --low-memory-load                  graph capture with meta tensors
    --allow-unconverted-weights        debug-only: skip the CQ-weights check
    --execute-after-transpile          run a reference execution after building
    --graph-filename <name>            override saved graph filename
    --skip-reference-compare           skip PyTorch comparison (with --execute-…)
    --no-fuse-rms-norm                 disable RMSNorm fusion
    --no-fuse-rope                     disable RoPE fusion
    --no-fuse-attention                disable attention fusion
    --no-fuse-attention-block          disable attention-block fusion
    --no-fuse-add-clipped              disable add-clipped fusion
    --no-fuse-gated-deltanet           disable gated DeltaNet fusion

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
    run_parser.add_argument("--backend", choices=["cpu", "metal"], default=None,
                            help="Inference backend (default: auto)")

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
    serve_parser.add_argument("--no-access-log", action="store_true",
                              help="Disable per-request HTTP access logging")
    serve_parser.add_argument("--backend", choices=["cpu", "metal"], default=None,
                              help="Inference backend (default: auto)")

    code_parser = subparsers.add_parser("code", help="Run the Cactus coding agent (TUI / print mode)",
                                        parents=[_build_parent()])
    code_parser.add_argument("--serve-model", default=None,
                             help="If no server is running, start `cactus serve` with this model")
    code_parser.add_argument("--host", default="127.0.0.1",
                             help="Server bind address to use/start (default: 127.0.0.1)")
    code_parser.add_argument("--port", type=_port_int, default=8080,
                             help="Server port to use/start (default: 8080)")
    code_parser.add_argument("--no-serve", action="store_true",
                             help="Do not auto-start a server; require one to already be running")
    code_parser.add_argument("--no-cloud-handoff", action="store_true",
                             help="Disable automatic cloud handoff on the auto-started server")
    code_parser.add_argument("--confidence-threshold", type=_unit_float, default=None,
                             help="Confidence threshold below which completions hand off to cloud")
    code_parser.add_argument("--cloud-timeout-ms", type=_non_negative_int, default=None,
                             help="Maximum time to wait for cloud handoff before falling back locally")
    code_parser.add_argument("--backend", choices=["cpu", "metal"], default=None,
                             help="Inference backend (default: auto)")
    code_parser.add_argument("agent_args", nargs=argparse.REMAINDER,
                             help="Arguments passed through to the coding agent (prefix with -- )")

    test_parser = subparsers.add_parser("test", help="Run the test suite",
                                        parents=[_build_parent(), _engine_test_parent()])
    test_parser.add_argument("--component", choices=COMPONENTS, default="all",
                             help="Component to test (default: all)")
    test_parser.add_argument("--suite", default=None,
                             help="Run a single test suite by name; resolved across all components (e.g. llm → engine)")
    test_parser.add_argument("--list", action="store_true",
                             help="List available components and engine tests, then exit")

    subparsers.add_parser("benchmark", help="Run the engine benchmark suite",
                          parents=[_build_parent(), _engine_test_parent()])

    auth_parser = subparsers.add_parser("auth", help="Manage cloud API key")
    auth_parser.add_argument("--clear", action="store_true",
                             help="Remove the saved API key")
    auth_parser.add_argument("--status", action="store_true",
                             help="Show current key status")

    clean_parser = subparsers.add_parser("clean", help="Delete build artifacts, downloaded weights, and venv")
    clean_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    subparsers.add_parser("list", help="List downloaded models")

    convert_parser = subparsers.add_parser("convert",
                                           help="Convert a HuggingFace model into a runnable cactus bundle")
    convert_parser.add_argument("model_id", type=_hf_id_or_path,
                                help="HuggingFace model id (e.g. openai/whisper-base)")
    convert_parser.add_argument("output_dir", nargs="?", default=None,
                                help="Output directory (default: weights/<model>)")
    convert_parser.add_argument("--bits", type=int, choices=[1, 2, 3, 4], default=4,
                                help="CQ quantization bits (default: 4)")
    convert_parser.add_argument("--token", help="HuggingFace token")
    convert_parser.add_argument("--lora",
                                help="Path or HF id of a LoRA adapter to merge before converting (requires `peft`)")
    convert_parser.add_argument("--reconvert", action="store_true",
                                help="Force conversion from source")
    convert_parser.add_argument("--skip-model-load", action="store_true",
                                help="Convert directly from checkpoint tensors without loading the full HF model object")
    convert_parser.add_argument("--low-memory-load", action="store_true",
                                help="Avoid loading checkpoint tensors during graph capture")
    convert_parser.add_argument("--weights-only", action="store_true",
                                help="Only quantize weights; skip building the runtime graph bundle")
    convert_parser.add_argument("--weights-dir",
                                help="CQ weights directory (default: weights/<model>)")
    convert_parser.add_argument("--task", default="auto",
                                choices=["auto", "causal_lm_logits", "multimodal_causal_lm_logits",
                                         "ctc_logits", "encoder_hidden_states",
                                         "seq2seq_transcription", "tdt_transcription"],
                                help="Graph task (default: auto, from model config)")
    convert_parser.add_argument("--prompt",
                                help="Prompt for causal/multimodal shape capture")
    convert_parser.add_argument("--system-prompt", default=None,
                                help="System prompt for multimodal chat formats")
    convert_parser.add_argument("--enable-thinking", action="store_true",
                                help="Enable thinking markers when the prompt supports them")
    convert_parser.add_argument("--input-ids", default=None,
                                help="Comma-separated token ids for causal-LM shape capture")
    convert_parser.add_argument("--image-file", action="append", default=[],
                                help="Image for multimodal shape capture (repeatable)")
    convert_parser.add_argument("--audio-file",
                                help="Audio file (WAV) for audio/multimodal shape capture")
    convert_parser.add_argument("--max-new-tokens", type=_positive_int, default=None,
                                help="Decode context to preallocate for causal LM (default: 32)")
    convert_parser.add_argument("--component-pipeline", default="auto", choices=["auto", "on", "off"],
                                help="Split-component graph when supported (default: auto)")
    convert_parser.add_argument("--components",
                                help="Comma-separated component subset (e.g. vision_encoder,decoder)")
    convert_parser.add_argument("--torch-dtype", default=None,
                                choices=["float16", "float32", "bfloat16"],
                                help="Torch dtype for HF loading (default: float16)")
    convert_parser.add_argument("--trust-remote-code", action="store_true",
                                help="Pass trust_remote_code=True to HF loaders")
    convert_parser.add_argument("--local-files-only", action="store_true",
                                help="Require model/processor to already be local")
    convert_parser.add_argument("--allow-unconverted-weights", action="store_true",
                                help="Debug: build the graph without CQ weights")
    convert_parser.add_argument("--execute-after-transpile", action="store_true",
                                help="Run a reference execution after building the graph")
    convert_parser.add_argument("--artifact-dir",
                                help="Graph bundle output directory (default: weights/<model>)")
    convert_parser.add_argument("--graph-filename", default=None,
                                help="Saved graph filename (default: graph.cactus)")
    convert_parser.add_argument("--skip-reference-compare", action="store_true",
                                help="Skip PyTorch comparison (requires --execute-after-transpile)")
    convert_parser.add_argument("--no-fuse-rms-norm", action="store_true",
                                help="Disable RMSNorm fusion")
    convert_parser.add_argument("--no-fuse-rope", action="store_true",
                                help="Disable RoPE fusion")
    convert_parser.add_argument("--no-fuse-attention", action="store_true",
                                help="Disable attention fusion")
    convert_parser.add_argument("--no-fuse-attention-block", action="store_true",
                                help="Disable attention-block fusion")
    convert_parser.add_argument("--no-fuse-add-clipped", action="store_true",
                                help="Disable add-clipped fusion")
    convert_parser.add_argument("--no-fuse-gated-deltanet", action="store_true",
                                help="Disable gated DeltaNet fusion")
    convert_parser.add_argument("--cache-context-length", default=None,
                                help="KV cache context length for cached decode graphs (default: model config)")

    upload_parser = subparsers.add_parser("upload",
                                          parents=[_build_parent(mixed=False)],
                                          help="Build a runnable bundle locally and upload it to Cactus-Compute on HuggingFace")
    upload_parser.add_argument("model_id", type=_hf_id_or_path,
                               help="HuggingFace model id (e.g. openai/whisper-base)")

    return parser



_COMMANDS = {
    "download":   cmd_download,
    "build":      cmd_build,
    "run":        cmd_run,
    "serve":      cmd_serve,
    "transcribe": cmd_transcribe,
    "test":       cmd_test,
    "benchmark":  cmd_benchmark,
    "list":       cmd_list,

    "auth":           cmd_auth,
    "clean":          cmd_clean,
    "code":           cmd_code,
    "convert":        cmd_convert,
    "upload":         cmd_upload,
}


_REPO_ONLY = {"build", "test", "benchmark", "clean"}


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
