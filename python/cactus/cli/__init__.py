import sys
import argparse

from .common import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEST_MODEL_ID,
)
from .download import cmd_download
from .compile import cmd_build
from .run import cmd_run
from .transcribe import cmd_transcribe, DEFAULT_ASR_MODEL_ID
from .test import cmd_test
from .convert import cmd_convert
from .eval import cmd_eval
from .misc import cmd_auth, cmd_clean, cmd_list


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

  cactus download <model>              downloads CQ model files
                                       auto-resolves Cactus-Compute CQ repo

    Optional flags:
    --language-bits 1|2|3|4            language quantization bits (default: 4)
    --vision-bits 1|2|3|4             vision quantization bits (default: 4)
    --audio-bits 1|2|3|4              audio quantization bits (default: 4)
    --token <token>                    HuggingFace API token
    --reconvert                        force FP16 conversion from source

  -----------------------------------------------------------------

  cactus convert <model> [output_dir]  converts HuggingFace model to CQ format
                                       and transpiles the graph into the same folder

    Optional flags:
    --bits 1|2|3|4                     CQ quantization bits (default: 4)
    --token <token>                    HuggingFace API token

  -----------------------------------------------------------------

  cactus run <bundle_dir>              runs a transpiled Cactus bundle

  -----------------------------------------------------------------

  cactus build                         builds cactus for ARM chips
                                       output: build/libcactus.a

    Optional flags:
    --apple                            build for Apple (iOS/macOS)
    --android                          build for Android
    --flutter                          build for Flutter (all platforms)
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
    --no-rebuild                       skip building library and tests
    --llm                              run only LLM tests
    --vlm                              run only VLM tests
    --stt                              run only speech-to-text tests
    --embed                            run only embedding tests
    --rag                              run only RAG tests
    --graph                            run only graph tests
    --index                            run only index tests
    --kernel                           run only kernel tests
    --kv_cache                         run only KV cache tests
    --performance                      run only performance benchmarks
    --ios                              run on connected iPhone
    --android                          run on connected Android

  -----------------------------------------------------------------

  cactus list                          list all supported models
                                       shows download status

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

    download_parser = subparsers.add_parser('download', help='Download CQ model files')
    download_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                                 help=f'HuggingFace model ID or Cactus-Compute CQ repo (default: {DEFAULT_MODEL_ID})')
    download_parser.add_argument('--language-bits', type=int, choices=[1, 2, 3, 4], default=4,
                                 help='Language weight quantization bits (default: 4)')
    download_parser.add_argument('--vision-bits', type=int, choices=[1, 2, 3, 4], default=4,
                                 help='Vision weight quantization bits (default: 4)')
    download_parser.add_argument('--audio-bits', type=int, choices=[1, 2, 3, 4], default=4,
                                 help='Audio weight quantization bits (default: 4)')
    download_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    download_parser.add_argument('--token', help='HuggingFace API token')
    download_parser.add_argument('--reconvert', action='store_true',
                                 help='Force FP16 conversion from source (skip CQ download)')

    build_parser = subparsers.add_parser('build', help='Build the chat application')
    build_parser.add_argument('--apple', action='store_true',
                              help='Build for Apple platforms (iOS/macOS)')
    build_parser.add_argument('--android', action='store_true',
                              help='Build for Android')
    build_parser.add_argument('--flutter', action='store_true',
                              help='Build for Flutter (iOS, macOS, Android)')
    build_parser.add_argument('--python', action='store_true',
                              help='Build shared library for Python FFI')

    run_parser = subparsers.add_parser('run', help='Build, download (if needed), and run chat')
    run_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                            help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    run_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    run_parser.add_argument('--token', help='HuggingFace API token')
    run_parser.add_argument('--no-cloud-tele', action='store_true',
                            help='Disable cloud telemetry (write to cache only)')
    run_parser.add_argument('--reconvert', action='store_true',
                            help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')
    run_parser.add_argument('--image',
                            help='Path to image file for VLM inference (attached to first message)')
    run_parser.add_argument('--audio',
                            help='Path to audio file (WAV) for audio chat (attached to first message)')
    run_parser.add_argument('--file', dest='audio_file', default=None,
                            help='Audio file for transpiled bundles when model_id points to a transpiled folder')
    run_parser.add_argument('--image-file', action='append', default=[],
                            help='Repeatable image input for transpiled bundles')
    run_parser.add_argument('--weights-dir', default=None,
                            help='Converted weights directory for transpiled bundles with bound weights')
    run_parser.add_argument('--system',
                            help='System prompt to prepend to all messages')
    run_parser.add_argument('--prompt',
                            help='Initial prompt to send immediately')
    run_parser.add_argument('--input-ids', default=None,
                            help='Comma-separated token ids for transpiled causal-LM bundles')
    run_parser.add_argument('--max-new-tokens', type=int, default=None,
                            help='Maximum tokens to generate for transpiled causal-LM bundles')
    run_parser.add_argument('--stop-sequence', action='append', default=[],
                            help='Optional stop sequence for transpiled causal-LM bundles. Repeatable.')
    run_parser.add_argument('--result-json', default=None,
                            help='Optional path to save transpiled bundle results as JSON')
    run_parser.add_argument('--thinking', action='store_true',
                            help='Enable thinking/reasoning for models that support it')

    transcribe_parser = subparsers.add_parser('transcribe', help='Download ASR model and run transcription')
    transcribe_parser.add_argument('model_id', nargs='?', default=DEFAULT_ASR_MODEL_ID,
                                   help=f'HuggingFace model ID (default: {DEFAULT_ASR_MODEL_ID})')
    transcribe_parser.add_argument('--file', dest='audio_file', default=None,
                                   help='Audio file to transcribe (WAV format). Omit for live microphone.')
    transcribe_parser.add_argument('--language', default='en',
                                   help='Language code for transcription (default: en). Examples: es, fr, de, zh, ja')
    transcribe_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    transcribe_parser.add_argument('--token', help='HuggingFace API token')
    transcribe_parser.add_argument('--no-cloud-tele', action='store_true',
                                   help='Disable cloud telemetry (write to cache only)')
    transcribe_parser.add_argument('--force-handoff', action='store_true',
                                   help='Force cloud handoff by assuming low confidence')
    transcribe_parser.add_argument('--reconvert', action='store_true',
                                   help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')
    transcribe_parser.add_argument('--android', action='store_true',
                                   help='Run transcription on a connected Android device (requires --file)')
    transcribe_parser.add_argument('--ios', action='store_true',
                                   help='Run transcription on a connected iOS device (requires --file)')
    transcribe_parser.add_argument('--device', default=None,
                                   help='ADB device ID to use with --android')

    eval_parser = subparsers.add_parser('eval', help='Run evaluation scripts outside the cactus submodule')
    eval_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                             help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    eval_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    eval_parser.add_argument('--token', help='HuggingFace API token')
    eval_parser.add_argument('--tools', action='store_true', help='Run tools evals (default)')
    eval_parser.add_argument('--vlm', action='store_true', help='Run VLM-specific evals')
    eval_parser.add_argument('--stt', action='store_true', help='Run speech-to-text evals')
    eval_parser.add_argument('--llm', action='store_true', help='Run LLM evals')
    eval_parser.add_argument('--embed', action='store_true', help='Run embedding evals')
    eval_parser.add_argument('--no-cloud-tele', action='store_true',
                             help='Disable cloud telemetry (write to cache only)')
    eval_parser.add_argument('--reconvert', action='store_true',
                             help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')

    test_parser = subparsers.add_parser('test', help='Run the test suite')
    test_parser.add_argument('--model', default=DEFAULT_TEST_MODEL_ID,
                             help='Model to use for tests (default: Gemma4)')
    test_parser.add_argument('--no-rebuild', action='store_true',
                             help='Skip building cactus library and tests')
    test_parser.add_argument('--token', help='HuggingFace API token')
    test_parser.add_argument('--android', action='store_true',
                             help='Run tests on Android')
    test_parser.add_argument('--ios', action='store_true',
                             help='Run tests on iOS')
    test_parser.add_argument('--only', help='(deprecated, use --<test_name> instead) Only run the specified test')
    for _test_name in ['llm', 'vlm', 'stt', 'embed', 'rag', 'graph', 'index', 'kernel', 'kv_cache', 'performance']:
        test_parser.add_argument(f'--{_test_name}', action='store_true',
                                 help=f'Only run the {_test_name} tests')
    test_parser.add_argument('--enable-telemetry', action='store_true',
                             help='Enable cloud telemetry (disabled by default in tests)')
    test_parser.add_argument('--reconvert', action='store_true',
                             help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')

    auth_parser = subparsers.add_parser('auth', help='Manage Cactus Cloud API key')
    auth_parser.add_argument('--clear', action='store_true',
                             help='Remove the saved API key')
    auth_parser.add_argument('--status', action='store_true',
                             help='Show current key status without prompting')

    clean_parser = subparsers.add_parser('clean', help='Remove all build artifacts')

    list_parser = subparsers.add_parser('list', help='List supported models')
    list_parser.add_argument('--downloaded', action='store_true',
                             help='Only show downloaded models')

    convert_parser = subparsers.add_parser('convert', help='Convert HuggingFace model to CQ format')
    convert_parser.add_argument('model_name', help='HuggingFace model name')
    convert_parser.add_argument('output_dir', nargs='?', default=None,
                                help='Output directory (default: weights/<model_name>)')
    convert_parser.add_argument('--bits', type=int, choices=[1, 2, 3, 4], default=4,
                                help='CQ quantization bits (default: 4)')
    convert_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    convert_parser.add_argument('--token', help='HuggingFace API token')
    convert_parser.add_argument('--task', default='auto',
                                choices=['auto', 'causal_lm_logits', 'multimodal_causal_lm_logits',
                                         'ctc_logits', 'encoder_hidden_states',
                                         'seq2seq_transcription', 'tdt_transcription'],
                                help='Transpile task after conversion (default: auto)')
    convert_parser.add_argument('--prompt',
                                help='Prompt used for causal or multimodal graph shape capture')
    convert_parser.add_argument('--system-prompt', default='',
                                help='Optional system prompt for multimodal prompt construction')
    convert_parser.add_argument('--image-file', action='append', default=[],
                                help='Representative image file for multimodal transpile')
    convert_parser.add_argument('--audio-file',
                                help='Representative audio file for audio/multimodal transpile')
    convert_parser.add_argument('--max-new-tokens', type=int, default=None,
                                help='Generation room to preallocate for causal decode graphs')
    convert_parser.add_argument('--component-pipeline', default='auto', choices=['auto', 'on', 'off'],
                                help='Use split component graph transpilation when supported')
    convert_parser.add_argument('--components',
                                help='Comma-separated component subset for component-pipeline models')
    convert_parser.add_argument('--trust-remote-code', action='store_true',
                                help='Allow HF remote code during the transpile phase')
    convert_parser.add_argument('--local-files-only', action='store_true',
                                help='Require HF model/processor files to already be local during transpile')

    return parser


def preprocess_eval_args(parser, argv):
    args, unknown = parser.parse_known_args(argv)

    if getattr(args, 'command', None) == 'eval':
        setattr(args, 'extra_args', unknown)
        return args

    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args


def main():
    """Main entry point for the Cactus CLI."""
    parser = create_parser()

    argv = sys.argv[1:]
    args = preprocess_eval_args(parser, argv)

    if args.command == 'download':
        sys.exit(cmd_download(args))
    elif args.command == 'build':
        sys.exit(cmd_build(args))
    elif args.command == 'run':
        sys.exit(cmd_run(args))
    elif args.command == 'transcribe':
        sys.exit(cmd_transcribe(args))
    elif args.command == 'test':
        sys.exit(cmd_test(args))
    elif args.command == 'eval':
        sys.exit(cmd_eval(args))
    elif args.command == 'auth':
        sys.exit(cmd_auth(args))
    elif args.command == 'clean':
        sys.exit(cmd_clean(args))
    elif args.command == 'list':
        sys.exit(cmd_list(args))
    elif args.command == 'convert':
        sys.exit(cmd_convert(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
