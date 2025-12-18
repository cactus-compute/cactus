#!/usr/bin/env python3
import sys
import os
import argparse
import re
import subprocess
import shutil
import platform
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_ID = "LiquidAI/LFM2-1.2B"

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def print_color(color, message):
    """Print a message with ANSI color codes."""
    print(f"{color}{message}{NC}")


def get_model_dir_name(model_id):
    """Convert HuggingFace model ID to local directory name."""
    model_name = model_id.split('/')[-1]
    model_name = model_name.lower()
    model_name = re.sub(r'-it$', '', model_name)
    return model_name


def get_weights_dir(model_id):
    """Get the weights directory path for a model."""
    model_dir = get_model_dir_name(model_id)
    return PROJECT_ROOT / "weights" / model_dir


def check_command(cmd):
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and optionally exit on failure."""
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def cmd_download(args):
    """Download and convert HuggingFace model weights to Cactus format."""
    model_id = args.model_id
    weights_dir = get_weights_dir(model_id)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return 0

    print()
    print_color(YELLOW, f"Model weights not found. Downloading {model_id}...")
    print("=" * 45)

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print_color(RED, "Error: Required Python packages not found.")
        print("Please run: ./setup")
        return 1

    from .converter_llm import convert_hf_model_weights
    from .converter_vlm import convert_processors
    from .tokenizer import convert_hf_tokenizer
    from .tensor_io import format_config_value
    from .config_utils import is_lfm2_vl, pick_torch_dtype, vision_weight_sanity_check

    weights_dir.mkdir(parents=True, exist_ok=True)

    precision = getattr(args, 'precision', 'INT8')
    cache_dir = getattr(args, 'cache_dir', None)
    token = getattr(args, 'token', None)

    print(f"Converting {model_id} to {precision}...")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoConfig

    try:
        from transformers import Lfm2VlForConditionalGeneration
    except ImportError:
        Lfm2VlForConditionalGeneration = None

    is_vlm = 'vl' in model_id.lower() or 'vlm' in model_id.lower()
    is_whisper = 'whisper' in model_id.lower()

    try:
        if is_vlm:
            missing_deps = []
            try:
                from PIL import Image
            except Exception:
                missing_deps.append('Pillow')
            try:
                import num2words
            except Exception:
                missing_deps.append('num2words')
            try:
                import torchvision
            except Exception:
                missing_deps.append('torchvision')

            if missing_deps:
                print_color(RED, f"Error: Missing packages for VLM: {', '.join(missing_deps)}")
                print(f"Install with: pip install {' '.join(missing_deps)}")
                return 1

            processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            dtype = pick_torch_dtype()

            if is_lfm2_vl(model_id, cfg) and Lfm2VlForConditionalGeneration is not None:
                model = Lfm2VlForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=dtype, token=token)
            else:
                model = AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=dtype, token=token)

            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

            if is_lfm2_vl(model_id, cfg) and not vision_weight_sanity_check(model):
                print_color(RED, "Vision embeddings look randomly initialized.")
                return 1

            try:
                convert_processors(processor, model_id, weights_dir, token=token)
            except Exception as e:
                print(f"  Warning: convert_processors failed: {e}")

        elif is_whisper:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            except ValueError:
                model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        config = convert_hf_model_weights(model, weights_dir, precision, args)

        model_name_l = model_id.lower()
        if 'extract' in model_name_l:
            config['model_variant'] = 'extract'
        elif 'vlm' in model_name_l:
            config['model_variant'] = 'vlm'
        elif 'rag' in model_name_l:
            config['model_variant'] = 'rag'
        else:
            config.setdefault('model_variant', 'default')

        if precision == 'INT8':
            config['precision'] = "FP16"
        else:
            config['precision'] = precision

        config_path = weights_dir / "config.txt"
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}={format_config_value(value)}\n")

        convert_hf_tokenizer(tokenizer, weights_dir, token=token)

        del model
        del tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_color(GREEN, f"Successfully downloaded and converted weights to {weights_dir}")
        return 0

    except Exception as e:
        print_color(RED, f"Error: {e}")
        return 1


def cmd_build(args):
    """Build the Cactus library and chat binary."""
    if getattr(args, 'apple', False):
        return cmd_build_apple(args)
    if getattr(args, 'android', False):
        return cmd_build_android(args)

    print_color(BLUE, "Building Cactus chat...")
    print("=" * 23)

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake")
        return 1

    cactus_dir = PROJECT_ROOT / "cactus"
    lib_path = cactus_dir / "build" / "libcactus.a"

    print_color(YELLOW, "Building Cactus library...")
    build_script = cactus_dir / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1
    result = run_command(str(build_script), cwd=cactus_dir, check=False)
    if result.returncode != 0:
        print_color(RED, "Failed to build cactus library")
        return 1

    tests_dir = PROJECT_ROOT / "tests"
    build_dir = tests_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling chat.cpp...")

    chat_cpp = tests_dir / "chat.cpp"
    if not chat_cpp.exists():
        print_color(RED, f"Error: chat.cpp not found at {chat_cpp}")
        return 1

    is_darwin = platform.system() == "Darwin"

    if is_darwin:
        compiler = "clang++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            f"-I{PROJECT_ROOT}",
            str(chat_cpp),
            str(lib_path),
            "-o", "chat",
            "-lcurl",
            "-framework", "Accelerate",
            "-framework", "CoreML",
            "-framework", "Foundation"
        ]
    else:
        compiler = "g++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            f"-I{PROJECT_ROOT}",
            str(chat_cpp),
            str(lib_path),
            "-o", "chat",
            "-lcurl",
            "-pthread"
        ]

    if not check_command(compiler):
        print_color(RED, f"Error: {compiler} is not installed")
        return 1

    result = subprocess.run(cmd, cwd=build_dir)
    if result.returncode != 0:
        print_color(RED, "Build failed")
        return 1

    print_color(GREEN, f"Build complete: {build_dir / 'chat'}")
    return 0


def cmd_build_apple(args):
    """Build Cactus for Apple platforms (iOS/macOS)."""
    print_color(BLUE, "Building Cactus for Apple platforms...")
    print("=" * 40)

    if platform.system() != "Darwin":
        print_color(RED, "Error: Apple builds require macOS")
        return 1

    build_script = PROJECT_ROOT / "apple" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "apple", check=False)
    if result.returncode != 0:
        print_color(RED, "Apple build failed")
        return 1

    print_color(GREEN, "Apple build complete!")
    return 0


def cmd_build_android(args):
    """Build Cactus for Android."""
    print_color(BLUE, "Building Cactus for Android...")
    print("=" * 32)

    build_script = PROJECT_ROOT / "android" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "android", check=False)
    if result.returncode != 0:
        print_color(RED, "Android build failed")
        return 1

    print_color(GREEN, "Android build complete!")
    return 0


def cmd_run(args):
    """Build, download model if needed, and start interactive chat."""
    model_id = args.model_id

    build_result = cmd_build(args)
    if build_result != 0:
        return build_result

    download_result = cmd_download(args)
    if download_result != 0:
        return download_result

    weights_dir = get_weights_dir(model_id)
    chat_binary = PROJECT_ROOT / "tests" / "build" / "chat"

    if not chat_binary.exists():
        print_color(RED, f"Error: Chat binary not found at {chat_binary}")
        return 1

    os.system('clear' if platform.system() != 'Windows' else 'cls')
    print_color(GREEN, f"Starting Cactus Chat with model: {model_id}")
    print()

    os.execv(str(chat_binary), [str(chat_binary), str(weights_dir)])


def cmd_test(args):
    """Run the Cactus test suite."""
    print_color(BLUE, "Running test suite...")
    print("=" * 20)

    test_script = PROJECT_ROOT / "tests" / "run.sh"

    if not test_script.exists():
        print_color(RED, f"Error: Test script not found at {test_script}")
        return 1

    cmd = [str(test_script)]

    if args.model:
        cmd.extend(["--model", args.model])
    if args.transcribe_model:
        cmd.extend(["--transcribe_model", args.transcribe_model])
    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT / "tests")
    return result.returncode


def cmd_clean(args):
    """Remove all build artifacts, caches, and downloaded weights."""
    print_color(BLUE, "Cleaning all build artifacts from Cactus project...")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    def remove_if_exists(path):
        if path.is_dir():
            print(f"Removing: {path}")
            shutil.rmtree(path)
        else:
            print(f"Not found: {path}")

    remove_if_exists(PROJECT_ROOT / "cactus" / "build")

    remove_if_exists(PROJECT_ROOT / "android" / "build")
    remove_if_exists(PROJECT_ROOT / "android" / "libs")
    remove_if_exists(PROJECT_ROOT / "android" / "arm64-v8a")

    remove_if_exists(PROJECT_ROOT / "apple" / "build")

    remove_if_exists(PROJECT_ROOT / "tests" / "build")

    remove_if_exists(PROJECT_ROOT / "venv")

    remove_if_exists(PROJECT_ROOT / "weights")

    print()
    print("Removing compiled libraries and frameworks...")

    so_count = 0
    for so_file in PROJECT_ROOT.rglob("*.so"):
        so_file.unlink()
        so_count += 1
    print(f"Removed {so_count} .so files" if so_count else "No .so files found")

    a_count = 0
    for a_file in PROJECT_ROOT.rglob("*.a"):
        a_file.unlink()
        a_count += 1
    print(f"Removed {a_count} .a files" if a_count else "No .a files found")

    xcf_count = 0
    for xcf_dir in PROJECT_ROOT.rglob("*.xcframework"):
        if xcf_dir.is_dir():
            shutil.rmtree(xcf_dir)
            xcf_count += 1
    print(f"Removed {xcf_count} .xcframework directories" if xcf_count else "No .xcframework directories found")

    pycache_count = 0
    for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            pycache_count += 1
    print(f"Removed {pycache_count} __pycache__ directories" if pycache_count else "No __pycache__ directories found")

    egg_count = 0
    for egg_dir in PROJECT_ROOT.rglob("*.egg-info"):
        if egg_dir.is_dir():
            shutil.rmtree(egg_dir)
            egg_count += 1
    print(f"Removed {egg_count} .egg-info directories" if egg_count else "No .egg-info directories found")

    print()
    print_color(GREEN, "Clean complete!")
    print("All build artifacts have been removed.")
    print()

    # Re-run setup automatically
    print_color(BLUE, "Re-running setup...")
    setup_script = PROJECT_ROOT / "setup"
    result = subprocess.run(
        ["bash", "-c", f"source {setup_script} && pip install -e {PROJECT_ROOT / 'tools'} --quiet"],
        cwd=PROJECT_ROOT
    )
    if result.returncode == 0:
        print_color(GREEN, "Setup complete!")
    else:
        print_color(YELLOW, "Setup had issues. Please run manually:")
        print("  source ./setup")
    return 0


def cmd_convert(args):
    """Convert a HuggingFace model to a custom output directory."""
    model_id = args.model_name
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = get_weights_dir(model_id)
    else:
        output_dir = Path(output_dir)

    class DownloadArgs:
        pass

    download_args = DownloadArgs()
    download_args.model_id = model_id
    download_args.precision = args.precision
    download_args.cache_dir = getattr(args, 'cache_dir', None)
    download_args.token = getattr(args, 'token', None)

    original_get_weights = get_weights_dir

    def custom_weights_dir(mid):
        return output_dir

    import src.cli as cli_module
    cli_module.get_weights_dir = custom_weights_dir

    try:
        return cmd_download(download_args)
    finally:
        cli_module.get_weights_dir = original_get_weights


def create_parser():
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=argparse.SUPPRESS,
        description="""
         
  -----------------------------------------------------------------

  How to use the Cactus Repo/CLI:

  -----------------------------------------------------------------

  cactus run <model>                   opens playground for the model
                                       auto downloads and spins up

    Optional flags:
    --precision INT8|FP16|FP32         default: INT8
    --token <token>                    HF token (for gated models)

   -----------------------------------------------------------------

  cactus download <model>              downloads model to ./weights
                                       see supported weights on ReadMe

    Optional flags:
    --precision INT8|FP16|FP32         quantization (default: INT8)
    --token <token>                    HuggingFace API token

  -----------------------------------------------------------------

  cactus build                         builds cactus for ARM chips
                                       output: build/libcactus.a

    Optional flags:
    --apple                            build for Apple (iOS/macOS)
    --android                          build for Android

  -----------------------------------------------------------------

  cactus test                          runs unit tests and benchmarks
                                       all must pass for contributions

    Optional flags:
    --model <model>                    default: LFM2-VL-450M
    --transcribe_model <model>         default: whisper-small
    --ios                              run on connected iPhone
    --android                          run on connected Android

  -----------------------------------------------------------------

  cactus clean                         removes all build artifacts

  -----------------------------------------------------------------

  cactus --help                        shows these instructions

  -----------------------------------------------------------------

  Python bindings:

  Cactus python package is auto installed for researchers and testing
  Please see tools/examples.py and run the following instructions.

  1. cactus build
  2. cactus download LiquidAI/LFM2-VL-450M
  3. python tools/example.py

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

    download_parser = subparsers.add_parser('download', help='Download and convert model weights')
    download_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                                 help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    download_parser.add_argument('--precision', choices=['INT8', 'FP16', 'FP32'], default='INT8',
                                 help='Quantization precision (default: INT8)')
    download_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    download_parser.add_argument('--token', help='HuggingFace API token')

    build_parser = subparsers.add_parser('build', help='Build the chat application')
    build_parser.add_argument('--apple', action='store_true',
                              help='Build for Apple platforms (iOS/macOS)')
    build_parser.add_argument('--android', action='store_true',
                              help='Build for Android')

    run_parser = subparsers.add_parser('run', help='Build, download (if needed), and run chat')
    run_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                            help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    run_parser.add_argument('--precision', choices=['INT8', 'FP16', 'FP32'], default='INT8',
                            help='Quantization precision (default: INT8)')
    run_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    run_parser.add_argument('--token', help='HuggingFace API token')

    test_parser = subparsers.add_parser('test', help='Run the test suite')
    test_parser.add_argument('--model', default='LiquidAI/LFM2-VL-450M',
                             help='Model to use for tests')
    test_parser.add_argument('--transcribe_model', default='openai/whisper-small',
                             help='Transcribe model to use')
    test_parser.add_argument('--android', action='store_true',
                             help='Run tests on Android')
    test_parser.add_argument('--ios', action='store_true',
                             help='Run tests on iOS')

    clean_parser = subparsers.add_parser('clean', help='Remove all build artifacts')

    convert_parser = subparsers.add_parser('convert', help='Convert model to custom output directory')
    convert_parser.add_argument('model_name', help='HuggingFace model name')
    convert_parser.add_argument('output_dir', nargs='?', default=None,
                                help='Output directory (default: weights/<model_name>)')
    convert_parser.add_argument('--precision', choices=['INT8', 'FP16', 'FP32'], default='INT8',
                                help='Quantization precision')
    convert_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    convert_parser.add_argument('--token', help='HuggingFace API token')

    return parser


def main():
    """Main entry point for the Cactus CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'download':
        sys.exit(cmd_download(args))
    elif args.command == 'build':
        sys.exit(cmd_build(args))
    elif args.command == 'run':
        sys.exit(cmd_run(args))
    elif args.command == 'test':
        sys.exit(cmd_test(args))
    elif args.command == 'clean':
        sys.exit(cmd_clean(args))
    elif args.command == 'convert':
        sys.exit(cmd_convert(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
