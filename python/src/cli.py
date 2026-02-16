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
    return model_name


def get_weights_dir(model_id):
    """Get the weights directory path for a model."""
    if 'silero-vad' in model_id.lower():
        return PROJECT_ROOT / "weights" / "silero-vad"
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
    from .config_utils import is_lfm2_vl, pick_dtype, vision_weight_sanity_check

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
    is_vad = 'silero-vad' in model_id.lower()

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

            processor = None
            try:
                processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            except Exception as proc_err:
                if "TokenizersBackend" in str(proc_err) or "does not exist or is not currently imported" in str(proc_err):
                    print(f"  Note: AutoProcessor failed, using fallback tokenizer loading...")
                else:
                    raise

            cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            dtype = pick_dtype()

            if is_lfm2_vl(model_id, cfg) and Lfm2VlForConditionalGeneration is not None:
                model = Lfm2VlForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, dtype=dtype, token=token)
            else:
                model = AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, dtype=dtype, token=token)

            tokenizer = getattr(processor, "tokenizer", None) if processor else None
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
                except Exception as tok_err:
                    if "TokenizersBackend" in str(tok_err) or "does not exist or is not currently imported" in str(tok_err):
                        from transformers import PreTrainedTokenizerFast
                        print(f"  Note: Using PreTrainedTokenizerFast fallback for invalid tokenizer_class...")
                        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, cache_dir=cache_dir, token=token)
                    else:
                        raise

            if is_lfm2_vl(model_id, cfg) and not vision_weight_sanity_check(model):
                print_color(RED, "Vision embeddings look randomly initialized.")
                return 1

            try:
                convert_processors(processor, model_id, weights_dir, token=token)
            except Exception as e:
                print(f"  Warning: convert_processors failed: {e}")

        elif 'moonshine' in model_id.lower():
            from transformers import MoonshineForConditionalGeneration
            print(f"  Note: Loading Moonshine model using MoonshineForConditionalGeneration...")
            model = MoonshineForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        elif is_whisper:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        elif is_vad:
            try:
                import torchaudio
            except ImportError:
                print_color(RED, "Error: torchaudio is required for Silero-VAD")
                print("Install with: pip install torchaudio")
                return 1

            from .converter_silero_vad import convert_silero_vad_weights

            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
            convert_silero_vad_weights(model, weights_dir, precision, args)

            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print_color(GREEN, f"Successfully downloaded and converted weights to {weights_dir}")
            return 0

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

        # Config precision stores the compute precision (weights are quantized, activations stay FP16)
        if precision in ('INT8', 'INT4'):
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


def check_libcurl():
    """Check if libcurl development libraries are installed."""
    import platform

    if platform.system() == 'Darwin':
        return True

    if check_command('pkg-config'):
        result = subprocess.run(['pkg-config', '--exists', 'libcurl'], capture_output=True)
        if result.returncode == 0:
            return True

    curl_paths = [
        '/usr/include/curl/curl.h',
        '/usr/include/x86_64-linux-gnu/curl/curl.h',
        '/usr/include/aarch64-linux-gnu/curl/curl.h',
        '/usr/local/include/curl/curl.h',
    ]
    for path in curl_paths:
        if Path(path).exists():
            return True

    return False


def cmd_build(args):
    """Build the Cactus library and chat binary."""
    if getattr(args, 'apple', False):
        return cmd_build_apple(args)
    if getattr(args, 'android', False):
        return cmd_build_android(args)
    if getattr(args, 'flutter', False):
        return cmd_build_flutter(args)
    if getattr(args, 'python', False):
        return cmd_build_python(args)

    print_color(BLUE, "Building Cactus chat...")
    print("=" * 23)

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake build-essential")
        return 1

    if not check_libcurl():
        print_color(RED, "Error: libcurl development libraries not found")
        print("  macOS: brew install curl")
        print("  Ubuntu: sudo apt-get install libcurl4-openssl-dev")
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

    asr_cpp = tests_dir / "asr.cpp"
    if asr_cpp.exists():
        print("Compiling asr.cpp...")

        # Check for SDL2 and get flags
        sdl2_available = False
        sdl2_include = ""
        sdl2_lib = ""

        if is_darwin:
            sdl2_check = subprocess.run(["brew", "list", "sdl2"], capture_output=True)
            if sdl2_check.returncode == 0:
                sdl2_prefix_result = subprocess.run(["brew", "--prefix", "sdl2"], capture_output=True, text=True)
                if sdl2_prefix_result.returncode == 0:
                    sdl2_prefix = sdl2_prefix_result.stdout.strip()
                    sdl2_include = f"-I{sdl2_prefix}/include"
                    sdl2_lib = f"-L{sdl2_prefix}/lib -lSDL2"
                    sdl2_available = True
        else:
            sdl2_check = subprocess.run(["pkg-config", "--exists", "sdl2"], capture_output=True)
            if sdl2_check.returncode == 0:
                cflags = subprocess.run(["pkg-config", "--cflags", "sdl2"], capture_output=True, text=True)
                libs = subprocess.run(["pkg-config", "--libs", "sdl2"], capture_output=True, text=True)
                if cflags.returncode == 0 and libs.returncode == 0:
                    sdl2_include = cflags.stdout.strip()
                    sdl2_lib = libs.stdout.strip()
                    sdl2_available = True

        if sdl2_available:
            print_color(GREEN, "SDL2 found - building with live transcription support")
        else:
            print_color(YELLOW, "SDL2 not found - live transcription will be disabled")
            print_color(YELLOW, "Install SDL2 for live mic support: brew install sdl2 (macOS)")

        if is_darwin:
            cmd = [
                compiler, "-std=c++20", "-O3",
                f"-I{PROJECT_ROOT}",
            ]
            if sdl2_available:
                cmd.extend(["-DHAVE_SDL2", sdl2_include])
            cmd.extend([
                str(asr_cpp),
                str(lib_path),
                "-o", "asr",
                "-lcurl",
                "-framework", "Accelerate",
                "-framework", "CoreML",
                "-framework", "Foundation"
            ])
            if sdl2_available:
                cmd.extend(sdl2_lib.split())
        else:
            cmd = [
                compiler, "-std=c++20", "-O3",
                f"-I{PROJECT_ROOT}",
            ]
            if sdl2_available:
                cmd.extend(["-DHAVE_SDL2", sdl2_include])
            cmd.extend([
                str(asr_cpp),
                str(lib_path),
                "-o", "asr",
                "-lcurl",
                "-pthread"
            ])
            if sdl2_available:
                cmd.extend(sdl2_lib.split())

        result = subprocess.run(cmd, cwd=build_dir)
        if result.returncode != 0:
            print_color(RED, "ASR build failed")
            return 1

        print_color(GREEN, f"Build complete: {build_dir / 'asr'}")

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


def cmd_build_flutter(args):
    """Build Cactus for Flutter (iOS, macOS, Android)."""
    print_color(BLUE, "Building Cactus for Flutter...")
    print("=" * 32)

    build_script = PROJECT_ROOT / "flutter" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "flutter", check=False)
    if result.returncode != 0:
        print_color(RED, "Flutter build failed")
        return 1

    print_color(GREEN, "Flutter build complete!")
    print()
    print("Output:")
    print(f"  flutter/libcactus.so")
    print(f"  flutter/cactus-ios.xcframework")
    print(f"  flutter/cactus-macos.xcframework")
    return 0


def cmd_build_python(args):
    """Build Cactus shared library for Python FFI."""
    print_color(BLUE, "Building Cactus for Python...")
    print("=" * 30)

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake")
        return 1

    cactus_dir = PROJECT_ROOT / "cactus"
    build_script = cactus_dir / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=cactus_dir, check=False)
    if result.returncode != 0:
        print_color(RED, "Build failed")
        return 1

    if platform.system() == "Darwin":
        lib_name = "libcactus.dylib"
    else:
        lib_name = "libcactus.so"

    lib_path = cactus_dir / "build" / lib_name
    if not lib_path.exists():
        print_color(RED, f"Shared library not found at {lib_path}")
        return 1

    print_color(GREEN, "Python build complete!")
    print(f"Library: {lib_path}")
    return 0


def cmd_run(args):
    """Download model if needed and start interactive chat."""
    model_id = args.model_id

    if getattr(args, 'no_cloud_tele', False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    lib_path = PROJECT_ROOT / "cactus" / "build" / "libcactus.a"
    if not lib_path.exists():
        print_color(RED, "Error: Cactus library not built. Run 'cactus build' first.")
        return 1

    local_path = Path(model_id)
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
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


DEFAULT_ASR_MODEL_ID = "UsefulSensors/moonshine-base"


def cmd_transcribe(args):
    """Download ASR model if needed and start transcription."""
    model_id = getattr(args, 'model_id', DEFAULT_ASR_MODEL_ID)
    audio_file = getattr(args, 'audio_file', None)

    if getattr(args, 'no_cloud_tele', False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    if model_id and model_id.lower().endswith(audio_extensions):
        audio_file = model_id
        model_id = DEFAULT_ASR_MODEL_ID
        args.model_id = model_id

    asr_binary = PROJECT_ROOT / "tests" / "build" / "asr"
    if not asr_binary.exists():
        print_color(RED, "Error: ASR binary not built. Run 'cactus build' first.")
        return 1

    local_path = Path(model_id)
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
        download_result = cmd_download(args)
        if download_result != 0:
            return download_result
        weights_dir = get_weights_dir(model_id)

    os.system('clear' if platform.system() != 'Windows' else 'cls')
    print_color(GREEN, f"Starting Cactus ASR with model: {model_id}")
    print()

    cmd_args = [str(asr_binary), str(weights_dir)]
    if audio_file:
        cmd_args.append(audio_file)

    os.execv(str(asr_binary), cmd_args)


def cmd_eval(args):
    model_id = getattr(args, 'model_id', DEFAULT_MODEL_ID)

    if PROJECT_ROOT.parent.name != 'evals':
        print_color(RED, "Skipping internal eval checks: companion repo not found.")
        return 1

    # Check if cactus library exists
    lib_path = PROJECT_ROOT / "cactus" / "build" / "libcactus.a"
    if not lib_path.exists():
        print_color(RED, "Error: Cactus library not built. Run 'cactus build' first.")
        return 1

    class DownloadArgs:
        pass

    dlargs = DownloadArgs()
    dlargs.model_id = model_id
    dlargs.precision = getattr(args, 'precision', 'INT8')
    dlargs.cache_dir = getattr(args, 'cache_dir', None)
    dlargs.token = getattr(args, 'token', None)

    download_result = cmd_download(dlargs)
    if download_result != 0:
        return download_result

    weights_dir = get_weights_dir(model_id)
    extra = getattr(args, 'extra_args', None) or []

    def extra_has_flag(flag: str) -> bool:
        for a in extra:
            if a == flag or a.startswith(flag + "="):
                return True
        return False

    mode_flags = []
    if getattr(args, 'tools', False): mode_flags.append('tools')
    if getattr(args, 'llm', False):   mode_flags.append('llm')
    if getattr(args, 'stt', False):   mode_flags.append('stt')
    if getattr(args, 'vlm', False):   mode_flags.append('vlm')
    if getattr(args, 'embed', False): mode_flags.append('embed')

    if len(mode_flags) > 1:
        print_color(RED, f"Error: choose only one eval mode flag, got: {' '.join(mode_flags)}")
        return 1

    mode = mode_flags[0] if mode_flags else "tools"
    repo_root = PROJECT_ROOT.parent  # evals/
    cwd = repo_root

    if mode == "tools":
        eval_runner = repo_root / "tool-evals" / "run_eval_berk.py"
    elif mode == "stt":
        eval_runner = repo_root / "speech-evals" / "speech_eval.py"
    elif mode == "llm":
        eval_runner = repo_root / "text-evals" / "perplexity_eval.py"
    elif mode == "vlm":
        eval_runner = repo_root / "video-evals" / "run_benchmarks.py"
    elif mode == "embed":
        print_color(RED, f"Error: eval mode '{mode}' is not supported in this repo layout")
        return 1
    else:
        print_color(RED, f"Error: unknown eval mode '{mode}'")
        return 1

    if not eval_runner.exists():
        print_color(RED, f"Eval runner not found at {eval_runner}")
        return 1

    cmd = [sys.executable, str(eval_runner)]

    if mode == "vlm":
        if not extra_has_flag("--model"):
            cmd += ["--model", str(weights_dir)]
        if not extra_has_flag("--all") and not extra_has_flag("--benchmarks"):
            cmd += ["--all"]
    else:
        if not extra_has_flag("--model-path"):
            cmd += ["--model-path", str(weights_dir)]

    if mode == "llm" and not extra_has_flag("--model-id"):
        cmd += ["--model-id", str(model_id)]

    if mode == "stt" and not extra_has_flag("--dataset-path"):
        default_dataset_path = repo_root / "speech-evals" / "dataset-retrieval"
        cmd += ["--dataset-path", str(default_dataset_path)]

    if not extra_has_flag("--output-dir"):
        if mode == "tools":
            default_out = repo_root / "tool-evals" / "results"
        elif mode == "stt":
            default_out = repo_root / "speech-evals" / "results"
        elif mode == "llm":
            default_out = repo_root / "text-evals" / "results"
        else:
            default_out = None
        if default_out is not None:
            cmd += ["--output-dir", str(default_out)]

    cmd += extra

    print_color(BLUE, f"[cactus] launching {mode} eval runner")
    print(" ".join(cmd))

    env = os.environ.copy()
    if getattr(args, 'no_cloud_tele', False):
        env["CACTUS_NO_CLOUD_TELE"] = "1"
    if mode == "vlm":
        ffi_dir = str(repo_root / "cactus" / "tools" / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ffi_dir if not existing else (ffi_dir + os.pathsep + existing)

    r = subprocess.run(cmd, cwd=str(cwd), env=env)
    return r.returncode



def cmd_test(args):
    """Run the Cactus test suite."""
    print_color(BLUE, "Running test suite...")
    print("=" * 20)

    if getattr(args, 'large', False):
        args.model = 'LiquidAI/LFM2.5-VL-1.6B'
        args.transcribe_model = 'openai/whisper-small'
        print_color(BLUE, f"Using large models: {args.model}, {args.transcribe_model}, {args.vad_model}")

    precision = getattr(args, 'precision', None)
    if precision:
        # Regenerate main model weights
        model_id = getattr(args, 'model', 'LiquidAI/LFM2-VL-450M')
        weights_dir = get_weights_dir(model_id)

        if weights_dir.exists():
            print_color(YELLOW, f"Removing existing weights at {weights_dir} to regenerate with {precision}...")
            shutil.rmtree(weights_dir)

        class DownloadArgs:
            pass
        dl_args = DownloadArgs()
        dl_args.model_id = model_id
        dl_args.precision = precision
        dl_args.cache_dir = None
        dl_args.token = getattr(args, 'token', None)

        download_result = cmd_download(dl_args)
        if download_result != 0:
            return download_result

        transcribe_model_id = getattr(args, 'transcribe_model', 'UsefulSensors/moonshine-base')
        transcribe_weights_dir = get_weights_dir(transcribe_model_id)

        if transcribe_weights_dir.exists():
            print_color(YELLOW, f"Removing existing weights at {transcribe_weights_dir} to regenerate with {precision}...")
            shutil.rmtree(transcribe_weights_dir)

        dl_args_transcribe = DownloadArgs()
        dl_args_transcribe.model_id = transcribe_model_id
        dl_args_transcribe.precision = precision
        dl_args_transcribe.cache_dir = None
        dl_args_transcribe.token = getattr(args, 'token', None)

        download_result = cmd_download(dl_args_transcribe)
        if download_result != 0:
            return download_result

        vad_model_id = getattr(args, 'vad_model', 'silero-vad')
        vad_weights_dir = get_weights_dir(vad_model_id)

        if vad_weights_dir.exists():
            print_color(YELLOW, f"Removing existing weights at {vad_weights_dir} to regenerate with {precision}...")
            shutil.rmtree(vad_weights_dir)

        dl_args_vad = DownloadArgs()
        dl_args_vad.model_id = vad_model_id
        dl_args_vad.precision = precision
        dl_args_vad.cache_dir = None
        dl_args_vad.token = getattr(args, 'token', None)

        download_result = cmd_download(dl_args_vad)
        if download_result != 0:
            return download_result

    test_script = PROJECT_ROOT / "tests" / "run.sh"

    if not test_script.exists():
        print_color(RED, f"Error: Test script not found at {test_script}")
        return 1

    cmd = [str(test_script)]

    if args.model:
        cmd.extend(["--model", args.model])
    if args.transcribe_model:
        cmd.extend(["--transcribe_model", args.transcribe_model])
    if args.vad_model:
        cmd.extend(["--vad_model", args.vad_model])
    if precision:
        cmd.extend(["--precision", precision])
    if getattr(args, 'no_rebuild', False):
        cmd.append("--no-rebuild")
    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")
    if args.only:
        cmd.extend(["--only", args.only])

    env = os.environ.copy()
    if getattr(args, 'no_cloud_tele', False):
        env["CACTUS_NO_CLOUD_TELE"] = "1"

    result = subprocess.run(cmd, cwd=PROJECT_ROOT / "tests", env=env)
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

    bin_count = 0
    for bin_file in PROJECT_ROOT.rglob("*.bin"):
        bin_file.unlink()
        bin_count += 1
    print(f"Removed {bin_count} .bin files" if bin_count else "No .bin files found")

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
        ["bash", "-c", f"source {setup_script}"],
        cwd=PROJECT_ROOT
    )
    if result.returncode == 0:
        print_color(GREEN, "Setup complete!")
    else:
        print_color(YELLOW, "Setup had issues. Please run manually:")
        print("  source ./setup")
    return 0


def merge_lora_adapter(base_model_id, lora_path, cache_dir=None, token=None):
    """Merge a LoRA adapter into a base model and return the merged model."""
    try:
        from peft import PeftModel
    except ImportError:
        print_color(RED, "Error: peft package required for LoRA merging")
        print("Install with: pip install peft")
        return None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_color(YELLOW, f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token
    )

    print_color(YELLOW, f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path, token=token)

    print_color(YELLOW, "Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print_color(GREEN, "LoRA merge complete")
    return merged_model, tokenizer


def cmd_convert(args):
    """Convert a HuggingFace model to a custom output directory."""
    import tempfile

    model_id = args.model_name
    output_dir = args.output_dir
    lora_path = getattr(args, 'lora', None)

    if output_dir is None:
        output_dir = get_weights_dir(model_id)
    else:
        output_dir = Path(output_dir)

    cache_dir = getattr(args, 'cache_dir', None)
    token = getattr(args, 'token', None)

    temp_merged_dir = None

    if lora_path:
        merged_model, tokenizer = merge_lora_adapter(model_id, lora_path, cache_dir, token)
        if merged_model is None:
            return 1

        temp_merged_dir = tempfile.mkdtemp(prefix="cactus_lora_merged_")
        print_color(YELLOW, f"Saving merged model to temp directory: {temp_merged_dir}")
        merged_model.save_pretrained(temp_merged_dir)
        tokenizer.save_pretrained(temp_merged_dir)

        del merged_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_id = temp_merged_dir

    class DownloadArgs:
        pass

    download_args = DownloadArgs()
    download_args.model_id = model_id
    download_args.precision = args.precision
    download_args.cache_dir = cache_dir
    download_args.token = token

    original_get_weights = get_weights_dir

    def custom_weights_dir(mid):
        return output_dir

    import src.cli as cli_module
    cli_module.get_weights_dir = custom_weights_dir

    try:
        result = cmd_download(download_args)
        return result
    finally:
        cli_module.get_weights_dir = original_get_weights
        if temp_merged_dir and Path(temp_merged_dir).exists():
            print_color(YELLOW, "Cleaning up temp directory...")
            shutil.rmtree(temp_merged_dir)


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
    --precision INT4|INT8|FP16   default: INT8
    --token <token>                    HF token (for gated models)

  -----------------------------------------------------------------

  cactus transcribe [model]            live microphone transcription
                                       default model: whisper-small

    Optional flags:
    --file <audio.wav>                 transcribe audio file instead of mic
    --precision INT4|INT8|FP16         default: INT8
    --token <token>                    HF token (for gated models)

    Examples:
    cactus transcribe                  live microphone transcription
    cactus transcribe --file audio.wav transcribe single file
    cactus transcribe openai/whisper-small   use different model
    cactus transcribe openai/whisper-small --file audio.wav

   -----------------------------------------------------------------

  cactus download <model>              downloads model to ./weights
                                       see supported weights on ReadMe

    Optional flags:
    --precision INT4|INT8|FP16   quantization (default: INT8)
    --token <token>                    HuggingFace API token

  -----------------------------------------------------------------

  cactus convert <model> [output_dir]  converts model to custom directory
                                       supports LoRA adapter merging

    Optional flags:
    --precision INT4|INT8|FP16   quantization (default: INT8)
    --lora <path>                      LoRA adapter path to merge
    --token <token>                    HuggingFace API token

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
    --transcribe_model <model>         default: whisper-small
    --large                            use larger models (LFM2.5-VL-1.6B + whisper-small)
    --precision INT4|INT8|FP16         regenerates weights with precision
    --no-rebuild                       skip building library and tests
    --only <test_name>                 run specific test (engine, graph, index, kernel, kv_cache, performance, etc)
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
  2. cactus download LiquidAI/LFM2-VL-450M
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

    download_parser = subparsers.add_parser('download', help='Download and convert model weights')
    download_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                                 help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    download_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT8',
                                 help='Quantization precision (default: INT8)')
    download_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    download_parser.add_argument('--token', help='HuggingFace API token')

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
    run_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT8',
                            help='Quantization precision (default: INT8)')
    run_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    run_parser.add_argument('--token', help='HuggingFace API token')
    run_parser.add_argument('--no-cloud-tele', action='store_true',
                            help='Disable cloud telemetry (write to cache only)')

    transcribe_parser = subparsers.add_parser('transcribe', help='Download ASR model and run transcription')
    transcribe_parser.add_argument('model_id', nargs='?', default=DEFAULT_ASR_MODEL_ID,
                                   help=f'HuggingFace model ID (default: {DEFAULT_ASR_MODEL_ID})')
    transcribe_parser.add_argument('--file', dest='audio_file', default=None,
                                   help='Audio file to transcribe (WAV format). Omit for live microphone.')
    transcribe_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT8',
                                   help='Quantization precision (default: INT8)')
    transcribe_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    transcribe_parser.add_argument('--token', help='HuggingFace API token')
    transcribe_parser.add_argument('--no-cloud-tele', action='store_true',
                                   help='Disable cloud telemetry (write to cache only)')

    eval_parser = subparsers.add_parser('eval', help='Run evaluation scripts outside the cactus submodule')
    eval_parser.add_argument('model_id', nargs='?', default=DEFAULT_MODEL_ID,
                             help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    eval_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT8',
                             help='Quantization precision (default: INT8)')
    eval_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    eval_parser.add_argument('--token', help='HuggingFace API token')
    eval_parser.add_argument('--tools', action='store_true', help='Run tools evals (default)')
    eval_parser.add_argument('--vlm', action='store_true', help='Run VLM-specific evals')
    eval_parser.add_argument('--stt', action='store_true', help='Run speech-to-text evals')
    eval_parser.add_argument('--llm', action='store_true', help='Run LLM evals')
    eval_parser.add_argument('--embed', action='store_true', help='Run embedding evals')
    eval_parser.add_argument('--no-cloud-tele', action='store_true',
                             help='Disable cloud telemetry (write to cache only)')

    test_parser = subparsers.add_parser('test', help='Run the test suite')
    test_parser.add_argument('--model', default='LiquidAI/LFM2-VL-450M',
                             help='Model to use for tests')
    test_parser.add_argument('--transcribe_model', default='UsefulSensors/moonshine-base',
                             help='Transcribe model to use')
    test_parser.add_argument('--vad_model', default='silero-vad',
                             help='VAD model to use')
    test_parser.add_argument('--large', action='store_true',
                             help='Use larger models (LFM2.5-VL-1.6B + whisper-small)')
    test_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'],
                             help='Regenerate weights with this precision (deletes existing weights)')
    test_parser.add_argument('--no-rebuild', action='store_true',
                             help='Skip building cactus library and tests')
    test_parser.add_argument('--token', help='HuggingFace API token')
    test_parser.add_argument('--android', action='store_true',
                             help='Run tests on Android')
    test_parser.add_argument('--ios', action='store_true',
                             help='Run tests on iOS')
    test_parser.add_argument('--only', help='Only run the specified test (engine, graph, index, kernel, kv_cache, performance, etc)')
    test_parser.add_argument('--no-cloud-tele', action='store_true',
                             help='Disable cloud telemetry (write to cache only)')

    clean_parser = subparsers.add_parser('clean', help='Remove all build artifacts')

    convert_parser = subparsers.add_parser('convert', help='Convert model to custom output directory')
    convert_parser.add_argument('model_name', help='HuggingFace model name')
    convert_parser.add_argument('output_dir', nargs='?', default=None,
                                help='Output directory (default: weights/<model_name>)')
    convert_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT8',
                                help='Quantization precision (default: INT8)')
    convert_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    convert_parser.add_argument('--token', help='HuggingFace API token')
    convert_parser.add_argument('--lora', help='Path to LoRA adapter (local path or HuggingFace ID) to merge before conversion')

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
    elif args.command == 'clean':
        sys.exit(cmd_clean(args))
    elif args.command == 'convert':
        sys.exit(cmd_convert(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
