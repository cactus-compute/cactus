#!/usr/bin/env python3
import sys
import os
import argparse
import re
import json
import subprocess
import shutil
import platform
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
DEFAULT_TEST_TRANSCRIBE_MODEL_ID = "UsefulSensors/moonshine-base"

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
    """Run a script or command and optionally exit on failure.

    Args:
        cmd: Script path (str) or command list. String paths are executed
             directly without shell interpretation to handle spaces safely.
        cwd: Working directory for the command.
        check: If True, exit on non-zero return code.
    """
    # Convert string paths to list to avoid shell=True and handle spaces safely
    if isinstance(cmd, str):
        cmd = [cmd]
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def ensure_vad_weights(model_id, weights_dir, precision='INT8'):
    """Bundle Silero VAD weights into <weights_dir>/vad/ for ASR models."""
    is_asr = (
        'whisper' in model_id.lower()
        or 'moonshine' in model_id.lower()
        or 'parakeet' in model_id.lower()
    )
    if not is_asr:
        return
    vad_dir = weights_dir / "vad"
    if (vad_dir / "config.txt").exists():
        return
    try:
        import torch
        import urllib.request
        import tempfile
        from .converter import convert_silero_vad_weights

        print_color(YELLOW, "Bundling VAD weights for speech model...")
        vad_jit_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
        with tempfile.NamedTemporaryFile(suffix='.jit', delete=False) as f:
            jit_path = f.name
        urllib.request.urlretrieve(vad_jit_url, jit_path)
        vad_model = torch.jit.load(jit_path, map_location='cpu')
        os.unlink(jit_path)

        convert_silero_vad_weights(vad_model, str(vad_dir), precision)
        del vad_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_color(GREEN, "VAD weights bundled successfully")
    except Exception as e:
        print_color(RED, f"Warning: Failed to bundle VAD weights: {e}")
        print("Transcription may fail without VAD. Try: cactus download snakers4/silero-vad")


def download_from_hf(model_id, weights_dir, precision):
    """Download pre-converted model from Cactus-Compute HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import zipfile
    except ImportError:
        print_color(RED, "Error: huggingface_hub package not found.")
        print("Please run: pip install huggingface_hub")
        return False

    model_name = get_model_dir_name(model_id)
    org = "Cactus-Compute"
    repo_id = f"{org}/{model_id.split('/')[-1]}"

    try:
        precision_lower = precision.lower()
        apple_zip = f"{model_name}-{precision_lower}-apple.zip"
        standard_zip = f"{model_name}-{precision_lower}.zip"

        repo_files = list_repo_files(repo_id, repo_type="model")

        zip_file = None
        if f"weights/{apple_zip}" in repo_files:
            zip_file = apple_zip
        elif f"weights/{standard_zip}" in repo_files:
            zip_file = standard_zip
        else:
            print_color(YELLOW, f"Pre-converted model not found in {repo_id}")
            return False

        print_color(BLUE, f"Downloading from {repo_id}...")

        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"weights/{zip_file}",
            repo_type="model"
        )

        weights_dir.mkdir(parents=True, exist_ok=True)

        print_color(YELLOW, "Extracting model weights...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)

        if not (weights_dir / "config.txt").exists():
            print_color(RED, f"Error: Downloaded model is missing config.txt")
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            return False

        print_color(GREEN, f"Successfully downloaded pre-converted model to {weights_dir}")
        return True

    except Exception:
        print_color(YELLOW, f"Could not download from {repo_id}")
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        return False


def cmd_download(args):
    """Download model weights. By default downloads pre-converted weights from Cactus-Compute."""
    model_id = args.model_id
    weights_dir = get_weights_dir(model_id)
    reconvert = getattr(args, 'reconvert', False)
    precision = getattr(args, 'precision', 'INT4')

    if reconvert and weights_dir.exists():
        print_color(YELLOW, f"Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        ensure_vad_weights(model_id, weights_dir, precision)
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return 0

    print()
    print_color(YELLOW, f"Model weights not found. Downloading {model_id}...")
    print("=" * 45)

    if not reconvert:
        if download_from_hf(model_id, weights_dir, precision):
            ensure_vad_weights(model_id, weights_dir, precision)
            return 0

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print_color(RED, "Error: Required Python packages not found.")
        print("Please run: ./setup")
        return 1

    from .converter import convert_hf_model_weights
    from .tokenizer import convert_hf_tokenizer
    from .tensor_io import format_config_value
    from .config_utils import is_lfm2_vl, pick_dtype, vision_weight_sanity_check

    weights_dir.mkdir(parents=True, exist_ok=True)

    precision = getattr(args, 'precision', 'INT4')
    cache_dir = getattr(args, 'cache_dir', None)
    token = getattr(args, 'token', None)

    print(f"Converting {model_id} to {precision}...")

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    import transformers
    transformers.logging.set_verbosity_error()

    def _download_config_json(repo_id):
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir, token=token)
        with open(config_path, 'r', encoding='utf-8') as fh:
            return json.load(fh)

    def _load_raw_hf_state_dict(repo_id):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors_file

        snapshot_path = Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json"],
        ))

        index_candidates = [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]

        shard_files = []
        for index_name in index_candidates:
            index_path = snapshot_path / index_name
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as fh:
                    index_data = json.load(fh)
                shard_files = sorted(set(index_data.get("weight_map", {}).values()))
                if shard_files:
                    break

        if not shard_files:
            shard_files = sorted([p.name for p in snapshot_path.glob("*.safetensors")])
        if not shard_files:
            shard_files = sorted([p.name for p in snapshot_path.glob("*.bin")])

        if not shard_files:
            raise RuntimeError("No checkpoint shard files found in HuggingFace snapshot.")

        merged_state_dict = {}
        for shard_name in shard_files:
            shard_path = snapshot_path / shard_name
            if shard_name.endswith(".safetensors"):
                shard_state = load_safetensors_file(str(shard_path), device="cpu")
            elif shard_name.endswith(".bin"):
                shard_state = torch.load(str(shard_path), map_location="cpu")
            else:
                continue
            merged_state_dict.update(shard_state)

        return merged_state_dict

    try:
        from transformers import Lfm2VlForConditionalGeneration
    except ImportError:
        Lfm2VlForConditionalGeneration = None

    is_vlm = 'vl' in model_id.lower() or 'vlm' in model_id.lower()
    is_whisper = 'whisper' in model_id.lower()
    is_parakeet = 'parakeet' in model_id.lower()
    is_vad = 'silero-vad' in model_id.lower()

    try:
        if is_vlm:
            from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
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

        elif 'moonshine' in model_id.lower():
            from transformers import MoonshineForConditionalGeneration
            print(f"  Note: Loading Moonshine model using MoonshineForConditionalGeneration...")
            model = MoonshineForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        elif is_whisper:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

        elif is_parakeet:
            from transformers import AutoConfig
            from huggingface_hub import hf_hub_download, snapshot_download
            from safetensors.torch import load_file as load_safetensors

            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            config_obj = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)

            state_dict = None
            try:
                weights_path = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                    token=token
                )
                state_dict = load_safetensors(weights_path, device="cpu")
            except Exception:
                snapshot_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, token=token)
                index_path = Path(snapshot_path) / "model.safetensors.index.json"
                if not index_path.exists():
                    raise
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                shard_files = sorted(set(index_data.get("weight_map", {}).values()))
                if not shard_files:
                    raise RuntimeError("Parakeet safetensors index has no shard entries")
                state_dict = {}
                for shard_name in shard_files:
                    shard_path = Path(snapshot_path) / shard_name
                    shard_state = load_safetensors(str(shard_path), device="cpu")
                    state_dict.update(shard_state)

            class _StateDictModel:
                def __init__(self, config, state_dict):
                    self.config = config
                    self._state_dict = state_dict

                def state_dict(self):
                    return self._state_dict

            model = _StateDictModel(config_obj, state_dict)

        elif is_vad:
            import urllib.request
            import tempfile
            from .converter import convert_silero_vad_weights

            vad_jit_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
            with tempfile.NamedTemporaryFile(suffix='.jit', delete=False) as f:
                jit_path = f.name
            urllib.request.urlretrieve(vad_jit_url, jit_path)
            model = torch.jit.load(jit_path, map_location='cpu')
            os.unlink(jit_path)
            convert_silero_vad_weights(model, weights_dir, precision, args)

            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print_color(GREEN, f"Successfully downloaded and converted weights to {weights_dir}")
            return 0

        else:
            config_json = _download_config_json(model_id)
            model_type = str(config_json.get('model_type', '')).lower()

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, token=token)
            except Exception as tok_err:
                if "TokenizersBackend" in str(tok_err) or "does not exist or is not currently imported" in str(tok_err):
                    from transformers import PreTrainedTokenizerFast
                    print("  Note: Using PreTrainedTokenizerFast fallback for invalid tokenizer_class...")
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, cache_dir=cache_dir, token=token)
                else:
                    raise

            if model_type == 'lfm2_moe':
                print("  Note: Loading raw checkpoint tensors for lfm2_moe conversion...")
                raw_state_dict = _load_raw_hf_state_dict(model_id)

                class _RawModelWrapper:
                    def __init__(self, state_dict, config):
                        self._state_dict = state_dict
                        self.config = config

                    def state_dict(self):
                        return self._state_dict

                model = _RawModelWrapper(raw_state_dict, config_json)
            else:
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
    vendored_curl = PROJECT_ROOT / "libs" / "curl" / "macos" / "libcurl.a"

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
        if not vendored_curl.exists():
            print_color(RED, f"Error: vendored libcurl not found at {vendored_curl}")
            print("Build it first and place it in libs/curl/macos/libcurl.a")
            return 1
        compiler = "clang++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            "-DACCELERATE_NEW_LAPACK",
            f"-I{PROJECT_ROOT}",
            str(chat_cpp),
            str(lib_path),
            "-o", "chat",
            str(vendored_curl),
            "-framework", "Accelerate",
            "-framework", "CoreML",
            "-framework", "Foundation",
            "-framework", "Security",
            "-framework", "SystemConfiguration",
            "-framework", "CFNetwork",
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
            print_color(YELLOW, "Then run `cactus build`")

        if is_darwin:
            cmd = [
                compiler, "-std=c++20", "-O3",
                "-DACCELERATE_NEW_LAPACK",
                f"-I{PROJECT_ROOT}",
            ]
            if sdl2_available:
                cmd.extend(["-DHAVE_SDL2", sdl2_include])
            cmd.extend([
                str(asr_cpp),
                str(lib_path),
                "-o", "asr",
                str(vendored_curl),
                "-framework", "Accelerate",
                "-framework", "CoreML",
                "-framework", "Foundation",
                "-framework", "Security",
                "-framework", "SystemConfiguration",
                "-framework", "CFNetwork",
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


def prompt_for_api_key(config):
    """Prompt user to set Cactus Cloud API key if not already configured. Returns the key or empty string."""
    api_key = config.get_api_key()
    if api_key:
        return api_key

    print("\n" + "="*50)
    print("  Cactus Cloud Setup (Optional)")
    print("="*50 + "\n")
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    print("to enable automatic cloud fallback.\n")

    api_key = input("Your Cactus Cloud key (press Enter to skip): ").strip()
    if api_key:
        config.set_api_key(api_key)
        masked = api_key[:4] + "..." + api_key[-4:]
        print_color(GREEN, f"API key saved: {masked}")
    print()
    return api_key


def cmd_run(args):
    """Download model if needed and start interactive chat."""
    from .config_utils import CactusConfig

    config = CactusConfig()
    api_key = prompt_for_api_key(config)

    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

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


DEFAULT_ASR_MODEL_ID = "nvidia/parakeet-ctc-1.1b"

def _pick_android_device_id(preferred_device=None):
    if preferred_device:
        return preferred_device

    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    if result.returncode != 0:
        return None

    devices = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])

    if len(devices) == 1:
        return devices[0]
    return None


def _cmd_transcribe_android(weights_dir, audio_file, args):
    if not audio_file:
        print_color(RED, "Error: --android requires --file <audio.wav>")
        return 1
    if not check_command("adb"):
        print_color(RED, "Error: adb not found in PATH")
        return 1

    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        print_color(RED, f"Error: audio file not found: {audio_path}")
        return 1

    device_id = _pick_android_device_id(getattr(args, "device", None))
    if not device_id:
        print_color(RED, "Error: could not select Android device. Use --device <adb_id>.")
        return 1

    print_color(BLUE, f"Using Android device: {device_id}")

    android_build_script = PROJECT_ROOT / "android" / "build.sh"
    if not android_build_script.exists():
        print_color(RED, f"Error: build.sh not found at {android_build_script}")
        return 1
    if run_command(str(android_build_script), cwd=PROJECT_ROOT / "android", check=False).returncode != 0:
        print_color(RED, "Android library build failed")
        return 1

    if not check_command("cmake"):
        print_color(RED, "Error: CMake is not installed")
        return 1

    android_test_dir = PROJECT_ROOT / "tests" / "android"
    android_build_dir = android_test_dir / "build"
    ndk_home = os.environ.get("ANDROID_NDK_HOME")
    if not ndk_home:
        android_home = os.environ.get("ANDROID_HOME") or str(Path.home() / "Library" / "Android" / "sdk")
        ndk_root = Path(android_home) / "ndk"
        if ndk_root.exists():
            ndk_versions = sorted([p for p in ndk_root.iterdir() if p.is_dir()])
            if ndk_versions:
                ndk_home = str(ndk_versions[-1])
    if not ndk_home or not Path(ndk_home).exists():
        print_color(RED, "Error: Android NDK not found. Set ANDROID_NDK_HOME.")
        return 1

    toolchain = Path(ndk_home) / "build" / "cmake" / "android.toolchain.cmake"
    if not toolchain.exists():
        print_color(RED, f"Error: Android toolchain not found at {toolchain}")
        return 1

    android_build_dir.mkdir(parents=True, exist_ok=True)
    cfg_cmd = [
        "cmake", "-S", str(android_test_dir), "-B", str(android_build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        "-DANDROID_ABI=arm64-v8a",
        f"-DANDROID_PLATFORM={os.environ.get('ANDROID_PLATFORM', 'android-21')}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if subprocess.run(cfg_cmd).returncode != 0:
        print_color(RED, "Failed to configure Android transcribe build")
        return 1
    build_cmd = ["cmake", "--build", str(android_build_dir), "--target", "asr", "-j", str(os.cpu_count() or 4)]
    if subprocess.run(build_cmd).returncode != 0:
        print_color(RED, "Failed to build Android asr binary")
        return 1

    asr_bin = android_build_dir / "asr"
    if not asr_bin.exists():
        print_color(RED, f"Error: Android asr binary not found at {asr_bin}")
        return 1

    model_name = Path(weights_dir).name
    device_root = "/data/local/tmp/cactus_transcribe"
    device_model_root = f"{device_root}/models"
    device_audio_root = f"{device_root}/audio"
    device_bin_root = f"{device_root}/bin"
    device_audio = f"{device_audio_root}/{audio_path.name}"
    device_model = f"{device_model_root}/{model_name}"

    subprocess.run(["adb", "-s", device_id, "shell", f"mkdir -p {device_bin_root} {device_model_root} {device_audio_root}"], check=False)
    if subprocess.run(["adb", "-s", device_id, "push", str(asr_bin), f"{device_bin_root}/asr"]).returncode != 0:
        print_color(RED, "Failed to push Android asr binary")
        return 1
    subprocess.run(["adb", "-s", device_id, "shell", f"chmod +x {device_bin_root}/asr"], check=False)
    if subprocess.run(["adb", "-s", device_id, "push", str(weights_dir), device_model_root]).returncode != 0:
        print_color(RED, "Failed to push ASR model weights to device")
        return 1
    if subprocess.run(["adb", "-s", device_id, "push", str(audio_path), device_audio]).returncode != 0:
        print_color(RED, "Failed to push audio file to device")
        return 1

    cloud_api_key = os.environ.get("CACTUS_CLOUD_KEY", os.environ.get("CACTUS_CLOUD_API_KEY", ""))
    cloud_strict_ssl = os.environ.get("CACTUS_CLOUD_STRICT_SSL", "")
    cloud_handoff_threshold = os.environ.get("CACTUS_CLOUD_HANDOFF_THRESHOLD", "")
    ca_bundle = os.environ.get("CACTUS_CA_BUNDLE", "")
    ca_path = os.environ.get("CACTUS_CA_PATH", "")
    force_handoff = os.environ.get("CACTUS_FORCE_HANDOFF", "")
    env_exports = []
    if cloud_api_key:
        env_exports.append(f"export CACTUS_CLOUD_KEY='{cloud_api_key}'")
    if cloud_strict_ssl:
        env_exports.append(f"export CACTUS_CLOUD_STRICT_SSL='{cloud_strict_ssl}'")
    if cloud_handoff_threshold:
        env_exports.append(f"export CACTUS_CLOUD_HANDOFF_THRESHOLD='{cloud_handoff_threshold}'")
    if ca_bundle:
        env_exports.append(f"export CACTUS_CA_BUNDLE='{ca_bundle}'")
    if ca_path:
        env_exports.append(f"export CACTUS_CA_PATH='{ca_path}'")
    if getattr(args, "no_cloud_tele", False):
        env_exports.append("export CACTUS_NO_CLOUD_TELE=1")
    if force_handoff:
        env_exports.append(f"export CACTUS_FORCE_HANDOFF='{force_handoff}'")

    shell_cmd = " && ".join(env_exports + [f"{device_bin_root}/asr {device_model} {device_audio}"])
    print_color(BLUE, "Running Android transcription...")
    return subprocess.run(["adb", "-s", device_id, "shell", shell_cmd]).returncode


def _cmd_transcribe_ios(weights_dir, audio_file, args):
    if not audio_file:
        print_color(RED, "Error: --ios requires --file <audio.wav>")
        return 1

    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.exists():
        print_color(RED, f"Error: audio file not found: {audio_path}")
        return 1

    ios_script = PROJECT_ROOT / "tests" / "ios" / "run.sh"
    if not ios_script.exists():
        print_color(RED, f"Error: iOS runner not found at {ios_script}")
        return 1

    transcribe_model_id = Path(weights_dir).name
    env = os.environ.copy()
    env["CACTUS_RUN_ASR"] = "1"
    env["CACTUS_ASR_AUDIO_SOURCE"] = str(audio_path)
    env["CACTUS_ASR_AUDIO_FILE"] = audio_path.name

    cmd = [str(ios_script), transcribe_model_id, transcribe_model_id, "snakers4/silero-vad"]
    print_color(BLUE, "Running iOS transcription...")
    return subprocess.run(cmd, cwd=PROJECT_ROOT / "tests" / "ios", env=env).returncode


def cmd_transcribe(args):
    """Download ASR model if needed and start transcription."""
    from .config_utils import CactusConfig

    config = CactusConfig()
    api_key = prompt_for_api_key(config)

    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

    model_id = getattr(args, 'model_id', DEFAULT_ASR_MODEL_ID)
    audio_file = getattr(args, 'audio_file', None)

    if getattr(args, 'no_cloud_tele', False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    if getattr(args, 'force_handoff', False):
        os.environ["CACTUS_FORCE_HANDOFF"] = "1"
    else:
        os.environ.pop("CACTUS_FORCE_HANDOFF", None)

    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    if model_id and model_id.lower().endswith(audio_extensions):
        audio_file = model_id
        model_id = DEFAULT_ASR_MODEL_ID
        args.model_id = model_id

    local_path = Path(model_id)
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
        download_result = cmd_download(args)
        if download_result != 0:
            return download_result
        weights_dir = get_weights_dir(model_id)

    if getattr(args, 'android', False) and getattr(args, 'ios', False):
        print_color(RED, "Error: choose only one of --android or --ios")
        return 1
    if getattr(args, 'android', False):
        return _cmd_transcribe_android(weights_dir, audio_file, args)
    if getattr(args, 'ios', False):
        return _cmd_transcribe_ios(weights_dir, audio_file, args)

    asr_binary = PROJECT_ROOT / "tests" / "build" / "asr"
    if not asr_binary.exists():
        print_color(RED, "Error: ASR binary not built. Run 'cactus build' first.")
        return 1

    os.system('clear' if platform.system() != 'Windows' else 'cls')
    print_color(GREEN, f"Starting Cactus ASR with model: {model_id}")
    print()

    cmd_args = [str(asr_binary), str(weights_dir)]
    if audio_file:
        cmd_args.append(audio_file)
    if hasattr(args, 'language') and args.language:
        cmd_args.extend(['--language', args.language])

    os.execv(str(asr_binary), cmd_args)


def cmd_auth(args):
    """Manage Cactus Cloud API key."""
    from .config_utils import CactusConfig

    config = CactusConfig()

    if args.clear:
        config.clear_api_key()
        print_color(GREEN, "API key cleared.")
        return 0

    api_key = config.get_api_key()

    if api_key:
        masked = api_key[:4] + "..." + api_key[-4:]
        print(f"Current API key: {masked}")
    else:
        print("No API key set.")

    if args.status:
        return 0

    print()
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    new_key = input("Enter new API key (press Enter to skip): ").strip()
    if new_key:
        config.set_api_key(new_key)
        masked = new_key[:4] + "..." + new_key[-4:]
        print_color(GREEN, f"API key saved: {masked}")
    return 0


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
    dlargs.precision = getattr(args, 'precision', 'INT4')
    dlargs.cache_dir = getattr(args, 'cache_dir', None)
    dlargs.token = getattr(args, 'token', None)
    dlargs.reconvert = getattr(args, 'reconvert', False)

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

    if getattr(args, 'ios', False) and not getattr(args, 'reconvert', False):
        print_color(
            YELLOW,
            "Warning: iOS tests without --reconvert may use stale or inconsistent local weights. "
            "If tests fail unexpectedly, rerun with --reconvert."
        )

    if getattr(args, 'benchmark', False):
        args.model = 'LiquidAI/LFM2.5-VL-1.6B'
        args.transcribe_model = 'nvidia/parakeet-ctc-1.1b'
        print_color(BLUE, f"Using large models: {args.model}, {args.transcribe_model}, {args.vad_model}")

    if getattr(args, 'reconvert', False):
        for model_id in [
            getattr(args, 'model', 'LiquidAI/LFM2-VL-450M'),
            getattr(args, 'transcribe_model', DEFAULT_TEST_TRANSCRIBE_MODEL_ID),
            getattr(args, 'vad_model', 'snakers4/silero-vad')
        ]:
            class DownloadArgs:
                pass
            dl_args = DownloadArgs()
            dl_args.model_id = model_id
            dl_args.reconvert = True
            dl_args.cache_dir = None
            if args.precision:
                dl_args.precision = args.precision
            else:
                is_asr = 'whisper' in model_id.lower() or 'moonshine' in model_id.lower() or 'silero-vad' in model_id.lower()
                dl_args.precision = 'INT8' if is_asr else 'INT4'
            if args.token:
                dl_args.token = args.token
            if cmd_download(dl_args) != 0:
                return 1

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
    if args.precision:
        cmd.extend(["--precision", args.precision])
    if getattr(args, 'no_rebuild', False):
        cmd.append("--no-rebuild")
    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")
    if getattr(args, 'exhaustive', False):
        cmd.append("--exhaustive")
    if args.only:
        cmd.extend(["--only", args.only])
    env = os.environ.copy()
    if getattr(args, 'enable_telemetry', False):
        env.pop("CACTUS_NO_CLOUD_TELE", None)
    else:
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

    # Clean telemetry cache
    telemetry_cache = Path.home() / "Library" / "Caches" / "cactus" / "telemetry"
    if telemetry_cache.exists():
        print(f"Removing telemetry cache: {telemetry_cache}")
        shutil.rmtree(telemetry_cache)
    else:
        print(f"Telemetry cache not found: {telemetry_cache}")

    # Re-cache API key from config so users don't need to run `cactus auth` again
    from .config_utils import CactusConfig
    config = CactusConfig()
    saved_key = config.load_config().get("api_key", "")
    if saved_key:
        config.cache_api_key(saved_key)
        masked = saved_key[:4] + "..." + saved_key[-4:]
        print(f"Restored cached API key: {masked}")

    print()
    print("Removing compiled libraries and frameworks...")

    preserve_roots = [
        PROJECT_ROOT / "libs" / "curl",
        PROJECT_ROOT / "android" / "mbedtls",
        PROJECT_ROOT / "libs" / "mbedtls",
    ]

    def should_preserve_artifact(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return False
        for root in preserve_roots:
            try:
                if resolved.is_relative_to(root.resolve()):
                    return True
            except FileNotFoundError:
                continue
        return False

    so_count = 0
    for so_file in PROJECT_ROOT.rglob("*.so"):
        so_file.unlink()
        so_count += 1
    print(f"Removed {so_count} .so files" if so_count else "No .so files found")

    a_count = 0
    a_preserved_count = 0
    for a_file in PROJECT_ROOT.rglob("*.a"):
        if should_preserve_artifact(a_file):
            a_preserved_count += 1
            continue
        a_file.unlink()
        a_count += 1
    if a_count or a_preserved_count:
        print(f"Removed {a_count} .a files (preserved {a_preserved_count} vendored static libs)")
    else:
        print("No .a files found")

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
    download_args.reconvert = True

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

  cactus auth                          manage Cactus Cloud API key
                                       shows status and prompts to set key

    Optional flags:
    --status                           show key status without prompting
    --clear                            remove the saved API key

  -----------------------------------------------------------------

  cactus run <model>                   opens playground for the model
                                       auto downloads and spins up

    Optional flags:
    --precision INT4|INT8|FP16         default: INT4
    --token <token>                    HF token (for gated models)
    --reconvert                        force model weights reconversion from source

  -----------------------------------------------------------------

  cactus transcribe [model]            live microphone transcription
                                       default model: parakeet-ctc-1.1b

    Optional flags:
    --file <audio.wav>                 transcribe audio file instead of mic
    --precision INT4|INT8|FP16         default: INT4
    --token <token>                    HF token (for gated models)
    --reconvert                        force model weights reconversion from source

    Examples:
    cactus transcribe                  live microphone transcription
    cactus transcribe --file audio.wav transcribe single file
    cactus transcribe nvidia/parakeet-ctc-1.1b   use different model
    cactus transcribe nvidia/parakeet-ctc-1.1b --file audio.wav

   -----------------------------------------------------------------

  cactus download <model>              downloads model to ./weights
                                       see supported weights on ReadMe

    Optional flags:
    --precision INT4|INT8|FP16         quantization (default: INT4)
    --token <token>                    HuggingFace API token
    --reconvert                        force model weights reconversion from source

  -----------------------------------------------------------------

  cactus convert <model> [output_dir]  converts model to custom directory
                                       supports LoRA adapter merging

    Optional flags:
    --precision INT4|INT8|FP16   quantization (default: INT4)
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
    --transcribe_model <model>         default: UsefulSensors/moonshine-base
    --benchmark                        use larger models (LFM2.5-VL-1.6B + nvidia/parakeet-ctc-1.1b)
    --precision INT4|INT8|FP16         regenerates weights with precision
    --reconvert                        force model weights reconversion from source
    --no-rebuild                       skip building library and tests
    --only <test_name>                 run specific test (llm, vlm, stt, embed, rag, graph, index, kernel, kv_cache, performance, etc)
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
    download_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT4',
                                 help='Quantization precision (default: INT4)')
    download_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    download_parser.add_argument('--token', help='HuggingFace API token')
    download_parser.add_argument('--reconvert', action='store_true',
                                 help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')

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
    run_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT4',
                            help='Quantization precision (default: INT4)')
    run_parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    run_parser.add_argument('--token', help='HuggingFace API token')
    run_parser.add_argument('--no-cloud-tele', action='store_true',
                            help='Disable cloud telemetry (write to cache only)')
    run_parser.add_argument('--reconvert', action='store_true',
                            help='Download original model and convert (instead of using pre-converted from Cactus-Compute)')

    transcribe_parser = subparsers.add_parser('transcribe', help='Download ASR model and run transcription')
    transcribe_parser.add_argument('model_id', nargs='?', default=DEFAULT_ASR_MODEL_ID,
                                   help=f'HuggingFace model ID (default: {DEFAULT_ASR_MODEL_ID})')
    transcribe_parser.add_argument('--file', dest='audio_file', default=None,
                                   help='Audio file to transcribe (WAV format). Omit for live microphone.')
    transcribe_parser.add_argument('--language', default='en',
                                   help='Language code for transcription (default: en). Examples: es, fr, de, zh, ja')
    transcribe_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT4',
                                   help='Quantization precision (default: INT4)')
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
    eval_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT4',
                             help='Quantization precision (default: INT4)')
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
    test_parser.add_argument('--model', default='LiquidAI/LFM2-VL-450M',
                             help='Model to use for tests')
    test_parser.add_argument('--transcribe_model', default=DEFAULT_TEST_TRANSCRIBE_MODEL_ID,
                             help='Transcribe model to use')
    test_parser.add_argument('--vad_model', default='snakers4/silero-vad',
                             help='VAD model to use')
    test_parser.add_argument('--benchmark', action='store_true',
                             help='Use larger models (LFM2.5-VL-1.6B + nvidia/parakeet-ctc-1.1b)')
    test_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'],
                             help='Regenerate weights with this precision (deletes existing weights)')
    test_parser.add_argument('--no-rebuild', action='store_true',
                             help='Skip building cactus library and tests')
    test_parser.add_argument('--token', help='HuggingFace API token')
    test_parser.add_argument('--android', action='store_true',
                             help='Run tests on Android')
    test_parser.add_argument('--ios', action='store_true',
                             help='Run tests on iOS')
    test_parser.add_argument('--exhaustive', action='store_true',
                             help='Run exhaustive golden tests for all model families and precisions')
    test_parser.add_argument('--only', help='Only run the specified test (llm, vlm, stt, embed, rag, graph, index, kernel, kv_cache, performance, etc)')
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

    convert_parser = subparsers.add_parser('convert', help='Convert model to custom output directory')
    convert_parser.add_argument('model_name', help='HuggingFace model name')
    convert_parser.add_argument('output_dir', nargs='?', default=None,
                                help='Output directory (default: weights/<model_name>)')
    convert_parser.add_argument('--precision', choices=['INT4', 'INT8', 'FP16'], default='INT4',
                                help='Quantization precision (default: INT4)')
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
    elif args.command == 'auth':
        sys.exit(cmd_auth(args))
    elif args.command == 'clean':
        sys.exit(cmd_clean(args))
    elif args.command == 'convert':
        sys.exit(cmd_convert(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
