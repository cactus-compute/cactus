import os
import subprocess
import platform
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    DEFAULT_TEST_MODEL_ID,
    check_command,
    run_command,
    get_weights_dir,
    print_color,
    RED, GREEN, YELLOW, BLUE,
)
from .download import cmd_download

DEFAULT_ASR_MODEL_ID = DEFAULT_TEST_MODEL_ID


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

    ios_script = PROJECT_ROOT / "tests" / "ios" / "test.sh"
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
    from .common import prompt_for_api_key

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
