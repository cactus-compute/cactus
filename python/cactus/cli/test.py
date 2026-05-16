import os
import subprocess

from .common import (
    PROJECT_ROOT,
    DEFAULT_TEST_MODEL_ID,
    print_color,
    RED, YELLOW, BLUE,
)
from .download import cmd_download


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

    if getattr(args, 'reconvert', False):
        model_id = getattr(args, 'model', DEFAULT_TEST_MODEL_ID)
        class DownloadArgs:
            pass
        dl_args = DownloadArgs()
        dl_args.model_id = model_id
        dl_args.reconvert = True
        dl_args.cache_dir = None
        if args.token:
            dl_args.token = args.token
        if cmd_download(dl_args) != 0:
            return 1

    test_filter = args.only
    for _test_name in ['llm', 'vlm', 'stt', 'embed', 'rag', 'graph', 'index', 'kernel', 'kv_cache', 'performance']:
        if getattr(args, _test_name, False):
            test_filter = _test_name
            break

    if test_filter == "kernel":
        test_script = PROJECT_ROOT / "cactus-kernels" / "test.sh"
        test_cwd = PROJECT_ROOT / "cactus-kernels"
    elif test_filter in ("graph", "kv_cache"):
        test_script = PROJECT_ROOT / "cactus-graph" / "test.sh"
        test_cwd = PROJECT_ROOT / "cactus-graph"
    else:
        test_script = PROJECT_ROOT / "cactus-engine" / "test.sh"
        test_cwd = PROJECT_ROOT / "cactus-engine"

    if not test_script.exists():
        print_color(RED, f"Error: Test script not found at {test_script}")
        return 1

    cmd = [str(test_script)]

    if args.model:
        cmd.extend(["--model", args.model])
    if getattr(args, 'no_rebuild', False):
        cmd.append("--no-rebuild")
    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")
    if test_filter:
        cmd.extend(["--only", test_filter])
    env = os.environ.copy()
    if getattr(args, 'enable_telemetry', False):
        env.pop("CACTUS_NO_CLOUD_TELE", None)
    else:
        env["CACTUS_NO_CLOUD_TELE"] = "1"

    result = subprocess.run(cmd, cwd=test_cwd, env=env)
    return result.returncode
