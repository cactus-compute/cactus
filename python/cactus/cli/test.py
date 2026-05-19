import os
import subprocess

from .common import (
    PROJECT_ROOT,
    DEFAULT_TEST_MODEL_ID,
    get_weights_dir,
    print_color,
    RED, YELLOW, BLUE,
)
from .download import cmd_download


def cmd_test(args):
    """Run the Cactus test suite."""
    print_color(BLUE, "Running test suite...")
    print("=" * 20)

    model_id = getattr(args, 'model', DEFAULT_TEST_MODEL_ID)

    if getattr(args, 'ios', False) and not getattr(args, 'reconvert', False):
        print_color(
            YELLOW,
            "Warning: iOS tests without --reconvert may use stale or inconsistent local weights. "
            "If tests fail unexpectedly, rerun with --reconvert."
        )

    from types import SimpleNamespace
    dl_args = SimpleNamespace(
        model_id=model_id,
        reconvert=getattr(args, 'reconvert', False),
        cache_dir=None,
        token=getattr(args, 'token', None),
    )
    if cmd_download(dl_args) != 0:
        print_color(RED, "Failed to download model weights")
        return 1

    test_filter = getattr(args, 'only', None)
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

    weights_dir = get_weights_dir(model_id)
    cmd.extend(["--model", str(weights_dir)])

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
