import os
import subprocess

from .common import (
    PROJECT_ROOT,
    print_color,
    RED, YELLOW, BLUE,
)


def cmd_test(args):
    """Run the Cactus test suite."""
    from .model import ensure_weights

    print_color(BLUE, "Running test suite...")
    print("=" * 20)

    model_id = args.model

    if args.ios and not args.reconvert:
        print_color(
            YELLOW,
            "Warning: iOS tests without --reconvert may use stale or inconsistent local weights. "
            "If tests fail unexpectedly, rerun with --reconvert."
        )

    try:
        weights_dir = ensure_weights(
            model_id,
            token=args.token,
            reconvert=args.reconvert,
        )
    except RuntimeError as e:
        print_color(RED, f"Failed to download model weights: {e}")
        return 1

    test_filter = args.only
    for name in ("llm", "vlm", "stt", "embed", "rag", "graph", "index", "kernel", "kv_cache", "performance"):
        if getattr(args, name, False):
            test_filter = name
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

    cmd = [str(test_script), "--model", str(weights_dir)]

    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")
    if test_filter:
        cmd.extend(["--only", test_filter])

    env = os.environ.copy()
    if args.enable_telemetry:
        env.pop("CACTUS_NO_CLOUD_TELE", None)
    else:
        env["CACTUS_NO_CLOUD_TELE"] = "1"

    return subprocess.run(cmd, cwd=test_cwd, env=env).returncode
