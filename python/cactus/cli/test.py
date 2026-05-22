import os
import subprocess

from .common import (
    PROJECT_ROOT,
    print_color,
    RED, YELLOW, BLUE,
)

LAYERS = {
    "kernel": PROJECT_ROOT / "cactus-kernels",
    "graph":  PROJECT_ROOT / "cactus-graph",
}

ENGINE_DIR = PROJECT_ROOT / "cactus-engine"


def discover_suites():
    suites = []
    for name, path in LAYERS.items():
        if (path / "test.sh").exists():
            suites.append(name)
    for f in sorted((ENGINE_DIR / "tests").glob("test_*.cpp")):
        name = f.stem.removeprefix("test_")
        if name != "utils":
            suites.append(name)
    return suites


def cmd_test(args):
    from .model import ensure_weights

    print_color(BLUE, "Running test suite...")

    if args.ios and not args.reconvert:
        print_color(YELLOW, "Warning: iOS tests without --reconvert may use stale weights.")

    try:
        weights_dir = ensure_weights(args.model, token=args.token, reconvert=args.reconvert)
    except RuntimeError as e:
        print_color(RED, f"Failed to download model weights: {e}")
        return 1

    suite = args.suite
    test_cwd = LAYERS.get(suite, ENGINE_DIR)
    test_script = test_cwd / "test.sh"

    if not test_script.exists():
        print_color(RED, f"Test script not found: {test_script}")
        return 1

    cmd = [str(test_script), "--model", str(weights_dir)]
    if args.android:
        cmd.append("--android")
    if args.ios:
        cmd.append("--ios")
    if suite:
        cmd.extend(["--only", suite])

    env = os.environ.copy()
    if args.enable_telemetry:
        env.pop("CACTUS_NO_CLOUD_TELE", None)
    else:
        env["CACTUS_NO_CLOUD_TELE"] = "1"

    return subprocess.run(cmd, cwd=test_cwd, env=env).returncode
