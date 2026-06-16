import os
import subprocess

from .common import (
    BLUE, DEFAULT_TEST_MODEL_ID, DEFAULT_TEST_TRANSCRIPTION_MODEL_ID,
    PROJECT_ROOT, RED, apply_cloud_api_key_env, print_color,
)

COMPONENTS = ("kernels", "graph", "engine", "all")


def _list_component_suites(component):
    tests_dir = PROJECT_ROOT / f"cactus-{component}" / "tests"
    if not tests_dir.exists():
        return []
    return sorted(
        f.stem.removeprefix("test_")
        for f in tests_dir.glob("test_*.cpp")
        if f.stem != "test_utils"
    )


def _components_with_suite(suite):
    return tuple(c for c in ("kernels", "graph", "engine") if suite in _list_component_suites(c))


def _component_args(component, args):
    cmd = [str(PROJECT_ROOT / f"cactus-{component}" / "test.sh")]
    if args.suite:
        cmd.extend(["--suite", args.suite])
    if component == "engine":
        cmd.extend(["--model", args.model_id])
        cmd.extend(["--transcription-model", args.transcription_model_id])
        if args.android:
            cmd.append("--android")
        if args.ios:
            cmd.append("--ios")
    return cmd


def cmd_test(args):
    if getattr(args, "list", False):
        print_color(BLUE, "Components:")
        for c in COMPONENTS:
            print(f"  {c}")
        print_color(BLUE, "\nSuites by component:")
        for c in ("kernels", "graph", "engine"):
            suites = _list_component_suites(c)
            if suites:
                print(f"  {c}: {', '.join(suites)}")
        return 0

    if args.suite:
        matches = _components_with_suite(args.suite)
        if not matches:
            print_color(RED, f"unknown suite '{args.suite}'.")
            print_color(RED, "Run `cactus test --list` to see available suites.")
            return 2
        if args.component != "all" and args.component not in matches:
            print_color(RED,
                f"suite '{args.suite}' does not exist in component '{args.component}'. "
                f"Found in: {', '.join(matches)}.")
            return 2
        targets = matches if args.component == "all" else (args.component,)
    else:
        targets = ("kernels", "graph", "engine") if args.component == "all" else (args.component,)

    if "engine" in targets:
        from .model import ensure_runnable_bundle
        try:
            args.model_id = str(ensure_runnable_bundle(args.model_id or DEFAULT_TEST_MODEL_ID))
            args.transcription_model_id = str(
                ensure_runnable_bundle(args.transcription_model_id or DEFAULT_TEST_TRANSCRIPTION_MODEL_ID)
            )
        except RuntimeError as exc:
            print_color(RED, f"Model setup failed: {exc}")
            return 1

    apply_cloud_api_key_env()
    env = os.environ.copy()
    if args.enable_telemetry:
        env.pop("CACTUS_NO_CLOUD_TELE", None)
    else:
        env["CACTUS_NO_CLOUD_TELE"] = "1"

    for c in targets:
        cmd = _component_args(c, args)
        print_color(BLUE, f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if rc != 0:
            return rc
    return 0
