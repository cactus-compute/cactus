import os
import subprocess

from .common import (
    BLUE, DEFAULT_MODEL_ID, DEFAULT_TRANSCRIPTION_MODEL_ID, PROJECT_ROOT, RED,
    YELLOW, apply_cloud_api_key_env, print_color, weights_root,
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
        cmd.extend(["--model", args.model_id or DEFAULT_MODEL_ID])
        cmd.extend(["--transcription-model", args.transcription_model_id or DEFAULT_TRANSCRIPTION_MODEL_ID])
        if args.android:
            cmd.append("--android")
        if args.ios:
            cmd.append("--ios")
    return cmd


def _bundle_dir(model_id):
    name = model_id.rsplit("/", 1)[-1].lower()
    return weights_root() / name


def _has_bundle(model_id):
    return (_bundle_dir(model_id) / "components" / "manifest.json").exists()


def _ensure_bundle(model_id):
    if _has_bundle(model_id):
        return True

    class _Args:
        pass

    # Try pre-built download first
    print_color(YELLOW, f"Bundle not found for {model_id}, trying download...")
    from .download import cmd_download
    dl = _Args()
    dl.model_id = model_id
    dl.bits = 4
    dl.platform = "cpu"
    dl.token = None
    if cmd_download(dl) == 0 and _has_bundle(model_id):
        return True

    # Fall back to convert + transpile
    print_color(YELLOW, f"Download unavailable, converting {model_id} from source...")
    from .convert import cmd_convert, cmd_transpile
    cv = _Args()
    cv.model_id = model_id
    cv.output_dir = None
    cv.bits = 4
    cv.token = None
    cv.lora = None
    cv.reconvert = False
    if cmd_convert(cv) != 0:
        print_color(RED, f"Failed to convert {model_id}")
        return False

    tp = _Args()
    tp.model_id = model_id
    tp.weights_dir = None
    tp.task = "auto"
    tp.prompt = None
    tp.system_prompt = None
    tp.enable_thinking = False
    tp.input_ids = None
    tp.image_file = []
    tp.audio_file = None
    tp.max_new_tokens = None
    tp.component_pipeline = "auto"
    tp.components = None
    tp.torch_dtype = None
    tp.token = None
    tp.trust_remote_code = False
    tp.local_files_only = False
    tp.allow_unconverted_weights = False
    tp.execute_after_transpile = False
    tp.artifact_dir = None
    tp.graph_filename = None
    tp.skip_reference_compare = False
    tp.no_fuse_rms_norm = False
    tp.no_fuse_rope = False
    tp.no_fuse_attention = False
    tp.no_fuse_attention_block = False
    tp.no_fuse_add_clipped = False
    tp.no_fuse_gated_deltanet = False
    tp.npu = False
    tp.npu_quantize = None
    tp.npu_audio_quantize = None
    tp.npu_vision_quantize = None
    tp.cache_context_length = None
    if cmd_transpile(tp) != 0:
        print_color(RED, f"Failed to transpile {model_id}")
        return False

    return _has_bundle(model_id)


def _ensure_engine_bundles(args):
    model_id = args.model_id or DEFAULT_MODEL_ID
    transcription_id = args.transcription_model_id or DEFAULT_TRANSCRIPTION_MODEL_ID
    for mid in (model_id, transcription_id):
        if not _ensure_bundle(mid):
            raise SystemExit(1)


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
        _ensure_engine_bundles(args)

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
