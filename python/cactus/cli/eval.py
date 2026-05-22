import sys
import os
import subprocess

from .common import (
    PROJECT_ROOT,
    print_color,
    RED, BLUE,
)


def cmd_eval(args):
    """Run evaluation scripts from the companion evals repo."""
    from .model import ensure_weights

    model_id = args.model_id

    if PROJECT_ROOT.parent.name != "evals":
        print_color(RED, "Skipping internal eval checks: companion repo not found.")
        return 1

    lib_path = PROJECT_ROOT / "cactus" / "build" / "libcactus.a"
    if not lib_path.exists():
        print_color(RED, "Error: Cactus library not built. Run 'cactus build' first.")
        return 1

    try:
        weights_dir = ensure_weights(
            model_id,
            token=args.token,
            cache_dir=args.cache_dir,
            reconvert=args.reconvert,
        )
    except RuntimeError as e:
        print_color(RED, str(e))
        return 1

    extra = getattr(args, "extra_args", None) or []

    def extra_has_flag(flag):
        return any(a == flag or a.startswith(flag + "=") for a in extra)

    mode = args.suite
    repo_root = PROJECT_ROOT.parent

    eval_runners = {
        "tools": repo_root / "tool-evals" / "run_eval_berk.py",
        "stt":   repo_root / "speech-evals" / "speech_eval.py",
        "llm":   repo_root / "text-evals" / "perplexity_eval.py",
        "vlm":   repo_root / "video-evals" / "run_benchmarks.py",
    }
    eval_runner = eval_runners.get(mode)
    if eval_runner is None:
        print_color(RED, f"Error: eval mode '{mode}' is not supported")
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
        cmd += ["--dataset-path", str(repo_root / "speech-evals" / "dataset-retrieval")]

    default_output_dirs = {
        "tools": repo_root / "tool-evals" / "results",
        "stt":   repo_root / "speech-evals" / "results",
        "llm":   repo_root / "text-evals" / "results",
    }
    if not extra_has_flag("--output-dir") and mode in default_output_dirs:
        cmd += ["--output-dir", str(default_output_dirs[mode])]

    cmd += extra

    print_color(BLUE, f"[cactus] launching {mode} eval runner")
    print(" ".join(cmd))

    env = os.environ.copy()
    if args.no_cloud_tele:
        env["CACTUS_NO_CLOUD_TELE"] = "1"
    if mode == "vlm":
        ffi_dir = str(repo_root / "cactus" / "tools" / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ffi_dir if not existing else (ffi_dir + os.pathsep + existing)

    return subprocess.run(cmd, cwd=str(repo_root), env=env).returncode
