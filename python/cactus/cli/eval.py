import sys
import os
import subprocess

from .common import (
    PROJECT_ROOT,
    DEFAULT_MODEL_ID,
    get_effective_weights_dir,
    print_color,
    RED, BLUE,
)
from .download import cmd_download


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
    dlargs.cache_dir = getattr(args, 'cache_dir', None)
    dlargs.token = getattr(args, 'token', None)
    dlargs.reconvert = getattr(args, 'reconvert', False)

    download_result = cmd_download(dlargs)
    if download_result != 0:
        return download_result

    weights_dir = get_effective_weights_dir(model_id, args)
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
