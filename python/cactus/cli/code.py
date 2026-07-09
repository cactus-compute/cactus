import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from .common import BLUE, DEFAULT_MODEL_ID, GREEN, RED, YELLOW, is_valid_bundle, print_color, weights_root

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent


def _find_agent_cli() -> Path | None:
    candidates = (
        _PACKAGE_DIR / "code" / "node_modules" / "@earendil-works" / "pi-coding-agent" / "dist" / "cli.js",
        _PACKAGE_DIR / "code" / "packages" / "coding-agent" / "dist" / "cli.js",
        _PROJECT_ROOT / "cactus-code" / "packages" / "coding-agent" / "dist" / "cli.js",
    )
    for cli in candidates:
        if cli.exists():
            return cli
    return None


def _ensure_agent_built() -> Path | None:
    cli = _find_agent_cli()
    if cli is not None:
        return cli
    source_dir = _PROJECT_ROOT / "cactus-code"
    if not (source_dir / "package.json").exists():
        return None
    npm = shutil.which("npm")
    if not npm:
        print_color(RED, "Error: npm is required to build the Cactus coding agent on first run.")
        print_color(YELLOW, "Install Node.js >= 22 (includes npm) from https://nodejs.org and retry.")
        return None
    print_color(BLUE, "Building the Cactus coding agent (first run, this can take a minute) ...")
    try:
        subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=source_dir, check=True)
        subprocess.run([npm, "run", "build"], cwd=source_dir, check=True)
    except subprocess.CalledProcessError:
        print_color(RED, "Error: failed to build the Cactus coding agent.")
        return None
    return _find_agent_cli()


def _discover_local_model() -> str | None:
    try:
        root = weights_root()
        if not root.is_dir():
            return None
        bundles = sorted(p for p in root.iterdir() if p.is_dir() and is_valid_bundle(p))
        return str(bundles[0]) if bundles else None
    except Exception:
        return None


def _clear_console() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[3J\033[H")
        sys.stdout.flush()


def _base_url(host: str, port: int) -> str:
    env = os.environ.get("CACTUS_BASE_URL")
    if env:
        return env.rstrip("/")
    return f"http://{host}:{port}/v1"


def _server_alive(base_url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _echo_log_tail(log_path: Path, pos: int) -> int:
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            new_pos = f.tell()
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return new_pos
    except Exception:
        return pos


def _wait_for_server(base_url: str, proc: "subprocess.Popen | None" = None, timeout: float = 1800.0,
                     log_path: "Path | None" = None) -> bool:
    deadline = time.time() + timeout
    pos = 0
    while time.time() < deadline:
        if log_path is not None:
            pos = _echo_log_tail(log_path, pos)
        if _server_alive(base_url):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def cmd_code(args) -> int:
    node = shutil.which("node")
    if not node:
        print_color(RED, "Error: Node.js (>=22) is required to run `cactus code`.")
        print_color(YELLOW, "Install it from https://nodejs.org and try again.")
        return 1

    agent_was_built = _find_agent_cli() is None
    agent_cli = _ensure_agent_built()
    if agent_cli is None:
        print_color(RED, "Error: the Cactus coding agent is not available in this installation.")
        return 1

    base_url = _base_url(args.host, args.port)
    started_server: subprocess.Popen | None = None

    if not _server_alive(base_url):
        if args.no_serve or os.environ.get("CACTUS_BASE_URL"):
            print_color(RED, f"Error: no Cactus server reachable at {base_url}.")
            print_color(YELLOW, "Start one with `cactus serve <model>` first.")
            return 1
        model_flags_given = (
            getattr(args, "reconvert", False)
            or getattr(args, "bits", 4) != 4
            or bool(getattr(args, "token", None))
        )
        if args.serve_model:
            serve_model = args.serve_model
        elif model_flags_given:
            serve_model = DEFAULT_MODEL_ID
        else:
            serve_model = _discover_local_model() or DEFAULT_MODEL_ID
        if serve_model == DEFAULT_MODEL_ID and not args.serve_model:
            print_color(BLUE, f"No model specified; using the default ({DEFAULT_MODEL_ID}), downloading on first run ...")
        print_color(BLUE, f"Starting Cactus server with {serve_model} ...")
        serve_cmd = [
            sys.executable, "-m", "cactus", "serve", serve_model,
            "--host", args.host, "--port", str(args.port), "--no-access-log",
            "--bits", str(getattr(args, "bits", 4)),
        ]
        if getattr(args, "backend", None):
            serve_cmd += ["--backend", args.backend]
        if getattr(args, "token", None):
            serve_cmd += ["--token", args.token]
        if getattr(args, "reconvert", False):
            serve_cmd.append("--reconvert")
        if getattr(args, "no_cloud_handoff", False):
            serve_cmd.append("--no-cloud-handoff")
        if getattr(args, "confidence_threshold", None) is not None:
            serve_cmd += ["--confidence-threshold", str(args.confidence_threshold)]
        if getattr(args, "cloud_timeout_ms", None) is not None:
            serve_cmd += ["--cloud-timeout-ms", str(args.cloud_timeout_ms)]
        serve_log_path = Path(tempfile.gettempdir()) / "cactus-code-serve.log"
        serve_log = open(serve_log_path, "w")
        started_server = subprocess.Popen(serve_cmd, stdout=serve_log, stderr=subprocess.STDOUT)
        if not _wait_for_server(base_url, started_server, log_path=serve_log_path):
            print_color(RED, "Error: the Cactus server did not become ready in time.")
            print_color(YELLOW, f"See server logs: {serve_log_path}")
            started_server.terminate()
            return 1
        print_color(GREEN, f"Cactus server ready (logs: {serve_log_path})")

    env = dict(os.environ)
    env["CACTUS_BASE_URL"] = base_url
    env["PI_PACKAGE_DIR"] = str(agent_cli.parent.parent)
    agent_args = [a for a in (args.agent_args or []) if a != "--"]
    interactive = not any(a in ("-p", "--print") for a in agent_args)
    if interactive and (started_server is not None or agent_was_built):
        _clear_console()
    try:
        return subprocess.run([node, str(agent_cli), *agent_args], env=env).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        if started_server is not None:
            started_server.terminate()
            try:
                started_server.wait(timeout=10)
            except Exception:
                started_server.kill()
