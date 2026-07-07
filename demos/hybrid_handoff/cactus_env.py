"""Environment + path bootstrap for the hybrid-handoff demo.

Importing this module (and calling setup_cactus_path) makes the in-repo `cactus`
package importable despite the stray top-level ./cactus dir, and loads the
untracked .env so CACTUS_CLOUD_KEY reaches the engine's cloud-handoff path.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_PKG = os.path.join(REPO_ROOT, "python")

DEFAULT_BUNDLE = os.environ.get("CACTUS_DEMO_BUNDLE") or os.path.join(REPO_ROOT, "weights", "gemma-4")

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def setup_cactus_path() -> None:
    """Ensure `import cactus` resolves to python/cactus, not the stray ./cactus,
    and select the inference backend so both TUIs run on the GPU.

    Backend comes from CACTUS_DEMO_BACKEND (default "auto" = Metal on Apple
    Silicon); the FFI otherwise defaults to CPU. Ignored on builds without a
    Metal backend (cactus_set_backend just fails and CPU stays selected).
    """
    if PYTHON_PKG not in sys.path:
        sys.path.insert(0, PYTHON_PKG)
    for bad in ("", REPO_ROOT, os.getcwd()):
        while bad in sys.path:
            sys.path.remove(bad)
    from cactus import cactus_set_backend
    cactus_set_backend(os.environ.get("CACTUS_DEMO_BACKEND", "auto"))


def load_env(path: str = _ENV_PATH) -> None:
    """Tiny .env reader (no python-dotenv dep). Lines of KEY=VALUE; # comments.

    Existing environment variables win (so an exported key isn't overwritten).
    Then set cloud defaults. We intentionally do NOT set CACTUS_CLOUD_STRICT_SSL,
    leaving SSL verification OFF -- required for the IP-based cloud endpoint.
    """
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    os.environ.setdefault("CACTUS_CLOUD_MODEL", "gemini-2.5-flash")
    os.environ.setdefault("CACTUS_CLOUD_API_BASE", "https://104.198.76.3/api/v1")


def have_cloud_key() -> bool:
    return bool(os.environ.get("CACTUS_CLOUD_KEY") or os.environ.get("CACTUS_CLOUD_API_KEY"))


_LOG_NOISE = ("[WARN] [cloud_handoff]", "[WARN] [tool_rag]")


@contextlib.contextmanager
def quiet_engine_logs():
    """Filter the engine's OS-level stderr (fd 2) to drop routing log noise
    (cloud_handoff / tool_rag WARN lines the C++ engine prints each turn) while
    letting genuine errors through. CLI-local: no engine change, no rebuild."""
    sys.stderr.flush()
    saved_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _pump() -> None:
        with os.fdopen(read_fd, "r", errors="replace") as reader:
            for line in reader:
                if not any(token in line for token in _LOG_NOISE):
                    os.write(saved_fd, line.encode("utf-8", "replace"))

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        thread.join(timeout=2)
        os.close(saved_fd)
