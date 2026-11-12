"""Keep optional Apple inference dependencies out of the base package import."""
import importlib.util
import platform
import sys


def require_runtime():
    if sys.version_info < (3, 12):
        raise RuntimeError("Cactus MLX inference requires Python 3.12 or newer.")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Cactus MLX inference requires an Apple silicon Mac.")
    version = platform.mac_ver()[0]
    if not version or int(version.split(".")[0]) < 26:
        raise RuntimeError("Cactus MLX inference requires macOS 26 or newer.")
    missing = [name for name in ("mlx", "mlx_lm", "transformers", "numpy")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Install MLX inference dependencies with: pip install -U 'cactus-compute[mlx]'"
        )
