"""Stamped by CI before building; dev fallback reads CACTUS_VERSION."""
from pathlib import Path as _Path


def _dev_version():
    vfile = _Path(__file__).resolve().parent.parent.parent / "CACTUS_VERSION"
    if vfile.exists():
        raw = vfile.read_text().strip().lstrip("v")
        return raw + ".0" if raw.count(".") == 1 else raw
    return "0.0.0+dev"


__version__ = _dev_version()
