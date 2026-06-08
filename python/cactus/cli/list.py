import stat as _stat
from pathlib import Path

from .common import BLUE, CYAN, print_color, transpiled_root, weights_root


def _read_config(path):
    pairs = (
        line.split("=", 1)
        for line in path.read_text(errors="ignore").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    return {k.strip(): v.strip() for k, v in pairs}


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _dir_size(path):
    total = 0
    for f in path.rglob("*"):
        try:
            st = f.lstat()
        except OSError:
            continue
        if _stat.S_ISREG(st.st_mode):
            total += st.st_size
    return total


def _categorize(roots):
    converted, transpiled = [], []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or not (p / "config.txt").is_file():
                continue
            cfg = _read_config(p / "config.txt")
            entry = (p, cfg.get("model_type", "?"), _dir_size(p))
            if (p / "components" / "manifest.json").is_file():
                transpiled.append(entry)
            else:
                converted.append(entry)
    return converted, transpiled


def _print_section(title: str, color: str, entries):
    print_color(color, title)
    if not entries:
        print("  (none)")
        return
    width = max(len(p.name) for p, _, _ in entries)
    for p, model_type, size in entries:
        print(f"  {p.name:<{width}}  {model_type:<12}  {_human_size(size):>10}  {p.parent}")


def cmd_list(_args):
    roots = (weights_root(), transpiled_root())
    converted, transpiled = _categorize(roots)
    _print_section("Converted weights (cactus convert)", BLUE, converted)
    print()
    _print_section("Transpiled bundles (cactus transpile / download)", CYAN, transpiled)
    return 0
