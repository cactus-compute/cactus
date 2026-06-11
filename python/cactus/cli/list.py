import itertools
import stat as _stat
import struct

from .common import BLUE, CYAN, print_color, transpiled_root, weights_root

_PREC_TO_BITS = {3: 1, 4: 2, 5: 3, 6: 4}


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


def _cq_precision(weights_file):
    """Read the precision code at byte offset 48 of a Cactus tensor header."""
    try:
        with weights_file.open("rb") as f:
            head = f.read(52)
    except OSError:
        return None
    if len(head) < 52 or head[:4] != b"CACT":
        return None
    return struct.unpack_from("<I", head, 48)[0]


def _quant_label(model_dir, *, sample_cap=64, scan_cap=2000):
    """Infer a model's CQ quantization level from its tensor headers.

    Scans `.weights` files (top-level converted weights, then nested
    bundle components) and returns the dominant CQ level, e.g. "CQ4".
    Returns "—" if no CQ-quantized tensors are present.
    """
    candidates = itertools.chain(
        model_dir.glob("*.weights"),
        (model_dir / "components").rglob("*.weights"),
    )
    counts = {}
    scanned = cq_hits = 0
    for wf in candidates:
        scanned += 1
        bits = _PREC_TO_BITS.get(_cq_precision(wf))
        if bits is not None:
            counts[bits] = counts.get(bits, 0) + 1
            cq_hits += 1
        if cq_hits >= sample_cap or scanned >= scan_cap:
            break
    if not counts:
        return "—"
    return f"CQ{max(counts, key=counts.get)}"


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
            entry = (p, cfg.get("model_type", "?"), _quant_label(p), _dir_size(p))
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
    width = max(len(p.name) for p, _, _, _ in entries)
    for p, model_type, quant, size in entries:
        print(f"  {p.name:<{width}}  {model_type:<12}  {quant:<5}  {_human_size(size):>10}  {p.parent}")


def cmd_list(_args):
    roots = (weights_root(), transpiled_root())
    converted, transpiled = _categorize(roots)
    _print_section("Converted weights (cactus convert)", BLUE, converted)
    print()
    _print_section("Transpiled bundles (cactus transpile / download)", CYAN, transpiled)
    return 0
