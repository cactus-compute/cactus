from __future__ import annotations

import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cactus.transpile.hf_model import main as transpile_main


def main() -> int:
    if "--task" not in sys.argv[1:]:
        sys.argv[1:1] = ["--task", "causal_lm_logits"]
    return transpile_main()


if __name__ == "__main__":
    raise SystemExit(main())
