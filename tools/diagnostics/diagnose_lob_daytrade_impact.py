from __future__ import annotations

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pathlib import Path
from runpy import run_path
import sys


def _source_path() -> Path:
    return Path.home() / "Downloads" / "diagnose_lob_daytrade_impact.py"


def main() -> None:
    src = _source_path()
    if not src.exists():
        print(f"[ERROR] Missing source script: {src}")
        sys.exit(1)
    run_path(str(src), run_name="__main__")


if __name__ == "__main__":
    main()
