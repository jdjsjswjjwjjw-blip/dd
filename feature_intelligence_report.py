from __future__ import annotations

from pathlib import Path
from runpy import run_path
import sys


def _source_path() -> Path:
    return Path.home() / "Downloads" / "feature_intelligence_report.py"


def main() -> None:
    src = _source_path()
    if not src.exists():
        print(f"[ERROR] Missing source script: {src}")
        sys.exit(1)
    run_path(str(src), run_name="__main__")


if __name__ == "__main__":
    main()
