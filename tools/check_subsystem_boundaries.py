#!/usr/bin/env python3
"""
tools/check_subsystem_boundaries.py
═══════════════════════════════════════════════════════════════════════════════
Lint check that enforces the subsystem boundaries documented in
`SUBSYSTEMS.md`. Run before every commit (or wire into pre-commit).

Boundaries enforced:
  1. day_trade pipeline (prepare_day_trading.py + helpers in modules/*.py at
     root level) must NOT import from modules.deep_lob or
     modules.price_cycle or modules.trading_intel
  2. modules.deep_lob must NOT import prepare_day_trading or
     modules.trading_intel
  3. modules.trading_intel must NOT import prepare_day_trading
  4. self_supervised/ scripts may read day_trade outputs (parquet/npy) but
     must NOT import prepare_day_trading code (only regime_config is OK
     because it's pure constants)

Exit code 0 = clean. Exit code 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Helpers shared across subsystems (pure feature engineering, not SSL models).
# These are explicitly allowed even when an otherwise-forbidden import path
# would match. Document each exception below with a one-line reason.
SHARED_HELPER_ALLOWLIST: set[str] = {
    # day_trade adds 12 Wyckoff/swing/fractal features by calling these
    # — pure numpy feature engineering, NOT the SSL training code
    "modules.price_cycle.data_structures",   # BarSequence numpy container
    "modules.price_cycle.feature_pipeline",  # build_cycle_features (causal)
}


# (regex_to_match, forbidden_import_pattern, reason)
FORBIDDEN_IMPORTS: list[tuple[str, str, str]] = [
    # day_trade pipeline must not depend on SSL training code or trading_intel.
    # Specific submodules are forbidden — SHARED_HELPER_ALLOWLIST exempts
    # pure feature-engineering helpers.
    (
        r"^prepare_day_trading\.py$",
        r"^(from|import) modules\.(deep_lob|price_cycle|trading_intel)",
        "day_trade rule pipeline must not import SSL training code or trading_intel "
        "(except items in SHARED_HELPER_ALLOWLIST)",
    ),
    # SSL backbone must not depend on day_trade or trading_intel
    (
        r"^modules/deep_lob/.*\.py$",
        r"^(from|import) (prepare_day_trading|modules\.trading_intel)",
        "deep_lob backbone must not import day_trade or trading_intel",
    ),
    (
        r"^modules/price_cycle/.*\.py$",
        r"^(from|import) (prepare_day_trading|modules\.trading_intel)",
        "price_cycle backbone must not import day_trade or trading_intel",
    ),
    # trading_intel must not depend on day_trade code (only its data outputs)
    (
        r"^modules/trading_intel/.*\.py$",
        r"^(from|import) prepare_day_trading",
        "trading_intel must consume day_trade OUTPUT (parquet/npy), "
        "not import its code",
    ),
    # self_supervised scripts must not import day_trade code
    # (regime_config is allowed because it's pure constants)
    (
        r"^self_supervised/.*\.py$",
        r"^(from|import) prepare_day_trading",
        "self_supervised must read day_trade outputs, not import its code",
    ),
]


def _imports_shared_helper(line: str) -> bool:
    """True iff the line imports a symbol from the SHARED_HELPER_ALLOWLIST."""
    line = line.strip()
    for allowed_module in SHARED_HELPER_ALLOWLIST:
        # `from <allowed_module> import X`  OR  `import <allowed_module>`
        if line.startswith(f"from {allowed_module} ") or line == f"import {allowed_module}":
            return True
    return False


def check_file(rel_path: Path, content: str, abs_path: Path) -> list[str]:
    """Return list of violation messages for this file."""
    violations: list[str] = []
    rel_str = rel_path.as_posix()
    for path_pattern, import_pattern, reason in FORBIDDEN_IMPORTS:
        if not re.match(path_pattern, rel_str):
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not re.match(import_pattern, stripped):
                continue
            # Skip lines that import items in the shared-helper allowlist
            if _imports_shared_helper(stripped):
                continue
            violations.append(
                f"{rel_str}:{line_no}: {stripped}\n"
                f"    → {reason}"
            )
    return violations


def main() -> int:
    violations: list[str] = []
    # Walk all .py files in repo (excluding _archive, __pycache__, tests)
    skip_dirs = {"_archive", "__pycache__", ".git", "tests", "_workspace"}
    for py in REPO_ROOT.rglob("*.py"):
        parts = py.relative_to(REPO_ROOT).parts
        if any(p in skip_dirs for p in parts):
            continue
        rel = py.relative_to(REPO_ROOT)
        try:
            content = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        violations.extend(check_file(rel, content, py))

    if not violations:
        print("✅ Subsystem boundaries clean — no forbidden imports detected.")
        return 0

    print("❌ Subsystem boundary violations:")
    print()
    for v in violations:
        print(v)
        print()
    print(f"Total: {len(violations)} violation(s)")
    print()
    print("See SUBSYSTEMS.md for the boundary rules.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
