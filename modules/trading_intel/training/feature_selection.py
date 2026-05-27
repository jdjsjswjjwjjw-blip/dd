"""
modules/trading_intel/training/feature_selection.py
═══════════════════════════════════════════════════════════════════════════════
Feature-selection helpers used by both training entry points
(modules/trading_intel/training/train_hybrid.py AND
tools/run_walk_forward_fold_enhanced.py).

Single source of truth for: applying a drop list emitted by
tools/audit_feature_redundancy.py.

Usage in a trainer:

    from modules.trading_intel.training.feature_selection import (
        load_drop_list_from_audit, apply_drop_list,
    )

    drop_set = load_drop_list_from_audit("audit_results/redundancy_summary.json")
    feature_cols = apply_drop_list(feature_cols, drop_set)
    # feature_cols is now the kept-after-dedup list to slice the parquet by
"""
from __future__ import annotations

import json
from pathlib import Path


class AuditFileError(ValueError):
    """Raised when the audit JSON is malformed or unusable."""


def load_drop_list_from_audit(audit_path: str | Path) -> set[str]:
    """Read the `drop_list` from a redundancy_summary.json emitted by
    `tools/audit_feature_redundancy.py`.

    Returns a set of column names to drop. Raises AuditFileError on
    schema mismatch or empty drop list (the latter is treated as an
    operator mistake — if you intended NO drop, just don't pass the
    flag at all).
    """
    p = Path(audit_path)
    if not p.exists():
        raise AuditFileError(f"audit file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise AuditFileError(f"audit file is not valid JSON: {p} — {e}")
    if "drop_list" not in data:
        raise AuditFileError(
            f"audit file {p} has no 'drop_list' key — "
            f"is this really a redundancy_summary.json?"
        )
    drop = data["drop_list"]
    if not isinstance(drop, list):
        raise AuditFileError(
            f"audit drop_list is not a list (got {type(drop).__name__})"
        )
    return set(str(x) for x in drop)


def apply_drop_list(
    feature_cols: list[str], drop_set: set[str],
) -> tuple[list[str], list[str]]:
    """Filter `feature_cols`, removing anything in `drop_set`.

    Returns (kept_cols, dropped_cols_actually_seen). The second value
    is logged by the trainer for traceability — it's the intersection
    of drop_set with the input, NOT the full drop_set (in case the
    audit was run on a different snapshot of the parquet).
    """
    kept = [c for c in feature_cols if c not in drop_set]
    dropped = [c for c in feature_cols if c in drop_set]
    return kept, dropped


def validate_audit_against_features(
    audit_path: str | Path, available_cols: list[str],
) -> dict:
    """Sanity-check an audit file against the columns ACTUALLY present
    in the parquet. Useful to catch the case where the audit was run on
    an old/different dataset and now references columns that don't
    exist.

    Returns a diagnostic dict (does not raise unless the audit itself
    is malformed). The caller decides whether to fail on mismatch.
    """
    drop_set = load_drop_list_from_audit(audit_path)
    available = set(available_cols)
    matched = drop_set & available
    only_in_audit = drop_set - available
    return {
        "audit_path": str(audit_path),
        "n_drop_in_audit": len(drop_set),
        "n_drop_actually_in_features": len(matched),
        "n_in_audit_but_missing_from_features": len(only_in_audit),
        "missing_from_features": sorted(only_in_audit),
        # If too many are missing, the audit is likely stale
        "stale_audit": len(only_in_audit) > len(drop_set) // 2,
    }
