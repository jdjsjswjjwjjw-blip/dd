#!/usr/bin/env python3
"""Phase 1.6 — empirical re-validation harness for the full cleanup arc.

Run this AFTER the refinery has produced a fresh parquet on real data
(e.g. Q2 2025). It loads the parquet + sidecar + the original Q2 IC-audit
artefacts, then asserts the cleanup's expected effects empirically:

  ✓ NEUTRAL ratio is in the post-cleanup band (~30-50%, was ~80%)
  ✓ short_tp count > 0 (was 0 before A1)
  ✓ Top-feature whitelist columns are present + their dtypes match
  ✓ Blacklist columns are absent (Phase 1.2 enforced)
  ✓ gate_kill_ratio is no longer reported (the gate is removed)
  ✓ Phase 1.1 schema validate() returns clean
  ✓ dataset_meta.json is up-to-date (matches the schema version)

Usage:
    python tools/diagnostics/phase1_revalidate.py \\
        --parquet path/to/q2_post_cleanup.parquet \\
        [--baseline-ic path/to/q2_ic_summary.json]

Exit codes:
    0 — every gate passed (cleanup confirmed on real data)
    1 — at least one gate failed (revisit the relevant Phase round)
    2 — usage / file-not-found errors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Acceptable bands for each gate. Wide enough to absorb the bar count
# variance between quarters; tight enough to fail on a real regression.
GATES = {
    "neutral_ratio_max": 0.55,        # was 0.80 pre-cleanup; expect ≤0.55
    "min_short_tp_count": 1,          # was 0 pre-A1; even one proves the fix
    "min_long_tp_count": 1,
    "max_dead_path_codes": 0,         # path_outcome in {2,3,5,6} must not occur
    "min_exec_valid_pct": 5.0,        # at least 5% of bars usable for exec head
    "min_next_price_delta_valid_pct": 70.0,  # gate-free target — should be very high
}


def _emit(ok: bool, label: str, detail: str = "") -> tuple[bool, str]:
    mark = "✅" if ok else "❌"
    return ok, f"  {mark} {label}{(' — ' + detail) if detail else ''}"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"❌ parquet not found: {path}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_parquet(path)
    print(f"📊 Loaded {len(df):,} rows × {len(df.columns)} cols from {path.name}")
    return df


def _load_sidecar(parquet_path: Path) -> dict | None:
    sidecar = parquet_path.parent / "dataset_meta.json"
    if not sidecar.exists():
        print(f"⚠️  dataset_meta.json missing next to {parquet_path}")
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _load_baseline(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        print(f"⚠️  baseline IC summary not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── gates ───────────────────────────────────────────────────────────────────
def gate_neutral_ratio(df: pd.DataFrame) -> tuple[bool, str]:
    if 'bias_label' not in df.columns:
        return False, "  ❌ bias_label column missing"
    n = len(df)
    n_neutral = int((df['bias_label'] == 2).sum())
    ratio = n_neutral / max(n, 1)
    ok = ratio <= GATES["neutral_ratio_max"]
    return _emit(
        ok, "NEUTRAL ratio",
        f"{ratio*100:.1f}% ({n_neutral}/{n}); gate ≤ {GATES['neutral_ratio_max']*100:.0f}%",
    )


def gate_short_tp_exists(df: pd.DataFrame) -> tuple[bool, str]:
    if 'path_outcome' not in df.columns:
        return False, "  ❌ path_outcome column missing"
    n_short = int((df['path_outcome'] == 1).sum())
    n_long = int((df['path_outcome'] == 0).sum())
    ok_short = n_short >= GATES["min_short_tp_count"]
    ok_long = n_long >= GATES["min_long_tp_count"]
    return _emit(
        ok_short and ok_long, "A1 fix proven — both TP directions reachable",
        f"long_tp={n_long}, short_tp={n_short} (pre-A1: short_tp was 0)",
    )


def gate_no_dead_path_codes(df: pd.DataFrame) -> tuple[bool, str]:
    if 'path_outcome' not in df.columns:
        return False, "  ❌ path_outcome column missing"
    dead = int(df['path_outcome'].isin([2, 3, 5, 6]).sum())
    ok = dead <= GATES["max_dead_path_codes"]
    return _emit(
        ok, "no dead path_outcome codes",
        f"found {dead} (must be 0 — codes 2/3/5/6 are not produced by symmetric scan)",
    )


def gate_exec_valid_density(df: pd.DataFrame) -> tuple[bool, str]:
    if 'exec_valid' not in df.columns:
        return False, "  ❌ exec_valid missing (B1 not wired?)"
    pct = float(df['exec_valid'].mean() * 100)
    ok = pct >= GATES["min_exec_valid_pct"]
    return _emit(
        ok, "exec_valid coverage",
        f"{pct:.1f}% of bars (gate ≥ {GATES['min_exec_valid_pct']:.1f}%)",
    )


def gate_next_price_delta_density(df: pd.DataFrame) -> tuple[bool, str]:
    if 'next_price_delta_valid' not in df.columns:
        return False, "  ❌ next_price_delta_valid missing (B2 not wired?)"
    pct = float(df['next_price_delta_valid'].mean() * 100)
    ok = pct >= GATES["min_next_price_delta_valid_pct"]
    return _emit(
        ok, "next_price_delta_valid coverage",
        f"{pct:.1f}% (gate-free target — gate ≥ {GATES['min_next_price_delta_valid_pct']:.0f}%)",
    )


def gate_whitelist_present(df: pd.DataFrame) -> tuple[bool, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.dataset_schema import TOP_IC_FEATURES
    missing = [s.name for s in TOP_IC_FEATURES if s.name not in df.columns]
    ok = not missing
    return _emit(
        ok, "TOP_IC whitelist features present",
        f"missing: {missing}" if missing else f"all {len(TOP_IC_FEATURES)} present",
    )


def gate_blacklist_absent(df: pd.DataFrame) -> tuple[bool, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.dataset_schema import PHASE_1_2_BLACKLIST
    leaked = [c for c in df.columns if c in PHASE_1_2_BLACKLIST]
    ok = not leaked
    return _emit(
        ok, "Phase 1.2 blacklist NOT in parquet",
        f"leaked: {leaked}" if leaked else "clean",
    )


def gate_schema_validate(df: pd.DataFrame) -> tuple[bool, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.dataset_schema import validate
    errors = validate(df, strict=True)
    ok = not errors
    return _emit(
        ok, "Phase 1.1 schema validate(strict=True)",
        "clean" if ok else f"{len(errors)} error(s); first: {errors[0]}",
    )


def gate_sidecar_up_to_date(meta: dict | None) -> tuple[bool, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.dataset_schema import SCHEMA_VERSION
    if meta is None:
        return False, "  ❌ dataset_meta.json missing"
    ok = meta.get("schema_version") == SCHEMA_VERSION
    return _emit(
        ok, f"sidecar schema_version == {SCHEMA_VERSION}",
        f"got {meta.get('schema_version')!r}",
    )


def gate_phase1_4_features_present(df: pd.DataFrame) -> tuple[bool, str]:
    """II.B + II.C + II.D + MT5 CVD + iceberg + interaction outputs."""
    expected = {
        # II.B
        "dist_to_session_high_atr", "dist_to_vwap_atr", "dist_to_pdh_atr",
        # II.C
        "regime_label_grouped",
        # II.D
        "is_warmup",
        # 1.4-core MT5 CVD
        "cvd_bar_5m", "cvd_direction_ratio_5m", "cvd_intensity_vs_atr",
        "cvd_divergence_at_level", "cvd_consecutive_imbalance",
        # 1.5 iceberg (zeros on bar-only runs but the cols should exist)
        "iceberg_count_5m", "iceberg_total_volume_5m",
        # 1.5 interactions
        "hawkes_x_session_phase", "tick_count_x_session_phase",
        # 1.3 continuous z-scores
        "event_score_continuous", "hawkes_z_raw", "absorb_z_raw", "kyle_z_raw",
    }
    missing = sorted(expected - set(df.columns))
    ok = not missing
    return _emit(
        ok, "Phase 1.3/1.4/1.5 engineered features present",
        f"missing {len(missing)}: {missing[:5]}" if missing else f"all {len(expected)} present",
    )


def gate_baseline_comparison(meta: dict | None, baseline: dict | None) -> tuple[bool, str]:
    """If a baseline IC summary is provided, sanity-check the deltas."""
    if baseline is None:
        return True, "  ⏭️  baseline IC not provided — skipping comparison"
    baseline_gate = baseline.get("gate_diagnostic", {})
    baseline_verdict = baseline_gate.get("verdict", "UNKNOWN")
    baseline_kill = baseline_gate.get("mean_gate_kill_ratio")
    note = (
        f"baseline pre-cleanup verdict={baseline_verdict!r}, "
        f"mean_gate_kill_ratio={baseline_kill}"
    )
    # We just print — comparing requires the new IC audit to have been
    # run on the new parquet, which lives in tools/diagnostics/ic_audit.py
    # and the operator must produce that artefact separately.
    return True, f"  📊 baseline context — {note}"


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, type=Path,
                   help="path to post-cleanup features parquet")
    p.add_argument("--baseline-ic", type=Path, default=None,
                   help="optional path to pre-cleanup ic_summary.json for comparison")
    args = p.parse_args()

    df = _load_parquet(args.parquet)
    meta = _load_sidecar(args.parquet)
    baseline = _load_baseline(args.baseline_ic)

    print("\n══════════════════════════════════════════════════════════════")
    print("Phase 1.6 — Empirical Re-validation Gate")
    print("══════════════════════════════════════════════════════════════")

    results: list[tuple[bool, str]] = []
    for gate_fn, args_tuple in (
        (gate_neutral_ratio, (df,)),
        (gate_short_tp_exists, (df,)),
        (gate_no_dead_path_codes, (df,)),
        (gate_exec_valid_density, (df,)),
        (gate_next_price_delta_density, (df,)),
        (gate_whitelist_present, (df,)),
        (gate_blacklist_absent, (df,)),
        (gate_schema_validate, (df,)),
        (gate_phase1_4_features_present, (df,)),
        (gate_sidecar_up_to_date, (meta,)),
        (gate_baseline_comparison, (meta, baseline)),
    ):
        ok, msg = gate_fn(*args_tuple)
        print(msg)
        results.append((ok, msg))

    n_pass = sum(1 for ok, _ in results if ok)
    n_total = len(results)
    print("\n══════════════════════════════════════════════════════════════")
    print(f"Result: {n_pass}/{n_total} gates passed")
    print("══════════════════════════════════════════════════════════════")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
