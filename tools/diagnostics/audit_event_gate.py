"""
tools/diagnostics/audit_event_gate.py
═══════════════════════════════════════════════════════════════════════════════
Audit the upstream event gate that decides which bars get labeled.

Why this exists
───────────────
`prepare_day_trading.detect_microstructure_events` (prepare_day_trading.py:2222)
gates which bars are eligible for directional labeling. Every bar with
`is_event == 0` is silently stamped `bias_label = NEUTRAL` (line 2493),
regardless of price action. The gate's docstring targets a 20-30 % event
rate; if the actual rate is far lower the entire downstream NEUTRAL
proportion is determined by gate failure, not by market reality.

What it checks
──────────────
1. Overall is_event rate vs the 20-30 % design target.
2. Per-regime rate vs `REGIME_EVENT_THRESHOLD` (volatile/low_liquidity use
   a stricter threshold so a low rate there is expected; a low rate in
   trending/ranging is a bug).
3. Per-session rate (asian/london/ny). Asian + warm-up periods are the
   usual suspects for chronically-low gates.
4. Component failure rates: % of bars where each of the four binary
   gate components (hawkes_z > 1, absorb_z > 1, kyle_z > 0.5,
   cvd_align > 0.6) is 0. A near-100 % failure on one component reveals
   a dominated denominator.
5. cvd_align NaN contamination: prepare_day_trading silently
   `fillna(0.5)` on `cvd_direction_pct` and the threshold is 0.6, so
   every NaN row is a guaranteed component miss.
6. Warm-up zero bias: `_zscore` returns 0 inside the rolling-window
   warm-up (`min_periods=20`). Each session start systematically
   produces near-zero event_score regardless of underlying activity.

Verdict cascade
───────────────
INCOMPLETE        no `is_event` column → can't audit.
INSUFFICIENT_DATA < min_rows.
STARVED           overall event_rate < 10 %.
LOW_RATE          10 % ≤ event_rate < 20 %.
WARMUP_HEAVY      ≥ 5 % of dataset killed by `_zscore` warm-up.
COMPONENT_DOMINATED one component fails on > 90 % of bars.
REGIME_STARVED    a non-volatile regime has < 15 % event rate.
HEALTHY           20 % ≤ event_rate ≤ 35 % and no per-regime/session
                  red flags.

Usage
─────
    python tools/diagnostics/audit_event_gate.py \\
        --features combined/day_trading_features.parquet \\
        --output   diagnostics/event_gate

Outputs
───────
    <output>/event_gate_summary.json   verdict + all metrics
    <output>/event_gate_report.txt     human-readable
    <output>/component_failures.csv    per-component failure breakdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from regime_config import (
        EVENT_SCORE_WEIGHTS,
        EVENT_ZSCORE_WINDOW,
        EVENT_ZSCORE_MIN_PERIODS,
        REGIME_EVENT_THRESHOLD,
    )
except Exception:
    # Fallback defaults — keep in sync with regime_config.py
    EVENT_SCORE_WEIGHTS = {
        "hawkes_z_above_1": 0.26,
        "absorb_z_above_1": 0.26,
        "kyle_z_above_05": 0.165,
        "cvd_align_above_06": 0.165,
        "mbp_roll_cov_above_cut": 0.075,
        "mbo_tick_cov_above_cut": 0.075,
    }
    EVENT_ZSCORE_WINDOW = 100
    EVENT_ZSCORE_MIN_PERIODS = 20
    REGIME_EVENT_THRESHOLD = {
        "trending": 0.60,
        "ranging": 0.60,
        "volatile": 0.75,
        "low_liquidity": 0.80,
    }


DESIGN_RATE_MIN = 0.20
DESIGN_RATE_MAX = 0.35
STARVED_RATE = 0.10
WARMUP_HEAVY_PCT = 0.05
COMPONENT_DOMINATION_PCT = 0.90
MIN_ROWS_FOR_AUDIT = 1000


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════
def compute_event_rate(df: pd.DataFrame, col: str = "is_event") -> float:
    """Overall fraction of bars with is_event == 1."""
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).mean())


def compute_per_group_rate(
    df: pd.DataFrame, group_col: str, event_col: str = "is_event"
) -> dict[str, dict]:
    """Per-group event rate + count.  Returns {group: {rate, n_rows, n_events}}."""
    if group_col not in df.columns or event_col not in df.columns:
        return {}
    ev = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(bool)
    out: dict[str, dict] = {}
    for group, idx in df.groupby(df[group_col].astype(str)).groups.items():
        sub = ev.loc[idx]
        out[str(group)] = {
            "rate": float(sub.mean()) if len(sub) else float("nan"),
            "n_rows": int(len(sub)),
            "n_events": int(sub.sum()),
        }
    return out


def _zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Mirror of prepare_day_trading._zscore — same rolling logic."""
    roll = series.rolling(window, min_periods=min_periods)
    return ((series - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)


def compute_component_failures(df: pd.DataFrame) -> dict[str, dict]:
    """
    For each of the four binary gate components, compute the fraction of
    bars where the component contributes 0.

    Mirrors prepare_day_trading.detect_microstructure_events:2289.
    Missing source columns are reported as `available: False`.
    """
    n = len(df)
    if n == 0:
        return {}

    out: dict[str, dict] = {}

    sources = {
        "hawkes_z_above_1": ("hawkes_intensity", 1.0, "z"),
        "absorb_z_above_1": ("absorption_intensity", 1.0, "z"),
        "kyle_z_above_05": ("kyle_lambda", 0.5, "z"),
        "cvd_align_above_06": ("cvd_direction_pct", 0.6, "raw"),
    }

    for comp_name, (src_col, thr, mode) in sources.items():
        if src_col not in df.columns:
            out[comp_name] = {"available": False}
            continue
        raw = pd.to_numeric(df[src_col], errors="coerce")
        nan_rows = int(raw.isna().sum())
        if mode == "z":
            transformed = _zscore(
                raw.fillna(0.0), EVENT_ZSCORE_WINDOW, EVENT_ZSCORE_MIN_PERIODS
            )
        else:  # raw — mirrors prepare_day_trading: fillna(0.5)
            transformed = raw.fillna(0.5)
        pass_mask = transformed > thr
        out[comp_name] = {
            "available": True,
            "fail_rate": float((~pass_mask).mean()),
            "pass_rate": float(pass_mask.mean()),
            "nan_rows": nan_rows,
            "nan_rate": float(nan_rows / n),
            "source_col": src_col,
            "threshold": float(thr),
        }
    return out


def detect_cvd_fillna_contamination(df: pd.DataFrame) -> dict:
    """
    Sub-ghost B: when `cvd_direction_pct` is NaN, prepare_day_trading
    fillna(0.5), and 0.5 < 0.6 — so every NaN bar silently fails the
    component. Quantify how many bars are affected.
    """
    n = len(df)
    if "cvd_direction_pct" not in df.columns or n == 0:
        return {"available": False}
    nan_rows = int(pd.to_numeric(df["cvd_direction_pct"], errors="coerce").isna().sum())
    return {
        "available": True,
        "nan_rows": nan_rows,
        "nan_rate": float(nan_rows / n),
        "auto_fail_rate": float(nan_rows / n),
    }


def detect_warmup_zero_bias(
    df: pd.DataFrame,
    window: int = EVENT_ZSCORE_WINDOW,
    min_periods: int = EVENT_ZSCORE_MIN_PERIODS,
    session_col: str = "session",
) -> dict:
    """
    Sub-ghost C: `_zscore` returns 0 inside the rolling-window warm-up
    (min_periods rows). If sessions are processed independently, the first
    `min_periods` bars of each session systematically score 0.

    If `session_col` is absent we use a single global warm-up at the head.
    """
    n = len(df)
    if n == 0:
        return {"available": False}

    if session_col in df.columns:
        session_arr = df[session_col].astype(str).to_numpy()
        warmup_mask = np.zeros(n, dtype=bool)
        prev = None
        run_start = 0
        for i in range(n):
            cur = session_arr[i]
            if cur != prev:
                run_start = i
                prev = cur
            if i - run_start < min_periods:
                warmup_mask[i] = True
        return {
            "available": True,
            "scope": "per_session",
            "warmup_min_periods": int(min_periods),
            "warmup_window": int(window),
            "warmup_rows": int(warmup_mask.sum()),
            "warmup_rate": float(warmup_mask.mean()),
        }
    return {
        "available": True,
        "scope": "global",
        "warmup_min_periods": int(min_periods),
        "warmup_window": int(window),
        "warmup_rows": int(min(min_periods, n)),
        "warmup_rate": float(min(min_periods, n) / n),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════
def compute_verdict(
    overall_rate: float,
    per_regime: dict[str, dict],
    per_session: dict[str, dict],
    components: dict[str, dict],
    warmup: dict,
    n_rows: int,
    min_rows: int = MIN_ROWS_FOR_AUDIT,
) -> tuple[str, dict]:
    """Single-source verdict cascade. Returns (verdict, diag)."""
    diag: dict = {
        "overall_rate": float(overall_rate) if overall_rate == overall_rate else None,
        "n_rows": int(n_rows),
    }

    if not (isinstance(overall_rate, float) and overall_rate == overall_rate):
        return "INCOMPLETE", diag
    if n_rows < min_rows:
        diag["min_rows"] = min_rows
        return "INSUFFICIENT_DATA", diag

    if overall_rate < STARVED_RATE:
        diag["threshold"] = STARVED_RATE
        return "STARVED", diag

    # warm-up domination — only meaningful when measurable
    if warmup.get("available") and warmup["warmup_rate"] >= WARMUP_HEAVY_PCT:
        diag["warmup_rate"] = float(warmup["warmup_rate"])
        return "WARMUP_HEAVY", diag

    # component domination — any available component failing > 90 %
    dominated = []
    for comp, info in components.items():
        if info.get("available") and info["fail_rate"] > COMPONENT_DOMINATION_PCT:
            dominated.append(comp)
    if dominated:
        diag["dominated_components"] = dominated
        return "COMPONENT_DOMINATED", diag

    # regime starvation — exclude regimes that are SUPPOSED to be strict
    strict = {"volatile", "low_liquidity"}
    starved_regimes = [
        r
        for r, info in per_regime.items()
        if r not in strict
        and info["n_rows"] >= 200
        and info["rate"] < 0.15
    ]
    if starved_regimes:
        diag["starved_regimes"] = starved_regimes
        return "REGIME_STARVED", diag

    if overall_rate < DESIGN_RATE_MIN:
        diag["target_min"] = DESIGN_RATE_MIN
        return "LOW_RATE", diag

    diag["target_range"] = [DESIGN_RATE_MIN, DESIGN_RATE_MAX]
    return "HEALTHY", diag


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════
def write_report(
    path: Path,
    verdict: str,
    diag: dict,
    overall_rate: float,
    per_regime: dict,
    per_session: dict,
    components: dict,
    cvd_contam: dict,
    warmup: dict,
    n_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("═" * 70)
    lines.append("EVENT GATE AUDIT")
    lines.append("═" * 70)
    lines.append(f"Verdict          : {verdict}")
    lines.append(f"Total bars       : {n_rows:,}")
    if overall_rate == overall_rate:
        lines.append(f"Event rate       : {overall_rate:.2%}")
        lines.append(f"Design target    : {DESIGN_RATE_MIN:.0%}–{DESIGN_RATE_MAX:.0%}")
    lines.append("")

    if per_regime:
        lines.append("Per-regime breakdown:")
        for r, info in sorted(per_regime.items()):
            thr = REGIME_EVENT_THRESHOLD.get(r, 0.60)
            lines.append(
                f"  {r:<14} rate={info['rate']:.2%}  "
                f"n={info['n_rows']:,}  threshold={thr:.2f}"
            )
        lines.append("")

    if per_session:
        lines.append("Per-session breakdown:")
        for s, info in sorted(per_session.items()):
            lines.append(
                f"  {s:<14} rate={info['rate']:.2%}  n={info['n_rows']:,}"
            )
        lines.append("")

    if components:
        lines.append("Component failure analysis (% of bars where component=0):")
        for comp, info in components.items():
            if not info.get("available"):
                lines.append(f"  {comp:<22} — source missing")
                continue
            lines.append(
                f"  {comp:<22} fail={info['fail_rate']:.2%}  "
                f"nan={info['nan_rate']:.2%}  ({info['source_col']})"
            )
        lines.append("")

    if cvd_contam.get("available"):
        lines.append("cvd_direction_pct fillna(0.5) contamination:")
        lines.append(
            f"  nan_rows={cvd_contam['nan_rows']:,}  "
            f"auto_fail_rate={cvd_contam['auto_fail_rate']:.2%}"
        )
        lines.append("")

    if warmup.get("available"):
        lines.append(
            f"Warm-up zero-bias ({warmup['scope']}, min_periods="
            f"{warmup['warmup_min_periods']}):"
        )
        lines.append(
            f"  rows={warmup['warmup_rows']:,}  "
            f"rate={warmup['warmup_rate']:.2%}"
        )
        lines.append("")

    lines.append("Diagnostics:")
    for k, v in diag.items():
        lines.append(f"  {k}: {v}")
    lines.append("═" * 70)
    # utf-8: the report contains non-ASCII (═, Arabic). Without explicit
    # encoding Path.write_text uses the OS default (cp1252 on Windows) and
    # raises UnicodeEncodeError — the audit was silently skipped on Windows.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(features_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path)
    n_rows = len(df)

    if "is_event" not in df.columns:
        raise ValueError(
            f"features parquet at {features_path} has no 'is_event' column — "
            f"this is required to audit the event gate. Available columns: "
            f"{list(df.columns)[:20]}..."
        )

    overall_rate = compute_event_rate(df)
    per_regime = compute_per_group_rate(df, "regime_label")
    per_session = compute_per_group_rate(df, "session")
    components = compute_component_failures(df)
    cvd_contam = detect_cvd_fillna_contamination(df)
    warmup = detect_warmup_zero_bias(df)

    verdict, diag = compute_verdict(
        overall_rate=overall_rate,
        per_regime=per_regime,
        per_session=per_session,
        components=components,
        warmup=warmup,
        n_rows=n_rows,
    )

    summary = {
        "verdict": verdict,
        "diag": diag,
        "n_rows": n_rows,
        "overall_event_rate": (
            float(overall_rate) if overall_rate == overall_rate else None
        ),
        "design_target": [DESIGN_RATE_MIN, DESIGN_RATE_MAX],
        "per_regime": per_regime,
        "per_session": per_session,
        "components": components,
        "cvd_fillna_contamination": cvd_contam,
        "warmup_zero_bias": warmup,
        "regime_thresholds": REGIME_EVENT_THRESHOLD,
        "event_score_weights": EVENT_SCORE_WEIGHTS,
    }
    (output_dir / "event_gate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    comp_rows = []
    for comp, info in components.items():
        if not info.get("available"):
            comp_rows.append({"component": comp, "available": False})
        else:
            comp_rows.append({"component": comp, **info})
    pd.DataFrame(comp_rows).to_csv(
        output_dir / "component_failures.csv", index=False, encoding="utf-8",
    )

    write_report(
        output_dir / "event_gate_report.txt",
        verdict=verdict,
        diag=diag,
        overall_rate=overall_rate,
        per_regime=per_regime,
        per_session=per_session,
        components=components,
        cvd_contam=cvd_contam,
        warmup=warmup,
        n_rows=n_rows,
    )

    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    summary = run_audit(args.features, args.output)
    print(f"verdict          : {summary['verdict']}")
    rate = summary["overall_event_rate"]
    print(f"event rate       : {rate:.2%}" if rate is not None else "event rate : n/a")
    print(f"output           : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
