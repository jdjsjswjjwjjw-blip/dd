"""
tools/calibrate_slippage.py
═══════════════════════════════════════════════════════════════════════════════
Reads a replay execution_log.jsonl and refits AdaptiveSlippage's three
constants so the model tracks realised slippage instead of over- or
under-shooting it.

Why three fits, not one global bias factor
──────────────────────────────────────────
AdaptiveSlippage.estimate() is a piecewise function of (size / L1_depth):

    Region A  ratio ≤ depth_floor (0.10)   → return base_ticks (flat)
    Region B  depth_floor < ratio ≤ 1.0    → linear: base + t·(mid − base)
                                              where t = (ratio − 0.10) / 0.90
    Region C  ratio > 1.0                  → mid + extra·(levels_consumed − 1)

A single global bias would deform the ramp shape. Each region has its
own constant; each is fit independently from its own subset of the log:

    base_ticks       = mean(realised) on region A
    mid_ticks        = base + OLS(realised − base ~ t · 1)  on region B
    extra_per_level  = OLS(realised − mid ~ (levels − 1) · 1)  on region C

Verdict cascade
───────────────
INSUFFICIENT_DATA   total events < min_events
WELL_CALIBRATED     |model − realised| / realised ≤ tolerance (default 0.25)
OVER_CALIBRATED     model > (1 + tolerance) · realised  (conservative — safer)
UNDER_CALIBRATED    model < (1 − tolerance) · realised  (dangerous — refit!)

Usage
─────
    python tools/calibrate_slippage.py \\
        --log    replay_results/baseline/execution_log.jsonl \\
        --output replay_results/baseline/calibration

Outputs
───────
<output>/calibration_summary.json       current + suggested constants + diag
<output>/slippage_config_suggested.json  drop-in for AdaptiveSlippage(**kwargs)
<output>/calibration_report.txt          human-readable summary
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEPTH_FLOOR = 0.10                # mirrors AdaptiveSlippage default
TOLERANCE_DEFAULT = 0.25          # ±25 % gap → well-calibrated
MIN_EVENTS_PER_REGION = 5
MIN_TOTAL_EVENTS = 10
# A region dominated by L0-fits (realised ≈ 0) makes OLS collapse toward
# zero or negative — the fit is mathematically valid but unsafe. Require
# at least this fraction of events with non-trivial realised slippage
# before trusting the fit; otherwise we surface SAFETY_CLAMPED.
MIN_NONZERO_REALISED_FRAC = 0.30
NONZERO_REALISED_TICKS = 1e-3     # below this, treat realised as zero


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CalibratedConstants:
    """Suggested AdaptiveSlippage params. Any field can be None if the
    log didn't have enough events in that region to refit it."""
    base_ticks: float | None
    mid_ticks: float | None
    extra_per_level: float | None
    depth_floor: float = DEPTH_FLOOR

    def to_kwargs(self, tick_size: float, fallback: dict | None = None) -> dict:
        """Drop-in kwargs for AdaptiveSlippage(...). Falls back to the
        previous calibration's values for any None field."""
        fallback = fallback or {}
        return {
            "tick_size": tick_size,
            "base_ticks": (
                self.base_ticks if self.base_ticks is not None
                else fallback.get("base_ticks", 0.5)
            ),
            "mid_ticks": (
                self.mid_ticks if self.mid_ticks is not None
                else fallback.get("mid_ticks", 1.5)
            ),
            "extra_per_level": (
                self.extra_per_level if self.extra_per_level is not None
                else fallback.get("extra_per_level", 1.0)
            ),
            "depth_floor": self.depth_floor,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Region classification
# ══════════════════════════════════════════════════════════════════════════════
def _ratio_and_region(
    intended_size: float, l1_depth: float, depth_floor: float = DEPTH_FLOOR
) -> tuple[float, str]:
    """Returns (size/L1_depth, region_name). Region D is the L1=0 edge case."""
    if l1_depth <= 0.0:
        return float("inf"), "D_zero_L1"
    ratio = float(intended_size) / float(l1_depth)
    if ratio <= depth_floor:
        return ratio, "A_sub_L1"
    if ratio <= 1.0:
        return ratio, "B_linear"
    return ratio, "C_walking"


def classify_events(df: pd.DataFrame, depth_floor: float = DEPTH_FLOOR) -> pd.DataFrame:
    """Add `ratio` + `region` columns to a copy of the events DataFrame."""
    out = df.copy()
    ratios, regions = [], []
    for _, row in out.iterrows():
        r, reg = _ratio_and_region(
            float(row["intended_size"]),
            float(row["decision_lob_l1_depth"]),
            depth_floor=depth_floor,
        )
        ratios.append(r)
        regions.append(reg)
    out["ratio"] = ratios
    out["region"] = regions
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Region-by-region fits
# ══════════════════════════════════════════════════════════════════════════════
def _ols_no_intercept(x: np.ndarray, y: np.ndarray) -> float | None:
    """Closed-form OLS slope through origin: β = Σxy / Σx². Returns None
    if the regressor is empty or has zero variance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return None
    return float(np.sum(x * y) / denom)


def _enough_nonzero_realised(events: pd.DataFrame) -> bool:
    """True iff at least MIN_NONZERO_REALISED_FRAC of events have
    non-trivial realised slippage. Guards OLS against L0-fit pathologies."""
    if events.empty:
        return False
    realised = events["realised_slippage_ticks"].abs().to_numpy()
    nonzero = float(np.mean(realised > NONZERO_REALISED_TICKS))
    return nonzero >= MIN_NONZERO_REALISED_FRAC


def fit_base_ticks(events_a: pd.DataFrame) -> float | None:
    """Region A: realised should be flat. Return the mean magnitude.

    Note: realised_slippage_ticks can legitimately be 0 in region A when
    the order fits entirely at L0 (fill_avg_px = best_ask). We use the
    mean rather than the median so the calibration is not artificially
    pulled to zero by a cluster of L0-fits."""
    if len(events_a) < MIN_EVENTS_PER_REGION:
        return None
    realised = events_a["realised_slippage_ticks"].abs().to_numpy()
    return float(np.mean(realised))


def fit_mid_ticks(
    events_b: pd.DataFrame, base_ticks: float, depth_floor: float = DEPTH_FLOOR
) -> float | None:
    """Region B linear ramp: realised − base = t · (mid − base)

    Refits (mid − base) via OLS-through-origin on t = (ratio − floor) / (1 − floor),
    then returns mid_ticks = base + slope.

    Returns None when region B is dominated by L0-fits (realised ≈ 0).
    In that case the OLS collapses toward −base, producing an unsafe
    suggestion. The caller falls back to the previous mid_ticks."""
    if len(events_b) < MIN_EVENTS_PER_REGION or base_ticks is None:
        return None
    if not _enough_nonzero_realised(events_b):
        return None
    span = max(1.0 - depth_floor, 1e-9)
    t = (events_b["ratio"].to_numpy() - depth_floor) / span
    targets = events_b["realised_slippage_ticks"].abs().to_numpy() - base_ticks
    slope = _ols_no_intercept(t, targets)
    if slope is None:
        return None
    return float(base_ticks + slope)


def fit_extra_per_level(
    events_c: pd.DataFrame, mid_ticks: float
) -> float | None:
    """Region C: realised − mid = extra · (levels − 1)

    OLS-through-origin on (levels_consumed − 1). Returns None when region C
    has too few non-zero realised events to refit safely."""
    if len(events_c) < MIN_EVENTS_PER_REGION or mid_ticks is None:
        return None
    if not _enough_nonzero_realised(events_c):
        return None
    levels_minus_1 = events_c["levels_consumed"].to_numpy() - 1.0
    targets = events_c["realised_slippage_ticks"].abs().to_numpy() - mid_ticks
    slope = _ols_no_intercept(levels_minus_1, targets)
    if slope is None:
        return None
    return float(slope)


def apply_safety_clamps(
    raw: CalibratedConstants, current: dict
) -> tuple[CalibratedConstants, list[str]]:
    """Enforce non-negativity + monotonicity (base ≤ mid, extra ≥ 0).

    Returns (clamped_constants, list_of_clamp_descriptions). When a
    suggestion violates a constraint we fall back to the corresponding
    `current` value rather than emit a value that would let the model
    *under*-predict slippage in production (the dangerous direction).
    """
    notes: list[str] = []
    base = raw.base_ticks
    mid = raw.mid_ticks
    extra = raw.extra_per_level

    if base is not None and base < 0:
        notes.append(f"base_ticks={base:.3f} < 0 → fallback to current")
        base = None

    if mid is not None:
        if mid < 0:
            notes.append(f"mid_ticks={mid:.3f} < 0 → fallback to current")
            mid = None
        elif base is not None and mid < base:
            notes.append(
                f"mid_ticks={mid:.3f} < base_ticks={base:.3f} → fallback to current"
            )
            mid = None

    if extra is not None and extra < 0:
        notes.append(f"extra_per_level={extra:.3f} < 0 → fallback to current")
        extra = None

    return CalibratedConstants(
        base_ticks=base, mid_ticks=mid, extra_per_level=extra,
        depth_floor=raw.depth_floor,
    ), notes


# ══════════════════════════════════════════════════════════════════════════════
# Verdict + summary
# ══════════════════════════════════════════════════════════════════════════════
def compute_verdict(
    events: pd.DataFrame, tolerance: float = TOLERANCE_DEFAULT
) -> tuple[str, dict]:
    """Compare the model's mean prediction against realised. Verdict is
    based on the directional gap, not abs-gap, so OVER vs UNDER stay
    distinct (over-calibration is safe, under-calibration is dangerous)."""
    diag: dict = {
        "n_events": int(len(events)),
        "tolerance": float(tolerance),
    }
    if len(events) < MIN_TOTAL_EVENTS:
        diag["min_events"] = MIN_TOTAL_EVENTS
        return "INSUFFICIENT_DATA", diag

    model = events["model_slippage_ticks"].abs().to_numpy()
    realised = events["realised_slippage_ticks"].abs().to_numpy()
    mean_model = float(np.mean(model))
    mean_realised = float(np.mean(realised))
    diag["mean_model_ticks"] = mean_model
    diag["mean_realised_ticks"] = mean_realised
    diag["ratio_model_over_realised"] = (
        mean_model / mean_realised if mean_realised > 1e-9 else float("inf")
    )

    if mean_realised <= 1e-9:
        # Edge case: log is dominated by L0-fits with exact-zero realised
        diag["note"] = "realised mean ≈ 0 — likely too many in-L0 fills"
        return "INSUFFICIENT_DATA", diag

    gap = (mean_model - mean_realised) / mean_realised
    diag["gap_pct"] = float(gap)
    if abs(gap) <= tolerance:
        return "WELL_CALIBRATED", diag
    if gap > tolerance:
        return "OVER_CALIBRATED", diag
    return "UNDER_CALIBRATED", diag


def summarise_regions(events: pd.DataFrame) -> dict[str, dict]:
    """Per-region count + realised stats. Useful for the report."""
    out: dict[str, dict] = {}
    for region, sub in events.groupby("region"):
        rs = sub["realised_slippage_ticks"].abs().to_numpy()
        ms = sub["model_slippage_ticks"].abs().to_numpy()
        out[str(region)] = {
            "n": int(len(sub)),
            "mean_realised": float(np.mean(rs)) if len(rs) else 0.0,
            "mean_model": float(np.mean(ms)) if len(ms) else 0.0,
            "ratio": (
                float(np.mean(ms) / np.mean(rs))
                if len(rs) and np.mean(rs) > 1e-9 else None
            ),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# I/O + orchestration
# ══════════════════════════════════════════════════════════════════════════════
def load_log(path: Path) -> pd.DataFrame:
    """Load JSON-Lines execution log. Required columns documented above."""
    if not path.exists():
        raise FileNotFoundError(f"execution log not found: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    required = {
        "intended_size", "decision_lob_l1_depth",
        "realised_slippage_ticks", "model_slippage_ticks",
        "levels_consumed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"log missing required columns: {sorted(missing)}")
    return df


def calibrate(
    df: pd.DataFrame, depth_floor: float = DEPTH_FLOOR
) -> CalibratedConstants:
    """End-to-end refit: classify → region-wise OLS → return CalibratedConstants."""
    if df.empty:
        return CalibratedConstants(None, None, None, depth_floor)
    classed = classify_events(df, depth_floor=depth_floor)
    by_region = {r: g for r, g in classed.groupby("region")}

    base = fit_base_ticks(by_region.get("A_sub_L1", classed.iloc[0:0]))
    mid = fit_mid_ticks(
        by_region.get("B_linear", classed.iloc[0:0]),
        base_ticks=base if base is not None else 0.5,
        depth_floor=depth_floor,
    )
    extra = fit_extra_per_level(
        by_region.get("C_walking", classed.iloc[0:0]),
        mid_ticks=mid if mid is not None else 1.5,
    )
    return CalibratedConstants(
        base_ticks=base, mid_ticks=mid,
        extra_per_level=extra, depth_floor=depth_floor,
    )


def write_report(
    path: Path,
    verdict: str,
    diag: dict,
    region_summary: dict,
    current: dict,
    suggested: CalibratedConstants,
    safety_clamps: list[str] | None = None,
) -> None:
    lines = ["═" * 70, "SLIPPAGE CALIBRATION", "═" * 70,
             f"Verdict        : {verdict}",
             f"N events       : {diag.get('n_events', 0)}",
             f"Mean model     : {diag.get('mean_model_ticks', 0):.3f} ticks",
             f"Mean realised  : {diag.get('mean_realised_ticks', 0):.3f} ticks",
             f"Ratio          : {diag.get('ratio_model_over_realised', 0):.2f}x",
             f"Tolerance      : ±{int(diag.get('tolerance', 0) * 100)}%",
             "", "Per-region breakdown:"]
    for region in sorted(region_summary):
        info = region_summary[region]
        ratio_str = (
            f"ratio={info['ratio']:.2f}x" if info["ratio"] is not None
            else "ratio=n/a"
        )
        lines.append(
            f"  {region:<14} n={info['n']:>4}  "
            f"model={info['mean_model']:.3f}  "
            f"realised={info['mean_realised']:.3f}  {ratio_str}"
        )
    lines.append("")
    lines.append("Constants:")
    lines.append(f"  {'':<18} {'current':>10}  {'suggested':>12}")
    for name in ("base_ticks", "mid_ticks", "extra_per_level"):
        cur = current.get(name)
        sug = getattr(suggested, name)
        cur_str = f"{cur:.3f}" if cur is not None else "    n/a"
        sug_str = f"{sug:.3f}" if sug is not None else "      n/a (insufficient region data)"
        lines.append(f"  {name:<18} {cur_str:>10}  {sug_str:>12}")
    if safety_clamps:
        lines.append("")
        lines.append("Safety clamps applied:")
        for note in safety_clamps:
            lines.append(f"  ⚠️  {note}")
    lines.append("═" * 70)
    path.write_text("\n".join(lines) + "\n")


def run_calibration(
    log_path: Path,
    output_dir: Path,
    tick_size: float = 0.0001,
    tolerance: float = TOLERANCE_DEFAULT,
    current: dict | None = None,
) -> dict:
    """End-to-end: load → calibrate → safety-clamp → write outputs."""
    df = load_log(log_path)
    verdict, diag = compute_verdict(df, tolerance=tolerance)
    classed = classify_events(df) if not df.empty else df
    region_summary = summarise_regions(classed) if not classed.empty else {}
    raw = calibrate(df)

    current = current or {"base_ticks": 0.5, "mid_ticks": 1.5, "extra_per_level": 1.0}
    suggested, safety_clamps = apply_safety_clamps(raw, current)
    suggested_kwargs = suggested.to_kwargs(tick_size=tick_size, fallback=current)

    summary = {
        "verdict": verdict,
        "diag": diag,
        "tick_size": float(tick_size),
        "tolerance": float(tolerance),
        "current_constants": current,
        "raw_constants": {
            "base_ticks": raw.base_ticks,
            "mid_ticks": raw.mid_ticks,
            "extra_per_level": raw.extra_per_level,
        },
        "suggested_constants": {
            "base_ticks": suggested.base_ticks,
            "mid_ticks": suggested.mid_ticks,
            "extra_per_level": suggested.extra_per_level,
            "depth_floor": suggested.depth_floor,
        },
        "suggested_kwargs": suggested_kwargs,
        "safety_clamps": safety_clamps,
        "region_summary": region_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (output_dir / "slippage_config_suggested.json").write_text(
        json.dumps(suggested_kwargs, indent=2, default=str)
    )
    write_report(
        output_dir / "calibration_report.txt",
        verdict=verdict, diag=diag,
        region_summary=region_summary,
        current=current, suggested=suggested,
        safety_clamps=safety_clamps,
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path,
                   help="execution_log.jsonl from a replay backtest")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tick-size", type=float, default=0.0001)
    p.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAULT,
                   help="±gap to count as well-calibrated (default 0.25)")
    p.add_argument("--current-base", type=float, default=0.5)
    p.add_argument("--current-mid", type=float, default=1.5)
    p.add_argument("--current-extra", type=float, default=1.0)
    args = p.parse_args()

    current = {
        "base_ticks": args.current_base,
        "mid_ticks": args.current_mid,
        "extra_per_level": args.current_extra,
    }
    summary = run_calibration(
        args.log, args.output,
        tick_size=args.tick_size, tolerance=args.tolerance, current=current,
    )
    diag = summary["diag"]
    print(f"verdict          : {summary['verdict']}")
    print(f"events           : {diag.get('n_events', 0)}")
    mr = diag.get("mean_realised_ticks")
    mm = diag.get("mean_model_ticks")
    if mr is not None and mm is not None:
        print(f"model / realised : {mm:.3f} / {mr:.3f} ticks  "
              f"({diag.get('ratio_model_over_realised', 0):.2f}x)")
    suggested = summary["suggested_constants"]
    print(f"suggested        : base={suggested['base_ticks']}  "
          f"mid={suggested['mid_ticks']}  extra={suggested['extra_per_level']}")
    if summary["safety_clamps"]:
        print(f"safety clamps    : {len(summary['safety_clamps'])} applied "
              f"(see report)")
    print(f"output           : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
