"""case_dissect — DESCRIBE one sweep+reversal case bar-by-bar (NOT a predictor).

The user described a specific May-2025 sequence: prior-week low at 1.3214 was
swept down to ~1.3124, then price reversed up over days. This tool extracts the
window around the sweep point and tabulates, per 5-min bar:
  - price path + the swept level (pwl) and pdl
  - cumulative CVD (does signed flow flip from selling to buying at the low?)
  - per-bar CVD (cvd_bar_5m / bar_cvd_delta)
  - absorption proxy = |Δcvd| / |Δprice|  (heavy signed flow + tiny price move
    = a passive bid soaking up selling). absorption_intensity itself is NOT in
    the post-D1 export, so we reconstruct the bar-level ratio from exported
    bar_cvd_delta + close.
  - order_flow_imbalance, tick_count (activity)

THREE GUARDS BAKED IN (the user's caveats, R7):
  1. ONE CASE IS A STORY, NOT EVIDENCE. The report header says so. A descriptive
     dissection shows WHAT HAPPENED in this instance, never that it predicts.
  2. NO HINDSIGHT THRESHOLDS. We report the raw numbers + percentiles relative
     to the bar's own recent history; we do not back-fit an "absorption
     threshold" chosen because we know the reversal happened.
  3. GENERALIZATION IS A SEPARATE STEP. Proving the signature (sweep + absorption
     + accumulation) precedes reversals — and how often it appears WITHOUT one —
     requires the event_study / a population test, not this single dissection.

This module is a microscope, not a probe. It produces no verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TICK = 0.0001


def find_sweep_window(
    df: pd.DataFrame, target_low: float, *,
    date_start: str | None = None, date_end: str | None = None,
    tol_ticks: float = 5.0, tick_size: float = DEFAULT_TICK,
    before: int = 48, after: int = 96,
) -> tuple[pd.DataFrame, int]:
    """Locate the sweep bar (lowest low near target_low within the date window)
    and return (window_df, sweep_idx_within_window).

    Causal note: this is a DESCRIPTIVE locator on historical data — we are
    deliberately looking at a known past episode. It is not used to build a
    feature, so 'finding the bottom' here is description, not look-ahead."""
    d = df.copy()
    d["ts_event"] = pd.to_datetime(d["ts_event"], utc=True, errors="coerce")
    if date_start:
        d = d[d["ts_event"] >= pd.Timestamp(date_start, tz="UTC")]
    if date_end:
        d = d[d["ts_event"] <= pd.Timestamp(date_end, tz="UTC")]
    d = d.reset_index(drop=True)
    if len(d) == 0:
        raise RuntimeError("no bars in the requested date window")

    low = pd.to_numeric(d["low"], errors="coerce").to_numpy(np.float64)
    tol = tol_ticks * tick_size
    near = np.where(low <= target_low + tol)[0]
    if len(near) == 0:
        # fall back to the global min low in the window
        sweep_global = int(np.nanargmin(low))
    else:
        # the actual bottom = min low among bars that reached the target zone
        sweep_global = int(near[np.nanargmin(low[near])])

    lo = max(0, sweep_global - before)
    hi = min(len(d), sweep_global + after + 1)
    window = d.iloc[lo:hi].reset_index(drop=True)
    sweep_in_window = sweep_global - lo
    return window, sweep_in_window


def dissect(window: pd.DataFrame, *, tick_size: float = DEFAULT_TICK,
            absorption_floor_ticks: float = 0.5) -> pd.DataFrame:
    """Add the descriptive columns to the window (absorption proxy, CVD path)."""
    w = window.copy()
    close = pd.to_numeric(w["close"], errors="coerce").to_numpy(np.float64)

    # per-bar signed flow: prefer bar_cvd_delta, else diff of cumulative cvd
    if "bar_cvd_delta" in w.columns:
        dcvd = pd.to_numeric(w["bar_cvd_delta"], errors="coerce").to_numpy(np.float64)
    elif "cvd" in w.columns:
        dcvd = np.diff(pd.to_numeric(w["cvd"], errors="coerce").to_numpy(np.float64),
                       prepend=np.nan)
    else:
        dcvd = np.full(len(w), np.nan)

    dprice = np.diff(close, prepend=np.nan)
    floor = absorption_floor_ticks * tick_size
    # absorption proxy: heavy signed flow with tiny price move → large ratio
    w["abs_dprice_ticks"] = np.abs(dprice) / tick_size
    w["bar_signed_flow"] = dcvd
    w["absorption_proxy"] = np.abs(dcvd) / np.maximum(np.abs(dprice), floor)

    # cumulative CVD within the window (relative to window start), to see the
    # selling→buying turn at the low
    w["cvd_cum_window"] = np.nancumsum(np.where(np.isfinite(dcvd), dcvd, 0.0))
    return w


def summarize_at_sweep(w: pd.DataFrame, sweep_idx: int, *, span: int = 12) -> dict[str, Any]:
    """Numbers around the sweep, WITHOUT a hindsight pass/fail threshold.

    Reports the absorption proxy AT the sweep vs its window distribution
    (percentile), the CVD slope before vs after the sweep, and the price stall
    (range in the `span` bars around the sweep)."""
    n = len(w)
    s = int(np.clip(sweep_idx, 0, n - 1))
    ap = pd.to_numeric(w["absorption_proxy"], errors="coerce").to_numpy()
    cvd_cum = pd.to_numeric(w["cvd_cum_window"], errors="coerce").to_numpy()
    close = pd.to_numeric(w["close"], errors="coerce").to_numpy()

    pre = slice(max(0, s - span), s + 1)
    post = slice(s, min(n, s + span + 1))

    ap_finite = ap[np.isfinite(ap)]
    ap_at = float(ap[s]) if np.isfinite(ap[s]) else float("nan")
    ap_pctile = (float((ap_finite < ap_at).mean() * 100)
                 if np.isfinite(ap_at) and len(ap_finite) else float("nan"))

    def _slope(a):
        a = a[np.isfinite(a)]
        if len(a) < 3:
            return float("nan")
        x = np.arange(len(a))
        return float(np.polyfit(x, a, 1)[0])

    return {
        "sweep_idx": s,
        "sweep_ts": str(w["ts_event"].iloc[s]),
        "sweep_low": float(pd.to_numeric(w["low"], errors="coerce").iloc[s]),
        "sweep_close": float(close[s]) if np.isfinite(close[s]) else float("nan"),
        "absorption_proxy_at_sweep": ap_at,
        "absorption_proxy_window_pctile": ap_pctile,
        "absorption_proxy_window_median": float(np.nanmedian(ap_finite)) if len(ap_finite) else float("nan"),
        "cvd_slope_pre": _slope(cvd_cum[pre]),       # selling pressure into the low?
        "cvd_slope_post": _slope(cvd_cum[post]),     # buying after the low?
        "cvd_turned_up": bool(np.isfinite(_slope(cvd_cum[pre])) and np.isfinite(_slope(cvd_cum[post]))
                              and _slope(cvd_cum[pre]) < 0 < _slope(cvd_cum[post])),
        "price_range_ticks_around_sweep": (
            float((np.nanmax(close[pre.start:post.stop]) - np.nanmin(close[pre.start:post.stop])) / DEFAULT_TICK)
            if post.stop > pre.start else float("nan")),
    }


def run_case_dissect(
    features_parquet: Path, output_dir: Path, *,
    target_low: float, date_start: str | None, date_end: str | None,
    before: int = 48, after: int = 96, tol_ticks: float = 5.0,
    tick_size: float = DEFAULT_TICK,
) -> dict[str, Any]:
    df = pd.read_parquet(features_parquet)
    window, sweep_idx = find_sweep_window(
        df, target_low, date_start=date_start, date_end=date_end,
        tol_ticks=tol_ticks, tick_size=tick_size, before=before, after=after)
    w = dissect(window, tick_size=tick_size)
    summ = summarize_at_sweep(w, sweep_idx)

    # persist the per-bar series for plotting
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("ts_event", "open", "high", "low", "close", "volume",
                        "tick_count", "cvd", "bar_signed_flow", "absorption_proxy",
                        "abs_dprice_ticks", "cvd_cum_window", "order_flow_imbalance",
                        "hawkes_intrabar_sum", "vwap_z_score", "pwl", "pdl")
            if c in w.columns]
    w[cols].to_csv(output_dir / "case_dissect_series.csv", index=False)

    summary = {
        "features_parquet": str(features_parquet),
        "target_low": target_low, "date_start": date_start, "date_end": date_end,
        "window_bars": int(len(w)), "sweep_idx_in_window": int(sweep_idx),
        "summary_at_sweep": summ,
        "series_csv": str(output_dir / "case_dissect_series.csv"),
        "DISCLAIMER": ("ONE CASE — DESCRIPTIVE, NOT PREDICTIVE (R7). This shows what "
                       "happened in this single instance. Proving the signature "
                       "predicts requires a population/event-study test, including "
                       "how often it appears WITHOUT a reversal."),
    }
    (output_dir / "case_dissect_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_report(output_dir / "case_dissect_report.txt", summary, w, sweep_idx)
    return summary


def _write_report(path: Path, s: dict, w: pd.DataFrame, sweep_idx: int) -> None:
    L = []
    L.append("═" * 100)
    L.append("Case dissection — sweep + reversal (ONE CASE; DESCRIPTIVE, NOT PREDICTIVE — R7)")
    L.append("═" * 100)
    L.append(f"parquet     : {s['features_parquet']}")
    L.append(f"target low  : {s['target_low']}   window: {s['window_bars']} bars   "
             f"sweep at index {s['sweep_idx_in_window']}")
    L.append("")
    a = s["summary_at_sweep"]
    L.append("AT THE SWEEP (numbers, no hindsight threshold):")
    L.append(f"  ts={a['sweep_ts']}   low={a['sweep_low']:.5f}  close={a['sweep_close']:.5f}")
    L.append(f"  absorption_proxy        = {a['absorption_proxy_at_sweep']:.1f}  "
             f"(window pctile {a['absorption_proxy_window_pctile']:.0f}%, "
             f"median {a['absorption_proxy_window_median']:.1f})")
    L.append(f"  CVD slope  pre={a['cvd_slope_pre']:+.2f}  post={a['cvd_slope_post']:+.2f}  "
             f"→ turned selling→buying: {a['cvd_turned_up']}")
    L.append(f"  price range around sweep = {a['price_range_ticks_around_sweep']:.1f} ticks")
    L.append("")
    # compact path table: every 4th bar around the sweep
    L.append("PATH (every 4th bar; * = sweep):")
    L.append(f"  {'ts':20s} {'close':>9s} {'low':>9s} {'signed_flow':>12s} {'abs_proxy':>10s} {'cvd_cum':>10s} {'OFI':>7s}")
    n = len(w)
    for i in range(0, n, 4):
        mark = "*" if abs(i - sweep_idx) < 2 else " "
        ts = str(w["ts_event"].iloc[i])[:19]
        cl = float(pd.to_numeric(w["close"], errors="coerce").iloc[i])
        lo = float(pd.to_numeric(w["low"], errors="coerce").iloc[i])
        sf = float(w["bar_signed_flow"].iloc[i]) if np.isfinite(w["bar_signed_flow"].iloc[i]) else float("nan")
        ap = float(w["absorption_proxy"].iloc[i]) if np.isfinite(w["absorption_proxy"].iloc[i]) else float("nan")
        cc = float(w["cvd_cum_window"].iloc[i])
        ofi = float(pd.to_numeric(w.get("order_flow_imbalance", pd.Series([np.nan])), errors="coerce").iloc[i]) if "order_flow_imbalance" in w.columns else float("nan")
        L.append(f" {mark}{ts:20s} {cl:>9.5f} {lo:>9.5f} {sf:>12.1f} {ap:>10.1f} {cc:>10.1f} {ofi:>+7.3f}")
    L.append("")
    L.append(f"DISCLAIMER: {s['DISCLAIMER']}")
    L.append(f"full per-bar series → {s['series_csv']}")
    L.append("═" * 100)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Descriptive dissection of one sweep+reversal case")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-low", required=True, type=float, help="the sweep low, e.g. 1.3124")
    p.add_argument("--date-start", default=None, help="ISO date, e.g. 2025-05-01")
    p.add_argument("--date-end", default=None, help="ISO date, e.g. 2025-05-31")
    p.add_argument("--before", type=int, default=48, help="bars before the sweep")
    p.add_argument("--after", type=int, default=96, help="bars after the sweep")
    p.add_argument("--tol-ticks", type=float, default=5.0)
    p.add_argument("--tick-size", type=float, default=DEFAULT_TICK)
    args = p.parse_args()
    s = run_case_dissect(
        args.features, args.output, target_low=args.target_low,
        date_start=args.date_start, date_end=args.date_end,
        before=args.before, after=args.after, tol_ticks=args.tol_ticks,
        tick_size=args.tick_size)
    a = s["summary_at_sweep"]
    print(f"\n  sweep at {a['sweep_ts']}  low={a['sweep_low']:.5f}")
    print(f"  absorption_proxy={a['absorption_proxy_at_sweep']:.1f} "
          f"(pctile {a['absorption_proxy_window_pctile']:.0f}%)  "
          f"CVD turned up: {a['cvd_turned_up']}")
    print(f"  ⚠️  ONE CASE — descriptive, not predictive (R7)")
    print(f"  report: {args.output}/case_dissect_report.txt   series: {s['series_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
