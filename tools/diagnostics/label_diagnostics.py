"""label_diagnostics — why is there 0 directional signal? Three decisive checks.

decoder_probe + sequence_probe both found ~0 directional signal on 6B Apr-Jun
2025, AND a contradiction: Lasso on forward_return = MODERATE (0.087 @ h=48)
while Logistic on bias_label = NOISE (AUC≈0.49). Three hypotheses — we test, we
do not assume:

  (1) a real BUG (label / forward_return / probe wiring) → would also kill a
      KNOWN-predictive baseline feature.
  (2) bias_label (triple-barrier) DESTROYS a signal that the raw forward_return
      retains → raw vs label comparison + a simpler sign() label would show it.
  (3) 6B Apr-Jun is genuinely hard intraday → a daily-horizon trend exists that
      intraday horizons don't capture.

CHECK 1 — SANITY BASELINE (rules out a bug):
  Pick momentum-style features that are KNOWN to carry at least mechanical
  autocorrelation predictivity (past return → next return persistence/reversal).
  If even these score 0 IC → the probe/label wiring is broken (a BUG).
  If they score non-zero → no wiring bug; the 0-signal on microstructure
  features is real.

CHECK 2 — RAW forward_return vs bias_label (isolates the label):
  Same feature set, three targets at the SAME horizon h:
    a) forward_return_h        (continuous — what Lasso liked)
    b) sign(forward_return_h)  (the SIMPLEST possible directional label)
    c) bias_label              (production triple-barrier, LONG/SHORT)
  If raw + sign() carry IC but bias_label is NOISE → the triple-barrier
  labelling is dropping the signal (hypothesis 2 confirmed).

CHECK 3 — DAILY horizon (tests hypothesis 3):
  Resample close to daily, compute next-day return, test whether a
  daily-momentum feature predicts the daily direction. If a clean daily trend
  exists that intraday horizons miss, this surfaces it.

R6: reuses ic_audit.compute_forward_returns / _safe_spearman / fdr_correct.
R1: every target is forward-looking ONLY; every feature is read at t.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.ic_audit import _safe_spearman, compute_forward_returns


# Known-predictive baselines for the sanity check (past returns → persistence).
SANITY_FEATURES: tuple[str, ...] = ("return_6b", "return_1h", "return_4h",
                                    "rsi_14", "macd_hist", "cvd_momentum")
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
IC_MODERATE = 0.05


def _ic(feat: np.ndarray, label: np.ndarray, min_n: int = 100) -> tuple[float, int]:
    mask = np.isfinite(feat) & np.isfinite(label)
    if mask.sum() < min_n:
        return float("nan"), int(mask.sum())
    rho, _ = _safe_spearman(feat[mask], label[mask])
    return float(rho), int(mask.sum())


# ── CHECK 1 — sanity baseline ──────────────────────────────────────────────
def check_1_sanity(df: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd = compute_forward_returns(close, horizons, session_breaks=breaks)
    rows = []
    present = [f for f in SANITY_FEATURES if f in df.columns]
    for f in present:
        feat = pd.to_numeric(df[f], errors="coerce").to_numpy(np.float64)
        ics = {h: _ic(feat, fwd[h])[0] for h in horizons}
        best = max((abs(v) for v in ics.values() if np.isfinite(v)), default=float("nan"))
        rows.append({"feature": f, "ic_per_h": {str(h): ics[h] for h in horizons},
                     "best_abs_ic": best})
    any_signal = any(r["best_abs_ic"] >= IC_MODERATE for r in rows if np.isfinite(r["best_abs_ic"]))
    return {
        "features_present": present,
        "results": rows,
        "any_baseline_reaches_moderate": bool(any_signal),
        "verdict": ("NO_BUG (a baseline carries signal)" if any_signal
                    else "POSSIBLE_BUG (even baselines score ~0 — check wiring)"),
    }


# ── CHECK 2 — raw forward_return vs sign() vs bias_label ───────────────────
def check_2_label_destruction(
    df: pd.DataFrame, test_features: tuple[str, ...], horizons: tuple[int, ...],
) -> dict[str, Any]:
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd = compute_forward_returns(close, horizons, session_breaks=breaks)

    bias = (pd.to_numeric(df["bias_label"], errors="coerce").to_numpy()
            if "bias_label" in df.columns else None)
    bias_signed = None
    if bias is not None:
        bias_signed = np.where(bias == 0, 1.0, np.where(bias == 1, -1.0, np.nan))

    present = [f for f in test_features if f in df.columns]
    rows = []
    for f in present:
        feat = pd.to_numeric(df[f], errors="coerce").to_numpy(np.float64)
        per_h = {}
        for h in horizons:
            ic_raw, _ = _ic(feat, fwd[h])
            ic_sign, _ = _ic(feat, np.sign(fwd[h]))
            per_h[str(h)] = {"ic_raw_return": ic_raw, "ic_sign_return": ic_sign}
        ic_bias = _ic(feat, bias_signed)[0] if bias_signed is not None else float("nan")
        per_h["bias_label_signed"] = {"ic": ic_bias}
        rows.append({"feature": f, "per_horizon": per_h})

    # Aggregate diagnosis: best |IC| on raw vs sign vs bias across features/horizons
    def _best(kind: str) -> float:
        vals = []
        for r in rows:
            for h in horizons:
                vals.append(abs(r["per_horizon"][str(h)][kind]))
        return float(max([v for v in vals if np.isfinite(v)], default=float("nan")))

    best_raw = _best("ic_raw_return")
    best_sign = _best("ic_sign_return")
    best_bias = float(max([abs(r["per_horizon"]["bias_label_signed"]["ic"])
                           for r in rows
                           if np.isfinite(r["per_horizon"]["bias_label_signed"]["ic"])],
                          default=float("nan")))
    label_destroys = (np.isfinite(best_raw) and best_raw >= IC_MODERATE
                      and np.isfinite(best_bias) and best_bias < IC_MODERATE)
    return {
        "features_tested": present,
        "results": rows,
        "best_abs_ic_raw_return": best_raw,
        "best_abs_ic_sign_return": best_sign,
        "best_abs_ic_bias_label": best_bias,
        "label_destroys_signal": bool(label_destroys),
        "verdict": ("LABEL_DESTROYS_SIGNAL (raw carries IC, bias_label does not)"
                    if label_destroys else
                    "consistent (raw and bias_label agree — label is not the culprit)"),
    }


# ── CHECK 3 — daily horizon ────────────────────────────────────────────────
def check_3_daily(df: pd.DataFrame) -> dict[str, Any]:
    if "ts_event" not in df.columns or "close" not in df.columns:
        return {"skipped": True, "reason": "ts_event/close missing"}
    d = df[["ts_event", "close"]].copy()
    d["ts_event"] = pd.to_datetime(d["ts_event"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts_event"]).set_index("ts_event")
    daily = d["close"].resample("1D").last().dropna()
    if len(daily) < 20:
        return {"skipped": True, "reason": f"only {len(daily)} daily bars"}

    # daily next-day return
    next_ret = daily.pct_change(1).shift(-1).to_numpy()
    # daily momentum features (past)
    mom_1 = daily.pct_change(1).to_numpy()            # yesterday's return
    mom_3 = daily.pct_change(3).to_numpy()            # 3-day momentum
    mom_5 = daily.pct_change(5).to_numpy()
    # Daily samples are intrinsically few (a 3-month run is ~65 trading days);
    # use a lower min_n so the daily IC is computed (flagged as indicative).
    out = {}
    for name, feat in (("daily_mom_1", mom_1), ("daily_mom_3", mom_3), ("daily_mom_5", mom_5)):
        ic_dir, n = _ic(feat, next_ret, min_n=30)
        ic_mag, _ = _ic(feat, np.abs(next_ret), min_n=30)
        out[name] = {"ic_directional": ic_dir, "ic_magnitude": ic_mag, "n_days": n}
    n_daily = int(len(daily))
    best_daily_dir = max((abs(v["ic_directional"]) for v in out.values()
                          if np.isfinite(v["ic_directional"])), default=float("nan"))

    # NULL TEST (critical — small daily samples produce large |IC| by chance):
    # shuffle next_ret many times, record the 95th percentile of |IC| under the
    # null. A real daily trend must exceed BOTH the absolute floor AND this null.
    rng = np.random.RandomState(0)
    null_abs = []
    nr = next_ret.copy()
    finite = np.isfinite(mom_1) & np.isfinite(nr)
    if finite.sum() >= 30:
        m1f, nrf = mom_1[finite], nr[finite]
        for _ in range(500):
            rho, _ = _safe_spearman(m1f, rng.permutation(nrf))
            null_abs.append(abs(float(rho)))
    null_p95 = float(np.percentile(null_abs, 95)) if null_abs else float("nan")

    above_null = (np.isfinite(best_daily_dir) and np.isfinite(null_p95)
                  and best_daily_dir > null_p95)
    real_trend = (np.isfinite(best_daily_dir) and best_daily_dir >= 0.15 and above_null)
    return {
        "n_daily_bars": n_daily,
        "results": out,
        "best_abs_daily_directional_ic": best_daily_dir,
        "null_p95_daily_ic": null_p95,
        "above_null": bool(above_null),
        "note": ("daily sample is small — IC compared against a shuffle null "
                 "to reject small-sample chance"
                 if n_daily < 120 else "daily sample adequate"),
        "verdict": ("DAILY_TREND_PRESENT (a daily-horizon directional IC exists, above null)"
                    if real_trend
                    else "no clear daily directional IC (within small-sample null)"),
    }


def run_label_diagnostics(
    features_parquet: Path, output_dir: Path,
    test_features: tuple[str, ...] = (
        "cvd_bar_5m", "order_flow_imbalance", "vwap_z_score",
        "dist_to_session_high_atr", "dist_to_vwap_atr", "hawkes_intrabar_sum",
    ),
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    df = pd.read_parquet(features_parquet)
    c1 = check_1_sanity(df, horizons)
    c2 = check_2_label_destruction(df, test_features, horizons)
    c3 = check_3_daily(df)
    summary = {
        "features_parquet": str(features_parquet),
        "n_rows": int(len(df)),
        "horizons": list(horizons),
        "check_1_sanity_baseline": c1,
        "check_2_label_destruction": c2,
        "check_3_daily_horizon": c3,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "label_diagnostics_report.txt", summary)
    return summary


def _write_report(path: Path, s: dict) -> None:
    L = []
    L.append("═" * 92)
    L.append("Label diagnostics — why 0 directional signal? (3 decisive checks)")
    L.append("═" * 92)
    L.append(f"parquet : {s['features_parquet']}   n_rows: {s['n_rows']:,}")
    L.append("")
    c1 = s["check_1_sanity_baseline"]
    L.append("CHECK 1 — SANITY BASELINE (known-predictive features; 0 here ⇒ BUG)")
    L.append("-" * 92)
    for r in c1["results"]:
        L.append(f"  {r['feature']:22s} best |IC| = {r['best_abs_ic']:.3f}")
    L.append(f"  → {c1['verdict']}")
    L.append("")
    c2 = s["check_2_label_destruction"]
    L.append("CHECK 2 — RAW forward_return vs sign() vs bias_label (same features)")
    L.append("-" * 92)
    L.append(f"  best |IC| raw forward_return : {c2['best_abs_ic_raw_return']:.3f}")
    L.append(f"  best |IC| sign(forward_ret)  : {c2['best_abs_ic_sign_return']:.3f}")
    L.append(f"  best |IC| bias_label(signed) : {c2['best_abs_ic_bias_label']:.3f}")
    L.append(f"  → {c2['verdict']}")
    L.append("")
    c3 = s["check_3_daily_horizon"]
    L.append("CHECK 3 — DAILY horizon")
    L.append("-" * 92)
    if c3.get("skipped"):
        L.append(f"  SKIPPED: {c3.get('reason')}")
    else:
        L.append(f"  daily bars: {c3['n_daily_bars']}   ({c3['note']})")
        for name, v in c3["results"].items():
            L.append(f"  {name:14s} IC(dir)={v['ic_directional']:+.3f}  IC(|r|)={v['ic_magnitude']:+.3f}")
        L.append(f"  → {c3['verdict']}")
    L.append("")
    L.append("DECISION MAP:")
    L.append("  C1 BUG          → fix wiring before any further interpretation")
    L.append("  C1 ok + C2 destroys → the triple-barrier label is dropping the weak signal")
    L.append("  C1 ok + C2 consistent + C3 daily-trend → intraday-hard, daily edge exists")
    L.append("  C1 ok + C2 consistent + C3 none → genuinely no directional edge this period")
    L.append("═" * 92)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Label diagnostics — 3 checks for the 0-directional puzzle")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    args = p.parse_args()
    s = run_label_diagnostics(args.features, args.output, horizons=tuple(args.horizons))
    print(f"\n  CHECK 1 (sanity): {s['check_1_sanity_baseline']['verdict']}")
    print(f"  CHECK 2 (label) : {s['check_2_label_destruction']['verdict']}")
    print(f"  CHECK 3 (daily) : {s['check_3_daily_horizon'].get('verdict', 'skipped')}")
    print(f"  report: {args.output}/label_diagnostics_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
