"""Leakage shift-test — focused causal probe on K features × K shifts × 2 labels.

Complements tools/diagnostics/ic_audit.py (broad feature audit). This tool asks
ONE focused question per feature: does its IC against the label behave like a
CAUSAL signal under shift permutations, or does it betray future information?

Methodology (López de Prado AFML §3.6 / §7):
    baseline    : IC(f[i],            y[i])
    past +k     : IC(f.shift(+k)[i],  y[i])    # using older feature
    future -1   : IC(f.shift(-1)[i],  y[i])    # using TOMORROW's feature ← killer

A causal feature decays under past shifts at a rate compatible with its own
lag-1 autocorrelation. A leaky feature shows one of:
  • past_shift collapse FAR faster than autocorr explains   → LEAK_PAST_ALIGNMENT
  • future_shift IC higher than baseline                    → LEAK_FUTURE
  • non-monotonic decay across past shifts                  → SUSPECT_NONMONOTONIC

R3 (purging/embargo): IC RATIOS — not p-values — drive the verdict; ratios are
robust to label overlap because the same overlap structure appears in baseline
and shifted columns and cancels. The last max(horizons) rows are embargoed
(forward_return unobservable).

R4 (suspect first): default verdict on borderline cases is SUSPECT, not PASS.

Imports primitives from ic_audit (no duplication of Spearman / forward_returns).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.ic_audit import _safe_spearman, compute_forward_returns


DEFAULT_FEATURES: tuple[str, ...] = (
    "event_score",
    "london_sess_high",
    "current_vwap",
    "dist_to_session_high_atr",
)
DEFAULT_HORIZONS: tuple[int, ...] = (6, 12, 24)
DEFAULT_SHIFTS: tuple[int, ...] = (-1, 0, 1, 5, 10)

# ── Decision-tree thresholds (calibrated; see tests/test_leakage_shift_test.py)
NOISE_BASELINE = 0.03
HIGH_BASELINE = 0.10
# Future-shift leak (universal). The KEY tell is multiplicative — future_IC
# significantly larger than baseline_IC means tomorrow's feature predicts today
# better than today's, which a causal signal cannot do. We use BOTH a relative
# ratio (catches modest-IC leaks like 0.06→0.08) and a tiny absolute margin
# (filters noise jitter).
FUTURE_LEAK_RATIO_HARD = 1.30        # decisive: future > baseline × 1.30
FUTURE_LEAK_RATIO_SOFT = 1.10        # suspect: future > baseline × 1.10
FUTURE_LEAK_ABS_MIN = 0.05           # absolute floor for any future-leak verdict
FUTURE_LEAK_HARD_MARGIN = 0.01
FUTURE_LEAK_SOFT_MARGIN = 0.005
# Past-shift leak (anchored to feature's lag-1 autocorr).
PAST_LEAK_RATIO_HARD = 0.4
PAST_LEAK_RATIO_SOFT = 0.7
LOW_RHO_FLOOR = 0.10
LOW_PAST_RATIO_LOW_BASELINE = 0.3
HIGH_AUTOCORR = 0.7
EVENT_SCORE_TIER_MARGIN = 0.05


def _shifted_ic(feature: np.ndarray, label: np.ndarray, shift: int) -> tuple[float, int]:
    """Spearman IC between feature.shift(shift) and label.

    pandas convention:
      shift=+k → result[i] = feature[i-k]  (older value)
      shift=-k → result[i] = feature[i+k]  (newer value)
    """
    f = pd.Series(feature).shift(shift).to_numpy()
    mask = np.isfinite(f) & np.isfinite(label)
    if mask.sum() < 30:
        return float("nan"), int(mask.sum())
    rho, _ = _safe_spearman(f[mask], label[mask])
    return float(rho), int(mask.sum())


def _lag1_autocorr(feature: np.ndarray) -> float:
    f = pd.Series(feature).astype(np.float64)
    a = f.iloc[1:].to_numpy()
    b = f.iloc[:-1].to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return 0.0
    if a[mask].std() < 1e-12 or b[mask].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _classify(baseline: float, past1: float, past5: float, future: float, rho: float) -> str:
    """Verdict per the design tree. Inputs are |IC| magnitudes; rho can be signed."""
    if not np.isfinite(baseline) or baseline < NOISE_BASELINE:
        return "NOISE"

    abs_rho = abs(rho)

    # ── Future-shift checks (universal — relative + small absolute floor) ──
    # A causal feature cannot have future_IC > baseline_IC (tomorrow's value
    # cannot predict today's outcome better than today's value). When it does,
    # the feature carries information from beyond bar i.
    if np.isfinite(future) and future > FUTURE_LEAK_ABS_MIN:
        if future > baseline * FUTURE_LEAK_RATIO_HARD + FUTURE_LEAK_HARD_MARGIN:
            return "LEAK_FUTURE"
        if future > baseline * FUTURE_LEAK_RATIO_SOFT + FUTURE_LEAK_SOFT_MARGIN and baseline > 0.04:
            return "SUSPECT_FUTURE"

    # ── Past-shift PEAK: past+1 IC exceeds baseline IC by a meaningful margin.
    # That's a structural mistimings signature — the feature's signal is
    # locked one bar AHEAD of where it should be (e.g. feature[i] secretly =
    # label[i+1]). Past-shift+1 then aligns feature[i-1]=label[i] with label[i]
    # → perfect IC, exceeding the baseline measurement.
    if np.isfinite(past1) and past1 > baseline + 0.03 and past1 > 0.10:
        return "LEAK_PAST_PEAK"

    # ── Past-shift checks (anchored to autocorr when baseline is high) ──
    if baseline >= HIGH_BASELINE:
        if abs_rho < LOW_RHO_FLOOR:
            # iid-style feature: past-shift collapse is normal (no autocorr carrier).
            # Only future_leak can flag it; past-collapse is non-diagnostic here.
            return "PASS_LOW_AUTOCORR"
        expected_past1 = baseline * abs_rho
        if np.isfinite(past1):
            if past1 < PAST_LEAK_RATIO_HARD * expected_past1:
                return "LEAK_PAST_ALIGNMENT"
            if past1 < PAST_LEAK_RATIO_SOFT * expected_past1:
                return "SUSPECT_HIGH_DECAY"
        if np.isfinite(past5) and np.isfinite(past1) and past5 > past1 + 0.02:
            return "SUSPECT_NONMONOTONIC"
        return "PASS"

    # 0.03 ≤ baseline < 0.10  (modest signal — past-shift sensitivity is weak)
    if abs_rho > HIGH_AUTOCORR and np.isfinite(past1) and past1 < LOW_PAST_RATIO_LOW_BASELINE * baseline:
        return "SUSPECT"
    return "PASS"


def _bias_label_signed(df: pd.DataFrame) -> np.ndarray | None:
    """LONG=0 → +1, SHORT=1 → -1, NEUTRAL=2 → NaN. Returns None if column missing."""
    if "bias_label" not in df.columns:
        return None
    b = pd.to_numeric(df["bias_label"], errors="coerce")
    s = np.full(len(b), np.nan, dtype=np.float64)
    s[b == 0] = 1.0
    s[b == 1] = -1.0
    return s


def _embargo(label: np.ndarray, embargo_bars: int) -> np.ndarray:
    """Set last embargo_bars rows to NaN (forward window unobservable)."""
    out = np.asarray(label, dtype=np.float64).copy()
    if embargo_bars > 0 and len(out) > embargo_bars:
        out[-embargo_bars:] = np.nan
    return out


def _mean_abs(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))


def run_shift_test(
    features_path: Path,
    output_dir: Path,
    feature_names: tuple[str, ...] = DEFAULT_FEATURES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    shifts: tuple[int, ...] = DEFAULT_SHIFTS,
) -> dict:
    """Run the targeted shift-test and write report + summary. Returns summary dict."""
    df = pd.read_parquet(features_path)
    if "close" not in df.columns:
        raise RuntimeError("features parquet missing 'close' — cannot build forward_return")
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd_by_h = compute_forward_returns(close, tuple(horizons), session_breaks=breaks)
    embargo_bars = max(horizons)
    bias_signed = _bias_label_signed(df)

    results: list[dict] = []
    skipped: list[str] = []
    for fname in feature_names:
        if fname not in df.columns:
            skipped.append(fname)
            continue
        feature = pd.to_numeric(df[fname], errors="coerce").to_numpy(dtype=np.float64)
        rho = _lag1_autocorr(feature)

        # ── vs forward_return (averaged across horizons) ──
        per_h: dict[int, dict[int, float]] = {}
        for h in horizons:
            label_h = _embargo(fwd_by_h[h], embargo_bars)
            ic_at_shift: dict[int, float] = {}
            for s in shifts:
                ic, _ = _shifted_ic(feature, label_h, s)
                ic_at_shift[s] = ic
            per_h[h] = ic_at_shift

        baseline = _mean_abs([per_h[h].get(0, np.nan) for h in horizons])
        past1 = _mean_abs([per_h[h].get(1, np.nan) for h in horizons]) if 1 in shifts else float("nan")
        past5 = _mean_abs([per_h[h].get(5, np.nan) for h in horizons]) if 5 in shifts else float("nan")
        future = _mean_abs([per_h[h].get(-1, np.nan) for h in horizons]) if -1 in shifts else float("nan")
        verdict_fwd = _classify(baseline, past1, past5, future, rho)

        # ── vs bias_label (signed, NEUTRAL dropped) ──
        ic_at_shift_b: dict[int, float] = {}
        if bias_signed is not None:
            lab_b = _embargo(bias_signed, embargo_bars)
            for s in shifts:
                ic, _ = _shifted_ic(feature, lab_b, s)
                ic_at_shift_b[s] = ic
            b_baseline = abs(ic_at_shift_b.get(0, float("nan")))
            b_past1 = abs(ic_at_shift_b.get(1, float("nan")))
            b_past5 = abs(ic_at_shift_b.get(5, float("nan")))
            b_future = abs(ic_at_shift_b.get(-1, float("nan")))
            verdict_bias = _classify(b_baseline, b_past1, b_past5, b_future, rho)
        else:
            verdict_bias = "MISSING_LABEL"

        results.append({
            "feature": fname,
            "autocorr_lag1": rho,
            "vs_forward_return": {
                "per_horizon": {str(h): {str(s): per_h[h][s] for s in shifts} for h in horizons},
                "baseline": baseline, "past1": past1, "past5": past5, "future": future,
                "verdict": verdict_fwd,
            },
            "vs_bias_label": {
                "shifts": {str(s): ic_at_shift_b.get(s, float("nan")) for s in shifts},
                "verdict": verdict_bias,
            },
        })

    # ── event_score TIER-leak structural probe ──
    extra: dict | None = None
    if "event_score" in df.columns and bias_signed is not None:
        f = pd.to_numeric(df["event_score"], errors="coerce").to_numpy(dtype=np.float64)
        lab_b = _embargo(bias_signed, embargo_bars)
        ic_0, _ = _shifted_ic(f, lab_b, 0)
        ic_m1, _ = _shifted_ic(f, lab_b, -1)
        verdict_extra = "OK"
        if np.isfinite(ic_0) and np.isfinite(ic_m1):
            if abs(ic_m1) > abs(ic_0) + EVENT_SCORE_TIER_MARGIN:
                verdict_extra = "LEAK_TIER_KNOWS_LABEL"
        extra = {
            "label": "bias_label (signed)",
            "ic_at_shift_0": ic_0,
            "ic_at_shift_-1": ic_m1,
            "verdict": verdict_extra,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "features_parquet": str(features_path),
        "feature_names": list(feature_names),
        "skipped_features_not_in_parquet": skipped,
        "horizons": list(horizons),
        "shifts": list(shifts),
        "embargo_bars": embargo_bars,
        "n_rows_total": int(len(df)),
        "results": results,
        "event_score_extra_probe": extra,
    }
    (output_dir / "leakage_shift_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "leakage_shift_report.txt", summary)
    return summary


def _write_report(path: Path, summary: dict) -> None:
    L: list[str] = []
    L.append("═" * 78)
    L.append("Leakage shift-test — focused causal probe")
    L.append("═" * 78)
    L.append(f"features  : {summary['features_parquet']}")
    L.append(f"n_rows    : {summary['n_rows_total']:,}   embargo: {summary['embargo_bars']} bars")
    L.append(f"shifts    : {summary['shifts']}")
    L.append(f"horizons  : {summary['horizons']}")
    if summary["skipped_features_not_in_parquet"]:
        L.append(f"skipped (not in parquet): {summary['skipped_features_not_in_parquet']}")
    L.append("")
    for r in summary["results"]:
        L.append("─" * 78)
        L.append(f"FEATURE: {r['feature']}    autocorr(lag-1) = {r['autocorr_lag1']:+.3f}")
        f = r["vs_forward_return"]
        L.append(f"  vs forward_return  baseline={f['baseline']:.3f}  past+1={f['past1']:.3f}  "
                 f"past+5={f['past5']:.3f}  future-1={f['future']:.3f}  → {f['verdict']}")
        b = r["vs_bias_label"]
        if b["verdict"] != "MISSING_LABEL":
            shifts_str = "  ".join(f"{k}={float(v):+.3f}" for k, v in b["shifts"].items())
            L.append(f"  vs bias_label      {shifts_str}  → {b['verdict']}")
        else:
            L.append("  vs bias_label      MISSING (bias_label column absent)")
    if summary.get("event_score_extra_probe"):
        e = summary["event_score_extra_probe"]
        L.append("")
        L.append("─" * 78)
        L.append("event_score TIER-leak structural probe:")
        L.append(f"  IC vs bias_label at shift= 0:  {e['ic_at_shift_0']:+.3f}")
        L.append(f"  IC vs bias_label at shift=-1:  {e['ic_at_shift_-1']:+.3f}")
        L.append(f"  verdict: {e['verdict']}")
    L.append("═" * 78)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Leakage shift-test for refinery features")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature", action="append", default=None,
                   help="repeatable; defaults to event_score / london_sess_high / "
                        "current_vwap / dist_to_session_high_atr")
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--shifts", nargs="+", type=int, default=list(DEFAULT_SHIFTS))
    args = p.parse_args()
    feature_names = tuple(args.feature) if args.feature else DEFAULT_FEATURES
    summary = run_shift_test(
        args.features, args.output,
        feature_names=feature_names,
        horizons=tuple(args.horizons),
        shifts=tuple(args.shifts),
    )
    for r in summary["results"]:
        print(f"  {r['feature']:30s}  fwd={r['vs_forward_return']['verdict']:25s}  "
              f"bias={r['vs_bias_label']['verdict']}")
    if summary.get("event_score_extra_probe"):
        print(f"  event_score TIER probe: {summary['event_score_extra_probe']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
