"""sequence_probe — does the SEQUENCE of features predict direction (or only vol)?

decoder_probe asks: does feature[t] predict forward_return[t..t+h]?
This probe asks the user's distinct question: does a SHORT SEQUENCE of feature
values (slope over 3-6 bars, consecutive-close runs, OFI/hawkes acceleration)
predict a DIRECTIONAL forward return — or only volatility magnitude?

R4: simpler-first. We do NOT skip to LSTM. We derive a handful of semantic
sequence statistics, apply the SAME decoder_probe machinery (Stage 1 + Stage 2
+ Stage 3 + null + causality + purged CV + a priori thresholds), and additionally
report IC(directional) vs IC(|return|) per derived feature so the user's
"direction or volatility" question is answered explicitly.

DERIVED FEATURES (semantic, user-specified, R1-causal by construction):
  • CVD slope:               cvd_bar_5m_slope_3, cvd_bar_5m_slope_6
  • OFI / hawkes acceleration: order_flow_imbalance_mom_3/6, hawkes_intrabar_sum_mom_3/6
  • level/vwap trajectories:  dist_to_session_high_atr_slope_3,
                               dist_to_vwap_atr_slope_3, vwap_z_score_slope_3
  • consecutive-close runs:   consec_up_close, consec_down_close (Markov-style counters)

Every derived value at row t uses ONLY rows [t-w+1 .. t] — verified by a
teeth test (mutating bars after t must NOT change derived[t]).

R6: imports decoder_probe.run_probe via its new df= argument, so the full
stage hierarchy + null-test + causality probe + R10 thresholds apply
uniformly. Adds one orthogonal pass: directional-vs-magnitude IC comparison.
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

from tools.diagnostics.decoder_probe import (
    run_probe as _decoder_run_probe,
    DEFAULT_HORIZONS, DEFAULT_K_FOLDS, DEFAULT_N_NULL,
    IC_NOISE, IC_MODERATE, IC_STRONG, IC_SUSPECT,
)
from tools.diagnostics.ic_audit import _safe_spearman, compute_forward_returns


# Base features used for sequence derivation (a priori — semantic, not fishing)
BASE_FEATURES_FOR_SEQUENCE: tuple[str, ...] = (
    "cvd_bar_5m",                  # CVD trajectory
    "order_flow_imbalance",        # OFI trajectory
    "hawkes_intrabar_sum",         # intrabar activity trajectory (≡ tick_count, kept for naming)
    "vwap_z_score",                # rolling-z trajectory
    "dist_to_session_high_atr",    # London-distance trajectory
    "dist_to_vwap_atr",            # vwap-distance trajectory
)

# Window sizes for slope / momentum
DEFAULT_WINDOWS: tuple[int, ...] = (3, 6)

# A directional verdict requires the directional IC to be at least this fraction
# of the magnitude IC (otherwise the feature is volatility-only).
DIRECTIONAL_TO_VOL_RATIO = 0.5


# ── Sequence statistic primitives (all causal) ─────────────────────────────
def _rolling_slope(series: pd.Series, w: int) -> np.ndarray:
    """OLS slope of last w values for each row. NaN before w samples are
    accumulated. R1-causal: row t uses [t-w+1 .. t] only."""
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    x_dev = x - x.mean()
    xx = float((x_dev ** 2).sum())
    if xx <= 0:
        return out
    for i in range(w - 1, n):
        win = y[i - w + 1: i + 1]
        if np.isfinite(win).all():
            y_dev = win - win.mean()
            out[i] = float((x_dev * y_dev).sum() / xx)
    return out


def _rolling_momentum(series: pd.Series, w: int) -> np.ndarray:
    """feature[t] - feature[t-w]. R1-causal."""
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(y), np.nan, dtype=np.float64)
    if len(y) > w:
        out[w:] = y[w:] - y[:-w]
    return out


def _consecutive_run(direction: np.ndarray, sign: int) -> np.ndarray:
    """Counter of consecutive bars where sign(direction) == sign, ending at t.
    Resets on direction change. R1-causal: row t uses [0..t]."""
    out = np.zeros(len(direction), dtype=np.int32)
    cnt = 0
    for i, d in enumerate(direction):
        if d == sign:
            cnt += 1
        else:
            cnt = 0
        out[i] = cnt
    return out


# ── Build the derived-feature frame ────────────────────────────────────────
def build_sequence_features(
    df: pd.DataFrame,
    base_features: tuple[str, ...] = BASE_FEATURES_FOR_SEQUENCE,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (df_with_derived, list_of_derived_column_names).

    For each base feature × window: adds `<feat>_slope_<w>` and `<feat>_mom_<w>`.
    Also adds `consec_up_close` and `consec_down_close` from sign(close.diff()).
    All derived columns are R1-causal — verified by teeth test."""
    out = df.copy()
    derived: list[str] = []

    # Consecutive close runs (semantic feature the user explicitly asked for)
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        cdir = np.sign(close.diff()).fillna(0).astype(int).to_numpy()
        out["consec_up_close"] = _consecutive_run(cdir, +1)
        out["consec_down_close"] = _consecutive_run(cdir, -1)
        derived.extend(["consec_up_close", "consec_down_close"])

    # Slope + momentum per (base feature × window)
    for fname in base_features:
        if fname not in df.columns:
            continue
        series = pd.to_numeric(df[fname], errors="coerce")
        for w in windows:
            slope_name = f"{fname}_slope_{w}"
            mom_name = f"{fname}_mom_{w}"
            out[slope_name] = _rolling_slope(series, w)
            out[mom_name] = _rolling_momentum(series, w)
            derived.extend([slope_name, mom_name])

    return out, derived


# ── Directional-vs-magnitude IC comparison (orthogonal to decoder_probe) ──
def directional_vs_magnitude(
    df: pd.DataFrame, derived: list[str], horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    """For each derived feature × horizon: compute IC(feature, forward_return)
    and IC(feature, |forward_return|). Classify per (feature, horizon):
        DIRECTIONAL     — |IC(dir)| >= IC_MODERATE AND |IC(dir)| >= |IC(|r|)| × ratio
        VOLATILITY_ONLY — |IC(|r|)| >= IC_MODERATE AND |IC(dir)| <  |IC(|r|)| × ratio
        SUSPECT_LEAKAGE — |IC(dir)| >= IC_SUSPECT
        NOISE           — both < IC_MODERATE"""
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd_by_h = compute_forward_returns(close, horizons, session_breaks=breaks)

    out: list[dict[str, Any]] = []
    for fname in derived:
        feat = pd.to_numeric(df[fname], errors="coerce").to_numpy(np.float64)
        per_h: dict[int, dict[str, Any]] = {}
        for h in horizons:
            label = fwd_by_h[h]
            # IC against directional forward_return
            ic_dir, _ = _safe_spearman(feat, label)
            # IC against |forward_return| (magnitude / vol)
            ic_mag, _ = _safe_spearman(feat, np.abs(label))
            ad, am = abs(float(ic_dir)), abs(float(ic_mag))
            if ad >= IC_SUSPECT:
                v = "SUSPECT_LEAKAGE"
            elif ad >= IC_MODERATE and ad >= am * DIRECTIONAL_TO_VOL_RATIO:
                v = "DIRECTIONAL"
            elif am >= IC_MODERATE and ad < am * DIRECTIONAL_TO_VOL_RATIO:
                v = "VOLATILITY_ONLY"
            else:
                v = "NOISE"
            per_h[h] = {
                "ic_directional": float(ic_dir), "ic_magnitude": float(ic_mag),
                "abs_dir": ad, "abs_mag": am, "verdict": v,
            }
        # Feature-level: directional if any horizon directional; else vol_only if any vol; etc.
        verdicts = [per_h[h]["verdict"] for h in horizons]
        if "SUSPECT_LEAKAGE" in verdicts:
            feature_v = "SUSPECT_LEAKAGE"
        elif "DIRECTIONAL" in verdicts:
            feature_v = "DIRECTIONAL"
        elif "VOLATILITY_ONLY" in verdicts:
            feature_v = "VOLATILITY_ONLY"
        else:
            feature_v = "NOISE"
        out.append({
            "feature": fname, "per_horizon": {str(h): per_h[h] for h in horizons},
            "verdict": feature_v,
        })
    return out


# ── Top-level orchestrator ────────────────────────────────────────────────
def run_sequence_probe(
    features_parquet: Path, output_dir: Path,
    base_features: tuple[str, ...] = BASE_FEATURES_FOR_SEQUENCE,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    k_folds: int = DEFAULT_K_FOLDS,
    n_null: int = DEFAULT_N_NULL,
    seed: int = 42,
) -> dict[str, Any]:
    df = pd.read_parquet(features_parquet)
    df_derived, derived_cols = build_sequence_features(df, base_features, windows)
    if not derived_cols:
        raise RuntimeError("no derived columns built — check base features exist in parquet")

    # Run the full decoder_probe machinery on derived features (R6 reuse)
    decoder_summary = _decoder_run_probe(
        features_parquet=None, output_dir=output_dir, df=df_derived,
        feature_names=tuple(derived_cols),
        horizons=horizons, k_folds=k_folds, n_null=n_null, seed=seed,
    )

    # Orthogonal pass: directional-vs-magnitude IC per derived feature
    dir_vs_mag = directional_vs_magnitude(df_derived, derived_cols, horizons)

    summary = {
        "features_parquet": str(features_parquet),
        "n_rows": int(len(df_derived)),
        "base_features": list(base_features),
        "windows": list(windows),
        "horizons": list(horizons),
        "derived_columns": derived_cols,
        "decoder_probe_summary": decoder_summary,
        "directional_vs_magnitude": dir_vs_mag,
        "directional_to_vol_ratio_threshold": DIRECTIONAL_TO_VOL_RATIO,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequence_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "sequence_probe_report.txt", summary)
    return summary


def _write_report(path: Path, s: dict) -> None:
    L: list[str] = []
    L.append("═" * 100)
    L.append("Sequence probe — does the SEQUENCE of features predict DIRECTION (or only volatility)?")
    L.append("═" * 100)
    L.append(f"features parquet : {s['features_parquet']}")
    L.append(f"n_rows           : {s['n_rows']:,}   horizons: {s['horizons']}   windows: {s['windows']}")
    L.append(f"derived columns  : {len(s['derived_columns'])}")
    L.append("")
    L.append("DIRECTIONAL vs VOLATILITY-ONLY (orthogonal IC comparison):")
    L.append("-" * 100)
    L.append(f"  {'feature':38s} {'best |IC(dir)|':>15s} {'best |IC(|r|)|':>15s} {'verdict':>20s}")
    for r in s["directional_vs_magnitude"]:
        per_h = r["per_horizon"]
        best_dir = max(per_h[h]["abs_dir"] for h in per_h)
        best_mag = max(per_h[h]["abs_mag"] for h in per_h)
        L.append(f"  {r['feature']:38s} {best_dir:>14.3f}  {best_mag:>14.3f}  {r['verdict']:>20s}")
    L.append("")
    L.append("DECODER_PROBE summary on derived (Stage 1 + 2 + 3, with null + causality + purged CV):")
    L.append("-" * 100)
    dp = s["decoder_probe_summary"]
    n_mod = sum(1 for r in dp["stage_1_univariate"]
                if isinstance(r, dict) and r.get("verdict") in {"MODERATE", "STRONG"})
    L.append(f"  Stage 1: {n_mod}/{len(dp['stage_1_univariate'])} derived features MODERATE+")
    s2 = dp["stage_2_lasso"]
    if s2.get("skipped"):
        L.append(f"  Stage 2: SKIPPED ({s2.get('reason')})")
    else:
        L.append(f"  Stage 2: best h={s2.get('best_horizon')}  |IC|={s2.get('best_ic',0):.3f}  → {s2.get('verdict')}")
    s3 = dp["stage_3_logistic_bias"]
    if s3.get("skipped"):
        L.append(f"  Stage 3: SKIPPED ({s3.get('reason')})")
    else:
        L.append(f"  Stage 3: AUC={s3.get('auc_mean',0):.3f} ± {s3.get('auc_std',0):.3f}  → {s3.get('verdict')}")
    L.append("")
    L.append(f"KEY: DIRECTIONAL = |IC(dir)|≥{IC_MODERATE} AND ≥ {DIRECTIONAL_TO_VOL_RATIO}× |IC(|r|)|")
    L.append(f"     VOLATILITY_ONLY = |IC(|r|)|≥{IC_MODERATE} AND |IC(dir)| < {DIRECTIONAL_TO_VOL_RATIO}× |IC(|r|)|")
    L.append("R7 NOTE: a DIRECTIONAL verdict shows correlation, NOT proof of trading edge.")
    L.append("═" * 100)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Sequence probe: derived-sequence-feature predictivity")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    p.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    summary = run_sequence_probe(
        args.features, args.output,
        horizons=tuple(args.horizons), k_folds=args.k_folds,
        n_null=args.n_null, seed=args.seed,
    )
    dp = summary["decoder_probe_summary"]
    print(f"\n  derived columns: {len(summary['derived_columns'])}")
    n_dir = sum(1 for r in summary["directional_vs_magnitude"] if r["verdict"] == "DIRECTIONAL")
    n_vol = sum(1 for r in summary["directional_vs_magnitude"] if r["verdict"] == "VOLATILITY_ONLY")
    n_noise = sum(1 for r in summary["directional_vs_magnitude"] if r["verdict"] == "NOISE")
    print(f"  DIRECTIONAL: {n_dir}   VOLATILITY_ONLY: {n_vol}   NOISE: {n_noise}")
    print(f"  Stage 2 (Lasso on derived) : {dp['stage_2_lasso'].get('verdict', 'skipped')}")
    print(f"  Stage 3 (Logistic, bias)   : {dp['stage_3_logistic_bias'].get('verdict', 'skipped')}")
    print(f"  report: {args.output}/sequence_probe_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
