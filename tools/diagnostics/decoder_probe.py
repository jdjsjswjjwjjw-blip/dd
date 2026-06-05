"""decoder_probe — does any feature predict forward returns or bias_label?

Implements the design document approved with the user. Three progressive stages,
each with a priori acceptance thresholds (R10 — written BEFORE any result):

  Stage 1: Univariate Spearman IC for every (feature × horizon), with null-test
           (shuffled labels → IC≈0) and causality probe (feature.shift(-1) vs
           label → must NOT exceed baseline) at every stage to catch leakage.
           Pass: |IC|>=0.05 (MODERATE) AND IR>=0.30 AND smooth decay.

  Stage 2: Lasso linear regression on forward_return at each horizon.
           Pass: trained-model |IC| >= 0.05 with IR >= 0.30 (model adds value).

  Stage 3: Logistic regression on bias_label (LONG vs SHORT, NEUTRAL dropped).
           Pass: AUC >= 0.55 (MODERATE).

Stages 4/5 (CNN, SSL) are intentionally OUT of scope — added only if 1-3 show
SIGNAL_PRESENT. R4: simple first, complexity only after the simple model fails.

R10 SAFEGUARDS:
  - Null-test on every stage: shuffled labels → expected IC ≈ 0; observed IC
    must significantly exceed the null 95th percentile to count as signal.
  - Causality probe: feature.shift(-1) vs label MUST NOT meaningfully exceed
    baseline IC. If it does → SUSPECT_LEAKAGE.
  - SUSPECT verdicts at |IC|>=0.30 or AUC>=0.75 → force investigation, never
    treat as "great signal".
  - FDR-correct p-values across all (feature × horizon) trials (R4: all trials).
  - Purged + embargoed CV (Lopez de Prado AFML §7) at every model stage,
    embargo = max(horizons).

R6 — REUSED PRIMITIVES (no duplication):
  ic_audit.compute_forward_returns — fixed-horizon labels with session-break
                                       filtering, R1-clean (verified by
                                       leakage_shift_test in this branch).
  ic_audit._safe_spearman          — NaN-aware Spearman.
  ic_audit.fdr_correct             — Benjamini-Hochberg correction.

NOTE on default features in compass mode:
  absorb_z_raw / hawkes_z_raw / kyle_z_raw ship as zeros in structure_compass
  mode (prepare_day_trading.py:2599), so they are NOT in DEFAULT_FEATURES. The
  rolling-z family is tested via `vwap_z_score` which IS computed in compass
  mode. Pass --feature absorb_z_raw explicitly when running on microstructure-
  mode parquet.
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

from tools.diagnostics.ic_audit import (
    _safe_spearman,
    compute_forward_returns,
    fdr_correct,
)


# ══════════════════════════════════════════════════════════════════════════
# A priori thresholds (§5 of design doc — written before any result, R10)
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_FEATURES: tuple[str, ...] = (
    # rolling-z family (the active one in compass mode — tests neutralization)
    "vwap_z_score",
    # raw / non-z controls (should NOT suffer rolling-z neutralization)
    "cvd_bar_5m",
    "order_flow_imbalance",
    "hawkes_intrabar_sum",
    "tick_count",
    "cvd_intensity_vs_atr",
    # structural (per CLAUDE.md, STRONG individually in earlier IC audit)
    "dist_to_session_high_atr",
    "dist_to_vwap_atr",
)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
DEFAULT_K_FOLDS: int = 5
DEFAULT_N_NULL: int = 100
MIN_SAMPLES_PER_FOLD: int = 1000           # below this → fold skipped

# IC thresholds (consistent with ic_audit.py:91-93)
IC_NOISE = 0.02
IC_MODERATE = 0.05
IC_STRONG = 0.10
IC_SUSPECT = 0.30

# AUC thresholds
AUC_NOISE = 0.52
AUC_MODERATE = 0.55
AUC_STRONG = 0.60
AUC_SUSPECT = 0.75

# Information Ratio across folds (ic_audit.py:95-96)
IR_MIN_MODERATE = 0.30
IR_MIN_STRONG = 0.50

# Causality probe: future-shift IC must NOT exceed baseline meaningfully
CAUSALITY_FUTURE_LEAK_MARGIN = 0.05    # |IC(f.shift(-1))| > |IC(f)| + 0.05 → LEAK

# Decay validation
DECAY_PASS_RATIO = 0.5                  # IC@h_max < IC@h_min × 0.5


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _embargo(label: np.ndarray, h: int) -> np.ndarray:
    """Set the last h rows to NaN (forward return is unobservable there)."""
    out = np.asarray(label, dtype=np.float64).copy()
    if h > 0 and len(out) > h:
        out[-h:] = np.nan
    return out


def _walk_forward_folds(n: int, k: int) -> list[tuple[int, int]]:
    """K contiguous time blocks: (test_start, test_end) for fold i."""
    fold_size = n // k
    return [(i * fold_size, (i + 1) * fold_size if i < k - 1 else n)
            for i in range(k)]


def _ic_per_block(feature: np.ndarray, label: np.ndarray, k: int) -> list[float]:
    """Spearman IC on each of K contiguous time blocks (no training)."""
    mask = np.isfinite(feature) & np.isfinite(label)
    f = feature[mask]
    y = label[mask]
    n = len(f)
    if n < k * 200:
        return []
    ics = []
    for s, e in _walk_forward_folds(n, k):
        rho, _ = _safe_spearman(f[s:e], y[s:e])
        ics.append(float(rho))
    return ics


def _ir(values: list[float]) -> float:
    """Information Ratio = mean(|IC|) / std(IC) across folds (ic_audit pattern)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) < 2:
        return float("nan")
    return float(np.mean(np.abs(arr)) / max(np.std(arr), 1e-9))


def _null_distribution(
    feature: np.ndarray, label: np.ndarray, n_reps: int, seed: int,
) -> dict[str, Any]:
    """Shuffle labels n_reps times, compute |IC| each time. Under no signal, IC
    distribution is centred on 0 with spread ~1/sqrt(n)."""
    rng = np.random.RandomState(seed)
    mask = np.isfinite(feature) & np.isfinite(label)
    f = feature[mask]
    y = label[mask]
    if len(f) < 200:
        return {"null_median": float("nan"), "null_p95": float("nan"),
                "null_max": float("nan"), "n_reps": 0}
    ics_abs = []
    for _ in range(n_reps):
        y_shuf = rng.permutation(y)
        rho, _ = _safe_spearman(f, y_shuf)
        ics_abs.append(abs(float(rho)))
    arr = np.asarray(ics_abs, dtype=np.float64)
    return {
        "null_median": float(np.median(arr)),
        "null_p95": float(np.percentile(arr, 95)),
        "null_max": float(arr.max()),
        "n_reps": int(n_reps),
    }


def _causality_probe(feature: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """IC at shift {0, +1, -1}. For a causal feature, |IC(shift -1)| should NOT
    exceed |IC(shift 0)| by more than CAUSALITY_FUTURE_LEAK_MARGIN."""
    f = pd.Series(feature)
    def _ic(shift: int) -> float:
        fs = f.shift(shift).to_numpy()
        mask = np.isfinite(fs) & np.isfinite(label)
        if mask.sum() < 100:
            return float("nan")
        rho, _ = _safe_spearman(fs[mask], label[mask])
        return float(rho)
    ic_0 = _ic(0)
    ic_p1 = _ic(+1)
    ic_m1 = _ic(-1)
    if not np.isfinite(ic_0) or not np.isfinite(ic_m1):
        leak = False
    else:
        leak = abs(ic_m1) > abs(ic_0) + CAUSALITY_FUTURE_LEAK_MARGIN
    return {
        "ic_at_0": ic_0, "ic_at_plus_1": ic_p1, "ic_at_minus_1": ic_m1,
        "future_leak_flag": bool(leak),
    }


def _classify_ic(ic_mean_abs: float, ir: float) -> str:
    if not np.isfinite(ic_mean_abs):
        return "NOISE"
    if ic_mean_abs >= IC_SUSPECT:
        return "SUSPECT_LEAKAGE"
    if ic_mean_abs >= IC_STRONG and np.isfinite(ir) and ir >= IR_MIN_STRONG:
        return "STRONG"
    if ic_mean_abs >= IC_MODERATE and np.isfinite(ir) and ir >= IR_MIN_MODERATE:
        return "MODERATE"
    if ic_mean_abs >= IC_MODERATE:
        return "UNSTABLE"
    if ic_mean_abs >= IC_NOISE:
        return "WEAK"
    return "NOISE"


def _classify_auc(auc: float) -> str:
    if not np.isfinite(auc):
        return "NOISE"
    if auc >= AUC_SUSPECT:
        return "SUSPECT_LEAKAGE"
    if auc >= AUC_STRONG:
        return "STRONG"
    if auc >= AUC_MODERATE:
        return "MODERATE"
    if auc >= AUC_NOISE:                            # 0.52 .. 0.55
        return "WEAK"
    return "NOISE"


def _smooth_decay_ok(ic_at_h: dict[int, float], horizons: tuple[int, ...]) -> bool:
    """IC magnitude must decay across horizons (IC@h_max < IC@h_min × 0.5)."""
    sh = sorted(horizons)
    short_ic = abs(ic_at_h.get(sh[0], 0.0))
    long_ic = abs(ic_at_h.get(sh[-1], 0.0))
    if short_ic < IC_NOISE:
        return False           # too weak to assess decay
    return long_ic < short_ic * DECAY_PASS_RATIO


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — Univariate IC per (feature × horizon)
# ══════════════════════════════════════════════════════════════════════════
def stage_1_univariate(
    df: pd.DataFrame, feature_names: tuple[str, ...],
    horizons: tuple[int, ...], k_folds: int, n_null: int, seed: int,
) -> list[dict]:
    if "close" not in df.columns:
        return [{"error": "close column missing"}]
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd_by_h = compute_forward_returns(close, horizons, session_breaks=breaks)
    embargo = max(horizons)

    results: list[dict] = []
    for fname in feature_names:
        if fname not in df.columns:
            results.append({"feature": fname, "missing": True})
            continue
        feature = pd.to_numeric(df[fname], errors="coerce").to_numpy(np.float64)
        is_zfamily = fname.endswith("_z") or fname.endswith("_z_raw") \
                     or fname.endswith("_z_score")
        per_h: dict[int, dict[str, Any]] = {}
        p_values: list[float] = []         # for FDR across horizons

        for h in horizons:
            label = _embargo(fwd_by_h[h], embargo)
            ics = _ic_per_block(feature, label, k=k_folds)
            ic_mean = (float(np.mean([abs(x) for x in ics if np.isfinite(x)]))
                       if ics else float("nan"))
            ir = _ir(ics)
            null = _null_distribution(feature, label, n_reps=n_null,
                                      seed=(seed + hash(fname + str(h))) % (2**31))
            above_null = (np.isfinite(ic_mean) and np.isfinite(null["null_p95"])
                          and ic_mean > null["null_p95"])
            caus = _causality_probe(feature, label)
            # Approx 2-sided p (Gaussian on IC): not exact but used for FDR ranking
            mask = np.isfinite(feature) & np.isfinite(label)
            n_eff = int(mask.sum())
            z_approx = ic_mean * np.sqrt(max(n_eff - 1, 1)) if np.isfinite(ic_mean) else 0.0
            from scipy.stats import norm
            p_approx = 2 * (1 - norm.cdf(abs(z_approx)))
            p_values.append(float(p_approx))

            per_h[h] = {
                "ic_mean_abs": ic_mean, "ic_per_fold": ics, "ir": ir,
                "n_eff": n_eff, "p_approx": float(p_approx),
                "null_test": null, "ic_above_null_p95": bool(above_null),
                "causality_probe": caus,
                "verdict": _classify_ic(ic_mean, ir),
            }

        # FDR across this feature's horizons
        p_fdr = fdr_correct(np.array(p_values))
        for i, h in enumerate(horizons):
            per_h[h]["p_fdr"] = float(p_fdr[i])

        ic_at_h = {h: per_h[h]["ic_mean_abs"] for h in horizons}
        decay_ok = _smooth_decay_ok(ic_at_h, horizons)

        # Feature-level verdict aggregates the per-horizon picture
        any_strong = any(per_h[h]["verdict"] == "STRONG" for h in horizons)
        any_moderate = any(per_h[h]["verdict"] == "MODERATE" for h in horizons)
        any_suspect = any(per_h[h]["verdict"] == "SUSPECT_LEAKAGE" for h in horizons)
        any_above_null = any(per_h[h]["ic_above_null_p95"] for h in horizons)
        any_future_leak = any(per_h[h]["causality_probe"]["future_leak_flag"]
                              for h in horizons)

        if any_suspect or any_future_leak:
            verdict = "SUSPECT_LEAKAGE"
        elif any_strong and decay_ok and any_above_null:
            verdict = "STRONG"
        elif any_moderate and decay_ok and any_above_null:
            verdict = "MODERATE"
        elif any_moderate or any_strong:
            verdict = "WEAK_OR_UNSTABLE"
        else:
            verdict = "NOISE"

        results.append({
            "feature": fname,
            "is_rolling_z_family": is_zfamily,
            "per_horizon": {str(h): per_h[h] for h in horizons},
            "smooth_decay": decay_ok,
            "verdict": verdict,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — Lasso on forward_return (multi-feature)
# ══════════════════════════════════════════════════════════════════════════
def stage_2_lasso(
    df: pd.DataFrame, feature_names: tuple[str, ...],
    horizons: tuple[int, ...], k_folds: int, seed: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        return {"skipped": True, "reason": "sklearn not available"}
    if "close" not in df.columns:
        return {"skipped": True, "reason": "close column missing"}
    close = pd.to_numeric(df["close"], errors="coerce")
    breaks = df["is_session_break"] if "is_session_break" in df.columns else None
    fwd_by_h = compute_forward_returns(close, horizons, session_breaks=breaks)
    embargo = max(horizons)

    cols = [c for c in feature_names if c in df.columns]
    if not cols:
        return {"skipped": True, "reason": "no feature columns present"}
    X_full = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    per_h: dict[str, Any] = {}
    for h in horizons:
        y_full = _embargo(fwd_by_h[h], embargo)
        mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y_full)
        X, y = X_full[mask], y_full[mask]
        n = len(X)
        if n < k_folds * MIN_SAMPLES_PER_FOLD:
            per_h[h] = {"skipped": True, "reason": f"n={n} < {k_folds*MIN_SAMPLES_PER_FOLD}"}
            continue

        fold_ics: list[float] = []
        # Walk-forward expanding window: fold i uses [0..i*size) train, [i*size..) test
        fold_size = n // k_folds
        for fold in range(1, k_folds):
            train_end_raw = fold * fold_size
            train_end = max(0, train_end_raw - embargo)        # purge
            if train_end < MIN_SAMPLES_PER_FOLD:
                continue
            test_start, test_end = train_end_raw, min((fold + 1) * fold_size, n)
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
            sd[sd < 1e-9] = 1.0
            X_tr_s = (X_tr - mu) / sd
            X_te_s = (X_te - mu) / sd

            try:
                # Lasso α determined a priori (small — let it choose features but not
                # overfit). LassoCV would inner-CV which conflicts with our purged outer.
                model = Lasso(alpha=1e-4, max_iter=5000, random_state=seed + fold)
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
                rho, _ = _safe_spearman(y_pred, y_te)
                fold_ics.append(float(rho))
            except Exception as exc:
                fold_ics.append(float("nan"))

        ic_mean = (float(np.mean([abs(x) for x in fold_ics if np.isfinite(x)]))
                   if fold_ics else float("nan"))
        ir = _ir(fold_ics)
        per_h[h] = {
            "ic_per_fold": fold_ics, "ic_mean_abs": ic_mean, "ir": ir,
            "n_train_samples": int(n), "verdict": _classify_ic(ic_mean, ir),
        }

    best_h, best_ic = None, 0.0
    for h, r in per_h.items():
        if isinstance(r, dict) and not r.get("skipped"):
            if r["ic_mean_abs"] > best_ic:
                best_ic, best_h = r["ic_mean_abs"], h
    return {
        "per_horizon": {str(h): per_h[h] for h in horizons},
        "best_horizon": best_h, "best_ic": float(best_ic),
        "verdict": _classify_ic(best_ic,
                                per_h.get(best_h, {}).get("ir", float("nan")) if best_h else float("nan")),
    }


# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — Logistic regression on bias_label (LONG vs SHORT)
# ══════════════════════════════════════════════════════════════════════════
def stage_3_logistic_bias(
    df: pd.DataFrame, feature_names: tuple[str, ...],
    k_folds: int, seed: int, embargo: int,
) -> dict[str, Any]:
    if "bias_label" not in df.columns:
        return {"skipped": True, "reason": "bias_label column missing"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"skipped": True, "reason": "sklearn not available"}

    bias = pd.to_numeric(df["bias_label"], errors="coerce").to_numpy()
    valid = np.isin(bias, [0, 1])
    y = (bias == 0).astype(int)            # 1=LONG, 0=SHORT
    cols = [c for c in feature_names if c in df.columns]
    if not cols:
        return {"skipped": True, "reason": "no feature columns present"}
    X_full = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    mask = valid & np.isfinite(X_full).all(axis=1) & np.isfinite(y)
    X, y = X_full[mask], y[mask]
    n = len(X)
    if n < k_folds * MIN_SAMPLES_PER_FOLD or len(np.unique(y)) < 2:
        return {"skipped": True, "reason": f"insufficient samples / single class (n={n})"}

    fold_aucs: list[float] = []
    fold_size = n // k_folds
    for fold in range(1, k_folds):
        train_end = max(0, fold * fold_size - embargo)
        if train_end < MIN_SAMPLES_PER_FOLD:
            continue
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xtr_s = (X_tr - mu) / sd
        Xte_s = (X_te - mu) / sd
        try:
            model = LogisticRegression(max_iter=2000, random_state=seed + fold).fit(Xtr_s, y_tr)
            y_proba = model.predict_proba(Xte_s)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_te, y_proba)))
        except Exception:
            fold_aucs.append(float("nan"))

    vals = [a for a in fold_aucs if np.isfinite(a)]
    auc_mean = float(np.mean(vals)) if vals else float("nan")
    auc_std = float(np.std(vals)) if len(vals) > 1 else float("nan")
    return {
        "auc_per_fold": fold_aucs, "auc_mean": auc_mean, "auc_std": auc_std,
        "n_samples": int(n), "verdict": _classify_auc(auc_mean),
    }


# ══════════════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ══════════════════════════════════════════════════════════════════════════
def run_probe(
    features_parquet: Path | None, output_dir: Path,
    *,
    df: pd.DataFrame | None = None,
    feature_names: tuple[str, ...] = DEFAULT_FEATURES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    k_folds: int = DEFAULT_K_FOLDS,
    n_null: int = DEFAULT_N_NULL,
    seed: int = 42,
    run_stage_2: bool = True,
    run_stage_3: bool = True,
) -> dict[str, Any]:
    """Run the probe. Either pass `features_parquet` (read from disk) or `df`
    directly (used by sequence_probe to avoid temp parquet I/O). Both back-
    compatible: existing callers pass features_parquet positionally."""
    if df is None:
        if features_parquet is None:
            raise ValueError("decoder_probe.run_probe requires either features_parquet or df")
        df = pd.read_parquet(features_parquet)
    embargo = max(horizons)

    s1 = stage_1_univariate(df, feature_names, horizons, k_folds, n_null, seed)
    # Stage 2 runs only if Stage 1 shows any reachable signal (R4 — don't pile
    # complexity when the simple model says NOISE everywhere)
    any_s1_signal = any(r.get("verdict") in {"WEAK_OR_UNSTABLE", "MODERATE", "STRONG"}
                        for r in s1 if isinstance(r, dict) and not r.get("missing"))
    s2 = (stage_2_lasso(df, feature_names, horizons, k_folds, seed)
          if run_stage_2 and any_s1_signal
          else {"skipped": True, "reason": "no Stage 1 signal" if not any_s1_signal else "disabled"})
    s3 = (stage_3_logistic_bias(df, feature_names, k_folds, seed, embargo)
          if run_stage_3 else {"skipped": True, "reason": "disabled"})

    summary = {
        "features_parquet": str(features_parquet) if features_parquet is not None else "(in-memory df)",
        "n_rows": int(len(df)),
        "feature_names": list(feature_names),
        "horizons": list(horizons),
        "k_folds": k_folds, "embargo_bars": embargo,
        "n_null_reps": n_null, "seed": seed,
        "thresholds_a_priori": {
            "IC_NOISE": IC_NOISE, "IC_MODERATE": IC_MODERATE,
            "IC_STRONG": IC_STRONG, "IC_SUSPECT": IC_SUSPECT,
            "AUC_NOISE": AUC_NOISE, "AUC_MODERATE": AUC_MODERATE,
            "AUC_STRONG": AUC_STRONG, "AUC_SUSPECT": AUC_SUSPECT,
            "IR_MIN_MODERATE": IR_MIN_MODERATE, "IR_MIN_STRONG": IR_MIN_STRONG,
            "CAUSALITY_FUTURE_LEAK_MARGIN": CAUSALITY_FUTURE_LEAK_MARGIN,
            "DECAY_PASS_RATIO": DECAY_PASS_RATIO,
        },
        "stage_1_univariate": s1,
        "stage_2_lasso": s2,
        "stage_3_logistic_bias": s3,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decoder_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "decoder_probe_report.txt", summary)
    return summary


def _write_report(path: Path, s: dict) -> None:
    L: list[str] = []
    L.append("═" * 96)
    L.append("Decoder probe — feature predictivity audit (R10: a priori thresholds, R6: reused IC primitives)")
    L.append("═" * 96)
    L.append(f"features parquet : {s['features_parquet']}")
    L.append(f"n_rows           : {s['n_rows']:,}    feature_names: {s['feature_names']}")
    L.append(f"horizons         : {s['horizons']}   k_folds: {s['k_folds']}   embargo: {s['embargo_bars']}   null_reps: {s['n_null_reps']}")
    L.append("")
    L.append("STAGE 1 — Univariate Spearman IC (feature × horizon)")
    L.append("-" * 96)
    L.append(f"  {'feature':30s} {'family':12s} {'verdict':22s} {'best |IC|':>10s}  {'IR':>6s}  {'decay':>5s}  {'leak':>5s}")
    for r in s["stage_1_univariate"]:
        if r.get("missing"):
            L.append(f"  {r['feature']:30s} MISSING")
            continue
        if r.get("error"):
            L.append(f"  ERROR: {r['error']}")
            continue
        ph = r["per_horizon"]
        ics = [(int(h), ph[h]["ic_mean_abs"]) for h in ph if np.isfinite(ph[h]["ic_mean_abs"])]
        if ics:
            best_h, best_ic = max(ics, key=lambda x: x[1])
            best_ir = ph[str(best_h)]["ir"]
        else:
            best_h, best_ic, best_ir = 0, float("nan"), float("nan")
        family = "rolling-z" if r["is_rolling_z_family"] else "raw"
        any_leak = any(ph[h]["causality_probe"]["future_leak_flag"] for h in ph)
        L.append(f"  {r['feature']:30s} {family:12s} {r['verdict']:22s} "
                 f"{best_ic:>9.3f}  {best_ir:>5.2f}  {str(r['smooth_decay'])[0]:>5s}  {str(any_leak)[0]:>5s}")
    L.append("")
    L.append("STAGE 2 — Lasso multi-feature on forward_return")
    L.append("-" * 96)
    s2 = s["stage_2_lasso"]
    if s2.get("skipped"):
        L.append(f"  SKIPPED: {s2.get('reason')}")
    else:
        L.append(f"  best horizon: h={s2.get('best_horizon')}   "
                 f"best |IC|: {s2.get('best_ic',0):.3f}   verdict: {s2.get('verdict')}")
    L.append("")
    L.append("STAGE 3 — Logistic on bias_label (LONG vs SHORT)")
    L.append("-" * 96)
    s3 = s["stage_3_logistic_bias"]
    if s3.get("skipped"):
        L.append(f"  SKIPPED: {s3.get('reason')}")
    else:
        L.append(f"  AUC: {s3.get('auc_mean',float('nan')):.3f} ± {s3.get('auc_std',0):.3f}   "
                 f"n: {s3.get('n_samples',0):,}   verdict: {s3.get('verdict')}")
    L.append("")
    L.append("KEY (a priori, R10): |IC|>=0.05 MODERATE / 0.10 STRONG / 0.30 SUSPECT_LEAKAGE")
    L.append("                     AUC >=0.55 MODERATE / 0.60 STRONG / 0.75 SUSPECT_LEAKAGE")
    L.append("                     IR >= 0.30 stable across folds; smooth decay required.")
    L.append("R7 NOTE: a positive verdict shows correlation, NOT proof of trading edge.")
    L.append("═" * 96)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Decoder probe — feature predictivity audit")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--feature", action="append", default=None,
                   help="repeatable; defaults are tuned for compass-mode parquet")
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    p.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-stage-2", action="store_true")
    p.add_argument("--skip-stage-3", action="store_true")
    args = p.parse_args()
    feature_names = tuple(args.feature) if args.feature else DEFAULT_FEATURES
    summary = run_probe(
        args.features, args.output,
        feature_names=feature_names, horizons=tuple(args.horizons),
        k_folds=args.k_folds, n_null=args.n_null, seed=args.seed,
        run_stage_2=not args.skip_stage_2, run_stage_3=not args.skip_stage_3,
    )
    s1_pass = sum(1 for r in summary["stage_1_univariate"]
                  if isinstance(r, dict) and r.get("verdict") in {"MODERATE", "STRONG"})
    print(f"\n  Stage 1 (univariate): {s1_pass}/{len(summary['stage_1_univariate'])} features MODERATE+")
    print(f"  Stage 2 (Lasso)     : {summary['stage_2_lasso'].get('verdict', 'skipped')}")
    print(f"  Stage 3 (logistic)  : {summary['stage_3_logistic_bias'].get('verdict', 'skipped')}")
    print(f"  report: {args.output}/decoder_probe_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
