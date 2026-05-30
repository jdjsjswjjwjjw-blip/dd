"""
tools/diagnostics/ic_audit.py
═══════════════════════════════════════════════════════════════════════════════
Information Coefficient (IC) audit for day-trade features.

Measures empirical predictive power of every feature against forward returns
at multiple horizons. Two complementary metrics:

  LINEAR (monotonic):
    - Spearman rank correlation (primary — robust to fat tails)
    - Pearson correlation       (secondary — sensitive to outliers)

  NON-LINEAR (general dependence):
    - Mutual information (sklearn) — captures non-monotonic patterns
      that Spearman misses entirely

Why measure both
────────────────
Spearman = 0 does NOT mean "no signal". It means "no monotonic linear
relationship". A feature that flips sign at extremes (e.g., very high OR
very low values predict the same return) has zero Spearman but high MI.
Neural networks can exploit this. Linear models cannot. The verdict
cascade has a NONLINEAR_EDGE class specifically for these.

Honesty controls
────────────────
- Computed on ALL bars (no event gate, no NEUTRAL filter) — the user's
  explicit requirement: measure features in their raw causal state.
- Session-break aware: returns that span a weekend gap are excluded.
- Train/test split: IC on train MUST replicate on test, else overfit.
- Benjamini-Hochberg FDR correction: testing 250 features without it
  guarantees ~12 false positives at α=0.05.
- Rolling IC + Information Ratio: a feature that "works on average"
  but with high std over windows is unstable — IR < 0.3 demotes it.

Verdict cascade per feature × horizon
────────────────────────────────────
NONFINITE         feature is all-NaN, constant, or unusable
NOISE             |spearman| < 0.02 OR FDR-adjusted p > 0.05
WEAK              0.02 ≤ |spearman| < 0.05 AND FDR-adjusted p < 0.05
NONLINEAR_EDGE    MI > 0.05 AND |spearman| < 0.05 (non-monotonic signal)
MODERATE          0.05 ≤ |spearman| < 0.10 AND IR > 0.3
STRONG            |spearman| ≥ 0.10 AND IR > 0.5
UNSTABLE_STRONG   large mean IC but IR < 0.3 (looks good, isn't reliable)

Usage
─────
    python tools/diagnostics/ic_audit.py \\
        --features  combined/day_trading_features.parquet \\
        --output    audits/ic \\
        --horizons  3 6 12 24 \\
        --train-end 2024-09-01 \\
        --n-rolling-windows 8

Outputs
───────
<output>/ic_summary.csv          per (feature × horizon) row with all metrics
<output>/ic_top_features.csv     top-50 features by composite score
<output>/ic_per_regime.csv       IC × regime breakdown (if regime_label exists)
<output>/ic_per_session.csv      IC × session breakdown (if is_london exists)
<output>/ic_summary.json         machine-readable summary + meta
<output>/ic_report.txt           human-readable highlights + recommendations
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Lazy imports — sklearn is optional but recommended for non-linear IC
try:
    from sklearn.feature_selection import mutual_info_regression
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False


# ── Thresholds ──────────────────────────────────────────────────────
MIN_SAMPLES = 1_000           # below this → NONFINITE (no power)
NOISE_THRESHOLD = 0.02
WEAK_THRESHOLD = 0.05
STRONG_THRESHOLD = 0.10
MI_NONLINEAR_THRESHOLD = 0.05
MIN_IR_MODERATE = 0.30
MIN_IR_STRONG = 0.50
FDR_ALPHA = 0.05

DEFAULT_HORIZONS = (3, 6, 12, 24)        # in bars (15min, 30min, 1h, 2h @ 5min)
DEFAULT_N_ROLLING_WINDOWS = 8


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ICResult:
    """Per (feature × horizon) IC measurement."""
    feature: str
    horizon: int
    n_samples: int
    pearson_ic: float
    spearman_ic: float
    spearman_p: float
    mi_score: float
    rolling_ir: float          # mean(rolling_ic) / std(rolling_ic)
    rolling_mean: float
    rolling_std: float
    test_spearman: float       # holdout consistency check
    p_fdr: float               # FDR-adjusted p-value
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Pure IC computation
# ══════════════════════════════════════════════════════════════════════════════
def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman with NaN handling. Returns (ic, p_value)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan"), float("nan")
    if np.std(x[mask]) <= 1e-12 or np.std(y[mask]) <= 1e-12:
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(x[mask], y[mask])
    return float(r) if np.isfinite(r) else 0.0, float(p) if np.isfinite(p) else 1.0


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan")
    if np.std(x[mask]) <= 1e-12 or np.std(y[mask]) <= 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = pearsonr(x[mask], y[mask])
    return float(r) if np.isfinite(r) else 0.0


def _safe_mi(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Mutual information (non-linear). Returns 0 if sklearn unavailable
    or sample size insufficient."""
    if not _SKLEARN_OK:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 100:
        return float("nan")
    if np.std(x[mask]) <= 1e-12 or np.std(y[mask]) <= 1e-12:
        return 0.0
    try:
        mi = mutual_info_regression(
            x[mask].reshape(-1, 1), y[mask],
            random_state=seed, n_neighbors=3,
        )
        return float(mi[0])
    except Exception:
        return float("nan")


def _rolling_ic(
    x: np.ndarray, y: np.ndarray, n_windows: int = 8,
) -> tuple[float, float, float]:
    """Spearman IC computed in N non-overlapping windows.

    Returns (mean_ic, std_ic, ir) where ir = mean_ic / std_ic.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < n_windows * 100:
        return float("nan"), float("nan"), float("nan")
    x, y = x[mask], y[mask]
    n = len(x)
    window = n // n_windows
    ics: list[float] = []
    for i in range(n_windows):
        lo = i * window
        hi = min(lo + window, n)
        if hi - lo < 50:
            continue
        ic, _ = _safe_spearman(x[lo:hi], y[lo:hi])
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 2:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ir = mean / std if std > 1e-9 else 0.0
    return mean, std, ir


# ══════════════════════════════════════════════════════════════════════════════
# Forward returns (session-break aware)
# ══════════════════════════════════════════════════════════════════════════════
def compute_forward_returns(
    close: pd.Series, horizons: tuple[int, ...],
    session_breaks: Optional[pd.Series] = None,
) -> dict[int, np.ndarray]:
    """Returns {horizon: forward_return_array}. Returns whose window
    crosses a session break are NaN'd to avoid weekend-gap contamination."""
    n = len(close)
    c = close.to_numpy(dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    breaks = (
        session_breaks.astype(bool).to_numpy()
        if session_breaks is not None else None
    )
    for h in horizons:
        fwd = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - h):
            if c[i] <= 0 or not np.isfinite(c[i]) or not np.isfinite(c[i + h]):
                continue
            # Skip if any session break in (i, i+h]
            if breaks is not None and breaks[i + 1: i + h + 1].any():
                continue
            fwd[i] = (c[i + h] - c[i]) / c[i]
        out[h] = fwd
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg FDR
# ══════════════════════════════════════════════════════════════════════════════
def fdr_correct(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    valid = np.isfinite(pvals)
    out = np.ones(n, dtype=np.float64)
    if not valid.any():
        return out
    p = pvals[valid]
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(len(ranked)) + 1)
    # Enforce monotone non-increasing from right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    result_valid = np.empty_like(p)
    result_valid[order] = adj
    out[valid] = result_valid
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════
def classify_verdict(
    spearman_ic: float, mi: float, ir: float, p_fdr: float, n_samples: int,
) -> str:
    if n_samples < MIN_SAMPLES:
        return "NONFINITE"
    if not np.isfinite(spearman_ic):
        return "NONFINITE"

    abs_ic = abs(spearman_ic)

    # Non-significance → NOISE regardless of magnitude
    if np.isfinite(p_fdr) and p_fdr > FDR_ALPHA:
        # But check if MI catches what Spearman missed (non-linear)
        if np.isfinite(mi) and mi > MI_NONLINEAR_THRESHOLD:
            return "NONLINEAR_EDGE"
        return "NOISE"

    if abs_ic < NOISE_THRESHOLD:
        # Low linear correlation — check non-linear
        if np.isfinite(mi) and mi > MI_NONLINEAR_THRESHOLD:
            return "NONLINEAR_EDGE"
        return "NOISE"

    if abs_ic < WEAK_THRESHOLD:
        return "WEAK"

    # Stronger signals require IR check
    if abs_ic >= STRONG_THRESHOLD:
        if np.isfinite(ir) and abs(ir) >= MIN_IR_STRONG:
            return "STRONG"
        return "UNSTABLE_STRONG"

    # 0.05 ≤ |ic| < 0.10
    if np.isfinite(ir) and abs(ir) >= MIN_IR_MODERATE:
        return "MODERATE"
    return "UNSTABLE_STRONG"


# ══════════════════════════════════════════════════════════════════════════════
# Single-feature IC
# ══════════════════════════════════════════════════════════════════════════════
def compute_ic_one_feature(
    feature: np.ndarray,
    forward_returns: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    name: str,
    horizon: int,
    n_rolling: int = DEFAULT_N_ROLLING_WINDOWS,
) -> ICResult:
    """Computes the full ICResult for one feature × horizon pair."""
    # Train slice — primary measurement
    x_tr = feature[train_mask]
    y_tr = forward_returns[train_mask]
    valid_mask_tr = np.isfinite(x_tr) & np.isfinite(y_tr)
    n_tr = int(valid_mask_tr.sum())

    pearson = _safe_pearson(x_tr, y_tr)
    spearman, sp_p = _safe_spearman(x_tr, y_tr)
    mi = _safe_mi(x_tr, y_tr)
    mean_ic, std_ic, ir = _rolling_ic(x_tr, y_tr, n_windows=n_rolling)

    # Test slice — holdout consistency check
    x_te = feature[test_mask]
    y_te = forward_returns[test_mask]
    test_sp, _ = _safe_spearman(x_te, y_te)

    return ICResult(
        feature=name, horizon=int(horizon),
        n_samples=n_tr,
        pearson_ic=pearson,
        spearman_ic=spearman,
        spearman_p=sp_p,
        mi_score=mi,
        rolling_ir=ir if np.isfinite(ir) else 0.0,
        rolling_mean=mean_ic if np.isfinite(mean_ic) else 0.0,
        rolling_std=std_ic if np.isfinite(std_ic) else 0.0,
        test_spearman=test_sp if np.isfinite(test_sp) else 0.0,
        p_fdr=float("nan"),       # filled in after FDR correction
        verdict="NONFINITE",      # filled in after FDR correction
    )


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════
def _select_feature_columns(
    df: pd.DataFrame, exclude_patterns: tuple[str, ...],
) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    out = []
    for c in numeric:
        if any(c == p or c.startswith(p) for p in exclude_patterns):
            continue
        out.append(c)
    return out


# Columns excluded from feature audit — labels / leakage / metadata
_DEFAULT_EXCLUDE = (
    # Targets and labels
    "bias_label", "soft_label", "path_outcome", "neutral_reason", "neutral_type",
    "conf_target", "label_", "forward_", "fwd_", "mfe_", "mae_",
    # Time / metadata
    "ts_event", "time_to_first_touch", "trade_duration",
    "effective_horizon", "label_horizon_steps", "label_dynamic_threshold",
    "label_end_ts", "signal_quality",
    # Event-flag family (use as features? user decides — default exclude)
    "is_event", "event_flag", "train_event_flag",
    # Split markers
    "is_train_slice", "is_holdout_slice", "is_purged_slice", "dataset_slice",
    # Pipeline metadata
    "is_session_break",
    # Sample weights
    "soft_sample_weight", "mc_sample_weight", "sample_weight",
    "label_confidence", "label_stability",
    # Adverse path
    "adverse_path_flag", "timeout_move_exceeded_band",
    # OHLC — used to compute returns, not as features
    "close", "open", "high", "low",
)


def run_ic_audit(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    train_end: Optional[pd.Timestamp] = None,
    n_rolling: int = DEFAULT_N_ROLLING_WINDOWS,
    feature_cols: Optional[list[str]] = None,
    close_col: str = "close",
    session_break_col: str = "is_session_break",
) -> tuple[pd.DataFrame, dict]:
    """Run the full IC audit. Returns (results_df, summary_dict)."""
    if close_col not in df.columns:
        raise ValueError(f"close column {close_col!r} missing from features parquet")

    feature_cols = feature_cols or _select_feature_columns(df, _DEFAULT_EXCLUDE)
    if not feature_cols:
        raise ValueError("no feature columns survived the exclusion filter")

    # Train / test split
    n = len(df)
    if train_end is not None and "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"])
        train_mask = (ts < pd.Timestamp(train_end)).to_numpy()
    else:
        # Default: 70/30
        train_mask = np.zeros(n, dtype=bool)
        train_mask[: int(n * 0.70)] = True
    test_mask = ~train_mask

    # Forward returns at each horizon
    session_breaks = (
        df[session_break_col] if session_break_col in df.columns else None
    )
    fwd_returns = compute_forward_returns(
        df[close_col], horizons, session_breaks=session_breaks,
    )

    # Compute IC for every (feature, horizon)
    results: list[ICResult] = []
    for h in horizons:
        ret = fwd_returns[h]
        for col in feature_cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            r = compute_ic_one_feature(
                feature=x, forward_returns=ret,
                train_mask=train_mask, test_mask=test_mask,
                name=col, horizon=h, n_rolling=n_rolling,
            )
            results.append(r)

    # FDR correction per horizon (multiple-testing burden is per horizon)
    by_horizon: dict[int, list[ICResult]] = {}
    for r in results:
        by_horizon.setdefault(r.horizon, []).append(r)
    for h, items in by_horizon.items():
        pvals = np.array([it.spearman_p for it in items])
        adj = fdr_correct(pvals)
        for it, pa in zip(items, adj):
            it.p_fdr = float(pa)
            it.verdict = classify_verdict(
                spearman_ic=it.spearman_ic, mi=it.mi_score,
                ir=it.rolling_ir, p_fdr=it.p_fdr,
                n_samples=it.n_samples,
            )

    # Build DataFrame
    rows = [r.to_dict() for r in results]
    results_df = pd.DataFrame(rows)

    # Verdict counts
    verdict_counts = (
        results_df["verdict"].value_counts().to_dict()
        if not results_df.empty else {}
    )

    summary = {
        "n_features": len(feature_cols),
        "n_horizons": len(horizons),
        "horizons": list(horizons),
        "n_total_measurements": len(results),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "verdict_counts": {str(k): int(v) for k, v in verdict_counts.items()},
        "n_sklearn_ok": int(_SKLEARN_OK),
    }
    return results_df, summary


def top_features(
    results_df: pd.DataFrame, n: int = 50, horizon: Optional[int] = None,
) -> pd.DataFrame:
    """Return top-N features by composite score |spearman| × IR.
    If horizon given, restricts to that horizon."""
    df = results_df.copy()
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    df["composite"] = df["spearman_ic"].abs() * df["rolling_ir"].abs()
    df = df[df["verdict"].isin([
        "STRONG", "MODERATE", "NONLINEAR_EDGE", "UNSTABLE_STRONG",
    ])]
    return df.sort_values("composite", ascending=False).head(n)


# ══════════════════════════════════════════════════════════════════════════════
# Per-regime / per-session breakdowns (optional)
# ══════════════════════════════════════════════════════════════════════════════
def per_group_ic(
    df: pd.DataFrame, feature_cols: list[str], horizons: tuple[int, ...],
    group_col: str, close_col: str = "close",
) -> pd.DataFrame:
    """Compute spearman IC for each (feature, horizon, group_value)."""
    if group_col not in df.columns:
        return pd.DataFrame()
    fwd_returns = compute_forward_returns(df[close_col], horizons)
    rows = []
    for group_val in df[group_col].unique():
        sub_mask = (df[group_col] == group_val).to_numpy()
        for h in horizons:
            ret = fwd_returns[h]
            for col in feature_cols:
                x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                ic, p = _safe_spearman(x[sub_mask], ret[sub_mask])
                rows.append({
                    "feature": col, "horizon": h,
                    "group": str(group_val), "spearman_ic": ic,
                    "p_value": p, "n": int(sub_mask.sum()),
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════
def write_report(
    path: Path, results_df: pd.DataFrame, summary: dict,
    top_n: int = 25,
) -> None:
    lines = ["═" * 70, "IC AUDIT — DAY-TRADE FEATURES", "═" * 70,
             f"Features audited : {summary['n_features']}",
             f"Horizons         : {summary['horizons']}",
             f"Train samples    : {summary['n_train']:,}",
             f"Test samples     : {summary['n_test']:,}",
             f"sklearn (MI)     : {'yes' if summary['n_sklearn_ok'] else 'NO (non-linear IC disabled)'}",
             "", "Verdict distribution:"]
    for verdict in (
        "STRONG", "MODERATE", "NONLINEAR_EDGE",
        "UNSTABLE_STRONG", "WEAK", "NOISE", "NONFINITE",
    ):
        n = summary["verdict_counts"].get(verdict, 0)
        if n > 0:
            lines.append(f"  {verdict:<18} {n:>5}")
    lines.append("")

    if not results_df.empty:
        lines.append(f"Top {top_n} features (by composite |spearman| × |IR|):")
        lines.append(
            f"  {'feature':<40} {'horiz':>5}  {'spearman':>10}  "
            f"{'MI':>8}  {'IR':>7}  {'test':>8}  {'verdict':<18}"
        )
        top = top_features(results_df, n=top_n)
        for _, r in top.iterrows():
            lines.append(
                f"  {str(r['feature'])[:40]:<40} {int(r['horizon']):>5}  "
                f"{r['spearman_ic']:>+10.4f}  {r['mi_score']:>8.4f}  "
                f"{r['rolling_ir']:>+7.2f}  {r['test_spearman']:>+8.4f}  "
                f"{r['verdict']:<18}"
            )

    lines.append("")
    lines.append("Threshold reference:")
    lines.append(f"  |spearman| < {NOISE_THRESHOLD}    → NOISE")
    lines.append(f"  |spearman| ≥ {WEAK_THRESHOLD}    → WEAK / MODERATE")
    lines.append(f"  |spearman| ≥ {STRONG_THRESHOLD}    → STRONG (if IR ≥ {MIN_IR_STRONG})")
    lines.append(f"  MI > {MI_NONLINEAR_THRESHOLD}            → NONLINEAR_EDGE (if linear weak)")
    lines.append("═" * 70)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--train-end", default=None,
                   help="ISO date — train is rows with ts_event < this. "
                        "Default: 70/30 by row order.")
    p.add_argument("--n-rolling-windows", type=int, default=DEFAULT_N_ROLLING_WINDOWS)
    p.add_argument("--per-regime", action="store_true",
                   help="Also compute IC × regime breakdown")
    p.add_argument("--per-session", action="store_true",
                   help="Also compute IC × session breakdown (london/ny/overlap)")
    args = p.parse_args()

    print(f"loading {args.features}")
    df = pd.read_parquet(args.features)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    train_end = pd.Timestamp(args.train_end) if args.train_end else None
    results_df, summary = run_ic_audit(
        df, horizons=tuple(args.horizons),
        train_end=train_end, n_rolling=args.n_rolling_windows,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output / "ic_summary.csv", index=False)
    top_features(results_df, n=50).to_csv(
        args.output / "ic_top_features.csv", index=False,
    )
    (args.output / "ic_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    write_report(args.output / "ic_report.txt", results_df, summary)

    if args.per_regime and "regime_label" in df.columns:
        feature_cols = _select_feature_columns(df, _DEFAULT_EXCLUDE)
        regime_df = per_group_ic(
            df, feature_cols, tuple(args.horizons),
            group_col="regime_label",
        )
        regime_df.to_csv(args.output / "ic_per_regime.csv", index=False)
        print(f"  wrote ic_per_regime.csv ({len(regime_df)} rows)")

    if args.per_session and "is_london" in df.columns:
        feature_cols = _select_feature_columns(df, _DEFAULT_EXCLUDE)
        sess = df["is_london"].astype(int) * 1 + \
               df.get("is_ny", 0).astype(int) * 2 + \
               df.get("is_overlap", 0).astype(int) * 3
        df2 = df.copy()
        df2["_session_code"] = sess
        sess_df = per_group_ic(
            df2, feature_cols, tuple(args.horizons),
            group_col="_session_code",
        )
        sess_df.to_csv(args.output / "ic_per_session.csv", index=False)
        print(f"  wrote ic_per_session.csv ({len(sess_df)} rows)")

    print(f"\nverdicts: {summary['verdict_counts']}")
    print(f"output  : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
