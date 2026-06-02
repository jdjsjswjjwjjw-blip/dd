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

# Production-grade hardening thresholds (Option 2)
LOOKAHEAD_DROP_THRESHOLD = 0.75    # drop > 75% from h=1 to longest horizon
WALK_FORWARD_MIN_CONSISTENCY = 0.625  # ≥ 5 of 8 windows same-sign
WARMUP_IC_DROP_THRESHOLD = 0.50    # > 50% drop when warmup excluded
COST_ADJUSTED_NOISE_THRESHOLD = 0.02

# Per-event-status thresholds (NEUTRAL diagnostic)
# gate_kill_ratio = |IC on non-event bars| / |IC on event bars|
GATE_KILLS_SIGNAL_THRESHOLD = 0.50   # ratio ≥ 0.5 → gate kills real signal
GATE_KILLS_SIGNAL_MIN_IC = 0.03      # only meaningful when event-IC > floor

DEFAULT_HORIZONS = (3, 6, 12, 24)        # in bars (15min, 30min, 1h, 2h @ 5min)
DEFAULT_N_ROLLING_WINDOWS = 8
DEFAULT_N_WALK_FORWARD = 8
DEFAULT_WARMUP_BARS = 20
DEFAULT_COST_PER_SIDE = 0.0          # in same units as returns (e.g., 5e-5 = 0.5 pip)


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ICResult:
    """Per (feature × horizon) IC measurement.

    Production-grade fields (Option 2 hardening)
    ─────────────────────────────────────────────
    walk_forward_mean/std/consistency: replaces single train/test split
        with N expanding windows. consistency = fraction of windows
        whose IC has the same sign as the mean.

    lookahead_score: 0 = clean signal decay across horizons,
                     > 0 = IC drops suspiciously fast (possible leakage).
                     Computed at the feature level (not per-horizon).
                     The same score is broadcast to every horizon row
                     for the same feature.

    cost_adjusted_spearman: IC where small forward returns
        (|r| < 2 × cost) have been collapsed to zero — captures only
        the portion of the signal that survives transaction costs.

    warmup_ic_drop: fractional drop in |IC| when warm-up bars (first N
        bars after each session break) are excluded. > 0 means the
        feature was riding warm-up bias.

    ic_event_bars / ic_non_event_bars / gate_kill_ratio:
        Per-event-status IC breakdown. Measures whether the
        prepare_day_trading event gate is throwing away bars that
        actually contained signal.

        gate_kill_ratio = |ic_non_event| / |ic_event|
          ≈ 1.0  → non-event bars carry the same edge → gate is wrong
          ≈ 0    → non-event bars are signal-free   → gate is justified
          NaN    → is_event column not in parquet OR one subset too
                   small to compute
    """
    feature: str
    horizon: int
    n_samples: int
    pearson_ic: float
    spearman_ic: float
    spearman_p: float
    mi_score: float
    rolling_ir: float
    rolling_mean: float
    rolling_std: float
    test_spearman: float
    p_fdr: float
    verdict: str
    # Option 2: production-grade fields
    walk_forward_mean: float = 0.0
    walk_forward_std: float = 0.0
    walk_forward_consistency: float = 0.0
    lookahead_score: float = 0.0
    cost_adjusted_spearman: float = 0.0
    warmup_ic_drop: float = 0.0
    # Per-event-status breakdown (NEUTRAL diagnostic)
    ic_event_bars: float = 0.0
    ic_non_event_bars: float = 0.0
    gate_kill_ratio: float = 0.0

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
# Option 2 — Production-grade IC primitives
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_ic(
    feature: np.ndarray, forward_returns: np.ndarray, n_windows: int = 8,
) -> tuple[float, float, float, list[float]]:
    """Expanding-window walk-forward IC.

    For N windows, build N (train_end, test_start, test_end) splits.
    For each split, compute Spearman IC on the test slice. The test
    sets are disjoint and forward-only — no test set ever sees data
    from any later window.

    Returns (wf_mean, wf_std, wf_consistency, individual_ics).
    wf_consistency = fraction of windows whose IC has the same sign
                     as the mean. 1.0 = unanimous, 0.5 = random.

    The first 1/N of the data is reserved for the initial training
    window and is never tested — that's by construction (we need a
    training history before we can have a test set).
    """
    mask = np.isfinite(feature) & np.isfinite(forward_returns)
    if mask.sum() < n_windows * 200:
        return float("nan"), float("nan"), float("nan"), []
    n = len(feature)
    step = n // n_windows
    test_ics: list[float] = []
    for i in range(1, n_windows):
        test_start = i * step
        test_end = min(test_start + step, n)
        if test_end - test_start < 100:
            continue
        test_feat = feature[test_start:test_end]
        test_ret = forward_returns[test_start:test_end]
        ic, _ = _safe_spearman(test_feat, test_ret)
        if np.isfinite(ic):
            test_ics.append(ic)
    if len(test_ics) < 3:
        return float("nan"), float("nan"), float("nan"), test_ics
    arr = np.asarray(test_ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # Consistency: how many windows have the same sign as the mean
    if abs(mean) <= 1e-9:
        consistency = 0.5
    else:
        same_sign = int((np.sign(arr) == np.sign(mean)).sum())
        consistency = float(same_sign) / float(len(arr))
    return mean, std, consistency, test_ics


def cost_adjusted_returns(
    forward_returns: np.ndarray, cost_per_side: float,
) -> np.ndarray:
    """Collapse forward returns to zero where |return| ≤ 2 × cost.

    Captures only the portion of the signal that would survive the
    round-trip transaction cost. A return of 0.0001 with cost 0.00005
    becomes zero — the trade would have been a wash.

    Pure: returns a NEW array, doesn't mutate input.
    """
    if cost_per_side <= 0.0:
        return forward_returns.copy()
    threshold = 2.0 * float(cost_per_side)
    out = forward_returns.copy()
    small_mask = np.abs(out) <= threshold
    out[small_mask] = 0.0
    return out


def detect_lookahead(
    spearman_ics_by_horizon: dict[int, float],
) -> float:
    """Return a lookahead score in [0, 1].

    Logic: a clean feature shows smooth IC decay (or rise) across
    horizons. Leakage typically shows |IC| at the shortest horizon
    that is much larger than at longer horizons.

    Score formula:
        ic_short = |IC(h_min)|
        ic_long  = |IC(h_max)|
        drop_ratio = (ic_short - ic_long) / max(ic_short, 1e-9)
        score = max(0, drop_ratio - 0.5)   # 0 unless drop > 50 %

    A score above 0.25 (= drop > 75 %) triggers SUSPICIOUS_LEAKAGE.
    """
    if len(spearman_ics_by_horizon) < 2:
        return 0.0
    horizons = sorted(spearman_ics_by_horizon.keys())
    h_min, h_max = horizons[0], horizons[-1]
    ic_short = abs(spearman_ics_by_horizon[h_min])
    ic_long = abs(spearman_ics_by_horizon[h_max])
    # A noise feature has all horizons near 0 — don't flag
    if ic_short < 0.03:
        return 0.0
    drop_ratio = (ic_short - ic_long) / max(ic_short, 1e-9)
    return float(max(0.0, drop_ratio - 0.5))


def compute_warmup_mask(
    df: pd.DataFrame, n_warmup_bars: int = DEFAULT_WARMUP_BARS,
    session_break_col: str = "is_session_break",
) -> np.ndarray:
    """Return boolean mask True for warm-up bars.

    A bar is warm-up if it is among the first `n_warmup_bars` rows
    after a session break (or after the start of the dataset).
    """
    n = len(df)
    if n_warmup_bars <= 0:
        return np.zeros(n, dtype=bool)
    if session_break_col in df.columns:
        breaks = df[session_break_col].astype(bool).to_numpy()
    else:
        breaks = np.zeros(n, dtype=bool)
        breaks[0] = True   # treat start of dataset as a break

    warmup = np.zeros(n, dtype=bool)
    counter = n_warmup_bars   # also warm up the very first bar
    for i in range(n):
        if breaks[i]:
            counter = n_warmup_bars
        if counter > 0:
            warmup[i] = True
            counter -= 1
    return warmup


def per_event_status_ic(
    feature: np.ndarray, forward_returns: np.ndarray,
    event_mask: np.ndarray, min_samples: int = 500,
) -> tuple[float, float, float]:
    """Spearman IC computed separately on event vs non-event bars.

    Returns (ic_event, ic_non_event, gate_kill_ratio).
      gate_kill_ratio = |ic_non_event| / |ic_event| when event-IC
                        clears the noise floor; 0 when event-IC is
                        itself noise (ratio would be meaningless).

    Pure: no side effects.

    Why this matters for the NEUTRAL problem:
      If non-event bars carry comparable signal to event bars, the
      event gate is killing real labeling opportunities — exactly the
      hypothesis the empirical 80%-vs-40% NEUTRAL gap suggests.
    """
    event_mask = event_mask.astype(bool)
    finite = np.isfinite(feature) & np.isfinite(forward_returns)
    ev_keep = event_mask & finite
    nonev_keep = (~event_mask) & finite
    if ev_keep.sum() < min_samples or nonev_keep.sum() < min_samples:
        return float("nan"), float("nan"), float("nan")
    ic_ev, _ = _safe_spearman(feature[ev_keep], forward_returns[ev_keep])
    ic_nonev, _ = _safe_spearman(feature[nonev_keep], forward_returns[nonev_keep])
    if not (np.isfinite(ic_ev) and np.isfinite(ic_nonev)):
        return float(ic_ev), float(ic_nonev), float("nan")
    if abs(ic_ev) < GATE_KILLS_SIGNAL_MIN_IC:
        # event-side is itself near-noise → ratio is uninformative
        return float(ic_ev), float(ic_nonev), 0.0
    ratio = abs(ic_nonev) / abs(ic_ev)
    return float(ic_ev), float(ic_nonev), float(ratio)


def warmup_drop_ratio(
    feature: np.ndarray, forward_returns: np.ndarray,
    warmup_mask: np.ndarray,
) -> float:
    """Fractional drop in |IC| when warm-up bars are excluded.

    Positive value = IC depended on warm-up bars.
    Returns 0.0 when neither subset has enough samples to compare.
    """
    full_ic, _ = _safe_spearman(feature, forward_returns)
    if not np.isfinite(full_ic) or abs(full_ic) < 1e-9:
        return 0.0
    keep = ~warmup_mask
    if keep.sum() < 1000:
        return 0.0
    no_warmup_ic, _ = _safe_spearman(feature[keep], forward_returns[keep])
    if not np.isfinite(no_warmup_ic):
        return 0.0
    drop = (abs(full_ic) - abs(no_warmup_ic)) / max(abs(full_ic), 1e-9)
    return float(max(0.0, drop))


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
    # Option 2 — production-grade fields (all optional, default 0.0 = "not measured")
    walk_forward_consistency: float = 1.0,
    lookahead_score: float = 0.0,
    cost_adjusted_ic: float | None = None,
    warmup_drop: float = 0.0,
) -> str:
    """Verdict cascade — extended with production-grade checks.

    Order of checks (early returns dominate):
      NONFINITE             (degenerate input / too few samples)
      SUSPICIOUS_LEAKAGE    (lookahead_score > threshold — possible cheating)
      WARMUP_RIDER          (>50% IC drop when warm-up bars excluded)
      COST_NEGATIVE         (linear signal disappears after costs)
      UNSTABLE_WALKFORWARD  (signal flips sign across walk-forward windows)
      NOISE / NONLINEAR_EDGE / WEAK / MODERATE / STRONG / UNSTABLE_STRONG
                            (original cascade — applies when no red flag fires)
    """
    if n_samples < MIN_SAMPLES:
        return "NONFINITE"
    if not np.isfinite(spearman_ic):
        return "NONFINITE"

    abs_ic = abs(spearman_ic)

    # ── Production-grade red flags (in priority order) ─────────────
    # Leakage takes precedence — feature MUST be excluded from training
    if lookahead_score > 0.25:
        return "SUSPICIOUS_LEAKAGE"

    # Warm-up dependency: the apparent IC came from a degenerate
    # rolling-window state. Real-world signal vanishes.
    if warmup_drop > WARMUP_IC_DROP_THRESHOLD and abs_ic > NOISE_THRESHOLD:
        return "WARMUP_RIDER"

    # Cost-adjusted check — only when the user supplied a non-zero
    # cost. Otherwise this branch is skipped (cost_adjusted_ic=None).
    if (
        cost_adjusted_ic is not None
        and abs_ic > WEAK_THRESHOLD
        and abs(cost_adjusted_ic) < COST_ADJUSTED_NOISE_THRESHOLD
    ):
        return "COST_NEGATIVE"

    # Walk-forward consistency — only enforced when measurement available
    if (
        walk_forward_consistency < WALK_FORWARD_MIN_CONSISTENCY
        and abs_ic > WEAK_THRESHOLD
    ):
        return "UNSTABLE_WALKFORWARD"

    # ── Original cascade ───────────────────────────────────────────
    if np.isfinite(p_fdr) and p_fdr > FDR_ALPHA:
        if np.isfinite(mi) and mi > MI_NONLINEAR_THRESHOLD:
            return "NONLINEAR_EDGE"
        return "NOISE"

    if abs_ic < NOISE_THRESHOLD:
        if np.isfinite(mi) and mi > MI_NONLINEAR_THRESHOLD:
            return "NONLINEAR_EDGE"
        return "NOISE"

    if abs_ic < WEAK_THRESHOLD:
        return "WEAK"

    if abs_ic >= STRONG_THRESHOLD:
        if np.isfinite(ir) and abs(ir) >= MIN_IR_STRONG:
            return "STRONG"
        return "UNSTABLE_STRONG"

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
    # Option 2 production-grade additions (all optional)
    warmup_mask: np.ndarray | None = None,
    cost_per_side: float = 0.0,
    n_walk_forward: int = 0,        # 0 = disabled
    event_mask: np.ndarray | None = None,
) -> ICResult:
    """Computes the full ICResult for one feature × horizon pair.

    With Option 2 enabled (warmup_mask / cost_per_side / n_walk_forward),
    populates walk_forward_*, cost_adjusted_spearman, and warmup_ic_drop.
    The lookahead_score is filled in later at the feature level (after
    all horizons for the same feature have been computed)."""
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

    # ── Walk-forward IC (full series, not train-only) ────────────
    if n_walk_forward > 0:
        wf_mean, wf_std, wf_cons, _ = walk_forward_ic(
            feature, forward_returns, n_windows=n_walk_forward,
        )
    else:
        wf_mean = wf_std = wf_cons = 0.0

    # ── Cost-adjusted IC (on train slice) ────────────────────────
    if cost_per_side > 0.0:
        cost_returns = cost_adjusted_returns(y_tr, cost_per_side)
        cost_ic, _ = _safe_spearman(x_tr, cost_returns)
        cost_adj = float(cost_ic) if np.isfinite(cost_ic) else 0.0
    else:
        cost_adj = 0.0

    # ── Warm-up dependency check ─────────────────────────────────
    if warmup_mask is not None:
        wm_tr = warmup_mask[train_mask]
        wu_drop = warmup_drop_ratio(x_tr, y_tr, wm_tr)
    else:
        wu_drop = 0.0

    # ── Per-event-status IC (NEUTRAL diagnostic) ─────────────────
    if event_mask is not None:
        em_tr = event_mask[train_mask]
        ic_ev, ic_nonev, kill_ratio = per_event_status_ic(x_tr, y_tr, em_tr)
        ic_ev = ic_ev if np.isfinite(ic_ev) else 0.0
        ic_nonev = ic_nonev if np.isfinite(ic_nonev) else 0.0
        kill_ratio = kill_ratio if np.isfinite(kill_ratio) else 0.0
    else:
        ic_ev = ic_nonev = kill_ratio = 0.0

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
        p_fdr=float("nan"),
        verdict="NONFINITE",
        walk_forward_mean=wf_mean if np.isfinite(wf_mean) else 0.0,
        walk_forward_std=wf_std if np.isfinite(wf_std) else 0.0,
        walk_forward_consistency=(
            wf_cons if np.isfinite(wf_cons) else 0.0
        ),
        lookahead_score=0.0,    # filled in at the feature level
        cost_adjusted_spearman=cost_adj,
        warmup_ic_drop=wu_drop,
        ic_event_bars=float(ic_ev),
        ic_non_event_bars=float(ic_nonev),
        gate_kill_ratio=float(kill_ratio),
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
    # Option 2 — production-grade hardening
    n_walk_forward: int = 0,           # 0 = disabled (Option 1 behaviour)
    cost_per_side: float = 0.0,
    exclude_warmup: bool = False,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    per_event_status: bool = False,
    event_col: str = "is_event",
) -> tuple[pd.DataFrame, dict]:
    """Run the full IC audit. Returns (results_df, summary_dict).

    Production hardening (Option 2)
    ──────────────────────────────
    n_walk_forward    > 0 enables expanding-window IC across N test
                      windows. Adds walk_forward_* fields + can trigger
                      UNSTABLE_WALKFORWARD verdict.
    cost_per_side     > 0 enables cost-adjusted IC. Adds
                      cost_adjusted_spearman field + can trigger
                      COST_NEGATIVE verdict (large naive IC, cost-
                      adjusted IC near zero).
    exclude_warmup    True enables warm-up dependency check. Adds
                      warmup_ic_drop field + can trigger WARMUP_RIDER
                      verdict. The warm-up bars are NOT removed from
                      the training data — only the dependency is
                      measured.
    """
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

    # Warm-up mask (Option 2) — computed once, reused across horizons
    warmup_mask_arr = (
        compute_warmup_mask(df, n_warmup_bars=warmup_bars,
                            session_break_col=session_break_col)
        if exclude_warmup else None
    )

    # Per-event-status mask (NEUTRAL diagnostic)
    event_mask_arr: np.ndarray | None = None
    if per_event_status:
        if event_col not in df.columns:
            raise ValueError(
                f"per-event-status requires '{event_col}' column "
                f"in features parquet"
            )
        event_mask_arr = (
            pd.to_numeric(df[event_col], errors="coerce")
            .fillna(0).astype(bool).to_numpy()
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
                warmup_mask=warmup_mask_arr,
                cost_per_side=cost_per_side,
                n_walk_forward=n_walk_forward,
                event_mask=event_mask_arr,
            )
            results.append(r)

    # Option 2: compute lookahead_score per feature (uses ICs across horizons)
    by_feature: dict[str, list[ICResult]] = {}
    for r in results:
        by_feature.setdefault(r.feature, []).append(r)
    for feat, items in by_feature.items():
        ic_by_h = {it.horizon: it.spearman_ic for it in items}
        score = detect_lookahead(ic_by_h)
        for it in items:
            it.lookahead_score = score

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
                # Option 2 fields — None/0.0 = "not measured" → branches skipped
                walk_forward_consistency=(
                    it.walk_forward_consistency if n_walk_forward > 0 else 1.0
                ),
                lookahead_score=it.lookahead_score,
                cost_adjusted_ic=(
                    it.cost_adjusted_spearman if cost_per_side > 0 else None
                ),
                warmup_drop=it.warmup_ic_drop if exclude_warmup else 0.0,
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
        # Option 2 production-grade params actually used
        "option2": {
            "n_walk_forward": int(n_walk_forward),
            "cost_per_side": float(cost_per_side),
            "exclude_warmup": bool(exclude_warmup),
            "warmup_bars": int(warmup_bars),
            "per_event_status": bool(per_event_status),
        },
    }

    # Per-event-status aggregate diagnostic (NEUTRAL hypothesis test)
    if per_event_status and not results_df.empty:
        # Features where event-bar IC clears the noise floor
        useful = results_df[
            results_df["ic_event_bars"].abs() >= GATE_KILLS_SIGNAL_MIN_IC
        ]
        if len(useful) >= 5:
            mean_kill_ratio = float(useful["gate_kill_ratio"].mean())
            n_above_threshold = int(
                (useful["gate_kill_ratio"] >= GATE_KILLS_SIGNAL_THRESHOLD).sum()
            )
            frac_above = n_above_threshold / len(useful)
            if frac_above >= 0.50:
                gate_verdict = "GATE_KILLS_SIGNAL"
            elif frac_above >= 0.25:
                gate_verdict = "GATE_PARTIALLY_JUSTIFIED"
            else:
                gate_verdict = "GATE_PROPERLY_FILTERING"
            summary["gate_diagnostic"] = {
                "verdict": gate_verdict,
                "mean_gate_kill_ratio": mean_kill_ratio,
                "n_features_kept_signal": n_above_threshold,
                "n_useful_features": int(len(useful)),
                "frac_kept_signal": float(frac_above),
                "threshold": float(GATE_KILLS_SIGNAL_THRESHOLD),
            }
        else:
            summary["gate_diagnostic"] = {
                "verdict": "INSUFFICIENT_SIGNAL_FEATURES",
                "n_useful_features": int(len(useful)),
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
def exclusive_session_label(df: pd.DataFrame) -> np.ndarray:
    """M1: assign each bar to EXACTLY ONE session by priority.

    After C1 the session flags overlap by construction (overlap ⊂ london ∩ ny:
    london 07-16, ny 13-22, overlap 13-16). The old per-session encoding summed
    them (is_london*1 + is_ny*2 + is_overlap*3) → ambiguous codes {0,1,2,6} that
    mis-attributed the per-session IC. Here, priority overlap > ny > london wins
    (most-specific last), so the groups are clean and human-readable:
        london  = 07-13   ny = 16-22   overlap = 13-16   other = off + asia
    'other' lumps off + asia because is_asia is not exported yet — separating it
    is M2, not M1. This only re-defines the per-session DIAGNOSTIC groups; the
    overall / per-event IC verdicts do not use it and are unchanged.
    """
    n = len(df)
    is_lon = df["is_london"].astype(bool).to_numpy()
    is_ny_ = df.get("is_ny", pd.Series(0, index=df.index)).astype(bool).to_numpy()
    is_ovl = df.get("is_overlap", pd.Series(0, index=df.index)).astype(bool).to_numpy()
    sess = np.full(n, "other", dtype=object)
    sess[is_lon] = "london"      # 07-13 (13-16 overwritten → overlap)
    sess[is_ny_] = "ny"          # 16-22 (13-16 overwritten → overlap)
    sess[is_ovl] = "overlap"     # 13-16 — most-specific, wins
    return sess


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

    # Gate diagnostic (per-event-status verdict)
    if "gate_diagnostic" in summary:
        gd = summary["gate_diagnostic"]
        lines.append("")
        lines.append("Event-gate diagnostic (NEUTRAL hypothesis test):")
        lines.append(f"  Verdict        : {gd['verdict']}")
        if "mean_gate_kill_ratio" in gd:
            lines.append(
                f"  Mean kill ratio: {gd['mean_gate_kill_ratio']:.3f}"
            )
            lines.append(
                f"  Features kept signal on non-event bars: "
                f"{gd['n_features_kept_signal']}/{gd['n_useful_features']} "
                f"({gd['frac_kept_signal']:.1%})"
            )
            lines.append(
                f"  Threshold      : kill_ratio ≥ {gd['threshold']:.2f}"
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
    # Option 2 — production-grade flags
    p.add_argument("--walk-forward", type=int, default=0,
                   help="Number of expanding walk-forward windows. "
                        "0 = disabled (Option 1 behaviour). "
                        "Typical production value: 8.")
    p.add_argument("--cost-per-side", type=float, default=0.0,
                   help="Transaction cost per side in return units "
                        "(e.g., 5e-5 for 0.5 pip on a $1.30 instrument). "
                        "0 = no cost adjustment.")
    p.add_argument("--exclude-warmup", action="store_true",
                   help="Measure IC dependency on warm-up bars (first "
                        "N bars after each session break). Adds a "
                        "warmup_ic_drop column + can trigger "
                        "WARMUP_RIDER verdict.")
    p.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS,
                   help=f"Warm-up window size in bars "
                        f"(default {DEFAULT_WARMUP_BARS}).")
    p.add_argument("--per-event-status", action="store_true",
                   help="Compute IC separately on event vs non-event "
                        "bars (requires is_event column). Adds "
                        "ic_event_bars / ic_non_event_bars / "
                        "gate_kill_ratio columns + a gate_diagnostic "
                        "verdict in summary.json. Directly answers: "
                        "is the event gate killing real signal?")
    p.add_argument("--event-col", default="is_event",
                   help="Column name for the event indicator "
                        "(default is_event).")
    args = p.parse_args()

    print(f"loading {args.features}")
    df = pd.read_parquet(args.features)
    print(f"  rows: {len(df):,}  cols: {df.shape[1]}")

    train_end = pd.Timestamp(args.train_end) if args.train_end else None
    results_df, summary = run_ic_audit(
        df, horizons=tuple(args.horizons),
        train_end=train_end, n_rolling=args.n_rolling_windows,
        n_walk_forward=args.walk_forward,
        cost_per_side=args.cost_per_side,
        exclude_warmup=args.exclude_warmup,
        warmup_bars=args.warmup_bars,
        per_event_status=args.per_event_status,
        event_col=args.event_col,
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
        # M1: exclusive priority session encoding (overlap > ny > london; rest
        # 'other'). Replaces the additive sum that produced ambiguous codes.
        sess = exclusive_session_label(df)
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
