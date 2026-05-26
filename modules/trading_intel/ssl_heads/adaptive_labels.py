"""
modules/deep_lob/adaptive_target_labels.py
═══════════════════════════════════════════════════════════════════════════════
Causal label builders for `AdaptiveTargetHeads` — pure numpy.

Three labels, all computed strictly from the forward window so the heads
learn what's actually achievable:

  • max_R_reached  (float)        max R-multiple in window (0 to ~5)
  • target_bucket  (int 0-3)      discretized: <1R, 1-2R, 2-3R, 3+R
  • regime_risk    (int 0-2)      normal / elevated / extreme volatility

Forward window respects `is_session_break`. Direction-aware: a LONG
sample's "max R reached" uses forward high; SHORT uses forward low.
"""
from __future__ import annotations

import numpy as np


def _forward_window_clean(
    is_session_break: np.ndarray, i: int, forward_bars: int,
) -> bool:
    n = len(is_session_break)
    end = min(i + forward_bars + 1, n)
    if end <= i + 1:
        return False
    return not bool(is_session_break[i + 1:end].any())


# ─────────────────────────────────────────────────────────────────────────────
# 1. max_R_reached — Direction-aware maximum favorable excursion in R units
# ─────────────────────────────────────────────────────────────────────────────
def build_max_R_reached_targets(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    bias_direction: np.ndarray,     # +1 LONG, -1 SHORT, 0 NEUTRAL
    is_session_break: np.ndarray,
    *,
    forward_bars: int = 6,           # day_trade default horizon
    r_cap: float = 5.0,              # clip to avoid tail blowup
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (max_R, valid):
        max_R[i] = max favorable R-multiple reached in (i, i+forward_bars]
        valid[i] = forward window clean + ATR > 0 + direction != 0

    For LONG: max_R = (max forward high - entry) / ATR
    For SHORT: max_R = (entry - min forward low) / ATR
    Clipped to [0, r_cap].
    """
    N = len(close)
    max_R = np.zeros(N, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if not _forward_window_clean(is_session_break, i, forward_bars):
            continue
        d = int(bias_direction[i])
        if d == 0:
            continue
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0:
            continue
        if not np.isfinite(close[i]) or close[i] <= 0:
            continue
        entry = float(close[i])
        end = min(i + forward_bars + 1, N)
        fwd_high = float(np.max(high[i + 1:end]))
        fwd_low = float(np.min(low[i + 1:end]))
        if d == 1:
            move = fwd_high - entry
        else:
            move = entry - fwd_low
        r = move / a
        max_R[i] = float(np.clip(r, 0.0, r_cap))
        valid[i] = True
    return max_R, valid


# ─────────────────────────────────────────────────────────────────────────────
# 2. target_bucket — Discretize max_R into 4 trader-relevant tiers
# ─────────────────────────────────────────────────────────────────────────────
def build_target_bucket_targets(
    max_R: np.ndarray, valid_max_R: np.ndarray,
    *,
    bucket_edges: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert continuous max_R into 4-way bucket label.

    Default boundaries (1.0, 2.0, 3.0) → buckets:
      0 = max_R < 1R       (no edge, dead trade)
      1 = 1R <= max_R < 2R (modest edge, take TP1)
      2 = 2R <= max_R < 3R (good edge, ride to TP2)
      3 = max_R >= 3R      (exceptional, ride further)

    valid mirrors the max_R validity.
    """
    n = len(max_R)
    bucket = np.zeros(n, dtype=np.int64)
    # np.digitize: edges = (1, 2, 3) → indices 0/1/2/3
    bucket = np.digitize(max_R, bucket_edges).astype(np.int64)
    return bucket, valid_max_R.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. regime_risk — Causal volatility tier (normal/elevated/extreme)
# ─────────────────────────────────────────────────────────────────────────────
def build_regime_risk_targets(
    atr: np.ndarray,
    is_session_break: np.ndarray,
    *,
    rolling_window: int = 100,
    elevated_ratio: float = 1.5,     # ATR > 1.5× trailing median = elevated
    extreme_ratio: float = 2.5,      # ATR > 2.5× = extreme
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar volatility regime tier — based on CURRENT atr vs trailing
    median. Strictly causal — uses only past bars in the median window.

    Returns (risk_label, valid):
      risk_label[i] ∈ {0=normal, 1=elevated, 2=extreme}
      valid[i]      = True iff enough history is available
    """
    n = len(atr)
    risk = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - rolling_window)
        # Strictly past bars
        past = atr[lo:i]
        # Filter out session-break-following NaN-prone bars
        if len(past) < 20:
            continue
        past_clean = past[(np.isfinite(past)) & (past > 0)]
        if len(past_clean) < 10:
            continue
        med = float(np.median(past_clean))
        if med <= 0:
            continue
        cur = float(atr[i])
        if not np.isfinite(cur) or cur <= 0:
            continue
        ratio = cur / med
        if ratio >= extreme_ratio:
            risk[i] = 2
        elif ratio >= elevated_ratio:
            risk[i] = 1
        else:
            risk[i] = 0
        valid[i] = True
    return risk, valid


# ─────────────────────────────────────────────────────────────────────────────
# Bundle builder
# ─────────────────────────────────────────────────────────────────────────────
def build_all_adaptive_targets(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    bias_direction: np.ndarray,
    is_session_break: np.ndarray,
    forward_bars: int = 6,
    bucket_edges: tuple[float, ...] = (1.0, 2.0, 3.0),
    regime_rolling_window: int = 100,
    elevated_ratio: float = 1.5,
    extreme_ratio: float = 2.5,
) -> dict[str, dict[str, np.ndarray]]:
    """Bundle all 3 adaptive-target builders into a single dict.

    Returns {task: {'label': ..., 'valid': ...}} for downstream training.
    """
    max_R, max_R_valid = build_max_R_reached_targets(
        close, high, low, atr, bias_direction, is_session_break,
        forward_bars=forward_bars,
    )
    bucket, bucket_valid = build_target_bucket_targets(
        max_R, max_R_valid, bucket_edges=bucket_edges,
    )
    risk, risk_valid = build_regime_risk_targets(
        atr, is_session_break,
        rolling_window=regime_rolling_window,
        elevated_ratio=elevated_ratio,
        extreme_ratio=extreme_ratio,
    )
    return {
        'max_R': {'label': max_R, 'valid': max_R_valid},
        'bucket': {'label': bucket, 'valid': bucket_valid},
        'regime_risk': {'label': risk, 'valid': risk_valid},
    }
