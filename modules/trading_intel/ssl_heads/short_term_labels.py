"""
modules/deep_lob/short_term_targets.py
═══════════════════════════════════════════════════════════════════════════════
Label builders for the short-term SSL heads (`short_term_heads.py`).

Each builder is a pure-numpy function:
  • Inputs: per-bar arrays (price, ATR, LOB tensors, trade volume, etc.)
  • Output: per-bar label vector + per-bar `valid` mask
  • Causal: never peeks past `i + forward_window`; never uses the row's own
    forward data without a clear forward-window contract

`valid[i] = False` whenever:
  • The forward window crosses a session_break
  • Required input data (ATR, walls, etc.) is missing/zero
  • The forward window extends past the dataset

The downstream training loop must filter on these masks before computing loss.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _forward_window_clean(
    is_session_break: np.ndarray, i: int, forward_bars: int,
) -> bool:
    """True iff bars (i+1, i+forward_bars] contain no session break.

    By convention `is_session_break[i]=1` means "bar i is the first bar
    of a new session" (a break occurred between i-1 and i). So the
    relevant indices to check are (i+1, i+forward_bars+1).
    """
    n = len(is_session_break)
    end = min(i + forward_bars + 1, n)
    if end <= i + 1:
        return False
    return not bool(is_session_break[i + 1:end].any())


# ─────────────────────────────────────────────────────────────────────────────
# 1. next_wall_break — Will the dominant wall be broken within K bars?
# ─────────────────────────────────────────────────────────────────────────────
def build_next_wall_break_targets(
    raw_depth_seq: np.ndarray,             # (N_bars, T, P) — last snapshot per bar matters
    high: np.ndarray,                       # (N_bars,)
    low: np.ndarray,                        # (N_bars,)
    bid_prices: np.ndarray,                 # (N_bars, half) best..deepest bid prices
    ask_prices: np.ndarray,                 # (N_bars, half) best..deepest ask prices
    is_session_break: np.ndarray,           # (N_bars,) bool/int
    *,
    forward_bars: int = 1,                  # for 15min bars, 1 = 15 minutes
    wall_multiplier: float = 5.0,           # use 5× walls (proposal recommends 3/5/10)
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (label, valid) where label[i] ∈ {0, 1, 2}:
       0 = wall broken downward (bid wall consumed)
       1 = no wall break
       2 = wall broken upward (ask wall consumed)

    "Broken" = price moved past the wall's price level in the forward window.

    The dominant wall = the level with the largest size in the last snapshot
    of bar i, on the side that has the strongest wall.
    """
    N = raw_depth_seq.shape[0]
    P = raw_depth_seq.shape[2]
    half = P // 2
    assert bid_prices.shape == (N, half)
    assert ask_prices.shape == (N, half)

    label = np.full(N, 1, dtype=np.int64)  # 1 = no break (default)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if not _forward_window_clean(is_session_break, i, forward_bars):
            continue
        # Last snapshot of bar i
        snap = raw_depth_seq[i, -1, :]  # (P,)
        # Find largest wall size and where it sits
        nz = snap[snap > 0]
        if len(nz) == 0:
            continue
        med = float(np.median(nz))
        if med <= 0:
            continue
        thr = wall_multiplier * med
        wall_mask = snap > thr
        if not wall_mask.any():
            continue
        # Among walls, pick the level with maximum size
        wall_idxs = np.where(wall_mask)[0]
        sizes = snap[wall_idxs]
        biggest = wall_idxs[int(np.argmax(sizes))]

        # Resolve wall price by side
        if biggest < half:
            # bid side — biggest is in [0..half-1], reversed: 0 deepest, half-1 best
            local = biggest
            # bid_prices is (best..deepest), so map back
            best_idx_in_local = half - 1 - local
            wall_price = float(bid_prices[i, best_idx_in_local])
            side = "bid"
        else:
            local = biggest - half
            wall_price = float(ask_prices[i, local])
            side = "ask"

        if not np.isfinite(wall_price) or wall_price <= 0:
            continue

        # Look forward
        end = min(i + forward_bars + 1, N)
        fwd_high = float(np.max(high[i + 1:end]))
        fwd_low = float(np.min(low[i + 1:end]))

        broken = False
        if side == "bid" and fwd_low < wall_price:
            label[i] = 0  # bid wall consumed → bearish break
            broken = True
        elif side == "ask" and fwd_high > wall_price:
            label[i] = 2  # ask wall consumed → bullish break
            broken = True
        # else: label stays 1 = no break

        valid[i] = True
        _ = broken  # silence lint
    return label, valid


# ─────────────────────────────────────────────────────────────────────────────
# 2. next_imbalance_shift — Will OBI sign flip within K bars?
# ─────────────────────────────────────────────────────────────────────────────
def build_next_imbalance_shift_targets(
    obi_signed: np.ndarray,                 # (N_bars,) — OBI value at end of bar
    is_session_break: np.ndarray,
    *,
    forward_bars: int = 2,                  # 30 min on 15min bars
    min_magnitude: float = 0.10,            # ignore noise around zero
) -> tuple[np.ndarray, np.ndarray]:
    """Label: 1 if OBI sign flips in the forward window, else 0.

    Both the current and future OBI must exceed `min_magnitude` in absolute
    value — otherwise we skip noise around zero (which would inflate label
    rate without giving useful information).
    """
    N = len(obi_signed)
    label = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)

    sign = np.sign(obi_signed)
    mag = np.abs(obi_signed)

    for i in range(N):
        if not _forward_window_clean(is_session_break, i, forward_bars):
            continue
        if mag[i] < min_magnitude:
            continue
        end = min(i + forward_bars + 1, N)
        fwd = sign[i + 1:end]
        fwd_mag = mag[i + 1:end]
        # Find any future bar with opposite sign AND meaningful magnitude
        opposite = (fwd != sign[i]) & (fwd != 0) & (fwd_mag >= min_magnitude)
        label[i] = int(opposite.any())
        valid[i] = True
    return label, valid


# ─────────────────────────────────────────────────────────────────────────────
# 3. next_micro_target_hit — Will price reach 0.5R or 1R within K bars?
# ─────────────────────────────────────────────────────────────────────────────
def build_next_micro_target_targets(
    close: np.ndarray,                      # (N_bars,)
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,                        # (N_bars,) — atr_14
    bias_direction: np.ndarray,             # (N_bars,) int in {-1, 0, +1} — current bias
    is_session_break: np.ndarray,
    *,
    forward_bars: int = 4,                  # 1 hour on 15min bars
    target_R_low: float = 0.5,              # half-R micro target
    target_R_high: float = 1.0,             # full-R target
) -> tuple[np.ndarray, np.ndarray]:
    """Label: 0 = none, 1 = 0.5R hit (but not 1R), 2 = 1R hit.

    A "hit" requires:
      • bias_direction != 0 (we need a direction to chase)
      • atr > 0 (need a unit to size the R)
      • forward window clean of session_break
      • forward high (long) or forward low (short) reaches target

    Direction-aware:
      LONG  → hit if forward_high >= entry + target_R * atr
      SHORT → hit if forward_low  <= entry - target_R * atr
    """
    N = len(close)
    label = np.zeros(N, dtype=np.int64)
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

        r = move / a if a > 0 else 0.0
        if r >= target_R_high:
            label[i] = 2
        elif r >= target_R_low:
            label[i] = 1
        else:
            label[i] = 0
        valid[i] = True
    return label, valid


# ─────────────────────────────────────────────────────────────────────────────
# 4. next_liquidity_sweep — Will a large sweep happen in next W bars?
# ─────────────────────────────────────────────────────────────────────────────
def build_next_liquidity_sweep_targets(
    trade_volume: np.ndarray,               # (N_bars,) — total volume per bar
    is_session_break: np.ndarray,
    *,
    forward_bars: int = 2,
    sweep_multiplier: float = 3.0,          # >=3× rolling median
    rolling_window: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Label: 1 if any future bar in (i, i+W] has volume >= sweep_multiplier
    × rolling_median(trade_volume).

    Uses TRAILING rolling median computed only from history (causal).
    """
    N = len(trade_volume)
    label = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)

    # Causal trailing median — at index i, median of (i-rolling_window..i]
    # We need med[j] for j > i (the future bars). Use j's OWN trailing window,
    # which is fine — it doesn't peek past j.
    for i in range(N):
        if not _forward_window_clean(is_session_break, i, forward_bars):
            continue
        end = min(i + forward_bars + 1, N)
        sweep_found = False
        for j in range(i + 1, end):
            lo = max(0, j - rolling_window)
            win = trade_volume[lo:j]  # strictly past from j's perspective
            if len(win) < 5:
                continue
            med = float(np.median(win[win > 0])) if (win > 0).any() else 0.0
            if med <= 0:
                continue
            if float(trade_volume[j]) >= sweep_multiplier * med:
                sweep_found = True
                break
        label[i] = int(sweep_found)
        valid[i] = True
    return label, valid


# ─────────────────────────────────────────────────────────────────────────────
# 5. next_gap_fill_attempt — Will price touch the nearest liquidity gap?
# ─────────────────────────────────────────────────────────────────────────────
def build_next_gap_fill_targets(
    raw_depth_seq: np.ndarray,              # (N_bars, T, P)
    bid_prices: np.ndarray,                 # (N_bars, half)
    ask_prices: np.ndarray,                 # (N_bars, half)
    high: np.ndarray,
    low: np.ndarray,
    is_session_break: np.ndarray,
    *,
    forward_bars: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Label: 1 if price (forward high/low) touches the price level of the
    nearest-to-best liquidity gap on either side; 0 otherwise.

    A "gap" = empty level surrounded by non-empty levels on the same side
    (matches `_liquidity_gaps` in `..lob.features`).
    """
    from modules.trading_intel.lob.features import _liquidity_gaps

    N = raw_depth_seq.shape[0]
    P = raw_depth_seq.shape[2]
    half = P // 2

    label = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if not _forward_window_clean(is_session_break, i, forward_bars):
            continue
        snap = raw_depth_seq[i, -1, :]  # (P,)
        gap_flag, _gap_pos = _liquidity_gaps(snap)
        if not gap_flag.any():
            continue

        # Find the closest gap to the best price
        # On the bid side, "closest to best" = highest local index (best bid = half-1)
        # On the ask side, "closest to best" = lowest local index (best ask = half)
        gap_idxs = np.where(gap_flag == 1)[0]

        # Pick the gap level whose mapped price is closest to the mid
        target_prices = []
        for g in gap_idxs:
            if g < half:
                # bid side gap; local g maps to bid_prices[i, half-1-g]
                p = float(bid_prices[i, half - 1 - g])
            else:
                # ask side; local g-half maps to ask_prices[i, g-half]
                p = float(ask_prices[i, g - half])
            if np.isfinite(p) and p > 0:
                target_prices.append(p)
        if not target_prices:
            continue

        end = min(i + forward_bars + 1, N)
        fwd_high = float(np.max(high[i + 1:end]))
        fwd_low = float(np.min(low[i + 1:end]))

        # Touched = any target price lies inside [fwd_low, fwd_high]
        touched = any(fwd_low <= p <= fwd_high for p in target_prices)
        label[i] = int(touched)
        valid[i] = True
    return label, valid


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: bundle all 5 builders into one call
# ─────────────────────────────────────────────────────────────────────────────
def build_all_short_term_targets(
    *,
    # bar-level series
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    obi_signed: np.ndarray,
    trade_volume: np.ndarray,
    bias_direction: np.ndarray,
    is_session_break: np.ndarray,
    # LOB tensors
    raw_depth_seq: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    # per-task forward windows
    wall_forward: int = 1,
    imbalance_forward: int = 2,
    micro_forward: int = 4,
    sweep_forward: int = 2,
    gap_forward: int = 2,
) -> dict[str, dict[str, np.ndarray]]:
    """Build all 5 short-term targets in one go.

    Returns a dict {task: {'label': arr, 'valid': arr}}.
    """
    return {
        "wall_break": dict(zip(
            ("label", "valid"),
            build_next_wall_break_targets(
                raw_depth_seq, high, low, bid_prices, ask_prices,
                is_session_break, forward_bars=wall_forward,
            ),
        )),
        "imbalance_shift": dict(zip(
            ("label", "valid"),
            build_next_imbalance_shift_targets(
                obi_signed, is_session_break, forward_bars=imbalance_forward,
            ),
        )),
        "micro_target": dict(zip(
            ("label", "valid"),
            build_next_micro_target_targets(
                close, high, low, atr, bias_direction, is_session_break,
                forward_bars=micro_forward,
            ),
        )),
        "liquidity_sweep": dict(zip(
            ("label", "valid"),
            build_next_liquidity_sweep_targets(
                trade_volume, is_session_break, forward_bars=sweep_forward,
            ),
        )),
        "gap_fill": dict(zip(
            ("label", "valid"),
            build_next_gap_fill_targets(
                raw_depth_seq, bid_prices, ask_prices, high, low,
                is_session_break, forward_bars=gap_forward,
            ),
        )),
    }
