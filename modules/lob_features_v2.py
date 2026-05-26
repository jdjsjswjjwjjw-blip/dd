"""
modules/lob_features_v2.py
═══════════════════════════════════════════════════════════════════════════════
LOB feature builder v2 — "human-style" enhancements over the original 9-channel
representation in prepare_day_trading.py.

Adds 4 critical channels for day-trading entry detection (per the development
proposal):
  • Multi-threshold wall detection (3×, 5×, 10× median)
  • Liquidity gap detection (empty levels in middle of book)
  • Depth gradient (signed rate of change → accumulation vs leakage)

Iceberg detection deliberately stays out of this module — it requires
order_id-level tracking from MBO ticks, which lives in
`self_supervised/build_order_batches.py`. The L2 snapshot here cannot see
hidden orders by definition.

Channel layout (13 channels per level, 20 levels per snapshot):
─────────────────────────────────────────────────────────────────────────────
  ch0   depth_combined_log    log1p(bid+ask size at level)              [legacy]
  ch1   trade_imbalance       per-level (buy-sell)/(buy+sell)            [legacy]
  ch2   trade_vol_log         log1p(total trade vol at level)            [legacy]
  ch3   bid_depth_log         log1p(bid_sz at level, 0 at ask side)      [legacy]
  ch4   ask_depth_log         log1p(ask_sz at level, 0 at bid side)      [legacy]
  ch5   buy_vol_log           log1p(buy trade vol at level)              [legacy]
  ch6   sell_vol_log          log1p(sell trade vol at level)             [legacy]
  ch7   wall_3x               size > 3× per-snapshot median              [legacy]
  ch8   wall_5x               size > 5× per-snapshot median              [NEW]
  ch9   wall_10x              size > 10× per-snapshot median             [NEW]
  ch10  liquidity_gap         1 iff this level is empty AND surrounded   [NEW]
                              by non-empty levels (gap inside the book)
  ch11  gap_position_norm     signed distance to nearest gap in levels;  [NEW]
                              positive = gap is "above" (toward best price),
                              negative = "below". 0 means no gap.
  ch12  depth_gradient        rolling rate of size change at this level  [NEW]
                              over the last N snapshots (signed, normalized).
                              + = accumulation, − = leakage.

Levels 0..9   = bid side (reversed: 0 = deepest bid, 9 = best bid)
Levels 10..19 = ask side (10 = best ask, 19 = deepest ask)

This module is COMPLETELY PURE — no I/O, no global state. Each function takes
arrays in and returns arrays out. That makes it trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Channel count — bump this and the snapshot builder together if you add more
N_LOB_CHANNELS_V2 = 13


@dataclass(frozen=True)
class LOBChannelsV2:
    """Index constants for v2 channels. Use these everywhere instead of magic
    numbers; if you ever reorder channels, the type system catches it."""
    DEPTH_LOG: int = 0
    TRADE_IMB: int = 1
    TRADE_VOL: int = 2
    BID_DEPTH: int = 3
    ASK_DEPTH: int = 4
    BUY_VOL: int = 5
    SELL_VOL: int = 6
    WALL_3X: int = 7
    WALL_5X: int = 8
    WALL_10X: int = 9
    LIQ_GAP: int = 10
    GAP_POS: int = 11
    DEPTH_GRAD: int = 12


CH = LOBChannelsV2()


# ─────────────────────────────────────────────────────────────────────────────
# Wall detection (multi-threshold)
# ─────────────────────────────────────────────────────────────────────────────
def _wall_mask_at_thresholds(
    raw_depth: np.ndarray, multipliers: tuple[float, ...] = (3.0, 5.0, 10.0),
) -> np.ndarray:
    """For a single snapshot's raw depth array of shape (P,), return a
    (P, len(multipliers)) boolean mask where col k is True iff
    raw_depth > multipliers[k] * median(nonzero_depths).

    Returns zeros (all False) if the snapshot is entirely empty.
    """
    P = raw_depth.shape[0]
    K = len(multipliers)
    out = np.zeros((P, K), dtype=np.float32)
    nz = raw_depth[raw_depth > 0]
    if len(nz) == 0:
        return out
    med = float(np.median(nz))
    if med <= 0:
        return out
    for k, mult in enumerate(multipliers):
        out[:, k] = (raw_depth > (mult * med)).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity gaps
# ─────────────────────────────────────────────────────────────────────────────
def _liquidity_gaps(raw_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect empty levels with non-empty neighbors → "gaps in the book".

    A gap matters more when it's:
      • surrounded by liquidity (someone pulled an order in a key zone)
      • close to the best price (impact on next price move)

    Returns two (P,) float32 arrays:
      gap_flag      — 1.0 iff level is empty AND has non-empty neighbor on
                      both sides within the same book side (bid or ask).
                      A side's edge (deepest level) is never a gap candidate.
      gap_pos_norm  — for each level i, the signed normalized distance to the
                      nearest gap. Positive = gap toward best price, negative
                      = gap toward deeper book. 0.0 = no gap on this side.
                      Normalized by half_levels.

    Why same-side only: an empty level at index 10 (best ask) doesn't form a
    gap with an empty index 9 (best bid). They're separate sides of the book.
    """
    P = raw_depth.shape[0]
    flag = np.zeros(P, dtype=np.float32)
    pos = np.zeros(P, dtype=np.float32)
    if P < 4:
        return flag, pos

    half = P // 2
    # bid side: indices 0..half-1
    # ask side: indices half..P-1
    for side_start, side_end in ((0, half), (half, P)):
        side_n = side_end - side_start
        if side_n < 3:
            continue
        side = raw_depth[side_start:side_end]
        # Gap = zero level with non-zero left AND non-zero right
        is_zero = side == 0
        if not is_zero.any():
            continue
        # left neighbor non-empty
        left_nonzero = np.concatenate([[False], side[:-1] > 0])
        right_nonzero = np.concatenate([side[1:] > 0, [False]])
        side_gap = is_zero & left_nonzero & right_nonzero
        flag[side_start:side_end] = side_gap.astype(np.float32)

        if not side_gap.any():
            continue
        gap_idxs = np.where(side_gap)[0]  # local indices
        # For each level in side, find signed distance to NEAREST gap
        for local_i in range(side_n):
            # signed dist: + = gap is at higher local idx (toward best for ask,
            # toward deeper bid). We unify by direction-to-best-price.
            dists = gap_idxs - local_i  # signed
            # Pick smallest absolute
            nearest = dists[np.argmin(np.abs(dists))]
            # Normalize by half size
            pos_global = side_start + local_i
            # For bid side (0..half-1), "best price" is at higher local idx
            # (index 9), so positive dist = toward best price.
            # For ask side (half..P-1), "best price" is at lower local idx
            # (index 10), so we flip sign.
            sign_flip = -1.0 if side_start >= half else 1.0
            pos[pos_global] = float(sign_flip * nearest / max(side_n - 1, 1))

    return flag, pos


# ─────────────────────────────────────────────────────────────────────────────
# Depth gradient (rolling)
# ─────────────────────────────────────────────────────────────────────────────
def _depth_gradient(
    raw_depth_seq: np.ndarray, window: int = 5,
) -> np.ndarray:
    """For a sequence of snapshots, compute per-level depth gradient.

    raw_depth_seq: (T, P) — raw size at each level across T snapshots.

    Returns gradient (T, P): for each (t, p), the signed-normalized
    rate-of-change over the trailing window of `window` snapshots:
        grad[t, p] = (raw[t, p] - raw[t - w, p]) / max(raw[t - w, p], 1)

    Clipped to [-10, +10] to avoid blow-up from sudden book resets.
    Snapshots with t < window are filled by shorter window (graceful).
    """
    T, P = raw_depth_seq.shape
    out = np.zeros((T, P), dtype=np.float32)
    for t in range(T):
        w = min(window, t)
        if w == 0:
            continue
        prev = raw_depth_seq[t - w]
        denom = np.maximum(prev, 1.0)
        grad = (raw_depth_seq[t] - prev) / denom
        out[t] = np.clip(grad, -10.0, 10.0).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API: build (T, P, 13) tensor for ONE bar
# ─────────────────────────────────────────────────────────────────────────────
def build_lob_tensor_v2_for_bar(
    raw_depth_seq: np.ndarray,           # (T, P)  raw bid+ask sizes per level
    depth_log_seq: np.ndarray,           # (T, P)  log1p of raw_depth_seq
    bid_sz_seq: np.ndarray,              # (T, half)  raw bid sizes, best..deepest
    ask_sz_seq: np.ndarray,              # (T, half)  raw ask sizes, best..deepest
    buy_vol_seq: np.ndarray,             # (T, P)  raw buy trade vol per level
    sell_vol_seq: np.ndarray,            # (T, P)  raw sell trade vol per level
    *,
    wall_multipliers: tuple[float, ...] = (3.0, 5.0, 10.0),
    gradient_window: int = 5,
) -> np.ndarray:
    """Build a (T, P, 13) feature tensor for one bar's worth of LOB snapshots.

    All inputs are pre-sliced for this bar by the caller; this function does
    no time-window logic of its own. That keeps it pure and easy to test.
    """
    T, P = raw_depth_seq.shape
    assert depth_log_seq.shape == (T, P)
    assert buy_vol_seq.shape == (T, P)
    assert sell_vol_seq.shape == (T, P)
    half = P // 2
    assert bid_sz_seq.shape == (T, half)
    assert ask_sz_seq.shape == (T, half)
    assert len(wall_multipliers) == 3, "v2 expects exactly 3 wall thresholds"

    out = np.zeros((T, P, N_LOB_CHANNELS_V2), dtype=np.float32)

    # Channels 0..6 are per-snapshot, vectorizable
    out[:, :, CH.DEPTH_LOG] = depth_log_seq.astype(np.float32)

    tot_fp_raw = buy_vol_seq + sell_vol_seq
    # Use 1.0 as the minimum denominator (= 1 share traded). 1e-9 made
    # the ratio blow up when total volume was a single tick.
    out[:, :, CH.TRADE_IMB] = np.divide(
        buy_vol_seq - sell_vol_seq, np.maximum(tot_fp_raw, 1.0),
    ).astype(np.float32)
    out[:, :, CH.TRADE_VOL] = np.log1p(np.maximum(tot_fp_raw, 0.0)).astype(np.float32)

    # ch3 — bid_depth: levels 0..half-1, reversed so best-bid is index half-1
    bid_block = np.log1p(np.maximum(bid_sz_seq[:, ::-1].astype(np.float32), 0.0))
    out[:, :half, CH.BID_DEPTH] = bid_block

    # ch4 — ask_depth: levels half..P-1
    ask_block = np.log1p(np.maximum(ask_sz_seq.astype(np.float32), 0.0))
    out[:, half:, CH.ASK_DEPTH] = ask_block

    out[:, :, CH.BUY_VOL] = np.log1p(np.maximum(buy_vol_seq.astype(np.float32), 0.0))
    out[:, :, CH.SELL_VOL] = np.log1p(np.maximum(sell_vol_seq.astype(np.float32), 0.0))

    # Channels 7..9 — multi-threshold walls (per snapshot)
    for t in range(T):
        walls = _wall_mask_at_thresholds(raw_depth_seq[t], wall_multipliers)
        out[t, :, CH.WALL_3X] = walls[:, 0]
        out[t, :, CH.WALL_5X] = walls[:, 1]
        out[t, :, CH.WALL_10X] = walls[:, 2]

    # Channels 10..11 — liquidity gap (per snapshot)
    for t in range(T):
        flag, pos = _liquidity_gaps(raw_depth_seq[t])
        out[t, :, CH.LIQ_GAP] = flag
        out[t, :, CH.GAP_POS] = pos

    # Channel 12 — depth gradient (rolling, needs full T)
    out[:, :, CH.DEPTH_GRAD] = _depth_gradient(raw_depth_seq, window=gradient_window)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build (N_bars, T, P, 13) for a whole dataset by stacking
# per-bar tensors. Use this only when memory allows — otherwise build per-bar
# in a streaming loop.
# ─────────────────────────────────────────────────────────────────────────────
def stack_bars(per_bar_tensors: list[np.ndarray]) -> np.ndarray:
    """Stack a list of (T, P, 13) tensors into (N, T, P, 13). Validates shapes."""
    if len(per_bar_tensors) == 0:
        raise ValueError("No tensors to stack")
    ref = per_bar_tensors[0].shape
    for i, a in enumerate(per_bar_tensors):
        if a.shape != ref:
            raise ValueError(f"shape mismatch at index {i}: {a.shape} != {ref}")
    return np.stack(per_bar_tensors, axis=0)
