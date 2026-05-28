"""
modules/replay_engine.book
══════════════════════════════════════════════════════════════════════════════
Pure data structures + pure functions for limit-order-book simulation.

Nothing here talks to disk, holds RNG state, or mutates anything other than
its own LOBState instance. This keeps the math testable in isolation and
keeps the orchestrator (engine.py) the only place that owns time.

Design contract
───────────────
LOBSnapshot   immutable view of the book at one instant (10 levels per side)
LOBState      mutable state machine; ingest MBP-10 rows → produce snapshots
fill_market_order(snapshot, side, size)
              pure: walk the book level by level, return fill price + residual
AdaptiveSlippage(snapshot, side, size)
              pure: expected slippage given level-1 depth ratio
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["BUY", "SELL"]

# Databento MBP-10 column convention: 00..09 (zero-padded, two-digit suffix)
MBP_DEPTH = 10


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot — immutable view of the book at one instant
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LOBSnapshot:
    """
    Frozen view of the book. Arrays are length-`MBP_DEPTH`, level 0 = best.

    Bid prices descend (level-0 is the best bid, level-1 < level-0).
    Ask prices ascend (level-0 is the best ask, level-1 > level-0).
    """
    ts_event: np.datetime64
    bid_px: np.ndarray   # shape (10,), float64
    bid_sz: np.ndarray   # shape (10,), float64
    ask_px: np.ndarray
    ask_sz: np.ndarray

    @property
    def best_bid(self) -> float:
        return float(self.bid_px[0])

    @property
    def best_ask(self) -> float:
        return float(self.ask_px[0])

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def total_depth(self, side: Side, n_levels: int = MBP_DEPTH) -> float:
        """Total visible size across the first `n_levels` of `side`."""
        n = max(1, min(int(n_levels), MBP_DEPTH))
        arr = self.ask_sz if side == "BUY" else self.bid_sz
        return float(np.nansum(arr[:n]))


def snapshot_from_mbp_row(row: dict | object) -> LOBSnapshot:
    """
    Build an LOBSnapshot from a single MBP-10 row (dict or pandas Series).

    Required columns: ts_event, bid_px_00..09, bid_sz_00..09,
                      ask_px_00..09, ask_sz_00..09.
    Missing levels are filled with NaN price and 0 size (treated as
    "no liquidity at this level").
    """
    get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(
        row, k, d
    ) if hasattr(row, k) else (row[k] if k in row else d)

    bid_px = np.full(MBP_DEPTH, np.nan, dtype=np.float64)
    bid_sz = np.zeros(MBP_DEPTH, dtype=np.float64)
    ask_px = np.full(MBP_DEPTH, np.nan, dtype=np.float64)
    ask_sz = np.zeros(MBP_DEPTH, dtype=np.float64)

    for i in range(MBP_DEPTH):
        s = f"{i:02d}"
        bp = get(f"bid_px_{s}")
        bs = get(f"bid_sz_{s}")
        ap = get(f"ask_px_{s}")
        as_ = get(f"ask_sz_{s}")
        if bp is not None and not _is_nan(bp):
            bid_px[i] = float(bp)
        if bs is not None and not _is_nan(bs):
            bid_sz[i] = float(bs)
        if ap is not None and not _is_nan(ap):
            ask_px[i] = float(ap)
        if as_ is not None and not _is_nan(as_):
            ask_sz[i] = float(as_)

    ts = get("ts_event")
    return LOBSnapshot(
        ts_event=np.datetime64(ts) if ts is not None else np.datetime64("NaT"),
        bid_px=bid_px, bid_sz=bid_sz,
        ask_px=ask_px, ask_sz=ask_sz,
    )


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# State machine — sequential snapshots
# ══════════════════════════════════════════════════════════════════════════════
class LOBState:
    """
    Sequential MBP-10 → snapshot stream.

    `feed_row(row)` updates the current snapshot. `current()` returns the
    last produced snapshot. This is intentionally thin — the orchestrator
    owns the iteration; we only own the state at one moment.
    """

    def __init__(self) -> None:
        self._snap: LOBSnapshot | None = None

    def feed_row(self, row: dict | object) -> LOBSnapshot:
        snap = snapshot_from_mbp_row(row)
        self._snap = snap
        return snap

    def current(self) -> LOBSnapshot | None:
        return self._snap


# ══════════════════════════════════════════════════════════════════════════════
# Market order fill — walk the book
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FillResult:
    avg_px: float          # volume-weighted average fill price
    filled_size: float     # total size successfully filled
    residual_size: float   # size left unfilled (book exhausted)
    levels_consumed: int   # how many price levels were touched
    worst_px: float        # the worst (deepest) price hit


def fill_market_order(
    snapshot: LOBSnapshot, side: Side, size: float
) -> FillResult:
    """
    Walk the book and consume liquidity level-by-level until `size` is met
    or the visible depth is exhausted.

    BUY  → consumes the ask side from level 0 upward
    SELL → consumes the bid side from level 0 downward

    Pure: snapshot is not modified.
    """
    if size <= 0.0:
        ref_px = snapshot.best_ask if side == "BUY" else snapshot.best_bid
        return FillResult(
            avg_px=ref_px, filled_size=0.0, residual_size=0.0,
            levels_consumed=0, worst_px=ref_px,
        )

    px_arr = snapshot.ask_px if side == "BUY" else snapshot.bid_px
    sz_arr = snapshot.ask_sz if side == "BUY" else snapshot.bid_sz

    remaining = float(size)
    notional = 0.0
    filled = 0.0
    levels = 0
    worst = float("nan")

    for i in range(MBP_DEPTH):
        avail = float(sz_arr[i])
        px = float(px_arr[i])
        if avail <= 0.0 or not np.isfinite(px):
            continue
        levels += 1
        take = min(avail, remaining)
        notional += take * px
        filled += take
        remaining -= take
        worst = px
        if remaining <= 1e-9:
            break

    avg_px = (notional / filled) if filled > 0 else float("nan")
    return FillResult(
        avg_px=avg_px,
        filled_size=filled,
        residual_size=max(remaining, 0.0),
        levels_consumed=levels,
        worst_px=worst if np.isfinite(worst) else avg_px,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive slippage — quantifies the level-1 depth squeeze
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class AdaptiveSlippage:
    """
    Slippage estimator that scales with how much of the level-1 depth
    the intended order consumes. Pure function wrapped in a dataclass
    so callers can swap implementations without changing call sites.

    Model
    ─────
    Let r = size / depth_at_level_1_for_this_side.
      r ≤ depth_floor       → constant `base_ticks` slippage
      depth_floor < r ≤ 1   → linear ramp from base_ticks → mid_ticks
      r > 1 (eats > L1)     → mid_ticks + extra (ticks/level_consumed)

    Outputs: expected slippage in ticks (positive number, always).
    """
    tick_size: float
    base_ticks: float = 0.5       # base spread crossing cost
    mid_ticks: float = 1.5        # full-L1 consumption cost
    extra_per_level: float = 1.0  # extra ticks per additional level eaten
    depth_floor: float = 0.10     # below this fraction, slippage is flat

    def estimate(self, snapshot: LOBSnapshot, side: Side, size: float) -> float:
        if size <= 0.0:
            return 0.0
        l1_depth = float(
            snapshot.ask_sz[0] if side == "BUY" else snapshot.bid_sz[0]
        )
        if l1_depth <= 0.0:
            # No level-1 liquidity → assume we walk the whole visible side
            return float(self.mid_ticks + self.extra_per_level * MBP_DEPTH)

        ratio = float(size) / l1_depth
        if ratio <= self.depth_floor:
            return float(self.base_ticks)
        if ratio <= 1.0:
            # Linear ramp inside L1
            span = max(1.0 - self.depth_floor, 1e-9)
            t = (ratio - self.depth_floor) / span
            return float(self.base_ticks + t * (self.mid_ticks - self.base_ticks))
        # We're past L1 — estimate levels we'll burn
        side_sz = snapshot.ask_sz if side == "BUY" else snapshot.bid_sz
        cum = float(side_sz[0])
        extra_levels = 0
        for i in range(1, MBP_DEPTH):
            if cum >= float(size):
                break
            cum += float(side_sz[i])
            extra_levels += 1
        return float(self.mid_ticks + self.extra_per_level * extra_levels)

    def estimate_dollars(
        self, snapshot: LOBSnapshot, side: Side, size: float
    ) -> float:
        return self.estimate(snapshot, side, size) * float(self.tick_size)
