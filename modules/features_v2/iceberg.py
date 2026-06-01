"""Phase 1.5 — Iceberg order detection from MBO tick streams.

Theoretical basis (Korajczyk & Murphy 2019, "Do High-Frequency Traders
Improve Liquidity Quality at the Open?"). An order is treated as an
iceberg when its EXECUTED volume at a given price level dramatically
exceeds the DISPLAYED size visible on the book — the trader is hiding
size, refilling the displayed quantity after each fill.

Detection rule used here (deliberately strict to keep false positives
low — the audit treats this as a sparse, high-confidence signal):

    For each price level touched within a bar:
      1. cumulative executed_volume_at_level >= ICEBERG_VOL_MULT × max_displayed_size
      2. AT LEAST ONE refill event (display restored after a fill) within
         ICEBERG_REFILL_SECS of the prior fill
      3. NOT a full clearout (some displayed size remained after final fill)

This module produces per-bar aggregates fit for joining onto the 5-min
refinery output:

    iceberg_count_5m         (int32) — distinct iceberg events in the bar
    iceberg_total_volume_5m  (float32) — total executed hidden volume

The implementation is intentionally pure-python+numpy on a small DataFrame
so it can be unit-tested on synthetic MBO scenarios without requiring
the live MBO feed. Integration into the refinery is conditional on the
operator providing a tick-level MBO frame; on bar-only inputs the cols
ship as zeros (and the validate() reporter notes them missing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IcebergConfig:
    """Detector knobs — frozen so unit tests can pin behaviour."""
    vol_mult: float = 2.5             # executed/displayed ratio threshold
    refill_seconds: float = 8.0       # max gap between fill and refill
    min_displayed_size: float = 1.0   # ignore levels too small to matter
    bar_freq: str = "5min"            # aggregation frequency for output


class IcebergDetector:
    """Pure-function detector. Stateless — pass MBO ticks to detect()."""

    def __init__(self, config: IcebergConfig | None = None) -> None:
        self.config = config or IcebergConfig()

    # ── Detection on raw MBO tick stream ───────────────────────────────
    def detect_events(self, mbo: pd.DataFrame) -> pd.DataFrame:
        """Returns one row per detected iceberg event with columns
        (ts_event, price_level, hidden_volume).

        Expected MBO columns:
            ts_event       (datetime64 UTC)
            price          (float — the price level the order sat at)
            side           ('B' for buy/bid, 'A' for ask/sell)
            displayed_size (float — visible size on the book at the level)
            executed_size  (float — filled size on this tick, 0 if not a fill)
            action         (str — one of 'FILL', 'REFILL', 'ADD', 'CANCEL')
        """
        required = ('ts_event', 'price', 'side', 'displayed_size',
                    'executed_size', 'action')
        for c in required:
            if c not in mbo.columns:
                raise ValueError(f"iceberg detector: MBO column missing: {c!r}")

        if len(mbo) == 0:
            return pd.DataFrame(columns=('ts_event', 'price_level', 'hidden_volume'))

        # Group by (price, side) — each price level on each side is an
        # independent level we scan for the iceberg pattern.
        df = mbo.sort_values('ts_event').reset_index(drop=True)
        events: list[dict] = []

        for (price, side), grp in df.groupby(['price', 'side']):
            if grp['displayed_size'].max() < self.config.min_displayed_size:
                continue
            evt = self._scan_level(grp, price)
            if evt is not None:
                events.append(evt)

        if not events:
            return pd.DataFrame(columns=('ts_event', 'price_level', 'hidden_volume'))
        return pd.DataFrame(events).sort_values('ts_event').reset_index(drop=True)

    def _scan_level(self, grp: pd.DataFrame, price: float) -> dict | None:
        """Scan a single (price, side) group for the iceberg rule.
        Returns event dict or None.
        """
        executed_cum = float(grp['executed_size'].sum())
        max_displayed = float(grp['displayed_size'].max())
        if max_displayed <= 0 or executed_cum < self.config.vol_mult * max_displayed:
            return None

        # Find refill timing — at least one REFILL within refill_seconds of a fill
        fills = grp[grp['action'] == 'FILL']
        refills = grp[grp['action'] == 'REFILL']
        if len(fills) == 0 or len(refills) == 0:
            return None
        # For every refill, find the prior fill; check time gap
        had_quick_refill = False
        for _, refill_row in refills.iterrows():
            prior_fills = fills[fills['ts_event'] < refill_row['ts_event']]
            if len(prior_fills) == 0:
                continue
            last_fill_ts = prior_fills['ts_event'].iloc[-1]
            gap = (refill_row['ts_event'] - last_fill_ts).total_seconds()
            if 0 < gap <= self.config.refill_seconds:
                had_quick_refill = True
                break
        if not had_quick_refill:
            return None

        # NOT a full clearout — some displayed size remained after the final fill
        last_displayed = float(grp['displayed_size'].iloc[-1])
        if last_displayed <= 0:
            return None

        return {
            'ts_event': grp['ts_event'].iloc[-1],
            'price_level': float(price),
            'hidden_volume': executed_cum,
        }

    # ── Per-bar aggregation for the refinery ───────────────────────────
    def aggregate_per_bar(
        self,
        events: pd.DataFrame,
        bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate event-level output into per-bar counts + hidden
        volume, aligned to `bars['ts_event']`.

        Returns a DataFrame indexed identically to `bars` with two cols:
            iceberg_count_5m, iceberg_total_volume_5m
        """
        if 'ts_event' not in bars.columns:
            raise ValueError("bars frame must have ts_event")
        out = pd.DataFrame({
            'iceberg_count_5m': np.zeros(len(bars), dtype=np.int32),
            'iceberg_total_volume_5m': np.zeros(len(bars), dtype=np.float32),
        }, index=bars.index)
        if events is None or len(events) == 0:
            return out

        ev = events.copy()
        ev['ts_event'] = pd.to_datetime(ev['ts_event'], utc=True)
        bars_ts = pd.to_datetime(bars['ts_event'], utc=True)
        # Floor each event to its bar
        ev['bar_floor'] = ev['ts_event'].dt.floor(self.config.bar_freq)
        # Build a Series aligned to bars
        counts = ev.groupby('bar_floor').size()
        vols = ev.groupby('bar_floor')['hidden_volume'].sum()

        bar_floor = bars_ts.dt.floor(self.config.bar_freq)
        for i, bf in enumerate(bar_floor):
            if bf in counts.index:
                out.iat[i, 0] = int(counts[bf])
                out.iat[i, 1] = float(vols.get(bf, 0.0))
        return out


# ── Convenience: drive detect + aggregate in one call ──────────────────────
def attach_iceberg_features(
    bars: pd.DataFrame,
    mbo: pd.DataFrame | None,
    *,
    config: IcebergConfig | None = None,
) -> pd.DataFrame:
    """Add iceberg_count_5m + iceberg_total_volume_5m to `bars`.

    If mbo is None (or empty), the columns ship as zeros — the refinery's
    sidecar will flag them as `feature_columns_missing` and validate()
    won't raise. Conditional integration keeps the bar-only Q2 baseline
    working while letting an MBO-equipped run produce real iceberg signal.
    """
    out = bars.copy()
    detector = IcebergDetector(config)
    if mbo is None or len(mbo) == 0:
        out['iceberg_count_5m'] = np.zeros(len(out), dtype=np.int32)
        out['iceberg_total_volume_5m'] = np.zeros(len(out), dtype=np.float32)
        return out
    events = detector.detect_events(mbo)
    agg = detector.aggregate_per_bar(events, bars)
    out['iceberg_count_5m'] = agg['iceberg_count_5m'].to_numpy()
    out['iceberg_total_volume_5m'] = agg['iceberg_total_volume_5m'].to_numpy()
    return out
