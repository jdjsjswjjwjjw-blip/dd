"""
modules/replay_engine.engine
══════════════════════════════════════════════════════════════════════════════
Replay orchestrator — owns time, owns the RNG, owns the execution log.

Pure data (LOBSnapshot, AdaptiveSlippage, fill_market_order) is imported
from `book.py`. Everything stateful (latency draws, log writing, signal
iteration) lives here so it's the only thing tests need to mock to
explore the system.

Components
──────────
LatencyModel        deterministic latency draws (configurable jitter)
ExecutionEvent      one row of the post-replay execution log
ReplayBacktest      ingests signals + tick stream → writes execution log
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from modules.replay_engine.book import (
    AdaptiveSlippage,
    FillResult,
    LOBSnapshot,
    LOBState,
    Side,
    fill_market_order,
    snapshot_from_mbp_row,
)


# ══════════════════════════════════════════════════════════════════════════════
# Latency model
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LatencyModel:
    """
    Models the gap between decision time and arrival time at the matching
    engine. Returns a `pd.Timedelta` per call.

    Distribution: lognormal-like — mean_us shifted by jitter_us * |N(0,1)|.
    Lognormal-ish keeps the model strictly positive without a hard floor.
    """
    mean_us: float = 5_000.0      # 5 ms baseline (round-trip retail)
    jitter_us: float = 2_000.0    # 2 ms jitter
    seed: int | None = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def draw(self) -> pd.Timedelta:
        # Lognormal centered around mean_us with jitter as scale.
        # Guarded floor at 100 ns so the orchestrator never sees zero.
        z = float(abs(self._rng.standard_normal()))
        us = max(self.mean_us + self.jitter_us * z, 0.1)
        return pd.Timedelta(microseconds=us)


# ══════════════════════════════════════════════════════════════════════════════
# Execution event — one row in the execution log
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExecutionEvent:
    signal_id: int                     # sequential index into the signals input
    side: str                          # "BUY" or "SELL"
    ts_decision: pd.Timestamp          # when the model emitted the order
    ts_arrival: pd.Timestamp           # when the order would hit the book
    latency_us: float                  # arrival - decision, microseconds
    intended_size: float               # requested order size
    decision_snapshot_mid: float       # mid at decision time
    arrival_snapshot_mid: float        # mid at arrival time (slippage reference)
    fill_avg_px: float                 # VWAP fill price
    fill_size: float
    residual_size: float               # unfilled
    levels_consumed: int               # how many price levels were touched
    worst_px: float                    # the deepest level we hit
    model_slippage_ticks: float        # what AdaptiveSlippage predicted
    realised_slippage_ticks: float     # (fill_avg - arrival_mid) in ticks, signed
    decision_lob_l1_depth: float       # for slippage attribution

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts_decision"] = self.ts_decision.isoformat()
        d["ts_arrival"] = self.ts_arrival.isoformat()
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Replay backtest — the orchestrator
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ReplayBacktest:
    """
    Replays a sequence of trading signals against a stream of MBP-10 ticks.

    Inputs
    ──────
    signals    DataFrame with columns: ts_event, side ("BUY"/"SELL"), size
    ticks      DataFrame with MBP-10 columns + ts_event
    slippage   AdaptiveSlippage instance (slip model is swappable)
    latency    LatencyModel instance

    Output
    ──────
    write_log(path) emits one JSON object per fill (newline-delimited)
    summarise()     returns dict of aggregate stats

    Algorithmic note
    ────────────────
    For each signal:
      1. Snapshot @ decision_ts (closest tick at or before decision_ts).
      2. Sample latency → arrival_ts.
      3. Snapshot @ arrival_ts.
      4. Fill against arrival snapshot.
      5. Log everything.

    Tick lookup is O(log n) via `np.searchsorted` on the sorted tick index.
    """
    signals: pd.DataFrame
    ticks: pd.DataFrame
    slippage: AdaptiveSlippage
    latency: LatencyModel
    events: list[ExecutionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "ts_event" not in self.ticks.columns:
            raise ValueError("ticks DataFrame must have a `ts_event` column")
        if "ts_event" not in self.signals.columns:
            raise ValueError("signals DataFrame must have a `ts_event` column")
        for col in ("side", "size"):
            if col not in self.signals.columns:
                raise ValueError(f"signals DataFrame missing `{col}` column")

        self._ticks = (
            self.ticks.sort_values("ts_event").reset_index(drop=True)
        )
        self._tick_ts = pd.to_datetime(self._ticks["ts_event"]).to_numpy()
        if not len(self._tick_ts):
            raise ValueError("ticks DataFrame is empty")

    # ── helpers ──────────────────────────────────────────────────────────
    def _snapshot_at_or_before(self, ts) -> LOBSnapshot | None:
        """Most recent tick at or before `ts`. None if `ts` < first tick."""
        idx = int(np.searchsorted(self._tick_ts, np.datetime64(ts), side="right")) - 1
        if idx < 0:
            return None
        return snapshot_from_mbp_row(self._ticks.iloc[idx])

    # ── main loop ────────────────────────────────────────────────────────
    def run(self) -> list[ExecutionEvent]:
        self.events.clear()
        tick_size = self.slippage.tick_size

        sig_ts = pd.to_datetime(self.signals["ts_event"]).to_numpy()
        sides = self.signals["side"].astype(str).to_numpy()
        sizes = pd.to_numeric(self.signals["size"], errors="coerce").to_numpy()

        for i in range(len(self.signals)):
            ts_dec = pd.Timestamp(sig_ts[i])
            side: Side = "BUY" if sides[i].upper() == "BUY" else "SELL"
            size = float(sizes[i])

            snap_dec = self._snapshot_at_or_before(ts_dec)
            if snap_dec is None:
                continue

            lat = self.latency.draw()
            ts_arr = ts_dec + lat
            snap_arr = self._snapshot_at_or_before(ts_arr) or snap_dec

            fill: FillResult = fill_market_order(snap_arr, side, size)
            model_slip = self.slippage.estimate(snap_arr, side, size)

            ref_px = snap_arr.best_ask if side == "BUY" else snap_arr.best_bid
            if np.isfinite(fill.avg_px) and np.isfinite(ref_px) and tick_size > 0:
                signed = 1.0 if side == "BUY" else -1.0
                realised_slip_ticks = signed * (fill.avg_px - ref_px) / tick_size
            else:
                realised_slip_ticks = float("nan")

            l1_depth = float(
                snap_dec.ask_sz[0] if side == "BUY" else snap_dec.bid_sz[0]
            )

            self.events.append(
                ExecutionEvent(
                    signal_id=i,
                    side=side,
                    ts_decision=ts_dec,
                    ts_arrival=ts_arr,
                    latency_us=float(lat.total_seconds() * 1e6),
                    intended_size=size,
                    decision_snapshot_mid=snap_dec.mid,
                    arrival_snapshot_mid=snap_arr.mid,
                    fill_avg_px=float(fill.avg_px),
                    fill_size=float(fill.filled_size),
                    residual_size=float(fill.residual_size),
                    levels_consumed=int(fill.levels_consumed),
                    worst_px=float(fill.worst_px),
                    model_slippage_ticks=float(model_slip),
                    realised_slippage_ticks=float(realised_slip_ticks),
                    decision_lob_l1_depth=l1_depth,
                )
            )
        return list(self.events)

    # ── I/O ──────────────────────────────────────────────────────────────
    def write_log(self, path: str | Path) -> None:
        """JSON Lines: one event per line so logs are append-friendly."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for ev in self.events:
                fh.write(json.dumps(ev.to_dict()) + "\n")

    def summarise(self) -> dict:
        """Aggregate stats — the slippage/latency/fill-quality numbers
        the model didn't see during training."""
        if not self.events:
            return {"n_signals": int(len(self.signals)), "n_filled": 0}
        df = pd.DataFrame(ev.to_dict() for ev in self.events)
        n_filled = int((df["fill_size"] > 0).sum())
        n_residual = int((df["residual_size"] > 0).sum())
        return {
            "n_signals": int(len(self.signals)),
            "n_executed": int(len(df)),
            "n_filled": n_filled,
            "n_with_residual": n_residual,
            "mean_latency_us": float(df["latency_us"].mean()),
            "p95_latency_us": float(df["latency_us"].quantile(0.95)),
            "mean_model_slippage_ticks": float(df["model_slippage_ticks"].mean()),
            "mean_realised_slippage_ticks": float(
                df["realised_slippage_ticks"].mean()
            ),
            "mean_levels_consumed": float(df["levels_consumed"].mean()),
            "max_levels_consumed": int(df["levels_consumed"].max()),
            "total_intended_size": float(df["intended_size"].sum()),
            "total_filled_size": float(df["fill_size"].sum()),
            "total_residual_size": float(df["residual_size"].sum()),
        }
