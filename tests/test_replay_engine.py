"""Tests for the event-by-event Replay Engine.

The engine has four independently-testable pieces:

    book.LOBSnapshot / snapshot_from_mbp_row
    book.fill_market_order
    book.AdaptiveSlippage
    engine.LatencyModel / ReplayBacktest

Each test crafts the minimal inputs needed to exercise one behavioural
contract, so a regression in one piece narrows to one failing test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.replay_engine import (
    MBP_DEPTH,
    AdaptiveSlippage,
    ExecutionEvent,
    LatencyModel,
    LOBSnapshot,
    LOBState,
    ReplayBacktest,
    fill_market_order,
    snapshot_from_mbp_row,
)


# ════════════════════════════════════════════════════════════════════
# Helpers — synthetic MBP-10 rows
# ════════════════════════════════════════════════════════════════════
def _mbp_row(
    bid_top: float = 1.2999,
    ask_top: float = 1.3001,
    tick: float = 0.0001,
    bid_sizes: tuple[float, ...] = (10, 8, 6, 4, 2, 1, 1, 1, 1, 1),
    ask_sizes: tuple[float, ...] = (10, 8, 6, 4, 2, 1, 1, 1, 1, 1),
    ts: str = "2024-01-01T00:00:00",
) -> dict:
    """Build a single MBP-10 row dict for tests."""
    row = {"ts_event": pd.Timestamp(ts)}
    for i in range(MBP_DEPTH):
        row[f"bid_px_{i:02d}"] = bid_top - i * tick
        row[f"ask_px_{i:02d}"] = ask_top + i * tick
        row[f"bid_sz_{i:02d}"] = float(bid_sizes[i]) if i < len(bid_sizes) else 0.0
        row[f"ask_sz_{i:02d}"] = float(ask_sizes[i]) if i < len(ask_sizes) else 0.0
    return row


# ════════════════════════════════════════════════════════════════════
# LOBSnapshot + snapshot_from_mbp_row
# ════════════════════════════════════════════════════════════════════
class TestSnapshot:
    def test_construction_basics(self):
        s = snapshot_from_mbp_row(_mbp_row())
        assert s.best_bid == pytest.approx(1.2999)
        assert s.best_ask == pytest.approx(1.3001)
        assert s.mid == pytest.approx(1.3000)
        assert s.spread == pytest.approx(0.0002)
        assert s.total_depth("BUY") == pytest.approx(35.0)   # 10+8+...+1
        assert s.total_depth("SELL") == pytest.approx(35.0)

    def test_missing_level_treated_as_zero_depth(self):
        row = _mbp_row(ask_sizes=(10, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        s = snapshot_from_mbp_row(row)
        assert s.total_depth("BUY") == pytest.approx(10.0)

    def test_immutable(self):
        s = snapshot_from_mbp_row(_mbp_row())
        with pytest.raises(Exception):
            s.bid_px = np.zeros(10)   # frozen dataclass


class TestLOBState:
    def test_feed_row_updates_current(self):
        st = LOBState()
        assert st.current() is None
        s1 = st.feed_row(_mbp_row(bid_top=1.2999))
        assert st.current() is s1
        s2 = st.feed_row(_mbp_row(bid_top=1.3001))
        assert st.current() is s2
        assert s2.best_bid > s1.best_bid


# ════════════════════════════════════════════════════════════════════
# fill_market_order
# ════════════════════════════════════════════════════════════════════
class TestFillMarketOrder:
    def test_zero_size_returns_no_fill(self):
        s = snapshot_from_mbp_row(_mbp_row())
        r = fill_market_order(s, "BUY", 0.0)
        assert r.filled_size == 0.0
        assert r.residual_size == 0.0
        assert r.levels_consumed == 0

    def test_small_buy_fills_at_best_ask(self):
        s = snapshot_from_mbp_row(_mbp_row())
        r = fill_market_order(s, "BUY", 5.0)
        assert r.avg_px == pytest.approx(1.3001)
        assert r.filled_size == 5.0
        assert r.residual_size == 0.0
        assert r.levels_consumed == 1

    def test_buy_walks_multiple_levels(self):
        # 10 on L0, 8 on L1 → 15 consumes both
        s = snapshot_from_mbp_row(_mbp_row())
        r = fill_market_order(s, "BUY", 15.0)
        # VWAP = (10 * 1.3001 + 5 * 1.3002) / 15
        expected = (10 * 1.3001 + 5 * 1.3002) / 15
        assert r.avg_px == pytest.approx(expected, abs=1e-6)
        assert r.levels_consumed == 2
        assert r.filled_size == 15.0
        assert r.worst_px == pytest.approx(1.3002)

    def test_sell_walks_down_bids(self):
        s = snapshot_from_mbp_row(_mbp_row())
        r = fill_market_order(s, "SELL", 15.0)
        # bid_top=1.2999 descending → L0=1.2999 (size 10), L1=1.2998 (size 8)
        expected = (10 * 1.2999 + 5 * 1.2998) / 15
        assert r.avg_px == pytest.approx(expected, abs=1e-6)
        assert r.worst_px == pytest.approx(1.2998)

    def test_residual_when_book_exhausted(self):
        s = snapshot_from_mbp_row(_mbp_row())
        # Total ask depth = 35; ask for 50 → 15 residual
        r = fill_market_order(s, "BUY", 50.0)
        assert r.filled_size == pytest.approx(35.0)
        assert r.residual_size == pytest.approx(15.0)
        assert r.levels_consumed == MBP_DEPTH


# ════════════════════════════════════════════════════════════════════
# AdaptiveSlippage
# ════════════════════════════════════════════════════════════════════
class TestAdaptiveSlippage:
    def test_small_order_returns_base_ticks(self):
        s = snapshot_from_mbp_row(_mbp_row())
        slip = AdaptiveSlippage(tick_size=0.0001, base_ticks=0.5, mid_ticks=1.5)
        # L1 ask depth = 10, size = 0.5 → ratio = 0.05 < depth_floor=0.10
        assert slip.estimate(s, "BUY", 0.5) == pytest.approx(0.5)

    def test_full_l1_consumption_returns_mid(self):
        s = snapshot_from_mbp_row(_mbp_row())
        slip = AdaptiveSlippage(tick_size=0.0001, base_ticks=0.5, mid_ticks=1.5)
        # L1 ask depth = 10, size = 10 → ratio = 1.0
        assert slip.estimate(s, "BUY", 10.0) == pytest.approx(1.5)

    def test_walking_levels_adds_extra(self):
        s = snapshot_from_mbp_row(_mbp_row())
        slip = AdaptiveSlippage(
            tick_size=0.0001, base_ticks=0.5, mid_ticks=1.5,
            extra_per_level=1.0,
        )
        # size = 15 eats L0 (10) + 5 of L1 → 1 extra level
        assert slip.estimate(s, "BUY", 15.0) == pytest.approx(2.5)

    def test_zero_l1_depth_assumes_full_walk(self):
        # All depth on L1+ (no L0 liquidity)
        s = snapshot_from_mbp_row(_mbp_row(ask_sizes=(0,) * 10))
        slip = AdaptiveSlippage(tick_size=0.0001, base_ticks=0.5, mid_ticks=1.5)
        # No L1 → mid_ticks + extra_per_level * MBP_DEPTH
        assert slip.estimate(s, "BUY", 5.0) == pytest.approx(1.5 + 1.0 * MBP_DEPTH)

    def test_dollars_scales_with_tick_size(self):
        s = snapshot_from_mbp_row(_mbp_row())
        slip = AdaptiveSlippage(tick_size=0.0001)
        ticks = slip.estimate(s, "BUY", 10.0)
        assert slip.estimate_dollars(s, "BUY", 10.0) == pytest.approx(
            ticks * 0.0001
        )


# ════════════════════════════════════════════════════════════════════
# LatencyModel
# ════════════════════════════════════════════════════════════════════
class TestLatencyModel:
    def test_deterministic_with_seed(self):
        m1 = LatencyModel(mean_us=5000, jitter_us=2000, seed=42)
        m2 = LatencyModel(mean_us=5000, jitter_us=2000, seed=42)
        draws_1 = [m1.draw() for _ in range(10)]
        draws_2 = [m2.draw() for _ in range(10)]
        assert draws_1 == draws_2

    def test_always_positive(self):
        m = LatencyModel(mean_us=5000, jitter_us=10_000, seed=0)
        for _ in range(100):
            d = m.draw()
            assert d.total_seconds() > 0

    def test_mean_in_neighbourhood(self):
        m = LatencyModel(mean_us=5000, jitter_us=2000, seed=0)
        draws = [m.draw().total_seconds() * 1e6 for _ in range(1000)]
        # Lognormal-ish with |N(0,1)|, mean is ~mean_us + jitter*sqrt(2/pi)
        mean_us = float(np.mean(draws))
        assert 5000 < mean_us < 8500


# ════════════════════════════════════════════════════════════════════
# ReplayBacktest
# ════════════════════════════════════════════════════════════════════
class TestReplayBacktest:
    def _make_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 100 ticks at 1-second intervals
        ts = pd.date_range("2024-01-01", periods=100, freq="1s")
        ticks = pd.DataFrame([
            {**_mbp_row(bid_top=1.2999 + i * 1e-5,
                       ask_top=1.3001 + i * 1e-5), "ts_event": ts[i]}
            for i in range(100)
        ])
        signals = pd.DataFrame({
            "ts_event": [ts[10], ts[30], ts[60]],
            "side": ["BUY", "SELL", "BUY"],
            "size": [5.0, 5.0, 30.0],   # third walks several levels
        })
        return signals, ticks

    def test_run_produces_one_event_per_signal(self):
        signals, ticks = self._make_data()
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(tick_size=0.0001),
            latency=LatencyModel(seed=0),
        )
        events = bt.run()
        assert len(events) == 3
        assert all(isinstance(e, ExecutionEvent) for e in events)
        assert events[0].side == "BUY"
        assert events[1].side == "SELL"

    def test_arrival_after_decision(self):
        signals, ticks = self._make_data()
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(tick_size=0.0001),
            latency=LatencyModel(mean_us=5000, jitter_us=2000, seed=0),
        )
        events = bt.run()
        for ev in events:
            assert ev.ts_arrival > ev.ts_decision
            assert ev.latency_us > 0

    def test_large_signal_shows_walk_penalty(self):
        signals, ticks = self._make_data()
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(
                tick_size=0.0001, base_ticks=0.5, mid_ticks=1.5,
                extra_per_level=1.0,
            ),
            latency=LatencyModel(seed=0),
        )
        events = bt.run()
        small_buy = events[0]      # size = 5
        big_buy = events[2]        # size = 30 → walks levels
        assert big_buy.model_slippage_ticks > small_buy.model_slippage_ticks
        assert big_buy.levels_consumed > small_buy.levels_consumed

    def test_signal_before_first_tick_skipped(self):
        signals, ticks = self._make_data()
        early = pd.DataFrame({
            "ts_event": [pd.Timestamp("2020-01-01")],
            "side": ["BUY"], "size": [1.0],
        })
        signals = pd.concat([early, signals], ignore_index=True)
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(tick_size=0.0001),
            latency=LatencyModel(seed=0),
        )
        events = bt.run()
        assert len(events) == 3   # the pre-tick signal is silently dropped

    def test_summarise_matches_event_count(self):
        signals, ticks = self._make_data()
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(tick_size=0.0001),
            latency=LatencyModel(seed=0),
        )
        bt.run()
        s = bt.summarise()
        assert s["n_signals"] == 3
        assert s["n_executed"] == 3
        assert s["mean_latency_us"] > 0

    def test_write_log_jsonl(self, tmp_path):
        signals, ticks = self._make_data()
        bt = ReplayBacktest(
            signals=signals, ticks=ticks,
            slippage=AdaptiveSlippage(tick_size=0.0001),
            latency=LatencyModel(seed=0),
        )
        bt.run()
        log = tmp_path / "execution_log.jsonl"
        bt.write_log(log)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "ts_decision" in obj
            assert "fill_avg_px" in obj
            assert "model_slippage_ticks" in obj

    def test_raises_on_missing_columns(self):
        with pytest.raises(ValueError, match="ts_event"):
            ReplayBacktest(
                signals=pd.DataFrame({"side": ["BUY"]}),
                ticks=pd.DataFrame(),
                slippage=AdaptiveSlippage(tick_size=0.0001),
                latency=LatencyModel(seed=0),
            )


# ════════════════════════════════════════════════════════════════════
# End-to-end via the CLI subprocess
# ════════════════════════════════════════════════════════════════════
class TestCLI:
    def test_cli_produces_outputs(self, tmp_path):
        ts = pd.date_range("2024-01-01", periods=100, freq="1s")
        ticks_df = pd.DataFrame([
            {**_mbp_row(), "ts_event": ts[i]} for i in range(100)
        ])
        mbp = tmp_path / "mbp.parquet"
        ticks_df.to_parquet(mbp)

        signals_df = pd.DataFrame({
            "ts_event": [ts[10], ts[20]],
            "side": ["BUY", "SELL"],
            "size": [3.0, 4.0],
        })
        sig = tmp_path / "signals.csv"
        signals_df.to_csv(sig, index=False)

        out = tmp_path / "replay_out"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "run_replay_backtest.py"),
                "--signals", str(sig),
                "--mbp", str(mbp),
                "--output", str(out),
                "--tick-size", "0.0001",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"failed:\n{result.stdout}\n{result.stderr}"
        )
        assert (out / "execution_log.jsonl").exists()
        assert (out / "replay_summary.json").exists()
        summary = json.loads((out / "replay_summary.json").read_text())
        assert summary["n_executed"] == 2
