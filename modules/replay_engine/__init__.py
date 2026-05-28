"""
modules.replay_engine
═══════════════════════════════════════════════════════════════════════════════
Event-by-event LOB replay simulator. Separate from `self_supervised/backtest_day_trade.py`
(which is a bar-level rule backtest with constant slippage). This module
models market impact, adaptive slippage, and execution latency against the
MBP-10 tick stream — the "Replay Engine" described in the architecture doc.

The pipeline:

    signals.csv  ─┐
                  ├─► ReplayBacktest ─► execution_log.jsonl + summary.json
    mbp_10.parquet ┘

CLI entry point: `tools/run_replay_backtest.py`.
"""
from modules.replay_engine.book import (
    AdaptiveSlippage,
    FillResult,
    LOBSnapshot,
    LOBState,
    MBP_DEPTH,
    fill_market_order,
    snapshot_from_mbp_row,
)
from modules.replay_engine.engine import (
    ExecutionEvent,
    LatencyModel,
    ReplayBacktest,
)

__all__ = [
    "AdaptiveSlippage",
    "ExecutionEvent",
    "FillResult",
    "LatencyModel",
    "LOBSnapshot",
    "LOBState",
    "MBP_DEPTH",
    "ReplayBacktest",
    "fill_market_order",
    "snapshot_from_mbp_row",
]
