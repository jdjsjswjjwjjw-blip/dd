"""
tools/run_replay_backtest.py
═══════════════════════════════════════════════════════════════════════════════
Stand-alone CLI for the event-by-event Replay Engine.

This is INTENTIONALLY separate from `self_supervised/backtest_day_trade.py`.
That script does bar-level fills with constant slippage. This one walks
the MBP-10 book level by level, samples per-signal latency, and writes
an execution log you can audit against the model's predicted slippage.

Usage
─────
    python tools/run_replay_backtest.py \\
        --signals  backtests/baseline/signals.csv \\
        --mbp      raw/mbp_10.parquet \\
        --output   replay_results/baseline \\
        --tick-size 0.0001 \\
        --latency-mean-us 5000 --latency-jitter-us 2000 \\
        --slip-base-ticks 0.5 --slip-mid-ticks 1.5

Inputs
──────
signals.csv     ts_event, side ("BUY"/"SELL"), size
mbp_10.parquet  ts_event, bid_px_00..09, bid_sz_00..09,
                ask_px_00..09, ask_sz_00..09

Outputs
───────
<output>/execution_log.jsonl   one fill per line
<output>/replay_summary.json   aggregate latency + slippage + fill quality
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.replay_engine import (
    AdaptiveSlippage,
    LatencyModel,
    ReplayBacktest,
)


def _load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "ts_event" not in df.columns:
        raise ValueError(f"signals at {path} missing `ts_event`")
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    return df


def _load_ticks(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "ts_event" not in df.columns:
        raise ValueError(f"ticks at {path} missing `ts_event`")
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2] if __doc__ else "")
    p.add_argument("--signals", required=True, type=Path)
    p.add_argument("--mbp", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tick-size", type=float, default=0.0001)
    p.add_argument("--latency-mean-us", type=float, default=5_000.0)
    p.add_argument("--latency-jitter-us", type=float, default=2_000.0)
    p.add_argument("--latency-seed", type=int, default=0)
    p.add_argument("--slip-base-ticks", type=float, default=0.5)
    p.add_argument("--slip-mid-ticks", type=float, default=1.5)
    p.add_argument("--slip-extra-per-level", type=float, default=1.0)
    args = p.parse_args()

    signals = _load_signals(args.signals)
    ticks = _load_ticks(args.mbp)

    slippage = AdaptiveSlippage(
        tick_size=args.tick_size,
        base_ticks=args.slip_base_ticks,
        mid_ticks=args.slip_mid_ticks,
        extra_per_level=args.slip_extra_per_level,
    )
    latency = LatencyModel(
        mean_us=args.latency_mean_us,
        jitter_us=args.latency_jitter_us,
        seed=args.latency_seed,
    )

    bt = ReplayBacktest(
        signals=signals, ticks=ticks, slippage=slippage, latency=latency,
    )
    bt.run()
    args.output.mkdir(parents=True, exist_ok=True)
    bt.write_log(args.output / "execution_log.jsonl")
    summary = bt.summarise()
    (args.output / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(f"signals          : {len(signals):,}")
    print(f"executed         : {summary.get('n_executed', 0):,}")
    print(f"filled           : {summary.get('n_filled', 0):,}")
    print(f"with residual    : {summary.get('n_with_residual', 0):,}")
    mean_slip = summary.get("mean_realised_slippage_ticks")
    if mean_slip is not None:
        print(f"realised slippage: {mean_slip:.3f} ticks (mean)")
    print(f"output           : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
