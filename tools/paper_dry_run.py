"""
tools/paper_dry_run.py — Sprint 9
─────────────────────────────────
محاكاة paper_v19 harness على synthetic data + IntegrationBridge.

يحلّ Phase 6 من timeline التقرير (Paper trading شهر) infrastructure-level.
الـ live feed يأتي لاحقاً، لكن الـ wiring يثبت إن:
  - الـ engine + bridge يدمجان decisions
  - الـ logs تُكتب في JSONL
  - الـ flow end-to-end شغّال

استخدام:
    python tools/paper_dry_run.py [--n-bars 500] [--alpha-set path/to/alphas.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def synthetic_bars(n_bars: int = 500, seed: int = 0):
    """Synthetic OHLCV + MBP بـ pattern قابل للتعلم."""
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.randn(n_bars) * 0.05)

    rows = []
    for i in range(n_bars):
        row = {
            "ts_event": ts[i],
            "price": float(close[i]),
            "high": float(close[i] + abs(rng.randn() * 0.05)),
            "low": float(close[i] - abs(rng.randn() * 0.05)),
            "volume": float(rng.uniform(100, 1000)),
            "atr_14": 0.01,
            "regime": int(rng.choice([0, 1, 2])),
            "label_horizon_steps": 12,
            "micro_atr": 0.005,
            "bid_px_00": float(close[i] - 0.01),
            "ask_px_00": float(close[i] + 0.01),
        }
        for k in range(10):
            row[f"bid_sz_{k:02d}"] = float(rng.uniform(0, 100))
            row[f"ask_sz_{k:02d}"] = float(rng.uniform(0, 100))
        # Bridge requires depth_pressure-like signals
        row["depth_pressure"] = float(rng.uniform(-1, 1))
        row["informed_prob"] = float(rng.uniform(0, 1))
        row["wall_strength"] = float(rng.uniform(0, 1))
        row["iceberg_score"] = float(rng.uniform(0, 1))
        row["trade_size"] = float(rng.uniform(1, 50))
        rows.append(row)
    return rows


def run_dry_paper(rows, alpha_set_path: str | None, output_dir: str) -> dict:
    """Mini-paper-runner: bridge decisions على synthetic rows."""
    import numpy as np
    from modules.integration_bridge import BridgeConfig, IntegrationBridge
    from modules.statistical_validation_layer import AlphaSet

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bridge_log = os.path.join(output_dir, "paper_bridge.jsonl")

    # Alpha set
    if alpha_set_path and os.path.exists(alpha_set_path):
        with open(alpha_set_path) as f:
            data = json.load(f)
        alpha_set = AlphaSet(candidates=data.get("candidates", []))
    else:
        # default: 1 alpha permissive
        alpha_set = AlphaSet(candidates=[{
            "zone": "T_REG_LONDON", "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }])

    bridge = IntegrationBridge(alpha_set, config=BridgeConfig())

    rng = np.random.RandomState(0)
    from collections import Counter
    action_counts: Counter = Counter()
    with open(bridge_log, "w", encoding="utf-8") as logf:
        for i, row in enumerate(rows):
            # Synthetic DL proba
            p_long = float(rng.uniform(0.2, 0.7))
            p_short = float(rng.uniform(0.0, 1.0 - p_long))
            p_neutral = max(0.0, 1.0 - p_long - p_short)
            dl_proba = np.array([p_long, p_short, p_neutral])
            try:
                decision = bridge.evaluate_row(row, dl_proba)
                action_counts[decision.action.name] += 1
                logf.write(json.dumps({
                    "i": i,
                    "ts": str(row["ts_event"]),
                    "action": decision.action.name,
                    "confidence": float(decision.confidence),
                    "reason": decision.reason,
                }, ensure_ascii=False) + "\n")
            except Exception as exc:
                logf.write(json.dumps({"i": i, "error": str(exc)}) + "\n")
                action_counts["ERROR"] += 1

    return {
        "n_bars": len(rows),
        "actions": dict(action_counts),
        "log_path": bridge_log,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Paper dry-run on synthetic data")
    p.add_argument("--n-bars", type=int, default=500)
    p.add_argument("--alpha-set", default=None)
    p.add_argument("--output-dir", default="artifacts/paper_dry_run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("═" * 70)
    print("Paper Dry-Run (Sprint 9 — infrastructure smoke)")
    print("═" * 70)

    print(f"\n📊 [1] Generating {args.n_bars:,} synthetic bars...")
    rows = synthetic_bars(args.n_bars, args.seed)

    print(f"\n🌉 [2] Running IntegrationBridge dry-paper...")
    result = run_dry_paper(rows, args.alpha_set, args.output_dir)

    print(f"\n📈 [3] Action distribution:")
    total = max(result["n_bars"], 1)
    for action, count in sorted(result["actions"].items()):
        pct = 100.0 * count / total
        print(f"   {action:12s}: {count:>5,} ({pct:.1f}%)")
    print(f"\n💾 Log: {result['log_path']}")

    print("\n" + "═" * 70)
    print("✅ Paper dry-run اكتمل.")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
