"""
tools/demo_integration_bridge.py — Phase 6 integration demo.

يوضّح كيفية ربط الـ IntegrationBridge مع:
  - V19.2 discovery (edge_scanner_v2) → AlphaSet
  - DL predictions (synthetic أو V19PredictionEngine)
  - Trading decisions per row

التشغيل (synthetic):
    python tools/demo_integration_bridge.py --synthetic

التشغيل (مع داتا حقيقية):
    python tools/demo_integration_bridge.py \\
        --enriched path/to/enriched.parquet \\
        --dl-proba path/to/dl_predictions.npy

الـ V19PredictionEngine integration:
    from predict_v19 import V19PredictionEngine
    engine = V19PredictionEngine(...)
    dl_proba = engine.predict(df)  # shape (n, 3)
    decisions = bridge.evaluate_batch(df, dl_proba)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd


def _make_synthetic_pipeline(n: int = 500) -> tuple[pd.DataFrame, np.ndarray]:
    """يولّد DataFrame + DL proba للـ demo."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.05)
    df = pd.DataFrame({
        "ts_event": ts,
        "open": close + rng.randn(n) * 0.02,
        "high": close + np.abs(rng.randn(n) * 0.1),
        "low": close - np.abs(rng.randn(n) * 0.1),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })
    # MBP-10
    for i in range(10):
        df[f"bid_sz_{i:02d}"] = rng.uniform(0, 100, n)
        df[f"ask_sz_{i:02d}"] = rng.uniform(0, 100, n)
    df["trade_size"] = rng.uniform(1, 50, n)
    df["side"] = rng.choice(["B", "S"], n)

    # DL proba (long, short, neutral)
    proba = rng.dirichlet([1.5, 1.5, 2.0], size=n)
    return df, proba


def main():
    ap = argparse.ArgumentParser(description="Integration Bridge demo")
    ap.add_argument("--synthetic", action="store_true",
                    help="استخدم synthetic data (للـ demo)")
    ap.add_argument("--enriched", default=None,
                    help="parquet ناتج من prepare_day_trading_enriched.py")
    ap.add_argument("--dl-proba", default=None,
                    help=".npy array بـ shape (n, 3) للـ DL predictions")
    ap.add_argument("--n-rows", type=int, default=500,
                    help="عدد الصفوف للـ synthetic mode")
    args = ap.parse_args()

    print("═" * 78)
    print("Phase 6: Integration Bridge Demo")
    print("═" * 78)

    # ── Step 1: تجهيز البيانات ────────────────────────────────────────────
    if args.synthetic:
        print(f"\n📊 [1] توليد {args.n_rows:,} synthetic rows...")
        df, dl_proba = _make_synthetic_pipeline(args.n_rows)
    else:
        if not args.enriched or not args.dl_proba:
            print("❌ يحتاج --enriched + --dl-proba، أو --synthetic")
            return 1
        print(f"\n📊 [1] قراءة {args.enriched}...")
        df = pd.read_parquet(args.enriched)
        dl_proba = np.load(args.dl_proba)

    # ── Step 2: Enrichment (لو الـ df مش enriched) ────────────────────────
    if "sim_depth_pressure" not in df.columns:
        print("\n🔮 [2] Phase 3: Feature Enrichment...")
        from modules.feature_enrichment import enrich_features, EnrichmentConfig
        cfg = EnrichmentConfig(
            add_simulators=True, add_wall_depth=False, add_iceberg=False,
            add_session_mapping=True, add_bell_pairs=True,
        )
        df = enrich_features(df, config=cfg)
    else:
        print("\n✓  [2] Df already enriched (sim_* columns موجودة)")

    # ── Step 3: Discovery (Phase 4) ───────────────────────────────────────
    print("\n🔍 [3] Phase 4: Discovery (statistical alpha extraction)...")
    from modules.statistical_validation_layer import discover_alphas
    try:
        alpha_set = discover_alphas(
            df, run_permutation=False, verbose=False,
        )
        print(f"   اكتُشف: {alpha_set.summary()}")
    except Exception as e:
        print(f"   ⚠️  discovery تخطّي (بيانات غير كافية): {e}")
        # fallback: manual alpha
        from modules.statistical_validation_layer import AlphaSet
        alpha_set = AlphaSet(candidates=[{
            "zone": df["zone_full"].iloc[0],
            "level": "PDH", "event": "touch",
            "combo": "LONG_dp_pos", "direction": 1, "horizon": 6,
            "filters": ["depth_pressure_pos"],
        }])
        print(f"   manual alpha for demo: {alpha_set.candidates[0]['combo']}")

    # ── Step 4: Bridge — قرارات ──────────────────────────────────────────
    print("\n🌉 [4] Phase 6: Integration Bridge — trading decisions...")
    from modules.integration_bridge import BridgeConfig, IntegrationBridge, Action
    bridge = IntegrationBridge(
        alpha_set,
        config=BridgeConfig(
            dl_confirm_threshold=0.55,
            wall_consumed_exit=0.70,
        ),
    )

    if len(dl_proba) != len(df):
        print(f"   ⚠️  truncating dl_proba من {len(dl_proba)} إلى {len(df)}")
        dl_proba = dl_proba[:len(df)]

    decisions = bridge.evaluate_batch(df, dl_proba)

    # ── Step 5: تحليل القرارات ────────────────────────────────────────────
    print("\n📈 [5] التحليل:")
    from collections import Counter
    action_counts = Counter(d.action.name for d in decisions)
    for action, count in sorted(action_counts.items()):
        pct = 100.0 * count / len(decisions)
        print(f"   {action:12s}: {count:>5,} ({pct:.1f}%)")

    # confidence stats
    confidences = [d.confidence for d in decisions if d.action != Action.HOLD]
    if confidences:
        print(f"\n   confidence mean: {np.mean(confidences):.2f}")
        print(f"   confidence max:  {np.max(confidences):.2f}")

    # عرض أول 3 trade decisions
    trade_decisions = [d for d in decisions if d.action != Action.HOLD][:3]
    if trade_decisions:
        print("\n   أول 3 trades:")
        for i, d in enumerate(trade_decisions, 1):
            print(f"     [{i}] {d.action.name:12s} reason={d.reason!r} "
                  f"conf={d.confidence:.2f}")

    print("\n" + "═" * 78)
    print("✅ Demo اكتمل.")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
