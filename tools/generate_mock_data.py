"""
tools/generate_mock_data.py
═══════════════════════════════════════════════════════════════════════════════
Generates a synthetic MBP-10 parquet + signals CSV for testing the Replay
Engine end-to-end without needing real Databento data.

The mock book has:
  - 10 bid + 10 ask levels per tick
  - Stepped price ladders (level i = best ± i × 0.01)
  - Random level-0 depths in [1, 50]; deeper levels in [1, 100]
  - Signal sizes deliberately span small / medium / "blow through L1"
    so AdaptiveSlippage exercises all three regions of its ramp.
"""
import os

import numpy as np
import pandas as pd


def generate_mock_data(output_dir="tests/data"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. توليد بيانات الـ MBP-10 (10 مستويات عمق)
    n_ticks = 1000
    base_price = 100.0

    data = {
        'ts_event': pd.date_range("2026-05-28", periods=n_ticks, freq="100ms"),
        'bid_px_00': base_price - np.random.uniform(0.01, 0.05, n_ticks),
        'ask_px_00': base_price + np.random.uniform(0.01, 0.05, n_ticks),
        'bid_sz_00': np.random.randint(1, 50, n_ticks),
        'ask_sz_00': np.random.randint(1, 50, n_ticks),
        'order_id': np.arange(n_ticks),
    }

    # تعبئة باقي المستويات (01-09) بأسعار متدرجة
    for i in range(1, 10):
        data[f'bid_px_{i:02d}'] = data['bid_px_00'] - (i * 0.01)
        data[f'ask_px_{i:02d}'] = data['ask_px_00'] + (i * 0.01)
        data[f'bid_sz_{i:02d}'] = np.random.randint(1, 100, n_ticks)
        data[f'ask_sz_{i:02d}'] = np.random.randint(1, 100, n_ticks)

    df_mbp = pd.DataFrame(data)
    df_mbp.to_parquet(os.path.join(output_dir, "mock_mbp_10.parquet"))

    # 2. توليد إشارات (Signals) بسيطة للاختبار
    signals = {
        'ts_event': df_mbp['ts_event'][10:500:50],
        'side': ['buy'] * 5 + ['sell'] * 5,
        'size': [10, 60, 20, 150, 5, 10, 80, 30, 200, 10],
    }
    pd.DataFrame(signals).to_csv(
        os.path.join(output_dir, "mock_signals.csv"), index=False
    )

    print(f"Mock data generated in {output_dir}")


if __name__ == "__main__":
    generate_mock_data()
