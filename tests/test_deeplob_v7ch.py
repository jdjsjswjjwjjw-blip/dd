"""tests/test_deeplob_v7ch.py — Phase 5 tests."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from modules.deeplob_v7ch import (
    EXTRA_CHANNELS,
    N_CHANNELS_V7,
    N_PRICE_LEVELS,
    N_TIME_STEPS,
    build_7ch_tensor,
    channel_names,
    estimate_tensor_size_mb,
)


def _make_lob_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cols = {}
    for i in range(10):
        cols[f"bid_sz_{i:02d}"] = rng.uniform(0, 100, n)
        cols[f"ask_sz_{i:02d}"] = rng.uniform(0, 100, n)
    cols["trade_size"] = rng.uniform(1, 50, n)
    cols["side"] = rng.choice(["B", "S"], n)
    # V19.2 extras
    cols["sim_iceberg_strength"] = rng.uniform(0, 1, n)
    cols["sim_wall_persist"] = rng.uniform(0, 1, n)
    cols["sim_informed_prob"] = rng.uniform(0, 1, n)
    cols["sim_depth_imbalance"] = rng.uniform(-1, 1, n)
    return pd.DataFrame(cols)


class TestTensorShape(unittest.TestCase):
    def test_output_shape_v7(self):
        df = _make_lob_df(n=100)
        t = build_7ch_tensor(df, time_steps=50, price_levels=20)
        self.assertEqual(t.shape, (51, 50, 20, 7), msg=f"got {t.shape}")

    def test_seven_channels(self):
        df = _make_lob_df(n=100)
        t = build_7ch_tensor(df)
        self.assertEqual(t.shape[-1], 7)

    def test_returns_empty_when_insufficient(self):
        df = _make_lob_df(n=30)
        t = build_7ch_tensor(df, time_steps=50)
        self.assertEqual(t.shape, (0, 50, 20, 7))

    def test_dtype_float32(self):
        df = _make_lob_df(n=80)
        t = build_7ch_tensor(df, time_steps=20)
        self.assertEqual(t.dtype, np.float32)


class TestChannelContents(unittest.TestCase):
    """تأكد إن الـ channels تحتوي البيانات الصحيحة."""

    def test_depth_channel_matches_input(self):
        df = _make_lob_df(n=80)
        t = build_7ch_tensor(df, time_steps=20)
        # depth = channel 0
        depth = t[:, :, :, 0]
        # في sample 0 timestep -1 (آخر شمعة في النافذة الأولى) = df row 19
        # bid_sz_00 -> position L-1 = 9 (المنتصف bid side)
        self.assertGreater(depth[:, :, 9].mean(), 0, msg="bid_sz_00 يجب أن يكون > 0")

    def test_iceberg_channel_broadcast(self):
        """channel 3 = iceberg، broadcasted على كل price levels."""
        df = _make_lob_df(n=80)
        t = build_7ch_tensor(df, time_steps=20)
        iceberg_ch = t[:, :, :, 3]
        # كل level في صف معين = نفس القيمة (broadcast)
        for sample_i in [0, 5, 10]:
            for ts in [0, 5, 19]:
                row = iceberg_ch[sample_i, ts, :]
                self.assertAlmostEqual(row.min(), row.max(), places=5,
                                       msg=f"sample {sample_i} ts {ts}: not broadcast")

    def test_missing_extra_channel_zeros(self):
        df = _make_lob_df(n=80).drop(columns=["sim_iceberg_strength"])
        t = build_7ch_tensor(df, time_steps=20, skip_missing=True)
        # channel 3 = iceberg = zeros
        self.assertEqual(t[:, :, :, 3].sum(), 0)

    def test_missing_raises_without_skip(self):
        df = _make_lob_df(n=80).drop(columns=["sim_iceberg_strength"])
        with self.assertRaises(KeyError):
            build_7ch_tensor(df, time_steps=20, skip_missing=False)


class TestMemoryEstimate(unittest.TestCase):
    def test_memory_estimate_reasonable(self):
        # 12K samples × 50 ts × 20 levels × 7 channels × 4 bytes ≈ 320 MB
        mb = estimate_tensor_size_mb(12_000)
        self.assertGreater(mb, 250.0)
        self.assertLess(mb, 400.0)


class TestChannelNames(unittest.TestCase):
    def test_seven_names(self):
        names = channel_names()
        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], "depth")
        self.assertIn("sim_iceberg_strength", names)
        self.assertIn("sim_wall_persist", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
