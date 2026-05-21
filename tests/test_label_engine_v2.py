"""Tests للـ label_engine_v2 (Triple Barrier + MFE/MAE).

شغّل بـ: python -m pytest tests/test_label_engine_v2.py -v
أو:      python tests/test_label_engine_v2.py
"""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from modules.label_engine_v2 import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    PATH_TIMEOUT_MFE_LONG,
    PATH_TIMEOUT_MFE_SHORT,
    PATH_TIMEOUT_NEUTRAL,
    PATH_TP_LONG,
    PATH_TP_SHORT,
    TripleBarrierConfig,
    compute_atr,
    label_distribution,
    label_triple_barrier_atr,
    label_triple_barrier_atr_vectorized,
)


class TestComputeATR(unittest.TestCase):
    def test_empty(self):
        atr = compute_atr(np.array([]), np.array([]), np.array([]), window=14)
        self.assertEqual(len(atr), 0)

    def test_constant_prices(self):
        # كل الأسعار ثابتة → ATR = 0
        closes = np.full(50, 100.0)
        atr = compute_atr(closes, closes, closes, window=14)
        self.assertEqual(len(atr), 50)
        np.testing.assert_allclose(atr, 0.0, atol=1e-10)

    def test_constant_range(self):
        # high-low ثابت = 1 → ATR يقترب من 1
        closes = np.full(100, 100.0)
        highs = closes + 0.5
        lows = closes - 0.5
        atr = compute_atr(highs, lows, closes, window=14)
        # بعد window يجب أن يقترب من 1.0
        self.assertAlmostEqual(atr[-1], 1.0, delta=0.05)

    def test_increasing_volatility(self):
        # تذبذب متزايد → ATR يزيد
        n = 100
        closes = 100.0 + np.cumsum(np.random.RandomState(0).randn(n) * 0.1)
        highs = closes + np.arange(n) * 0.01
        lows = closes - np.arange(n) * 0.01
        atr = compute_atr(highs, lows, closes, window=14)
        self.assertGreater(atr[-1], atr[20])


class TestTripleBarrierBasic(unittest.TestCase):
    def test_empty_input(self):
        result = label_triple_barrier_atr(np.array([]), np.array([]))
        for key in ("bias", "path", "mfe", "mae", "end_idx"):
            self.assertEqual(len(result[key]), 0)

    def test_length_mismatch_atr(self):
        with self.assertRaises(ValueError):
            label_triple_barrier_atr(np.zeros(10), np.zeros(5))

    def test_length_mismatch_horizons(self):
        with self.assertRaises(ValueError):
            label_triple_barrier_atr(np.zeros(10), np.ones(10), horizons=np.zeros(5))

    def test_constant_prices_all_neutral(self):
        # أسعار ثابتة → كل الـ labels NEUTRAL
        prices = np.full(50, 100.0)
        atr = np.full(50, 1.0)
        result = label_triple_barrier_atr(prices, atr)
        # last row دائماً NEUTRAL (لا توجد بيانات مستقبلية)
        np.testing.assert_array_equal(result["bias"], DIR_NEUTRAL)


class TestTripleBarrierHits(unittest.TestCase):
    """التأكد من أن الـ TP/SL barriers تُكتشف بشكل صحيح."""

    def test_clean_long_hit(self):
        # السعر يرتفع فوق upper barrier
        prices = np.array([100.0, 100.5, 101.0, 102.0, 103.0])
        atr = np.full(5, 1.0)  # → upper = 102 (tp=2.0×ATR), lower = 99
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=1.0, horizon_default=10)
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        # t=0: السعر يصل 102 عند j=3
        self.assertEqual(result["bias"][0], DIR_LONG)
        self.assertEqual(result["path"][0], PATH_TP_LONG)
        self.assertEqual(result["end_idx"][0], 3)

    def test_clean_short_hit(self):
        prices = np.array([100.0, 99.5, 99.0, 98.0, 97.0])
        atr = np.full(5, 1.0)  # → upper = 102, lower = 98
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0, horizon_default=10)
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        self.assertEqual(result["bias"][0], DIR_SHORT)
        self.assertEqual(result["path"][0], PATH_TP_SHORT)
        self.assertEqual(result["end_idx"][0], 3)

    def test_tp_beats_sl_when_simultaneous_in_time(self):
        # السعر يلامس upper barrier أولاً قبل lower
        prices = np.array([100.0, 102.5, 95.0])
        atr = np.full(3, 1.0)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0, horizon_default=10)
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        self.assertEqual(result["bias"][0], DIR_LONG)


class TestMFEMAEFallback(unittest.TestCase):
    """عند timeout، MFE/MAE يحدد الاتجاه."""

    def test_mfe_dominant_long(self):
        # السعر يرتفع +5 ثم يرجع 0 (mean-revert)
        # MFE=5, MAE=0 → LONG (mfe > ratio*mae)
        prices = np.array([100.0, 102.0, 105.0, 103.0, 100.0])
        atr = np.full(5, 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=10.0,  # بعيد جداً ما يتلامسش
            sl_atr_mult=10.0,
            mfe_mae_ratio=2.0,
            horizon_default=4,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        # t=0: future = [102,105,103,100] → MFE=5, MAE=0 → LONG
        self.assertEqual(result["bias"][0], DIR_LONG)
        self.assertEqual(result["path"][0], PATH_TIMEOUT_MFE_LONG)
        self.assertAlmostEqual(result["mfe"][0], 5.0, places=6)
        self.assertAlmostEqual(result["mae"][0], 0.0, places=6)

    def test_mae_dominant_short(self):
        # السعر ينخفض ثم يرجع
        prices = np.array([100.0, 97.0, 95.0, 98.0, 100.0])
        atr = np.full(5, 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=10.0,
            sl_atr_mult=10.0,
            mfe_mae_ratio=2.0,
            horizon_default=4,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        # t=0: future=[97,95,98,100] → MFE=0, MAE=5 → SHORT
        self.assertEqual(result["bias"][0], DIR_SHORT)
        self.assertEqual(result["path"][0], PATH_TIMEOUT_MFE_SHORT)

    def test_mfe_equal_mae_neutral(self):
        # MFE = MAE → NEUTRAL (لا يحقق ratio=2)
        prices = np.array([100.0, 102.0, 98.0, 102.0, 98.0])
        atr = np.full(5, 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=10.0,
            sl_atr_mult=10.0,
            mfe_mae_ratio=2.0,
            horizon_default=4,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        self.assertEqual(result["bias"][0], DIR_NEUTRAL)
        self.assertEqual(result["path"][0], PATH_TIMEOUT_NEUTRAL)

    def test_solves_legacy_bug_mean_revert(self):
        """🎯 الـ regression test الأهم: السيناريو المعطوب في dynamic_labels.

        قبل: future[-1] فقط → MFE=+30p ثم return=-25p ينتج future[-1]=+5p → NEUTRAL
        بعد: MFE/MAE → يكتشف الحركة الربحية في الوسط → LONG
        """
        # السعر: 100 → 130 (peak) → 105 (close)
        # المعطوب: future[-1] - entry = 5 → NEUTRAL لو threshold=5
        # المُصلَح: MFE=30, MAE=0 → LONG واضح
        prices = np.linspace(100, 130, 16).tolist() + np.linspace(130, 105, 15).tolist()
        prices = np.array(prices, dtype=np.float64)
        atr = np.full(len(prices), 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=100.0,    # بعيد جداً (محاكاة "نقطة النهاية فقط")
            sl_atr_mult=100.0,
            mfe_mae_ratio=2.0,
            horizon_default=30,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        self.assertEqual(
            result["bias"][0], DIR_LONG,
            msg="Triple Barrier فشل في اكتشاف الحركة الربحية اللي ضاعت بـ future[-1]"
        )


class TestATRAdaptive(unittest.TestCase):
    """العتبة تتكيّف مع التذبذب."""

    def test_low_atr_more_signals(self):
        # ATR منخفض → barriers قريبة → labels أكثر
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(200) * 0.5)
        atr_low = np.full(200, 0.1)
        atr_high = np.full(200, 5.0)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=1.0, horizon_default=20)

        result_low = label_triple_barrier_atr(prices, atr_low, config=cfg)
        result_high = label_triple_barrier_atr(prices, atr_high, config=cfg)

        dist_low = label_distribution(result_low["bias"])
        dist_high = label_distribution(result_high["bias"])

        # ATR منخفض → نسبة NEUTRAL أقل
        self.assertLess(
            dist_low["neutral"], dist_high["neutral"],
            msg=f"low={dist_low}, high={dist_high}",
        )


class TestClassBalance(unittest.TestCase):
    """التحقق من أن التوزيع صحي على داتا عشوائية - الـ regression test الأهم."""

    def test_random_walk_balanced_labels(self):
        """على random walk مع ATR-adaptive thresholds، توزيع يجب أن يكون أفضل من 98%/1%/1%.

        التقرير: قبل = 98% NEUTRAL، بعد = 40/30/30.
        هنا نتأكد على الأقل أن NEUTRAL < 80% على random walk صحي.
        """
        np.random.seed(0)
        n = 5000
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
        highs = prices + np.abs(np.random.randn(n) * 0.1)
        lows = prices - np.abs(np.random.randn(n) * 0.1)
        atr = compute_atr(highs, lows, prices, window=14)

        # symmetric config عشان random walk يعطي LONG ≈ SHORT
        cfg = TripleBarrierConfig(
            tp_atr_mult=2.0,
            sl_atr_mult=2.0,
            mfe_mae_ratio=2.0,
            min_move_atr_mult=0.5,
            horizon_default=20,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        dist = label_distribution(result["bias"])

        # التوزيع المتوقع: NEUTRAL مرتفع لكن أقل من 80% (مش 98%)
        self.assertLess(
            dist["neutral"], 0.80,
            msg=f"NEUTRAL أعلى من 80% — fix غير فعّال. dist={dist}",
        )
        # كل من LONG و SHORT يجب أن يحصلوا على >= 5%
        self.assertGreater(dist["long"], 0.05, msg=f"LONG قليل: {dist}")
        self.assertGreater(dist["short"], 0.05, msg=f"SHORT قليل: {dist}")
        # السيمتري نسبي - random walk symmetric config → |LONG - SHORT| < 0.20
        self.assertLess(
            abs(dist["long"] - dist["short"]), 0.20,
            msg=f"LONG/SHORT غير متماثلتين على random walk: {dist}",
        )

    def test_asymmetric_extreme_skews_short(self):
        """tp_atr_mult >> sl_atr_mult: upper barrier يكاد لا يُلامَس → SHORT يهيمن.

        upper = entry + 10*ATR (بعيد)، lower = entry - 0.5*ATR (قريب).
        السعر يلامس lower أسرع بكثير.
        """
        np.random.seed(1)
        n = 3000
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
        atr = np.full(n, 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=10.0, sl_atr_mult=0.5,  # extreme asymmetric
            min_move_atr_mult=0.3, horizon_default=20,
        )
        result = label_triple_barrier_atr(prices, atr, config=cfg)
        dist = label_distribution(result["bias"])
        self.assertGreater(
            dist["short"], dist["long"],
            msg=f"upper بعيد، lower قريب → SHORT يجب أن يهيمن. dist={dist}",
        )


class TestAdaptiveHorizons(unittest.TestCase):
    """آفاق per-row بدل ثابت."""

    def test_per_row_horizons(self):
        prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        atr = np.full(6, 0.3)
        # row 0: horizon=1 (يشوف فقط t+1) → 100→100.5، حركة 0.5 = barrier @ 0.6 → ما يلامسش
        # row 0: horizon=5 (يشوف t+1..t+5) → 100→102.5 → يلامس
        horizons_short = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        horizons_long = np.array([5, 5, 5, 5, 5, 5], dtype=np.int32)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0, min_move_atr_mult=2.0)

        r_short = label_triple_barrier_atr(prices, atr, horizons=horizons_short, config=cfg)
        r_long = label_triple_barrier_atr(prices, atr, horizons=horizons_long, config=cfg)

        # row 0 بأفق قصير → ما يلامسش barrier
        self.assertEqual(r_short["bias"][0], DIR_NEUTRAL)
        # row 0 بأفق طويل → يلامس upper
        self.assertEqual(r_long["bias"][0], DIR_LONG)


class TestVectorizedParity(unittest.TestCase):
    """Phase A: النسخة vectorized تعطي نفس نتائج النسخة loop-based."""

    def _compare(self, prices, atr, horizons=None, cfg=None):
        cfg = cfg or TripleBarrierConfig()
        r_loop = label_triple_barrier_atr(prices, atr, horizons=horizons, config=cfg)
        r_vec = label_triple_barrier_atr_vectorized(prices, atr, horizons=horizons, config=cfg)
        np.testing.assert_array_equal(r_loop["bias"], r_vec["bias"], err_msg="bias mismatch")
        np.testing.assert_array_equal(r_loop["path"], r_vec["path"], err_msg="path mismatch")
        np.testing.assert_allclose(r_loop["mfe"], r_vec["mfe"], rtol=1e-9, err_msg="mfe mismatch")
        np.testing.assert_allclose(r_loop["mae"], r_vec["mae"], rtol=1e-9, err_msg="mae mismatch")
        np.testing.assert_array_equal(r_loop["end_idx"], r_vec["end_idx"], err_msg="end_idx mismatch")

    def test_parity_empty(self):
        self._compare(np.array([]), np.array([]))

    def test_parity_clean_long(self):
        prices = np.array([100.0, 100.5, 101.0, 102.0, 103.0])
        atr = np.full(5, 1.0)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0, horizon_default=10)
        self._compare(prices, atr, cfg=cfg)

    def test_parity_random_walk(self):
        np.random.seed(123)
        n = 1000
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
        highs = prices + np.abs(np.random.randn(n) * 0.1)
        lows = prices - np.abs(np.random.randn(n) * 0.1)
        atr = compute_atr(highs, lows, prices, window=14)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=1.5, min_move_atr_mult=0.5)
        self._compare(prices, atr, cfg=cfg)

    def test_parity_mean_revert_bug_scenario(self):
        # السيناريو المعطوب: MFE في الوسط، النهاية تعود
        prices = np.linspace(100, 130, 16).tolist() + np.linspace(130, 105, 15).tolist()
        prices = np.array(prices, dtype=np.float64)
        atr = np.full(len(prices), 1.0)
        cfg = TripleBarrierConfig(
            tp_atr_mult=100.0, sl_atr_mult=100.0,
            mfe_mae_ratio=2.0, horizon_default=30,
        )
        self._compare(prices, atr, cfg=cfg)

    def test_parity_per_row_horizons(self):
        np.random.seed(7)
        n = 500
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.4)
        atr = np.full(n, 0.6)
        horizons = np.random.randint(5, 25, size=n).astype(np.int32)
        cfg = TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=2.0)
        self._compare(prices, atr, horizons=horizons, cfg=cfg)

    def test_parity_constant_prices(self):
        prices = np.full(50, 100.0)
        atr = np.full(50, 1.0)
        self._compare(prices, atr)


class TestVectorizedSpeed(unittest.TestCase):
    """Phase A: vectorized أسرع بشكل ملموس."""

    def test_vectorized_faster_on_large_input(self):
        import time
        np.random.seed(0)
        n = 12_000  # نطاق الـ production
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
        atr = np.full(n, 0.5)
        cfg = TripleBarrierConfig(horizon_default=20)

        t0 = time.perf_counter()
        r_loop = label_triple_barrier_atr(prices, atr, config=cfg)
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_vec = label_triple_barrier_atr_vectorized(prices, atr, config=cfg)
        t_vec = time.perf_counter() - t0

        np.testing.assert_array_equal(r_loop["bias"], r_vec["bias"])
        speedup = t_loop / max(t_vec, 1e-9)
        # threshold محافظ — الـ benchmark الفعلي مع pandas overhead
        # يعطي 5-10× سرعة في الإنتاج. هنا فقط نتأكد من تحسّن واضح.
        self.assertGreater(
            speedup, 2.0,
            msg=f"speedup={speedup:.1f}× (loop={t_loop*1000:.1f}ms, vec={t_vec*1000:.1f}ms)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
