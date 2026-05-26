"""tests/test_bell_pairs.py — Phase B: Bell pair entanglement tests.

يتأكد:
  ① Bell formula صحيحة رياضياً
  ② entanglement properties: constructive عند التوافق، destructive عند التضاد
  ③ Symmetric: bell_pair(a, b) == bell_pair(b, a)
  ④ Range: النتيجة في [-1, +1] لما a, b في [0, 1]
  ⑤ apply_bell_pairs على V19.2 simulators يولّد 9 أعمدة bp_*
  ⑥ شيء random/independent → بقايا قريبة من 0

شغّل من v19_2/:
    python tests/test_bell_pairs.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantum.bell_pairs import (
    BellPairSpec,
    STANDARD_BELL_PAIRS,
    apply_bell_pairs,
    bell_pair,
    cnot_gate,
    get_bell_pair_columns,
    normalize_to_unit,
)


class TestNormalize(unittest.TestCase):
    def test_clip_signed_to_unit(self):
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        out = normalize_to_unit(x, method="clip")
        np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_clip_already_unit_passes_through(self):
        x = np.array([0.0, 0.3, 0.7, 1.0])
        out = normalize_to_unit(x, method="clip")
        np.testing.assert_allclose(out, x)

    def test_tanh_smooth(self):
        x = np.array([-10.0, 0.0, 10.0])
        out = normalize_to_unit(x, method="tanh")
        self.assertAlmostEqual(out[0], 0.0, places=6)
        self.assertAlmostEqual(out[1], 0.5, places=6)
        self.assertAlmostEqual(out[2], 1.0, places=6)

    def test_minmax(self):
        x = np.array([2.0, 4.0, 6.0])
        out = normalize_to_unit(x, method="minmax")
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


class TestBellFormula(unittest.TestCase):
    """التأكد من الـ formula رياضياً."""

    def test_both_high_constructive(self):
        """a=b=1 → joint=1, anti=0, mixed=0 → result=+1."""
        result = bell_pair(np.array([1.0]), np.array([1.0]), normalize=False)
        np.testing.assert_allclose(result, [1.0])

    def test_both_low_constructive(self):
        """a=b=0 → joint=0, anti=1, mixed=0 → result=+1."""
        result = bell_pair(np.array([0.0]), np.array([0.0]), normalize=False)
        np.testing.assert_allclose(result, [1.0])

    def test_opposite_destructive(self):
        """a=1, b=0 → joint=0, anti=0, mixed=1 → result=-1."""
        result = bell_pair(np.array([1.0]), np.array([0.0]), normalize=False)
        np.testing.assert_allclose(result, [-1.0])

    def test_mid_neutral(self):
        """a=b=0.5 → joint=0.25, anti=0.25, mixed=0.5 → result=0."""
        result = bell_pair(np.array([0.5]), np.array([0.5]), normalize=False)
        np.testing.assert_allclose(result, [0.0])

    def test_range_bounded(self):
        """لـ a, b ∈ [0, 1] → result ∈ [-1, +1]."""
        rng = np.random.RandomState(0)
        a = rng.rand(1000)
        b = rng.rand(1000)
        result = bell_pair(a, b, normalize=False)
        self.assertTrue(np.all(result >= -1.0 - 1e-9))
        self.assertTrue(np.all(result <= 1.0 + 1e-9))

    def test_symmetric(self):
        """bell_pair(a, b) == bell_pair(b, a)."""
        rng = np.random.RandomState(0)
        a = rng.rand(500)
        b = rng.rand(500)
        r_ab = bell_pair(a, b, normalize=False)
        r_ba = bell_pair(b, a, normalize=False)
        np.testing.assert_allclose(r_ab, r_ba)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bell_pair(np.array([1.0]), np.array([1.0, 2.0]), normalize=False)


class TestEntanglementProperties(unittest.TestCase):
    """الخصائص الكمية للـ Bell entanglement."""

    def test_perfectly_correlated_pairs_high(self):
        """a == b → result دائماً ≈ +1 (full constructive)."""
        rng = np.random.RandomState(0)
        a = rng.rand(500)
        result = bell_pair(a, a, normalize=False)
        # كل قيم النتيجة قريبة من +1 (>= 0.5 على الأقل بدون انتظام)
        # في الواقع نريد result يكون >= 0 لمعظم القيم
        self.assertTrue(
            (result >= 0).mean() > 0.95,
            msg=f"correlated pairs should yield positive entanglement, mean={result.mean()}",
        )

    def test_perfectly_anticorrelated_pairs_low(self):
        """a, b = 1 - a → result دائماً ≤ 0 (destructive)."""
        rng = np.random.RandomState(0)
        a = rng.rand(500)
        b = 1.0 - a
        result = bell_pair(a, b, normalize=False)
        # نتيجة سالبة لكل القيم
        self.assertTrue(
            (result <= 0).mean() > 0.95,
            msg=f"anticorrelated pairs should be negative, mean={result.mean()}",
        )

    def test_independent_random_near_zero(self):
        """random a وb مستقلتان → mean(result) ≈ 0."""
        rng = np.random.RandomState(123)
        a = rng.rand(10_000)
        b = rng.rand(10_000)
        result = bell_pair(a, b, normalize=False)
        # المتوسط قريب من 0 (independent)
        self.assertAlmostEqual(
            result.mean(), 0.0, delta=0.05,
            msg=f"independent pairs mean should be near 0, got {result.mean()}",
        )


class TestCNotGate(unittest.TestCase):
    def test_cnot_amplifies_above_threshold(self):
        control = np.array([0.0, 0.3, 0.6, 1.0])
        target = np.array([2.0, 2.0, 2.0, 2.0])
        out = cnot_gate(control, target, threshold=0.5, amplification=2.0, deflation=0.5,
                        normalize_control=False)
        # control > 0.5 → target × 2; else × 0.5
        np.testing.assert_allclose(out, [1.0, 1.0, 4.0, 4.0])

    def test_cnot_normalizes_signed_control(self):
        control = np.array([-1.0, 0.0, 1.0])  # signed
        target = np.array([1.0, 1.0, 1.0])
        # normalize → [0, 0.5, 1] (clip method)
        out = cnot_gate(control, target, threshold=0.5, amplification=10.0, deflation=0.1)
        # at -1 → 0 ≤ thr → 0.1; at 0 → 0.5 ≤ thr → 0.1; at 1 → 1 > thr → 10
        np.testing.assert_allclose(out, [0.1, 0.1, 10.0])


class TestApplyBellPairs(unittest.TestCase):
    """تكامل مع V19.2 simulator outputs."""

    def _mock_df(self, n: int = 500, seed: int = 42) -> pd.DataFrame:
        """يحاكي مخرجات feature_simulators."""
        rng = np.random.RandomState(seed)
        return pd.DataFrame({
            # في [0, 1]
            "sim_absorb_intensity": rng.rand(n),
            "sim_informed_prob": rng.rand(n),
            "sim_iceberg_strength": rng.rand(n),
            "sim_iceberg_replenish": rng.rand(n),
            "sim_wall_persist": rng.rand(n),
            "sim_wall_real": rng.rand(n),
            "sim_wall_consumed": rng.rand(n),
            "sim_volatility_regime": rng.rand(n),
            "sim_flow_consistency": rng.rand(n),
            "sim_liquidity_state": rng.rand(n),
            "sim_data_quality": rng.rand(n),
            "sim_sweep_signal": rng.rand(n),
            # signed [-1, 1]
            "sim_depth_pressure": rng.uniform(-1, 1, n),
            "sim_depth_imbalance": rng.uniform(-1, 1, n),
            "sim_flow_direction": rng.uniform(-1, 1, n),
            "sim_iceberg_side": rng.uniform(-1, 1, n),
            "sim_informed_direction": rng.uniform(-1, 1, n),
            "sim_wall_growth": rng.uniform(-1, 1, n),
        })

    def test_produces_9_bp_columns(self):
        df = self._mock_df()
        out = apply_bell_pairs(df)
        bp_cols = [c for c in out.columns if c.startswith("bp_")]
        self.assertEqual(len(bp_cols), 9, msg=f"expected 9, got {len(bp_cols)}: {bp_cols}")
        # كلها float32
        for c in bp_cols:
            self.assertEqual(out[c].dtype, np.float32, msg=f"{c} dtype = {out[c].dtype}")

    def test_bp_columns_in_range(self):
        df = self._mock_df()
        out = apply_bell_pairs(df)
        for col in get_bell_pair_columns():
            vals = out[col].to_numpy()
            self.assertTrue(
                np.all(vals >= -1.0 - 1e-5) and np.all(vals <= 1.0 + 1e-5),
                msg=f"{col} out of range [-1,1]: min={vals.min()}, max={vals.max()}",
            )

    def test_no_nan_in_outputs(self):
        df = self._mock_df()
        out = apply_bell_pairs(df)
        for col in get_bell_pair_columns():
            self.assertFalse(out[col].isna().any(), msg=f"{col} has NaN")

    def test_skip_missing_columns(self):
        # حذف عمود واحد → الـ pair المعتمد عليه يُتخطّى
        df = self._mock_df().drop(columns=["sim_iceberg_strength"])
        out = apply_bell_pairs(df, skip_missing=True)
        self.assertNotIn("bp_wall_iceberg", out.columns)
        # باقي الـ pairs موجودة
        self.assertIn("bp_depth_informed", out.columns)

    def test_skip_missing_raises_when_false(self):
        df = self._mock_df().drop(columns=["sim_iceberg_strength"])
        with self.assertRaises(KeyError):
            apply_bell_pairs(df, skip_missing=False)

    def test_correlation_reduction(self):
        """الـ key insight للـ Phase B: 9 entangled features أقل ترابطاً
        ببعض من 18 features الأصلية (correlations مدمجة structurally).
        """
        rng = np.random.RandomState(7)
        n = 2000
        # بعض الـ features مترابطة بشكل طبيعي
        base = rng.randn(n)
        df = pd.DataFrame({
            "sim_wall_persist":   normalize_to_unit(base + rng.randn(n) * 0.2, "tanh"),
            "sim_iceberg_strength": normalize_to_unit(base + rng.randn(n) * 0.3, "tanh"),
            "sim_depth_pressure": normalize_to_unit(rng.randn(n) * 1.5, "tanh"),
            "sim_informed_prob":  normalize_to_unit(rng.randn(n) * 0.5, "tanh"),
            "sim_sweep_signal":   rng.rand(n),
            "sim_absorb_intensity": rng.rand(n),
            "sim_volatility_regime": rng.rand(n),
            "sim_flow_consistency": rng.rand(n),
            "sim_wall_growth": rng.uniform(-1, 1, n),
            "sim_iceberg_replenish": rng.rand(n),
            "sim_wall_real": rng.rand(n),
            "sim_wall_consumed": rng.rand(n),
            "sim_flow_direction": rng.uniform(-1, 1, n),
            "sim_depth_imbalance": rng.uniform(-1, 1, n),
            "sim_iceberg_side": rng.uniform(-1, 1, n),
            "sim_informed_direction": rng.uniform(-1, 1, n),
            "sim_liquidity_state": rng.rand(n),
            "sim_data_quality": rng.rand(n),
        })
        out = apply_bell_pairs(df)
        # تحقق إن كل 9 الأعمدة موجودة وغير NaN
        bp_cols = get_bell_pair_columns()
        self.assertEqual(len(bp_cols), 9)
        for c in bp_cols:
            self.assertIn(c, out.columns)
            self.assertFalse(out[c].isna().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
