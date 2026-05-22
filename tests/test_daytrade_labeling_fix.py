"""
tests/test_daytrade_labeling_fix.py — Sprint 19

يحرس إصلاح مشكلة الـ 99% NEUTRAL في prepare_day_trading.label_by_outcome
و build_day_trading_labels.

المشكلة الأصلية:
    - timeout → bias_label = NEUTRAL تلقائياً
    - long_sl / short_sl → NEUTRAL تلقائياً
    → على 5min/6B الـ TP barrier نادراً يُضرب → ≈99% NEUTRAL

الإصلاح:
    كل حالة non-TP → قرار MFE/MAE (منطق modules.label_engine_v2):
        mfe > ratio×mae & mfe >= min_move → LONG
        mae > ratio×mfe & mae >= min_move → SHORT
        else → NEUTRAL

يتحقق:
    1. timeout_mfe_mae=True (default) يقلّل NEUTRAL جذرياً
    2. timeout_mfe_mae=False يحافظ على السلوك القديم (backward compat)
    3. الـ MFE/MAE logic صحيح (ratio + min_move)
    4. allow_long/allow_short (event-direction veto) محترَم
    5. build_day_trading_labels مُصلَح كذلك
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_pdt():
    """Load prepare_day_trading with heavy deps mocked."""
    for m in ['modules.context_features', 'modules.tick_intrabar_slices',
              'modules.intrabar_mbp_microstructure', 'modules.manifest_v19']:
        sys.modules.setdefault(m, type(sys)('mock'))
    sys.modules['modules.context_features'].compute_daily_weekly_levels = lambda *a, **k: None
    sys.modules['modules.tick_intrabar_slices'].enrich_bars_with_intrabar = lambda *a, **k: None
    sys.modules['modules.intrabar_mbp_microstructure'].enrich_bars_with_intrabar_mbp = lambda *a, **k: None
    sys.modules['modules.manifest_v19'].write_manifest = lambda *a, **k: None
    spec = importlib.util.spec_from_file_location(
        'pdt_test', os.path.join(_ROOT, 'prepare_day_trading.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PDT = _load_pdt()


def _make_bars(n=400, seed=7, drift_choices=(-0.01, 0.0, 0.01),
               event_direction=0, atr=0.05):
    """Synthetic bars محاكية لـ 6B 5min — trends بطيئة، barrier نادراً يُضرب."""
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice(drift_choices, n // 40 + 1), 40)[:n]
    close = 100.0 + np.cumsum(drift + rng.randn(n) * 0.004)
    return pd.DataFrame({
        'ts_event': pd.date_range('2025-04-01', periods=n, freq='5min',
                                  tz='UTC').tz_localize(None),
        'open': close, 'high': close + 0.003, 'low': close - 0.003, 'close': close,
        'atr_14': np.full(n, atr),
        'is_event': np.ones(n, dtype=np.int8),
        'event_score': np.full(n, 0.6),
        'event_direction': np.full(n, event_direction, dtype=np.int8),
        'kalman_direction': np.zeros(n, dtype=np.int8),
        'is_london': np.ones(n, dtype=np.int8),
        'is_overlap': np.zeros(n, dtype=np.int8),
        'regime_label': ['trending'] * n,
    })


class TestLabelByOutcomeFix(unittest.TestCase):

    def test_fix_reduces_neutral_drastically(self):
        """timeout_mfe_mae=True → NEUTRAL ينخفض جذرياً."""
        df = _make_bars(n=400)
        off = _PDT.label_by_outcome(df, timeout_mfe_mae=False)
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)

        neutral_off = int((off['bias_label'] == 2).sum())
        neutral_on = int((on['bias_label'] == 2).sum())

        # NEUTRAL يجب أن ينخفض على الأقل 40 نقطة مئوية
        self.assertLess(neutral_on, neutral_off)
        self.assertLess(neutral_on / len(df), 0.60,
                        f"NEUTRAL لا يزال مرتفعاً: {neutral_on}/{len(df)}")

    def test_fix_produces_directional_labels(self):
        """بعد الإصلاح: عشرات/مئات الـ directional labels."""
        df = _make_bars(n=400)
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)
        n_long = int((on['bias_label'] == 0).sum())
        n_short = int((on['bias_label'] == 1).sum())
        self.assertGreater(n_long + n_short, 50,
                           "الإصلاح يجب أن ينتج directional labels كثيرة")
        # كلا الاتجاهين موجودان
        self.assertGreater(n_long, 0)
        self.assertGreater(n_short, 0)

    def test_backward_compat_when_disabled(self):
        """timeout_mfe_mae=False → السلوك القديم (NEUTRAL سائد)."""
        df = _make_bars(n=400)
        off = _PDT.label_by_outcome(df, timeout_mfe_mae=False)
        neutral_off = int((off['bias_label'] == 2).sum())
        # في السيناريو الواقعي البطيء، الـ off يعطي NEUTRAL عالٍ جداً
        self.assertGreater(neutral_off / len(df), 0.90,
                           "مع الإصلاح معطّلاً، السلوك القديم (≈99% NEUTRAL) محفوظ")

    def test_event_direction_veto_long(self):
        """event_direction=+1 → لا SHORT labels (allow_short=False)."""
        df = _make_bars(n=400, event_direction=1)
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)
        n_short = int((on['bias_label'] == 1).sum())
        self.assertEqual(n_short, 0,
                         "event_direction=+1 يجب أن يمنع كل SHORT labels")

    def test_event_direction_veto_short(self):
        """event_direction=-1 → لا LONG labels."""
        df = _make_bars(n=400, event_direction=-1)
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)
        n_long = int((on['bias_label'] == 0).sum())
        self.assertEqual(n_long, 0,
                         "event_direction=-1 يجب أن يمنع كل LONG labels")

    def test_min_move_filter(self):
        """min_move عالٍ جداً → كل شيء NEUTRAL (الحركة أصغر من العتبة)."""
        df = _make_bars(n=300)
        # min_move = 100×ATR — مستحيل تجاوزه
        on = _PDT.label_by_outcome(
            df, timeout_mfe_mae=True, timeout_mfe_min_move_atr=100.0,
        )
        neutral = int((on['bias_label'] == 2).sum())
        self.assertEqual(neutral, len(df),
                         "min_move مستحيل → كل شيء NEUTRAL")

    def test_ratio_filter(self):
        """ratio عالٍ جداً → الفائز نادراً يكون ضِعف الخاسر → NEUTRAL أكثر."""
        df = _make_bars(n=400)
        low_ratio = _PDT.label_by_outcome(
            df, timeout_mfe_mae=True, timeout_mfe_mae_ratio=1.2,
        )
        high_ratio = _PDT.label_by_outcome(
            df, timeout_mfe_mae=True, timeout_mfe_mae_ratio=10.0,
        )
        n_low = int((low_ratio['bias_label'] != 2).sum())
        n_high = int((high_ratio['bias_label'] != 2).sum())
        # ratio أعلى = شروط أصعب = directional أقل
        self.assertGreaterEqual(n_low, n_high)

    def test_uptrend_yields_long(self):
        """trend صاعد واضح → غالبية الـ labels = LONG."""
        df = _make_bars(n=400, drift_choices=(0.012, 0.012, 0.012))  # كله صاعد
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)
        n_long = int((on['bias_label'] == 0).sum())
        n_short = int((on['bias_label'] == 1).sum())
        self.assertGreater(n_long, n_short,
                           "uptrend واضح يجب أن يعطي LONG > SHORT")

    def test_downtrend_yields_short(self):
        """trend هابط واضح → غالبية الـ labels = SHORT."""
        df = _make_bars(n=400, drift_choices=(-0.012, -0.012, -0.012))
        on = _PDT.label_by_outcome(df, timeout_mfe_mae=True)
        n_long = int((on['bias_label'] == 0).sum())
        n_short = int((on['bias_label'] == 1).sum())
        self.assertGreater(n_short, n_long,
                           "downtrend واضح يجب أن يعطي SHORT > LONG")


class TestBuildDayTradingLabelsFix(unittest.TestCase):

    def _make_simple(self, n=400, seed=3, drift=(-0.01, 0.0, 0.01)):
        rng = np.random.RandomState(seed)
        d = np.repeat(rng.choice(drift, n // 40 + 1), 40)[:n]
        close = 100.0 + np.cumsum(d + rng.randn(n) * 0.004)
        return pd.DataFrame({
            'ts_event': pd.date_range('2025-04-01', periods=n, freq='5min',
                                      tz='UTC').tz_localize(None),
            'open': close, 'high': close + 0.003, 'low': close - 0.003, 'close': close,
            'atr_14': np.full(n, 0.05),
            'is_london': np.ones(n, dtype=np.int8),
            'is_overlap': np.zeros(n, dtype=np.int8),
        })

    def test_fix_reduces_neutral(self):
        df = self._make_simple(n=400)
        off = _PDT.build_day_trading_labels(df, horizon_bars=12, timeout_mfe_mae=False)
        on = _PDT.build_day_trading_labels(df, horizon_bars=12, timeout_mfe_mae=True)
        neutral_off = int((off['bias_label'] == 2).sum())
        neutral_on = int((on['bias_label'] == 2).sum())
        self.assertLess(neutral_on, neutral_off)

    def test_fix_produces_directional(self):
        df = self._make_simple(n=400)
        on = _PDT.build_day_trading_labels(df, horizon_bars=12, timeout_mfe_mae=True)
        directional = int((on['bias_label'] != 2).sum())
        self.assertGreater(directional, 30)

    def test_backward_compat(self):
        df = self._make_simple(n=400)
        off = _PDT.build_day_trading_labels(df, horizon_bars=12, timeout_mfe_mae=False)
        # default للدالة القديمة: السلوك القديم محفوظ عند False
        self.assertGreater(int((off['bias_label'] == 2).sum()) / len(df), 0.85)


class TestMFEMAEConsistencyWithLabelEngine(unittest.TestCase):
    """يتأكد أن منطق MFE/MAE هنا متسق مع modules.label_engine_v2."""

    def test_ratio_and_min_move_defaults_match(self):
        """الـ defaults يطابقان TripleBarrierConfig في label_engine_v2."""
        from modules.label_engine_v2 import TripleBarrierConfig
        cfg = TripleBarrierConfig()
        import inspect
        sig = inspect.signature(_PDT.label_by_outcome)
        self.assertEqual(
            sig.parameters['timeout_mfe_mae_ratio'].default,
            cfg.mfe_mae_ratio,
            "ratio يجب أن يطابق label_engine_v2.TripleBarrierConfig.mfe_mae_ratio",
        )
        self.assertEqual(
            sig.parameters['timeout_mfe_min_move_atr'].default,
            cfg.min_move_atr_mult,
            "min_move يجب أن يطابق label_engine_v2",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
