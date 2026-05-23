"""Tests for 4-class regime unification across regime_config + PR #17."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest


class TestRegimeConfig4Class(unittest.TestCase):
    """Verify all REGIME_* dicts cover the 4 classes."""

    def test_regimes_tuple(self):
        from regime_config import REGIMES
        self.assertEqual(set(REGIMES), {'trending', 'ranging', 'volatile', 'low_liquidity'})
        self.assertEqual(len(REGIMES), 4)

    def test_all_dicts_cover_4_regimes(self):
        from regime_config import (
            REGIMES, REGIME_EVENT_THRESHOLD, REGIME_TP_SL, REGIME_MAX_BARS,
            REGIME_MODEL_CONFIGS, REGIME_EXTRA_FEATURES, REGIME_DRIFT_CONFIG,
            REGIME_PRED_THRESHOLD,
        )
        for d_name, d in [
            ('REGIME_EVENT_THRESHOLD', REGIME_EVENT_THRESHOLD),
            ('REGIME_TP_SL', REGIME_TP_SL),
            ('REGIME_MAX_BARS', REGIME_MAX_BARS),
            ('REGIME_MODEL_CONFIGS', REGIME_MODEL_CONFIGS),
            ('REGIME_EXTRA_FEATURES', REGIME_EXTRA_FEATURES),
            ('REGIME_DRIFT_CONFIG', REGIME_DRIFT_CONFIG),
            ('REGIME_PRED_THRESHOLD', REGIME_PRED_THRESHOLD),
        ]:
            self.assertEqual(
                set(d.keys()), set(REGIMES),
                f"{d_name} keys mismatch: got {sorted(d.keys())}, expected {sorted(REGIMES)}",
            )

    def test_low_liquidity_sensible_values(self):
        """low_liquidity should have stricter thresholds + smaller targets."""
        from regime_config import (
            REGIME_EVENT_THRESHOLD, REGIME_TP_SL, REGIME_PRED_THRESHOLD,
        )
        # Event threshold ≥ trending (low_liquidity = noisy → strict)
        self.assertGreaterEqual(
            REGIME_EVENT_THRESHOLD['low_liquidity'],
            REGIME_EVENT_THRESHOLD['trending'],
        )
        # TP smaller than trending (limited movement)
        self.assertLess(
            REGIME_TP_SL['low_liquidity'][0],
            REGIME_TP_SL['trending'][0],
        )
        # Pred threshold ≥ trending (be selective)
        self.assertGreaterEqual(
            REGIME_PRED_THRESHOLD['low_liquidity'],
            REGIME_PRED_THRESHOLD['trending'],
        )

    def test_print_summary_works(self):
        """print_regime_summary should not crash with 4 regimes."""
        from regime_config import print_regime_summary
        # Just verify it runs without error
        print_regime_summary()


class TestPR17RegimeClasses(unittest.TestCase):
    """Verify PR #17 multi_task_heads now uses 4 regime classes."""

    def test_default_regime_n_classes_is_4(self):
        from modules.deep_lob.config import MultiTaskHeadsConfig
        cfg = MultiTaskHeadsConfig()
        self.assertEqual(cfg.regime_n_classes, 4)

    def test_regime_head_output_shape(self):
        """MultiTaskHeads regime logits should be (B, 4)."""
        import torch
        from modules.deep_lob.config import MultiTaskHeadsConfig
        from modules.deep_lob.multi_task_heads import MultiTaskHeads
        cfg = MultiTaskHeadsConfig()
        heads = MultiTaskHeads(cfg)
        shared = torch.randn(2, cfg.shared_dim)
        out = heads(shared)
        self.assertEqual(out.next_regime_logits.shape, (2, 4))
        # softmax → probs should sum to 1
        probs = out.regime_probs()
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
