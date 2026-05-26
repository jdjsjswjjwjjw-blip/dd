"""Tests for adaptive target heads + labels + decision policy."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.deep_lob.adaptive_target_heads import (
    AdaptiveTargetConfig,
    AdaptiveTargetHeads,
    AdaptiveTargetOutput,
    AdaptiveTargetTargets,
    compute_adaptive_target_loss,
)
from modules.deep_lob.adaptive_target_labels import (
    build_all_adaptive_targets,
    build_max_R_reached_targets,
    build_regime_risk_targets,
    build_target_bucket_targets,
)
from modules.trade_decision_policy import (
    TradeDecision,
    TradeDecisionPolicy,
    decide,
)


# ════════════════════════════════════════════════════════════════════
# Heads
# ════════════════════════════════════════════════════════════════════
class TestAdaptiveTargetHeads:
    def test_forward_shapes(self):
        cfg = AdaptiveTargetConfig(shared_dim=64)
        model = AdaptiveTargetHeads(cfg)
        emb = torch.randn(8, 64)
        out = model(emb)
        assert isinstance(out, AdaptiveTargetOutput)
        assert out.max_r_pred.shape == (8,)
        assert out.bucket_logits.shape == (8, 4)
        assert out.regime_risk_logits.shape == (8, 3)
        # max_r_pred must be non-negative (softplus)
        assert (out.max_r_pred >= 0).all()

    def test_adaptive_tp_mult(self):
        cfg = AdaptiveTargetConfig(shared_dim=32)
        model = AdaptiveTargetHeads(cfg)
        emb = torch.randn(4, 32)
        out = model(emb)
        tp = out.adaptive_tp_mult(base_tp_mult=1.5, max_tp_mult=3.0)
        assert tp.shape == (4,)
        assert (tp >= 1.5).all()
        assert (tp <= 3.0).all()

    def test_position_size_scale(self):
        cfg = AdaptiveTargetConfig(shared_dim=32)
        model = AdaptiveTargetHeads(cfg)
        emb = torch.randn(4, 32)
        out = model(emb)
        scale = out.position_size_scale()
        assert scale.shape == (4,)
        assert (scale >= 0).all()
        assert (scale <= 1.0).all()

    def test_loss_backward(self):
        cfg = AdaptiveTargetConfig(shared_dim=64)
        model = AdaptiveTargetHeads(cfg)
        emb = torch.randn(8, 64, requires_grad=True)
        out = model(emb)
        targets = AdaptiveTargetTargets(
            max_r_reached=torch.rand(8) * 3,
            target_bucket=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            regime_risk=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        )
        loss, metrics = compute_adaptive_target_loss(out, targets, cfg)
        loss.backward()
        assert emb.grad is not None
        assert "loss_max_r" in metrics
        assert "loss_bucket" in metrics
        assert "loss_regime" in metrics


# ════════════════════════════════════════════════════════════════════
# Labels
# ════════════════════════════════════════════════════════════════════
class TestMaxRReachedLabels:
    def test_long_2r_move(self):
        # Entry 1.30, ATR 0.005, LONG, fwd high reaches 1.310 (+2R)
        close = np.array([1.30, 1.305, 1.31, 1.31])
        high = np.array([1.301, 1.306, 1.312, 1.311])
        low = np.array([1.299, 1.302, 1.308, 1.309])
        atr = np.full(4, 0.005)
        bias = np.array([1, 1, 1, 1])
        sb = np.zeros(4, dtype=bool)
        max_R, valid = build_max_R_reached_targets(
            close, high, low, atr, bias, sb, forward_bars=3,
        )
        # bar 0: entry=1.30, fwd_high max in (0, 3] = max(1.306, 1.312, 1.311) = 1.312
        # max_R = (1.312 - 1.30) / 0.005 = 2.4
        assert valid[0]
        assert abs(max_R[0] - 2.4) < 0.01

    def test_short_1r_move(self):
        close = np.array([1.30, 1.295, 1.292, 1.293])
        high = np.array([1.302, 1.297, 1.294, 1.295])
        low = np.array([1.298, 1.294, 1.290, 1.291])
        atr = np.full(4, 0.005)
        bias = np.array([-1, -1, -1, -1])
        sb = np.zeros(4, dtype=bool)
        max_R, valid = build_max_R_reached_targets(
            close, high, low, atr, bias, sb, forward_bars=3,
        )
        # bar 0: entry=1.30, fwd_low min in (0, 3] = min(1.294, 1.290, 1.291) = 1.290
        # max_R = (1.30 - 1.290) / 0.005 = 2.0
        assert valid[0]
        assert abs(max_R[0] - 2.0) < 0.01

    def test_zero_bias_no_label(self):
        close = np.array([1.30, 1.305, 1.31])
        high = close + 0.001
        low = close - 0.001
        atr = np.full(3, 0.005)
        bias = np.zeros(3, dtype=int)
        sb = np.zeros(3, dtype=bool)
        max_R, valid = build_max_R_reached_targets(
            close, high, low, atr, bias, sb,
        )
        assert valid.sum() == 0


class TestBucketLabels:
    def test_bucket_boundaries(self):
        max_R = np.array([0.3, 1.0, 1.5, 2.0, 2.7, 3.5, 4.5])
        valid = np.ones(7, dtype=bool)
        bucket, _ = build_target_bucket_targets(
            max_R, valid, bucket_edges=(1.0, 2.0, 3.0),
        )
        expected = np.array([0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(bucket, expected)


class TestRegimeRiskLabels:
    def test_normal_regime(self):
        # Stable ATR — all normal
        atr = np.full(150, 0.005)
        sb = np.zeros(150, dtype=bool)
        risk, valid = build_regime_risk_targets(atr, sb)
        # At least the bars with enough history should be normal=0
        assert valid[100:].all()
        assert risk[100:].sum() == 0   # all normal

    def test_extreme_regime(self):
        atr = np.full(150, 0.005)
        atr[120] = 0.020   # 4× median = extreme
        sb = np.zeros(150, dtype=bool)
        risk, valid = build_regime_risk_targets(
            atr, sb, elevated_ratio=1.5, extreme_ratio=2.5,
        )
        assert valid[120]
        assert risk[120] == 2  # extreme

    def test_insufficient_history_invalid(self):
        atr = np.full(15, 0.005)   # less than rolling_window=100
        sb = np.zeros(15, dtype=bool)
        risk, valid = build_regime_risk_targets(atr, sb)
        # Some early bars should be invalid
        assert not valid[5]


class TestBundle:
    def test_all_three(self):
        N = 150
        rng = np.random.RandomState(42)
        close = 1.30 + rng.randn(N) * 0.005
        high = close + 0.001
        low = close - 0.001
        atr = np.full(N, 0.005)
        atr[100] = 0.015   # spike for regime
        bias = rng.choice([-1, 0, 1], N)
        sb = np.zeros(N, dtype=bool)
        out = build_all_adaptive_targets(
            close=close, high=high, low=low, atr=atr,
            bias_direction=bias, is_session_break=sb,
        )
        assert set(out.keys()) == {'max_R', 'bucket', 'regime_risk'}
        for k in out:
            assert 'label' in out[k] and 'valid' in out[k]
            assert out[k]['label'].shape == (N,)


# ════════════════════════════════════════════════════════════════════
# TradeDecisionPolicy
# ════════════════════════════════════════════════════════════════════
class TestDecisionPolicy:
    def _kwargs(self, **overrides):
        base = dict(
            rule_event_flag=1,
            rule_event_direction=1,
            rule_signal_quality=1,
            hybrid_event_prob=0.7,
            hybrid_p_long=0.6,
            hybrid_p_short=0.2,
            hybrid_p_neutral=0.2,
            hybrid_confidence=0.7,
            adaptive_max_r_pred=2.5,
            adaptive_bucket=2,
            adaptive_regime_risk=0,
        )
        base.update(overrides)
        return base

    def test_ok_trade_returns_take(self):
        d = decide(**self._kwargs())
        assert d.take_trade
        assert d.direction == 1
        assert d.position_size_scale > 0
        assert d.tp_mult >= 1.5

    def test_no_rule_event_skip(self):
        d = decide(**self._kwargs(rule_event_flag=0))
        assert not d.take_trade
        assert d.reason == "no_rule_event"

    def test_extreme_regime_skip(self):
        d = decide(**self._kwargs(adaptive_regime_risk=2))
        assert not d.take_trade
        assert d.reason == "regime_extreme"

    def test_weak_signal_quality_skip(self):
        d = decide(**self._kwargs(rule_signal_quality=0))
        assert not d.take_trade
        assert d.reason == "weak_signal_quality"

    def test_low_confidence_skip(self):
        d = decide(**self._kwargs(hybrid_confidence=0.3))
        assert not d.take_trade
        assert d.reason == "low_hybrid_confidence"

    def test_rule_hybrid_disagreement_skip(self):
        # Rule says LONG, hybrid says SHORT
        d = decide(**self._kwargs(
            rule_event_direction=1,
            hybrid_p_long=0.1, hybrid_p_short=0.7, hybrid_p_neutral=0.2,
        ))
        assert not d.take_trade
        assert d.reason == "rule_hybrid_disagree"

    def test_adaptive_tp_high_bucket(self):
        # Bucket 3 (3+R) + high max_r_pred → TP up to 2.5
        d = decide(**self._kwargs(adaptive_bucket=3, adaptive_max_r_pred=4.0))
        assert d.take_trade
        assert d.tp_mult == 2.5  # capped by bucket_tp_mult[3]

    def test_adaptive_tp_low_bucket(self):
        # Bucket 0 → TP stays at base
        d = decide(**self._kwargs(adaptive_bucket=0, adaptive_max_r_pred=0.5))
        # bucket 0 has size_scale 0.5 * regime 1.0 * quality 1.0 * conf 0.7 = 0.35
        # TP would clamp to 1.5
        assert d.tp_mult == 1.5

    def test_elevated_regime_reduces_size(self):
        d_norm = decide(**self._kwargs(adaptive_regime_risk=0))
        d_elev = decide(**self._kwargs(adaptive_regime_risk=1))
        if d_norm.take_trade and d_elev.take_trade:
            assert d_elev.position_size_scale < d_norm.position_size_scale

    def test_premium_quality_boosts_size(self):
        d_q1 = decide(**self._kwargs(rule_signal_quality=1))
        d_q2 = decide(**self._kwargs(rule_signal_quality=2))
        if d_q1.take_trade and d_q2.take_trade:
            assert d_q2.position_size_scale > d_q1.position_size_scale

    def test_decision_to_dict(self):
        d = decide(**self._kwargs())
        out = d.to_dict()
        assert set(out.keys()) == {'take_trade', 'direction',
                                    'position_size_scale', 'tp_mult',
                                    'sl_mult', 'reason'}

    def test_regime_out_of_bounds_clamps(self):
        """Defensive: regime > 2 should not crash (MEDIUM-1 from review)."""
        d = decide(**self._kwargs(adaptive_regime_risk=5))  # out of bounds
        # Should clamp to 2 (extreme) and skip
        assert d.reason == "regime_extreme"

    def test_position_size_scale_nan_handling(self):
        """NaN logits → expected = 0 (skip rather than gamble) (MEDIUM-3)."""
        cfg = AdaptiveTargetConfig(shared_dim=4)
        model = AdaptiveTargetHeads(cfg)
        # Manually inject NaN by feeding NaN through the backbone
        emb = torch.tensor([[float('nan'), 0.0, 0.0, 0.0],
                            [1.0, 2.0, 3.0, 4.0]])
        out = model(emb)
        scale = out.position_size_scale()
        # First row (NaN) → 0 ; second (clean) → finite, in [0,1]
        assert scale[0].item() == 0.0
        assert 0.0 <= scale[1].item() <= 1.0


# ════════════════════════════════════════════════════════════════════
# Numerical-stability fixes
# ════════════════════════════════════════════════════════════════════
class TestNumericalStabilityFixes:
    def test_trade_imbalance_no_blowup_tiny_volume(self):
        """1 share traded must not blow up the imbalance ratio (MEDIUM-4)."""
        from modules.lob_features_v2 import build_lob_tensor_v2_for_bar
        T, P = 5, 20
        half = P // 2
        raw_depth = np.ones((T, P)) * 10
        depth_log = np.log1p(raw_depth)
        bid_sz = np.ones((T, half))
        ask_sz = np.ones((T, half))
        # Tiny trade volume (way below 1.0 floor)
        buy = np.full((T, P), 0.01)
        sell = np.full((T, P), 0.01)
        out = build_lob_tensor_v2_for_bar(
            raw_depth, depth_log, bid_sz, ask_sz, buy, sell,
        )
        # Trade imbalance should be a sane ratio (in [-1, 1]), not ±inf or NaN
        imb = out[:, :, 1]
        assert np.isfinite(imb).all()
        assert (imb >= -1.0).all() and (imb <= 1.0).all()
