"""Tests for the HybridModel fusion architecture."""
from __future__ import annotations

import torch
import pytest

from modules.trading_intel.hybrid.model import (
    HybridConfig,
    HybridModel,
    HybridOutput,
    HybridTargets,
    compute_hybrid_loss,
)


class TestHybridConfig:
    def test_total_input_dim(self):
        cfg = HybridConfig(daytrade_feature_dim=100, ssl_embed_dim=50, cnn_embed_dim=20)
        assert cfg.total_input_dim == 170

    def test_total_input_dim_no_cnn(self):
        cfg = HybridConfig(daytrade_feature_dim=100, ssl_embed_dim=50, cnn_embed_dim=0)
        assert cfg.total_input_dim == 150


class TestHybridModel:
    def test_default_forward_shapes(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        B = 8
        out = model(
            torch.randn(B, cfg.daytrade_feature_dim),
            torch.randn(B, cfg.ssl_embed_dim),
            torch.randn(B, cfg.cnn_embed_dim),
        )
        assert out.event_logit.shape == (B,)
        assert out.direction_logits.shape == (B, 3)
        assert out.confidence_logit.shape == (B,)

    def test_no_cnn_fallback_zeros(self):
        """Passing None for cnn_embedding should not crash when cnn_embed_dim > 0."""
        cfg = HybridConfig(cnn_embed_dim=32)
        model = HybridModel(cfg)
        B = 4
        out = model(
            torch.randn(B, cfg.daytrade_feature_dim),
            torch.randn(B, cfg.ssl_embed_dim),
            cnn_embedding=None,  # gracefully fill with zeros
        )
        assert out.event_logit.shape == (B,)

    def test_no_cnn_configured(self):
        """Setting cnn_embed_dim=0 means we don't pass cnn at all."""
        cfg = HybridConfig(cnn_embed_dim=0)
        model = HybridModel(cfg)
        B = 4
        out = model(
            torch.randn(B, cfg.daytrade_feature_dim),
            torch.randn(B, cfg.ssl_embed_dim),
        )
        assert out.event_logit.shape == (B,)

    def test_probability_methods(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        out = model(
            torch.randn(4, cfg.daytrade_feature_dim),
            torch.randn(4, cfg.ssl_embed_dim),
            torch.randn(4, cfg.cnn_embed_dim),
        )
        # event prob in [0, 1]
        ep = out.event_prob()
        assert (ep >= 0).all() and (ep <= 1).all()
        # direction sum to 1
        dp = out.direction_probs()
        assert torch.allclose(dp.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_disable_event_head(self):
        cfg = HybridConfig(use_event_head=False)
        model = HybridModel(cfg)
        out = model(
            torch.randn(2, cfg.daytrade_feature_dim),
            torch.randn(2, cfg.ssl_embed_dim),
            torch.randn(2, cfg.cnn_embed_dim),
        )
        assert out.event_logit is None
        assert out.direction_logits is not None

    def test_all_heads_disabled_raises(self):
        cfg = HybridConfig(use_event_head=False, use_direction_head=False,
                           use_confidence_head=False)
        with pytest.raises(ValueError, match="All heads"):
            HybridModel(cfg)

    def test_zero_input_dim_raises(self):
        cfg = HybridConfig(daytrade_feature_dim=0, ssl_embed_dim=0, cnn_embed_dim=0)
        with pytest.raises(ValueError, match="All input dims"):
            HybridModel(cfg)

    def test_shape_validation(self):
        cfg = HybridConfig(daytrade_feature_dim=100, ssl_embed_dim=50, cnn_embed_dim=32)
        model = HybridModel(cfg)
        with pytest.raises(AssertionError, match="daytrade_features"):
            model(
                torch.randn(4, 99),   # wrong dim
                torch.randn(4, 50),
                torch.randn(4, 32),
            )


class TestHybridLoss:
    def test_full_targets(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        B = 8
        out = model(
            torch.randn(B, cfg.daytrade_feature_dim),
            torch.randn(B, cfg.ssl_embed_dim),
            torch.randn(B, cfg.cnn_embed_dim),
        )
        targets = HybridTargets(
            event_flag=torch.randint(0, 2, (B,)).float(),
            direction=torch.randint(0, 3, (B,)),
            confidence=torch.rand(B),
        )
        loss, metrics = compute_hybrid_loss(out, targets, cfg)
        assert loss.item() > 0
        for k in ("loss_event", "loss_direction", "loss_confidence",
                  "loss_total_hybrid"):
            assert k in metrics

    def test_partial_targets(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        B = 4
        out = model(
            torch.randn(B, cfg.daytrade_feature_dim),
            torch.randn(B, cfg.ssl_embed_dim),
            torch.randn(B, cfg.cnn_embed_dim),
        )
        # Only event head has targets
        targets = HybridTargets(event_flag=torch.randint(0, 2, (B,)).float())
        loss, metrics = compute_hybrid_loss(out, targets, cfg)
        assert "loss_event" in metrics
        assert "loss_direction" not in metrics
        assert "loss_confidence" not in metrics

    def test_no_targets_zero_loss(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        out = model(
            torch.randn(2, cfg.daytrade_feature_dim),
            torch.randn(2, cfg.ssl_embed_dim),
            torch.randn(2, cfg.cnn_embed_dim),
        )
        targets = HybridTargets()  # all None
        loss, _ = compute_hybrid_loss(out, targets, cfg)
        assert loss.item() == 0.0

    def test_backward_through_full_pipeline(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        B = 8
        xd = torch.randn(B, cfg.daytrade_feature_dim, requires_grad=True)
        xs = torch.randn(B, cfg.ssl_embed_dim, requires_grad=True)
        xc = torch.randn(B, cfg.cnn_embed_dim, requires_grad=True)
        out = model(xd, xs, xc)
        targets = HybridTargets(
            event_flag=torch.randint(0, 2, (B,)).float(),
            direction=torch.randint(0, 3, (B,)),
            confidence=torch.rand(B),
        )
        loss, _ = compute_hybrid_loss(out, targets, cfg)
        loss.backward()
        # All three inputs should receive gradient
        for t, name in [(xd, "daytrade"), (xs, "ssl"), (xc, "cnn")]:
            assert t.grad is not None, f"{name} got no grad"
            assert t.grad.abs().sum().item() > 0, f"{name} grad is zero"


class TestHybridParameterCount:
    def test_default_size(self):
        cfg = HybridConfig()
        model = HybridModel(cfg)
        n = model.num_parameters()
        # Default: 135+161+32=328 input → 256 hidden × 3 layers
        # rough estimate ~200k-400k
        assert 100_000 < n < 700_000, f"unexpected param count: {n}"

    def test_lean_size(self):
        """Smaller config → fewer parameters."""
        cfg = HybridConfig(
            daytrade_feature_dim=50, ssl_embed_dim=64, cnn_embed_dim=0,
            hidden_dim=64, n_layers=2,
        )
        model = HybridModel(cfg)
        n = model.num_parameters()
        assert n < 50_000, f"lean config too big: {n}"
