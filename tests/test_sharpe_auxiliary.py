"""Tests for the Sharpe-aware auxiliary loss in self_supervised/sharpe_auxiliary.py.

Three layers:
  1. Config validation
  2. _to_sharpe_trinity (pred → p_long, p_short, magnitude conversion)
  3. compute_sharpe_aux_loss end-to-end including the disabled fast path
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.sharpe_auxiliary import (
    SharpeAuxConfig,
    _to_sharpe_trinity,
    compute_sharpe_aux_loss,
)


# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_defaults_are_off(self):
        cfg = SharpeAuxConfig()
        assert cfg.enabled is False
        assert cfg.weight == pytest.approx(0.10)
        assert cfg.objective == "sharpe"

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            SharpeAuxConfig(weight=-0.1)

    def test_non_positive_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SharpeAuxConfig(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            SharpeAuxConfig(temperature=-1.0)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_unit_position"):
            SharpeAuxConfig(cost_per_unit_position=-1e-5)

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="objective"):
            SharpeAuxConfig(objective="rmse")

    def test_sortino_accepted(self):
        cfg = SharpeAuxConfig(objective="sortino")
        assert cfg.objective == "sortino"


# ════════════════════════════════════════════════════════════════════
# _to_sharpe_trinity
# ════════════════════════════════════════════════════════════════════
class TestTrinity:
    def test_positive_pred_long_active(self):
        pred = torch.tensor([1.0, 0.5, 0.1])
        p_long, p_short, mag = _to_sharpe_trinity(pred, temperature=1.0)
        assert (p_long > 0).all()
        assert (p_short == 0).all()
        assert torch.allclose(mag, p_long)

    def test_negative_pred_short_active(self):
        pred = torch.tensor([-1.0, -0.5, -0.1])
        p_long, p_short, mag = _to_sharpe_trinity(pred, temperature=1.0)
        assert (p_long == 0).all()
        assert (p_short > 0).all()
        assert torch.allclose(mag, p_short)

    def test_zero_pred_zero_magnitude(self):
        pred = torch.zeros(5)
        p_long, p_short, mag = _to_sharpe_trinity(pred, temperature=2.0)
        assert torch.all(p_long == 0)
        assert torch.all(p_short == 0)
        assert torch.all(mag == 0)

    def test_magnitude_equals_sum(self):
        # By construction: p_long + p_short == magnitude for any pred
        pred = torch.randn(20)
        p_long, p_short, mag = _to_sharpe_trinity(pred, temperature=1.5)
        assert torch.allclose(p_long + p_short, mag, atol=1e-6)

    def test_temperature_amplifies(self):
        pred = torch.tensor([0.5])
        _, _, mag_low = _to_sharpe_trinity(pred, temperature=0.5)
        _, _, mag_high = _to_sharpe_trinity(pred, temperature=5.0)
        assert mag_high.item() > mag_low.item()

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            _to_sharpe_trinity(torch.zeros(2, 3), temperature=1.0)


# ════════════════════════════════════════════════════════════════════
# compute_sharpe_aux_loss
# ════════════════════════════════════════════════════════════════════
class TestSharpeLoss:
    def test_disabled_returns_zero(self):
        cfg = SharpeAuxConfig(enabled=False)
        pred = torch.randn(32)
        target = torch.randn(32)
        loss, diag = compute_sharpe_aux_loss(pred, target, cfg)
        assert loss.item() == 0.0
        assert diag["loss_kind"] == "disabled"

    def test_disabled_skips_shape_check(self):
        # When disabled, the function should NOT raise even on
        # mismatched shapes — it's a no-op fast path
        cfg = SharpeAuxConfig(enabled=False)
        pred = torch.randn(10)
        target = torch.randn(5)  # mismatch
        loss, _ = compute_sharpe_aux_loss(pred, target, cfg)
        assert loss.item() == 0.0

    def test_enabled_with_perfect_alignment_negative_loss(self):
        # When pred == target (perfect alignment), the model is
        # consistently long when target is positive and short when
        # negative → high Sharpe → negative loss (because the loss is
        # −Sharpe)
        torch.manual_seed(0)
        target = torch.randn(64) * 0.01    # realistic return magnitudes
        pred = target.clone() * 100         # amplified for direction
        cfg = SharpeAuxConfig(enabled=True, temperature=2.0)
        loss, diag = compute_sharpe_aux_loss(pred, target, cfg)
        assert loss.item() < 0  # negative Sharpe loss = profitable
        assert diag["emp_sharpe"] > 0

    def test_enabled_with_anti_alignment_positive_loss(self):
        # When pred = -target, the model takes the wrong side every
        # time → negative Sharpe → positive loss
        torch.manual_seed(0)
        target = torch.randn(64) * 0.01
        pred = -target.clone() * 100
        cfg = SharpeAuxConfig(enabled=True, temperature=2.0)
        loss, diag = compute_sharpe_aux_loss(pred, target, cfg)
        assert loss.item() > 0
        assert diag["emp_sharpe"] < 0

    def test_valid_mask_drops_samples(self):
        cfg = SharpeAuxConfig(enabled=True)
        target = torch.randn(64) * 0.01
        pred = target.clone() * 100
        # Mask out half the samples
        mask = torch.zeros(64, dtype=torch.bool)
        mask[:32] = True
        loss, diag = compute_sharpe_aux_loss(
            pred, target, cfg, valid_mask=mask,
        )
        assert diag["n_samples_used"] == 32

    def test_too_few_samples_returns_zero(self):
        cfg = SharpeAuxConfig(enabled=True)
        pred = torch.tensor([0.5])      # single sample
        target = torch.tensor([0.01])
        loss, diag = compute_sharpe_aux_loss(pred, target, cfg)
        assert loss.item() == 0.0
        assert diag["loss_kind"] == "too_few_samples"

    def test_shape_mismatch_raises_when_enabled(self):
        cfg = SharpeAuxConfig(enabled=True)
        pred = torch.randn(10)
        target = torch.randn(5)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_sharpe_aux_loss(pred, target, cfg)

    def test_backward_pass_flows(self):
        # Critical: the loss must be differentiable wrt pred
        cfg = SharpeAuxConfig(enabled=True)
        torch.manual_seed(0)
        target = torch.randn(64) * 0.01
        pred = (target.clone() * 50).requires_grad_(True)
        loss, _ = compute_sharpe_aux_loss(pred, target, cfg)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert pred.grad.abs().sum() > 0   # actual gradient signal

    def test_sortino_objective_works(self):
        cfg = SharpeAuxConfig(enabled=True, objective="sortino")
        torch.manual_seed(0)
        target = torch.randn(64) * 0.01
        pred = target.clone() * 50
        loss, diag = compute_sharpe_aux_loss(pred, target, cfg)
        # Sortino is computed — just verify it doesn't crash and
        # produces a finite value
        assert torch.isfinite(loss).all()
        assert "emp_sharpe" in diag

    def test_zero_temperature_inherited_validation(self):
        # The config catches this; just confirms compute path is safe
        with pytest.raises(ValueError, match="temperature"):
            SharpeAuxConfig(enabled=True, temperature=0.0)


# ════════════════════════════════════════════════════════════════════
# Integration with compute_ssl_loss (smoke)
# ════════════════════════════════════════════════════════════════════
class TestIntegrationWithComputeSSLLoss:
    """We can't easily import compute_ssl_loss without the full deep_lob
    model, but we can verify the wiring is correct by checking the
    signature change and the disabled-default behavior contract."""

    def test_compute_ssl_loss_accepts_optional_kwarg(self):
        import inspect
        from self_supervised.pretrain_lob import compute_ssl_loss
        sig = inspect.signature(compute_ssl_loss)
        assert "sharpe_aux_config" in sig.parameters
        # Default is None → fully backward-compatible
        assert sig.parameters["sharpe_aux_config"].default is None
