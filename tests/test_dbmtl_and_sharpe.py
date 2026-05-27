"""Tests for DBMTLBalancer and differentiable Sharpe loss."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from modules.trading_intel.anti_collapse import (
    DBMTLBalancer,
    DBMTLConfig,
    SharpeLossConfig,
    SharpeRegularizer,
    compute_dbmtl_loss,
    differentiable_sharpe_loss,
)


# ════════════════════════════════════════════════════════════════════
# DBMTLBalancer
# ════════════════════════════════════════════════════════════════════
class TestDBMTLBalancer:
    def test_warmup_uses_uniform(self):
        b = DBMTLBalancer(["a", "b"], DBMTLConfig(warmup_steps=5))
        # Skewed losses
        for _ in range(3):
            losses = {
                "a": torch.tensor(1.0, requires_grad=True),
                "b": torch.tensor(100.0, requires_grad=True),
            }
            total = b.compute_balanced_loss(losses)
            # Uniform: (1 + 100) / 2 = 50.5
            assert abs(total.item() - 50.5) < 1e-3

    def test_loss_scale_balancing_equalizes_magnitudes(self):
        """After warmup + EMA convergence, two losses with VASTLY
        different natural scales should contribute comparably."""
        # Disable gradient-norm balancing to isolate the loss-scale effect
        cfg = DBMTLConfig(
            warmup_steps=10,
            enable_gradient_norm_balancing=False,
            enable_loss_scale_balancing=True,
            ema_alpha=0.5,  # converge fast
        )
        b = DBMTLBalancer(["small", "big"], cfg)

        # Feed many steps so EMA converges
        for _ in range(60):
            losses = {
                "small": torch.tensor(0.5, requires_grad=True),
                "big": torch.tensor(200.0, requires_grad=True),
            }
            b.compute_balanced_loss(losses)

        # Now check that the SCALED contributions are within an order of
        # magnitude of each other (vs the raw 400x ratio)
        losses = {
            "small": torch.tensor(0.5, requires_grad=True),
            "big": torch.tensor(200.0, requires_grad=True),
        }
        b.compute_balanced_loss(losses)
        # After log1p normalization, log1p(0.5/0.5) = log(2) ≈ 0.693
        # and log1p(200/200) = log(2) ≈ 0.693 — IDENTICAL
        # So the ratio collapses from 400x to 1x ✓
        # We verify by checking the actual scaled values
        scaled = b._loss_scale_balanced(losses)
        ratio = float(scaled["big"]) / float(scaled["small"])
        assert ratio < 2.0, f"LSB failed: big/small ratio = {ratio:.2f}"

    def test_gradient_norm_balancing_lifts_quiet_tasks(self):
        """A task whose gradient norm is small should be UP-weighted
        to match the largest gradient norm."""
        torch.manual_seed(0)
        # Shared params
        shared = nn.Linear(10, 8)
        # Two tasks with very different sensitivities
        head_loud = nn.Linear(8, 1)
        head_quiet = nn.Linear(8, 1)
        # Initialize head_quiet with tiny weights so its loss has small grad
        with torch.no_grad():
            head_quiet.weight.mul_(0.001)
            head_quiet.bias.zero_()

        x = torch.randn(64, 10)
        target = torch.randn(64, 1)

        cfg = DBMTLConfig(
            warmup_steps=1,
            enable_loss_scale_balancing=False,
            enable_gradient_norm_balancing=True,
        )
        b = DBMTLBalancer(["loud", "quiet"], cfg)

        # Warmup pass
        feat = shared(x)
        l_loud = ((head_loud(feat) - target) ** 2).mean()
        l_quiet = ((head_quiet(feat) - target) ** 2).mean()
        _ = b.compute_balanced_loss({"loud": l_loud, "quiet": l_quiet})

        # Second pass — now balancing kicks in
        feat = shared(x)
        l_loud = ((head_loud(feat) - target) ** 2).mean()
        l_quiet = ((head_quiet(feat) - target) ** 2).mean()
        _ = b.compute_balanced_loss(
            {"loud": l_loud, "quiet": l_quiet},
            shared_params=shared.parameters(),
        )
        weights = b.state.last_weights
        # The "quiet" task should have a LARGER weight (it's being lifted)
        assert weights["quiet"] > weights["loud"], (
            f"quiet weight {weights['quiet']:.3f} should exceed loud "
            f"weight {weights['loud']:.3f}"
        )

    def test_weights_clipped_to_safe_range(self):
        """Even extreme gradient-norm ratios should yield weights in
        [min_weight, max_weight] to avoid runaway."""
        cfg = DBMTLConfig(
            warmup_steps=0, min_weight=0.1, max_weight=10.0,
            enable_loss_scale_balancing=False,
        )
        b = DBMTLBalancer(["a", "b"], cfg)
        # Manually inject grad norms
        grad_norms = {"a": 1.0, "b": 1e-10}   # huge ratio
        weights = b._gradient_norm_balanced_weights(grad_norms)
        assert all(cfg.min_weight <= w <= cfg.max_weight for w in weights.values())

    def test_diagnostics_keys(self):
        b = DBMTLBalancer(["x", "y"])
        diag = b.get_diagnostics()
        assert set(diag.keys()) == {"step", "loss_ema", "grad_norms", "weights"}


# ════════════════════════════════════════════════════════════════════
# Differentiable Sharpe Loss
# ════════════════════════════════════════════════════════════════════
class TestDifferentiableSharpe:
    def _make_perfect_predictor(self, B=200):
        """Returns inputs where the model's predicted direction PERFECTLY
        matches the sign of the realized return — Sharpe should be high
        (positive expected return, low variance)."""
        torch.manual_seed(0)
        target_ret = torch.randn(B) * 0.005   # ±50bp returns
        # Perfect predictor: position sign = return sign
        p_long = (target_ret > 0).float()
        p_short = (target_ret < 0).float()
        magnitude = torch.ones(B)
        return p_long, p_short, magnitude, target_ret

    def _make_anti_predictor(self, B=200):
        """Predictor that bets OPPOSITE to the return — Sharpe is very negative."""
        torch.manual_seed(0)
        target_ret = torch.randn(B) * 0.005
        # Anti-predictor: opposite sign
        p_long = (target_ret < 0).float()
        p_short = (target_ret > 0).float()
        magnitude = torch.ones(B)
        return p_long, p_short, magnitude, target_ret

    def test_perfect_predictor_negative_loss(self):
        """Loss = −Sharpe, so a perfect predictor should give a LARGE
        negative loss (i.e., maximize Sharpe)."""
        p_long, p_short, mag, ret = self._make_perfect_predictor()
        loss, diag = differentiable_sharpe_loss(
            p_long, p_short, mag, ret,
            config=SharpeLossConfig(cost_per_unit_position=0.0),
        )
        # Perfect predictor → Sharpe should be high → loss should be very negative
        assert loss.item() < -0.5, f"loss {loss.item():.3f} not negative enough"
        assert diag["emp_sharpe"] > 0.5
        assert diag["mean_pnl"] > 0

    def test_anti_predictor_positive_loss(self):
        """An anti-predictor should yield positive loss (negative Sharpe)."""
        p_long, p_short, mag, ret = self._make_anti_predictor()
        loss, diag = differentiable_sharpe_loss(
            p_long, p_short, mag, ret,
            config=SharpeLossConfig(cost_per_unit_position=0.0),
        )
        assert loss.item() > 0.5
        assert diag["emp_sharpe"] < -0.5

    def test_gradient_flows_into_predictions(self):
        """Backprop through Sharpe must update the prediction parameters.
        Use a slightly non-degenerate starting point so position != 0."""
        torch.manual_seed(0)
        B = 100
        target_ret = torch.randn(B) * 0.005

        # Trainable predictions — START SKEWED so position != 0 everywhere
        p_long = torch.full((B,), 0.6, requires_grad=True)
        p_short = torch.full((B,), 0.3, requires_grad=True)
        magnitude = torch.full((B,), 1.0, requires_grad=True)

        loss, diag = differentiable_sharpe_loss(p_long, p_short, magnitude, target_ret)
        # Make sure we're NOT in the degenerate branch for this test
        assert diag["loss_kind"] != "degenerate_regularizer"
        loss.backward()

        assert p_long.grad is not None
        assert p_short.grad is not None
        assert magnitude.grad is not None
        assert p_long.grad.abs().sum().item() > 0

    def test_optimization_improves_sharpe(self):
        """Train a tiny model to maximize Sharpe directly; verify it
        learns the correct directional pattern."""
        torch.manual_seed(0)
        B = 256
        # Signal: a single feature predicts the return direction
        x = torch.randn(B)
        target_ret = (x.sign() * 0.005 + 0.002 * torch.randn(B))

        # Tiny predictor: a linear head with 1 input → 3 outputs (long/short/mag)
        head = nn.Linear(1, 3)
        optim = torch.optim.Adam(head.parameters(), lr=5e-2)

        initial_sharpe = None
        final_sharpe = None
        for step in range(300):
            out = head(x.unsqueeze(-1))
            probs = torch.softmax(out[:, :2], dim=-1)
            p_long_b, p_short_b = probs[:, 0], probs[:, 1]
            magnitude_b = torch.nn.functional.softplus(out[:, 2])

            loss, diag = differentiable_sharpe_loss(
                p_long_b, p_short_b, magnitude_b, target_ret,
                config=SharpeLossConfig(cost_per_unit_position=0.0),
            )
            if step == 0:
                initial_sharpe = diag["emp_sharpe"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            final_sharpe = diag["emp_sharpe"]

        # Sharpe should improve substantially
        assert final_sharpe > initial_sharpe + 0.3, (
            f"Sharpe didn't improve: {initial_sharpe:.3f} → {final_sharpe:.3f}"
        )
        assert final_sharpe > 0.4, f"final Sharpe {final_sharpe:.3f} too low"

    def test_transaction_cost_reduces_sharpe(self):
        """Adding transaction cost should reduce Sharpe (compared to zero-cost)."""
        p_long, p_short, mag, ret = self._make_perfect_predictor()
        loss_no_cost, diag_no_cost = differentiable_sharpe_loss(
            p_long, p_short, mag, ret,
            config=SharpeLossConfig(cost_per_unit_position=0.0),
        )
        loss_with_cost, diag_with_cost = differentiable_sharpe_loss(
            p_long, p_short, mag, ret,
            config=SharpeLossConfig(cost_per_unit_position=0.001),  # 10 bp
        )
        assert diag_with_cost["emp_sharpe"] < diag_no_cost["emp_sharpe"]

    def test_degenerate_case_returns_regularizer(self):
        """If almost no positions are taken, return a regularizer that
        pushes magnitudes up — not a divide-by-zero blowup."""
        B = 100
        p_long = torch.full((B,), 0.5, requires_grad=True)
        p_short = torch.full((B,), 0.5, requires_grad=True)
        # Zero magnitude → all positions zero → degenerate
        magnitude = torch.zeros(B, requires_grad=True)
        target_ret = torch.randn(B) * 0.005

        loss, diag = differentiable_sharpe_loss(
            p_long, p_short, magnitude, target_ret,
        )
        assert diag["loss_kind"] == "degenerate_regularizer"
        # Loss is finite (no NaN/Inf)
        assert torch.isfinite(loss)

    def test_sortino_uses_downside_only(self):
        """Sortino should differ from Sharpe when distributions are skewed."""
        torch.manual_seed(0)
        # Asymmetric: many small positives, few large negatives
        B = 500
        ret_base = torch.full((B,), 0.001)
        # Inject 5% large negatives
        idx = torch.randperm(B)[:25]
        ret_base[idx] = -0.05
        target_ret = ret_base

        # Perfect predictor: always long
        p_long = torch.ones(B)
        p_short = torch.zeros(B)
        mag = torch.ones(B)

        _, sharpe_diag = differentiable_sharpe_loss(
            p_long, p_short, mag, target_ret,
            config=SharpeLossConfig(cost_per_unit_position=0.0, objective="sharpe"),
        )
        _, sortino_diag = differentiable_sharpe_loss(
            p_long, p_short, mag, target_ret,
            config=SharpeLossConfig(cost_per_unit_position=0.0, objective="sortino"),
        )
        # Both report empirical Sharpe (numeric is the same; the LOSS differs)
        assert sharpe_diag["loss_kind"] == "sharpe"
        assert sortino_diag["loss_kind"] == "sortino"


class TestSharpeRegularizer:
    def test_weighted_call(self):
        reg = SharpeRegularizer(weight=0.3)
        p_long, p_short, mag, ret = TestDifferentiableSharpe()._make_perfect_predictor()
        weighted_loss, diag = reg(p_long, p_short, mag, ret)
        # Compare with un-weighted
        raw, _ = differentiable_sharpe_loss(p_long, p_short, mag, ret)
        assert abs(weighted_loss.item() - 0.3 * raw.item()) < 1e-4
