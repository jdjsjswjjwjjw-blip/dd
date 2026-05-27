"""
modules/trading_intel/anti_collapse/dbmtl_balancer.py
═══════════════════════════════════════════════════════════════════════════════
DB-MTL (Dual-Balancing Multi-Task Learning) — gradient + loss balancer.

Addresses Mechanism #7 (Multi-Task Optimization Tradeoff): in our 6-month
run we observed:
  • v1 (β=5 magnitude): magnitude head rank-IC = 0.25, direction collapsed
  • v2 (α=5 direction): direction still failed, magnitude lost (0.04)
  → static weights guarantee one head dominates, never both

Solution (Chinese DB-MTL, Lin et al. 2023 / Liu et al. 2022):
  Apply TWO complementary balancing passes per training step:

  Pass 1 — Loss-Scale Balancing (LSB):
    Convert raw losses to a common log-scale where they're comparable.
    Each task's effective weight is normalized by a running average of
    its own loss, so a task with naturally larger losses doesn't dominate
    just because its scale is bigger.

      L̃_i(t) = log(1 + L_i(t) / EMA[L_i])

  Pass 2 — Gradient-Norm Balancing (GNB):
    Compute the L2 norm of each task's gradient w.r.t. the shared backbone.
    Rescale each task's loss so that all per-task gradient norms become
    EQUAL TO THE MAXIMUM (instead of all becoming equal — that would shrink
    everything). This way, no task is "muted", they're all "lifted" to the
    most active task's contribution.

      g_i = ||∇_θ L_i||_2,  g_max = max_i g_i,
      α_i = g_max / g_i  (clipped to [α_min, α_max] for stability)
      L_total = Σ α_i · L̃_i

Combined: each task is comparable in scale (LSB) AND contributes equally
to the gradient update (GNB). No task can be muted by another.

Key references:
  • Lin et al. (2023) "DB-MTL: Dual-Balancing Multi-Task Learning"
  • Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss
    Balancing" — predecessor with similar gradient-norm logic
  • Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh
    Losses" — alternative weighting scheme using homoscedastic uncertainty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class DBMTLConfig:
    """Hyperparameters for the dual-balancing scheme."""
    ema_alpha: float = 0.95           # EMA smoothing for loss scales
    eps: float = 1e-8                 # numerical stability
    min_weight: float = 0.1           # floor on per-task weight to keep all alive
    max_weight: float = 10.0          # cap to prevent runaway
    warmup_steps: int = 50            # use uniform weights for the first N steps
    enable_loss_scale_balancing: bool = True
    enable_gradient_norm_balancing: bool = True


@dataclass
class DBMTLState:
    """Mutable state tracked across training steps."""
    step: int = 0
    loss_ema: dict[str, float] = field(default_factory=dict)
    last_grad_norms: dict[str, float] = field(default_factory=dict)
    last_weights: dict[str, float] = field(default_factory=dict)


class DBMTLBalancer:
    """Stateful balancer that computes a balanced total loss from
    per-task losses + an optional shared backbone (for gradient-norm
    balancing).

    Usage:
        balancer = DBMTLBalancer(task_names=["event", "direction", "confidence"])
        for step in range(num_steps):
            losses = compute_per_task_losses(...)  # dict[str, Tensor]
            balanced_loss = balancer.compute_balanced_loss(
                losses, shared_params=model.backbone.parameters(),
            )
            optimizer.zero_grad()
            balanced_loss.backward()
            optimizer.step()
    """

    def __init__(
        self, task_names: Iterable[str], config: DBMTLConfig | None = None,
    ):
        self.task_names = list(task_names)
        self.config = config or DBMTLConfig()
        self.state = DBMTLState()

    # ── Loss-Scale Balancing ──────────────────────────────────────────
    def _update_loss_ema(self, losses: dict[str, torch.Tensor]) -> None:
        a = self.config.ema_alpha
        for name in self.task_names:
            if name not in losses:
                continue
            l = float(losses[name].detach())
            if name not in self.state.loss_ema:
                self.state.loss_ema[name] = l
            else:
                # Running EMA, gives 0 the first step but converges to mean
                self.state.loss_ema[name] = a * self.state.loss_ema[name] + (1 - a) * l

    def _loss_scale_balanced(
        self, losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map raw L_i to log(1 + L_i / EMA[L_i]). Result is in a common
        scale across tasks. Tasks with naturally-large loss (e.g., 3-class
        CE vs binary BCE) won't dominate just because of their range.
        """
        out: dict[str, torch.Tensor] = {}
        for name, l in losses.items():
            scale = max(self.state.loss_ema.get(name, float(l.detach())),
                        self.config.eps)
            # Log-transform keeps gradients bounded
            out[name] = torch.log1p(l / scale)
        return out

    # ── Gradient-Norm Balancing ───────────────────────────────────────
    def _compute_grad_norms(
        self,
        losses: dict[str, torch.Tensor],
        shared_params: list[torch.nn.Parameter],
    ) -> dict[str, float]:
        """For each task loss, compute ||∇L_i / ∇θ_shared||_2.

        Uses autograd.grad with retain_graph=True so the original graph
        survives for the final backward pass on the balanced loss.
        """
        grad_norms: dict[str, float] = {}
        for name, l in losses.items():
            if not l.requires_grad:
                grad_norms[name] = self.config.eps
                continue
            grads = torch.autograd.grad(
                l, shared_params, retain_graph=True, allow_unused=True,
            )
            sq_sum = 0.0
            for g in grads:
                if g is not None:
                    sq_sum += float(g.detach().pow(2).sum())
            grad_norms[name] = float(sq_sum ** 0.5)
        return grad_norms

    def _gradient_norm_balanced_weights(
        self, grad_norms: dict[str, float],
    ) -> dict[str, float]:
        """Compute α_i = g_max / g_i, clipped to [min_weight, max_weight]."""
        # Floor the denominators to avoid divide-by-zero
        floored = {k: max(v, self.config.eps) for k, v in grad_norms.items()}
        g_max = max(floored.values()) if floored else 0.0
        if g_max <= self.config.eps:
            # All zero → uniform weights (degenerate case)
            return {k: 1.0 for k in grad_norms}
        weights = {k: g_max / v for k, v in floored.items()}
        # Clip
        weights = {
            k: max(self.config.min_weight, min(self.config.max_weight, w))
            for k, w in weights.items()
        }
        return weights

    # ── Main entry point ──────────────────────────────────────────────
    def compute_balanced_loss(
        self,
        losses: dict[str, torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter] | None = None,
    ) -> torch.Tensor:
        """Combine per-task losses into a single balanced total loss.

        Args:
          losses: dict mapping task_name → scalar Tensor (requires_grad)
          shared_params: iterable of nn.Parameter from the shared backbone.
            Required if config.enable_gradient_norm_balancing is True.

        Returns: scalar Tensor = Σ α_i · L̃_i, ready for .backward()
        """
        self.state.step += 1
        cfg = self.config

        # During warmup: just sum with uniform weights so EMA accumulates
        if self.state.step <= cfg.warmup_steps:
            uniform = sum(losses.values())
            self._update_loss_ema(losses)
            return uniform / max(len(losses), 1)

        # Update EMA before applying LSB
        self._update_loss_ema(losses)

        # Step 1: Loss-Scale Balancing
        if cfg.enable_loss_scale_balancing:
            scaled = self._loss_scale_balanced(losses)
        else:
            scaled = losses

        # Step 2: Gradient-Norm Balancing
        if cfg.enable_gradient_norm_balancing and shared_params is not None:
            shared_list = [p for p in shared_params if p.requires_grad]
            if shared_list:
                grad_norms = self._compute_grad_norms(scaled, shared_list)
                weights = self._gradient_norm_balanced_weights(grad_norms)
                self.state.last_grad_norms = grad_norms
            else:
                weights = {k: 1.0 for k in scaled}
        else:
            weights = {k: 1.0 for k in scaled}

        self.state.last_weights = weights

        # Final combined loss
        total = sum(weights[k] * scaled[k] for k in scaled)
        # Normalize by number of active tasks so the gradient magnitude
        # stays comparable to a single-task baseline
        n_active = max(len(scaled), 1)
        return total / n_active

    # ── Diagnostics ───────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        """Snapshot for logging / debugging."""
        return {
            "step": self.state.step,
            "loss_ema": dict(self.state.loss_ema),
            "grad_norms": dict(self.state.last_grad_norms),
            "weights": dict(self.state.last_weights),
        }


def compute_dbmtl_loss(
    losses: dict[str, torch.Tensor],
    shared_params: Iterable[torch.nn.Parameter],
    balancer: DBMTLBalancer,
) -> tuple[torch.Tensor, dict]:
    """Convenience wrapper: compute balanced loss + return diagnostics."""
    total = balancer.compute_balanced_loss(losses, shared_params)
    return total, balancer.get_diagnostics()
