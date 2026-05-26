"""
modules/deep_lob/adaptive_target_heads.py
═══════════════════════════════════════════════════════════════════════════════
Adaptive target heads — address the proposal's 4 day-trade success criteria:

  (1) Raise TP2 dynamically per trade (instead of fixed 1.5 × ATR)
  (2) Improve hit rate by per-trade confidence
  (3) Accurate prediction of 1R / 2R / 3R reach within forward window
  (4) Reduce drawdown in volatile regimes via regime-aware sizing

3 heads attached to the shared backbone embedding:

  • max_R_reached       (regression) — the maximum R-multiple reached in
                                       the forward window. Used directly
                                       to size adaptive TP per trade.
  • target_bucket       (4-way CE)   — 0=<1R, 1=1R-2R, 2=2R-3R, 3=3R+
                                       Discrete tier for trade selection.
  • regime_risk         (3-way CE)   — 0=normal, 1=elevated, 2=extreme.
                                       Used to scale position size down
                                       in extreme volatility.

These complement `short_term_heads.py` — that one predicts micro events
(walls, sweeps), this one predicts TARGET REACH (the trader's edge).

Labels live in `adaptive_target_labels.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptiveTargetConfig:
    shared_dim: int = 64
    dropout: float = 0.1
    # Loss weights
    w_max_r: float = 1.5
    w_bucket: float = 1.0
    w_regime: float = 0.7
    # Bucket boundaries in R-multiple — must match labels module
    bucket_edges: tuple[float, ...] = (1.0, 2.0, 3.0)  # → 4 buckets


@dataclass
class AdaptiveTargetOutput:
    max_r_pred: torch.Tensor          # (B,) — predicted max R reached
    bucket_logits: torch.Tensor       # (B, 4)
    regime_risk_logits: torch.Tensor  # (B, 3)

    def bucket_probs(self) -> torch.Tensor:
        return F.softmax(self.bucket_logits, dim=-1)

    def regime_risk_probs(self) -> torch.Tensor:
        return F.softmax(self.regime_risk_logits, dim=-1)

    def adaptive_tp_mult(
        self, base_tp_mult: float = 1.5, max_tp_mult: float = 3.0,
    ) -> torch.Tensor:
        """Derive a per-trade TP multiplier from max_r_pred.

        We don't trust the raw regression — we cap it at max_tp_mult and
        floor it at base_tp_mult. The idea: never SHORTEN the TP below
        the baseline (those trades work fine), only ALLOW EXTENSION when
        the model is bullish on magnitude.
        """
        return torch.clamp(self.max_r_pred, min=base_tp_mult, max=max_tp_mult)

    def position_size_scale(
        self, base_scale: float = 1.0, min_scale: float = 0.0,
        threshold_extreme: float = 0.6,
    ) -> torch.Tensor:
        """Derive a position-size scaler from regime_risk:
           normal   → 1.0  ×
           elevated → 0.5  ×
           extreme  → 0.0  × (skip trade)
        Returns (B,) scale factor.
        """
        probs = self.regime_risk_probs()
        # Expected scale = P(normal)*1 + P(elevated)*0.5 + P(extreme)*0
        scales = torch.tensor(
            [1.0, 0.5, 0.0], device=probs.device, dtype=probs.dtype,
        )
        expected = (probs * scales).sum(dim=-1)
        # If extreme prob > threshold_extreme, force skip
        force_skip = probs[:, 2] > threshold_extreme
        expected = torch.where(force_skip, torch.zeros_like(expected), expected)
        return torch.clamp(expected * base_scale, min=min_scale, max=base_scale)


@dataclass
class AdaptiveTargetTargets:
    max_r_reached: Optional[torch.Tensor] = None    # (B,) float — actual max R
    target_bucket: Optional[torch.Tensor] = None    # (B,) int in {0, 1, 2, 3}
    regime_risk: Optional[torch.Tensor] = None      # (B,) int in {0, 1, 2}


def _make_head(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    hidden = max(in_dim // 2, 16)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class AdaptiveTargetHeads(nn.Module):
    """3 adaptive-target heads on top of a shared backbone embedding."""

    def __init__(self, config: AdaptiveTargetConfig | None = None):
        super().__init__()
        self.config = config or AdaptiveTargetConfig()
        D = self.config.shared_dim
        d = self.config.dropout
        self.head_max_r = _make_head(D, 1, d)
        # 4 buckets: [<1R, 1-2R, 2-3R, 3+R]
        self.head_bucket = _make_head(D, 4, d)
        # 3 regime tiers: [normal, elevated, extreme]
        self.head_regime = _make_head(D, 3, d)

    def forward(self, shared_emb: torch.Tensor) -> AdaptiveTargetOutput:
        # max_r through softplus so it's always >= 0 (R-multiples are non-negative)
        max_r_raw = self.head_max_r(shared_emb).squeeze(-1)
        return AdaptiveTargetOutput(
            max_r_pred=F.softplus(max_r_raw),
            bucket_logits=self.head_bucket(shared_emb),
            regime_risk_logits=self.head_regime(shared_emb),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_adaptive_target_loss(
    outputs: AdaptiveTargetOutput,
    targets: AdaptiveTargetTargets,
    config: AdaptiveTargetConfig,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted multi-task loss. Skips heads whose target is None."""
    losses: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}

    if targets.max_r_reached is not None:
        # Smooth L1 for robustness to tail values; max_R can be 0 to ~5
        l = F.smooth_l1_loss(
            outputs.max_r_pred, targets.max_r_reached.float(),
            reduction=reduction, beta=0.1,
        )
        losses["max_r"] = l
        weighted["max_r"] = config.w_max_r * l

    if targets.target_bucket is not None:
        l = F.cross_entropy(
            outputs.bucket_logits, targets.target_bucket.long(),
            reduction=reduction,
        )
        losses["bucket"] = l
        weighted["bucket"] = config.w_bucket * l

    if targets.regime_risk is not None:
        l = F.cross_entropy(
            outputs.regime_risk_logits, targets.regime_risk.long(),
            reduction=reduction,
        )
        losses["regime"] = l
        weighted["regime"] = config.w_regime * l

    if weighted:
        total = sum(weighted.values())
    else:
        total = torch.tensor(
            0.0, device=outputs.max_r_pred.device, requires_grad=True,
        )
    metrics = {f"loss_{k}": float(v.detach()) for k, v in losses.items()}
    metrics["loss_total_adaptive"] = float(total.detach()) if total.requires_grad else 0.0
    return total, metrics
