"""
modules/deep_lob/multi_task_heads.py
─────────────────────────────────────
Multi-Task prediction heads.

Forces the backbone to learn deep market structure by predicting:
    - direction (main task, 3-way classification)
    - next_price (regression, log-return)
    - next_imbalance (regression, [-1, 1])
    - next_volatility (regression, ATR-relative)
    - next_regime (3-way classification)
    - wall_persist (regression, bars until wall consumed)
    - time_to_event (regression, bars until next significant event)

Each head is a small MLP. Shared backbone + task-specific heads = classical MTL setup.

Loss aggregation:
    L_total = w_main * L_direction + Σ w_aux * L_aux

References:
    - Caruana (1997): "Multitask Learning" - foundational
    - Kendall et al. (2018): "Multi-Task Learning Using Uncertainty to Weigh Losses"
    - Liu et al. (2019): "End-to-End Multi-Task Learning with Attention"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiTaskHeadsConfig


@dataclass
class MultiTaskOutput:
    """Container للـ multi-task predictions."""

    direction_logits: torch.Tensor       # (B, 3) - LONG/SHORT/NEUTRAL
    next_price: torch.Tensor             # (B,) - log-return
    next_imbalance: torch.Tensor         # (B,) - signed
    next_volatility: torch.Tensor        # (B,) - log ATR or relative
    next_regime_logits: torch.Tensor     # (B, 3)
    wall_persist: torch.Tensor           # (B,) - bars
    time_to_event: torch.Tensor          # (B,) - bars
    shared_embedding: torch.Tensor       # (B, shared_dim) - useful للـ downstream

    def direction_probs(self) -> torch.Tensor:
        """Softmax of direction logits."""
        return F.softmax(self.direction_logits, dim=-1)

    def regime_probs(self) -> torch.Tensor:
        return F.softmax(self.next_regime_logits, dim=-1)

    def to_dict(self) -> dict:
        return {
            "direction_logits": self.direction_logits,
            "direction_probs": self.direction_probs(),
            "next_price": self.next_price,
            "next_imbalance": self.next_imbalance,
            "next_volatility": self.next_volatility,
            "next_regime_logits": self.next_regime_logits,
            "next_regime_probs": self.regime_probs(),
            "wall_persist": self.wall_persist,
            "time_to_event": self.time_to_event,
        }


@dataclass
class MultiTaskTargets:
    """Container للـ training targets (matches MultiTaskOutput fields).

    All optional — if None, that task's loss is skipped (useful for partial labels).
    """

    direction: Optional[torch.Tensor] = None      # (B,) int — 0/1/2
    next_price: Optional[torch.Tensor] = None     # (B,) float
    next_imbalance: Optional[torch.Tensor] = None # (B,) float in [-1, 1]
    next_volatility: Optional[torch.Tensor] = None# (B,) float (positive)
    next_regime: Optional[torch.Tensor] = None    # (B,) int — 0/1/2
    wall_persist: Optional[torch.Tensor] = None   # (B,) float (positive)
    time_to_event: Optional[torch.Tensor] = None  # (B,) float (positive)


class _SingleHead(nn.Module):
    """Single prediction head: 2-layer MLP."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(input_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHeads(nn.Module):
    """The 7 prediction heads + loss computation."""

    def __init__(self, config: MultiTaskHeadsConfig):
        super().__init__()
        self.config = config
        D = config.shared_dim

        # 7 heads
        self.head_direction = _SingleHead(D, config.direction_n_classes, config.dropout)
        self.head_next_price = _SingleHead(D, 1, config.dropout)
        self.head_next_imbalance = _SingleHead(D, 1, config.dropout)
        self.head_next_volatility = _SingleHead(D, 1, config.dropout)
        self.head_next_regime = _SingleHead(D, config.regime_n_classes, config.dropout)
        self.head_wall_persist = _SingleHead(D, 1, config.dropout)
        self.head_time_to_event = _SingleHead(D, 1, config.dropout)

    def forward(self, shared_emb: torch.Tensor) -> MultiTaskOutput:
        """
        Parameters
        ----------
        shared_emb : (B, shared_dim)

        Returns
        -------
        MultiTaskOutput dataclass
        """
        return MultiTaskOutput(
            direction_logits=self.head_direction(shared_emb),
            next_price=self.head_next_price(shared_emb).squeeze(-1),
            next_imbalance=torch.tanh(self.head_next_imbalance(shared_emb).squeeze(-1)),
            next_volatility=F.softplus(self.head_next_volatility(shared_emb).squeeze(-1)),
            next_regime_logits=self.head_next_regime(shared_emb),
            wall_persist=F.softplus(self.head_wall_persist(shared_emb).squeeze(-1)),
            time_to_event=F.softplus(self.head_time_to_event(shared_emb).squeeze(-1)),
            shared_embedding=shared_emb,
        )

    def compute_loss(
        self,
        outputs: MultiTaskOutput,
        targets: MultiTaskTargets,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Computes multi-task loss with weighted aggregation.

        Returns dict with:
            'total': scalar — final loss
            '<task>': scalar — individual loss
            'weighted_<task>': scalar — task * its weight
        """
        losses: dict[str, torch.Tensor] = {}
        weighted: dict[str, torch.Tensor] = {}
        cfg = self.config

        # Direction (main)
        if targets.direction is not None:
            loss_dir = F.cross_entropy(
                outputs.direction_logits, targets.direction.long(),
                reduction=reduction,
            )
            losses["direction"] = loss_dir
            weighted["direction"] = cfg.direction_weight * loss_dir

        # Next price (regression)
        if targets.next_price is not None:
            loss_np = F.mse_loss(outputs.next_price, targets.next_price, reduction=reduction)
            losses["next_price"] = loss_np
            weighted["next_price"] = cfg.next_price_weight * loss_np

        # Next imbalance
        if targets.next_imbalance is not None:
            loss_ni = F.mse_loss(outputs.next_imbalance, targets.next_imbalance, reduction=reduction)
            losses["next_imbalance"] = loss_ni
            weighted["next_imbalance"] = cfg.next_imbalance_weight * loss_ni

        # Next volatility (log-MSE اقوى ضد outliers)
        if targets.next_volatility is not None:
            log_pred = torch.log(outputs.next_volatility + 1e-6)
            log_target = torch.log(targets.next_volatility + 1e-6)
            loss_nv = F.mse_loss(log_pred, log_target, reduction=reduction)
            losses["next_volatility"] = loss_nv
            weighted["next_volatility"] = cfg.next_volatility_weight * loss_nv

        # Next regime
        if targets.next_regime is not None:
            loss_nr = F.cross_entropy(
                outputs.next_regime_logits, targets.next_regime.long(),
                reduction=reduction,
            )
            losses["next_regime"] = loss_nr
            weighted["next_regime"] = cfg.next_regime_weight * loss_nr

        # Wall persist (Huber loss = robust)
        if targets.wall_persist is not None:
            loss_wp = F.smooth_l1_loss(outputs.wall_persist, targets.wall_persist, reduction=reduction)
            losses["wall_persist"] = loss_wp
            weighted["wall_persist"] = cfg.wall_persist_weight * loss_wp

        # Time to event
        if targets.time_to_event is not None:
            loss_te = F.smooth_l1_loss(outputs.time_to_event, targets.time_to_event, reduction=reduction)
            losses["time_to_event"] = loss_te
            weighted["time_to_event"] = cfg.time_to_event_weight * loss_te

        # Aggregate
        if not weighted:
            raise ValueError("No targets provided — at least 1 task must have target")

        total = sum(weighted.values())

        result = {"total": total}
        for k, v in losses.items():
            result[k] = v
        for k, v in weighted.items():
            result[f"weighted_{k}"] = v

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
