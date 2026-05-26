"""
modules/deep_lob/short_term_heads.py
═══════════════════════════════════════════════════════════════════════════════
Short-term SSL heads for day-trading horizons (per development proposal).

These 5 heads complement the existing MultiTaskHeads (next_price, next_imbalance,
next_volatility, next_regime, wall_persist, time_to_event). They focus on what
a day-trader actually cares about: the next 1-5 bars.

  • next_wall_break        — Will the dominant wall be broken within K bars?
                             Trinary: -1 broken downward, 0 no break, +1 upward
  • next_imbalance_shift   — Will OBI sign flip within K bars? Binary.
  • next_micro_target_hit  — Will price reach 0.5R or 1R within K bars?
                             Trinary: 0 none, 1 = 0.5R hit, 2 = 1R hit
  • next_liquidity_sweep   — Will a large sweep (>=K×median trade) happen
                             within W bars? Binary.
  • next_gap_fill_attempt  — Will price touch the nearest liquidity gap
                             within W bars? Binary.

All heads share the backbone embedding `shared_embedding` produced by the
existing HierarchicalLOBTransformer. They are deliberately separate from the
main `MultiTaskHeads` so we can:
  1. Train them independently if needed
  2. Plug them into an already-pretrained backbone without checkpoint conflict
  3. Ablate them one by one

The labels themselves are built by `short_term_targets.py` (pure numpy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShortTermHeadsConfig:
    shared_dim: int = 64              # backbone embedding dim
    dropout: float = 0.1
    # Loss weights — multiplied into the per-task loss
    w_wall_break: float = 1.0
    w_imbalance_shift: float = 0.5
    w_micro_target: float = 1.5
    w_liquidity_sweep: float = 0.7
    w_gap_fill: float = 0.5


@dataclass
class ShortTermOutput:
    wall_break_logits: torch.Tensor       # (B, 3) — down/none/up
    imbalance_shift_logit: torch.Tensor   # (B,)   — sigmoid for binary
    micro_target_logits: torch.Tensor     # (B, 3) — none/0.5R/1R
    liquidity_sweep_logit: torch.Tensor   # (B,)   — sigmoid for binary
    gap_fill_logit: torch.Tensor          # (B,)   — sigmoid for binary

    def wall_break_probs(self) -> torch.Tensor:
        return F.softmax(self.wall_break_logits, dim=-1)

    def micro_target_probs(self) -> torch.Tensor:
        return F.softmax(self.micro_target_logits, dim=-1)

    def imbalance_shift_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.imbalance_shift_logit)

    def liquidity_sweep_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.liquidity_sweep_logit)

    def gap_fill_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.gap_fill_logit)


@dataclass
class ShortTermTargets:
    """All optional — heads with target=None skip their loss term."""
    wall_break: Optional[torch.Tensor] = None       # (B,) int in {0, 1, 2} = {down, none, up}
    imbalance_shift: Optional[torch.Tensor] = None  # (B,) float in {0.0, 1.0}
    micro_target: Optional[torch.Tensor] = None     # (B,) int in {0, 1, 2}
    liquidity_sweep: Optional[torch.Tensor] = None  # (B,) float in {0.0, 1.0}
    gap_fill: Optional[torch.Tensor] = None         # (B,) float in {0.0, 1.0}


def _make_head(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    hidden = max(in_dim // 2, 16)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class ShortTermHeads(nn.Module):
    """5 short-term SSL heads attached to a shared backbone embedding."""

    def __init__(self, config: ShortTermHeadsConfig | None = None):
        super().__init__()
        self.config = config or ShortTermHeadsConfig()
        D = self.config.shared_dim
        d = self.config.dropout
        # Multi-class heads
        self.head_wall_break = _make_head(D, 3, d)
        self.head_micro_target = _make_head(D, 3, d)
        # Binary heads — single logit each
        self.head_imbalance_shift = _make_head(D, 1, d)
        self.head_liquidity_sweep = _make_head(D, 1, d)
        self.head_gap_fill = _make_head(D, 1, d)

    def forward(self, shared_emb: torch.Tensor) -> ShortTermOutput:
        return ShortTermOutput(
            wall_break_logits=self.head_wall_break(shared_emb),
            imbalance_shift_logit=self.head_imbalance_shift(shared_emb).squeeze(-1),
            micro_target_logits=self.head_micro_target(shared_emb),
            liquidity_sweep_logit=self.head_liquidity_sweep(shared_emb).squeeze(-1),
            gap_fill_logit=self.head_gap_fill(shared_emb).squeeze(-1),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_short_term_loss(
    outputs: ShortTermOutput,
    targets: ShortTermTargets,
    config: ShortTermHeadsConfig,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the weighted sum of per-task losses. Skips any task whose
    target is None (so partial-label batches work)."""
    losses: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}

    # ── Wall break (3-class CE) ──
    if targets.wall_break is not None:
        l = F.cross_entropy(
            outputs.wall_break_logits, targets.wall_break.long(),
            reduction=reduction,
        )
        losses["wall_break"] = l
        weighted["wall_break"] = config.w_wall_break * l

    # ── Imbalance shift (BCE) ──
    if targets.imbalance_shift is not None:
        l = F.binary_cross_entropy_with_logits(
            outputs.imbalance_shift_logit, targets.imbalance_shift.float(),
            reduction=reduction,
        )
        losses["imbalance_shift"] = l
        weighted["imbalance_shift"] = config.w_imbalance_shift * l

    # ── Micro target (3-class CE) ──
    if targets.micro_target is not None:
        l = F.cross_entropy(
            outputs.micro_target_logits, targets.micro_target.long(),
            reduction=reduction,
        )
        losses["micro_target"] = l
        weighted["micro_target"] = config.w_micro_target * l

    # ── Liquidity sweep (BCE) ──
    if targets.liquidity_sweep is not None:
        l = F.binary_cross_entropy_with_logits(
            outputs.liquidity_sweep_logit, targets.liquidity_sweep.float(),
            reduction=reduction,
        )
        losses["liquidity_sweep"] = l
        weighted["liquidity_sweep"] = config.w_liquidity_sweep * l

    # ── Gap fill (BCE) ──
    if targets.gap_fill is not None:
        l = F.binary_cross_entropy_with_logits(
            outputs.gap_fill_logit, targets.gap_fill.float(),
            reduction=reduction,
        )
        losses["gap_fill"] = l
        weighted["gap_fill"] = config.w_gap_fill * l

    total = sum(weighted.values()) if weighted else torch.tensor(
        0.0, device=outputs.wall_break_logits.device,
    )
    metrics = {f"loss_{k}": float(v.detach()) for k, v in losses.items()}
    metrics["loss_total_short_term"] = float(total.detach())
    return total, metrics
