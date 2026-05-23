"""
modules/price_cycle/cycle_model.py
───────────────────────────────────
Full Price Cycle Model — macro-scale deep learning.

Architecture:
    BarSequence → derived features (16-dim/bar)
        ↓
    Multi-resolution TCN encoder
        ↓
    Cycle embedding (64-dim)
        ↓
    5 multi-task heads:
        - phase           (Wyckoff: accumulation/markup/distribution/markdown)
        - trend_maturity  (young/mature/exhausted)
        - swing_direction (up/down/neutral)
        - reversal_proximity (regression: bars to likely reversal)
        - cycle_position  (regression: [0,1] within full cycle)

This is the MACRO counterpart to the micro-scale LOB Transformer.
The two combine in multi_scale_fusion.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PriceCycleConfig
from .cycle_encoder import CycleEncoder


# ════════════════════════════════════════════════════════════════════════════
# Output / target containers
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class CycleOutput:
    """Price Cycle Model predictions."""

    phase_logits: torch.Tensor          # (B, 4)
    maturity_logits: torch.Tensor       # (B, 3)
    swing_logits: torch.Tensor          # (B, 3)
    reversal_proximity: torch.Tensor    # (B,) — bars to reversal (softplus)
    cycle_position: torch.Tensor        # (B,) — [0, 1] (sigmoid)
    cycle_embedding: torch.Tensor       # (B, embed_dim) — for fusion

    def phase_probs(self) -> torch.Tensor:
        return F.softmax(self.phase_logits, dim=-1)

    def maturity_probs(self) -> torch.Tensor:
        return F.softmax(self.maturity_logits, dim=-1)

    def swing_probs(self) -> torch.Tensor:
        return F.softmax(self.swing_logits, dim=-1)

    def to_dict(self) -> dict:
        return {
            "phase_probs": self.phase_probs(),
            "maturity_probs": self.maturity_probs(),
            "swing_probs": self.swing_probs(),
            "reversal_proximity": self.reversal_proximity,
            "cycle_position": self.cycle_position,
        }


@dataclass
class CycleTargets:
    """Training targets — all optional (partial labels supported)."""

    phase: Optional[torch.Tensor] = None              # (B,) int 0-3
    maturity: Optional[torch.Tensor] = None           # (B,) int 0-2
    swing_direction: Optional[torch.Tensor] = None    # (B,) int 0-2
    reversal_proximity: Optional[torch.Tensor] = None # (B,) float >= 0
    cycle_position: Optional[torch.Tensor] = None     # (B,) float [0,1]


# ════════════════════════════════════════════════════════════════════════════
# Heads
# ════════════════════════════════════════════════════════════════════════════


class _Head(nn.Module):
    """2-layer MLP head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(in_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CycleHeads(nn.Module):
    """5 multi-task heads + loss computation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.shared_dim
        self.head_phase = _Head(D, config.phase_n_classes, config.dropout)
        self.head_maturity = _Head(D, config.maturity_n_classes, config.dropout)
        self.head_swing = _Head(D, config.swing_n_classes, config.dropout)
        self.head_reversal = _Head(D, 1, config.dropout)
        self.head_cycle_pos = _Head(D, 1, config.dropout)

    def forward(self, embedding: torch.Tensor) -> CycleOutput:
        return CycleOutput(
            phase_logits=self.head_phase(embedding),
            maturity_logits=self.head_maturity(embedding),
            swing_logits=self.head_swing(embedding),
            reversal_proximity=F.softplus(self.head_reversal(embedding).squeeze(-1)),
            cycle_position=torch.sigmoid(self.head_cycle_pos(embedding).squeeze(-1)),
            cycle_embedding=embedding,
        )

    def compute_loss(
        self, outputs: CycleOutput, targets: CycleTargets,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Weighted multi-task loss."""
        cfg = self.config
        losses: dict[str, torch.Tensor] = {}
        weighted: dict[str, torch.Tensor] = {}

        if targets.phase is not None:
            l = F.cross_entropy(outputs.phase_logits, targets.phase.long(), reduction=reduction)
            losses["phase"] = l
            weighted["phase"] = cfg.phase_weight * l

        if targets.maturity is not None:
            l = F.cross_entropy(outputs.maturity_logits, targets.maturity.long(), reduction=reduction)
            losses["maturity"] = l
            weighted["maturity"] = cfg.maturity_weight * l

        if targets.swing_direction is not None:
            l = F.cross_entropy(outputs.swing_logits, targets.swing_direction.long(), reduction=reduction)
            losses["swing"] = l
            weighted["swing"] = cfg.swing_weight * l

        if targets.reversal_proximity is not None:
            l = F.smooth_l1_loss(outputs.reversal_proximity, targets.reversal_proximity, reduction=reduction)
            losses["reversal"] = l
            weighted["reversal"] = cfg.reversal_weight * l

        if targets.cycle_position is not None:
            l = F.mse_loss(outputs.cycle_position, targets.cycle_position, reduction=reduction)
            losses["cycle_position"] = l
            weighted["cycle_position"] = cfg.cycle_position_weight * l

        if not weighted:
            raise ValueError("No targets provided — at least 1 task needed")

        total = sum(weighted.values())
        result = {"total": total}
        result.update(losses)
        for k, v in weighted.items():
            result[f"weighted_{k}"] = v
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════════
# Full model
# ════════════════════════════════════════════════════════════════════════════


class PriceCycleModel(nn.Module):
    """Macro-scale deep learning model — the price-cycle counterpart
    to the micro-scale LOB Transformer."""

    def __init__(self, config: PriceCycleConfig):
        super().__init__()
        self.config = config
        self.encoder = CycleEncoder(config.encoder)
        self.heads = CycleHeads(config.heads)

    def forward(self, bar_features: torch.Tensor) -> CycleOutput:
        """
        Parameters
        ----------
        bar_features : (B, T, n_input_features) — 16-dim/bar derived features

        Returns
        -------
        CycleOutput
        """
        embedding = self.encoder(bar_features)
        return self.heads(embedding)

    def predict(self, bar_features: torch.Tensor) -> dict:
        """Inference (eval + no_grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(bar_features).to_dict()

    def get_cycle_embedding(self, bar_features: torch.Tensor) -> torch.Tensor:
        """Macro embedding للـ multi-scale fusion."""
        self.eval()
        with torch.no_grad():
            return self.encoder(bar_features)

    def count_parameters(self) -> dict[str, int]:
        return {
            "encoder": self.encoder.count_parameters(),
            "heads": self.heads.count_parameters(),
            "total": sum(p.numel() for p in self.parameters()),
            "receptive_field": self.config.encoder.receptive_field,
        }

    def save_checkpoint(self, path: str, extra_state: Optional[dict] = None):
        from dataclasses import asdict
        state = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "model_name": self.config.model_name,
            "version": self.config.version,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "PriceCycleModel":
        from .config import (
            SwingConfig, PhaseConfig, FractalConfig,
            CycleEncoderConfig, CycleHeadsConfig, MultiScaleFusionConfig,
        )
        state = torch.load(path, map_location=map_location, weights_only=False)
        cd = state["config"]
        if isinstance(cd, PriceCycleConfig):
            config = cd
        else:
            config = PriceCycleConfig(
                swing=SwingConfig(**cd["swing"]),
                phase=PhaseConfig(**cd["phase"]),
                fractal=FractalConfig(**cd["fractal"]),
                encoder=CycleEncoderConfig(**cd["encoder"]),
                heads=CycleHeadsConfig(**cd["heads"]),
                fusion=MultiScaleFusionConfig(**cd["fusion"]),
                model_name=cd.get("model_name", "loaded"),
                version=cd.get("version", 1),
            )
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model
