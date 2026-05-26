"""
modules/hybrid_model.py
═══════════════════════════════════════════════════════════════════════════════
Hybrid model — fuses three feature sources into a single trading decision:

  1. day_trade hand-crafted features (~135 dims)
     The hand-tuned microstructure features that already give 78.3% hit
     rate via rules. These are the "expert knowledge" channel.

  2. SSL embeddings (161 dims by default)
     The 161-dim representation from the pretrained LOB transformer +
     price cycle model. These encode patterns the rules don't explicitly
     name. Empty/zero-filled rows are masked by `embedding_valid`.

  3. Human-LOB CNN embedding (32 dims, OPTIONAL)
     The CNN-based "human reading" of the order book (modules/human_lob_cnn.py).
     This is the channel the proposal asks us to add for stronger entry timing.
     Optional because the CNN needs to be trained first — the hybrid model
     gracefully falls back to (1)+(2) without it.

The fusion network is a small MLP — interpretable, fast, doesn't need GPU at
inference. The architecture deliberately mirrors what XGBoost would do, so
porting to xgboost.train(...) on a richer server later is a small change.

Output heads:
  • event_logit   — binary: is this bar a tradeable event? (replaces rule's
                    event_flag with a learned version)
  • direction_logits — 3-way: LONG / SHORT / NEUTRAL (replaces rule's
                    event_direction with a learned version)
  • confidence    — scalar in [0, 1] — soft mask for sizing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridConfig:
    """Hybrid model dimensions + regularization."""

    # Input dims
    daytrade_feature_dim: int = 135        # hand-crafted columns
    ssl_embed_dim: int = 161               # LOB + cycle SSL combined
    cnn_embed_dim: int = 32                # human_lob_cnn output (0 disables)
    # Architecture
    hidden_dim: int = 256
    dropout: float = 0.3
    n_layers: int = 3                       # 2-4 makes sense
    # Heads
    use_event_head: bool = True
    use_direction_head: bool = True
    use_confidence_head: bool = True
    # Loss weights
    w_event: float = 1.0
    w_direction: float = 1.5
    w_confidence: float = 0.5

    @property
    def total_input_dim(self) -> int:
        return self.daytrade_feature_dim + self.ssl_embed_dim + self.cnn_embed_dim


@dataclass
class HybridOutput:
    """Forward pass output."""
    event_logit: Optional[torch.Tensor] = None     # (B,)
    direction_logits: Optional[torch.Tensor] = None  # (B, 3)
    confidence_logit: Optional[torch.Tensor] = None  # (B,)

    def event_prob(self) -> Optional[torch.Tensor]:
        return torch.sigmoid(self.event_logit) if self.event_logit is not None else None

    def direction_probs(self) -> Optional[torch.Tensor]:
        if self.direction_logits is None:
            return None
        return F.softmax(self.direction_logits, dim=-1)

    def confidence(self) -> Optional[torch.Tensor]:
        return torch.sigmoid(self.confidence_logit) if self.confidence_logit is not None else None


@dataclass
class HybridTargets:
    """Training targets — all optional for partial-label batches."""
    event_flag: Optional[torch.Tensor] = None    # (B,) float in {0, 1}
    direction: Optional[torch.Tensor] = None     # (B,) int in {0, 1, 2} (LONG/SHORT/NEUTRAL)
    confidence: Optional[torch.Tensor] = None    # (B,) float in [0, 1]


def _make_backbone(input_dim: int, hidden_dim: int, n_layers: int,
                   dropout: float) -> nn.Sequential:
    """Build a stack of (Linear → GELU → Dropout) blocks."""
    layers: list[nn.Module] = []
    in_d = input_dim
    for _ in range(n_layers):
        layers += [
            nn.Linear(in_d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        in_d = hidden_dim
    return nn.Sequential(*layers)


class HybridModel(nn.Module):
    """Fuse day_trade features + SSL embeddings + (optional) CNN embedding
    into event / direction / confidence predictions."""

    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()
        cfg = self.config

        if cfg.total_input_dim <= 0:
            raise ValueError("All input dims are zero — nothing to fuse")

        self.backbone = _make_backbone(
            cfg.total_input_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout,
        )

        h = cfg.hidden_dim
        self.event_head = nn.Linear(h, 1) if cfg.use_event_head else None
        self.direction_head = nn.Linear(h, 3) if cfg.use_direction_head else None
        self.confidence_head = nn.Linear(h, 1) if cfg.use_confidence_head else None

        if all(x is None for x in (self.event_head, self.direction_head, self.confidence_head)):
            raise ValueError("All heads disabled — nothing to predict")

    def forward(
        self,
        daytrade_features: torch.Tensor,           # (B, daytrade_feature_dim)
        ssl_embedding: torch.Tensor,                # (B, ssl_embed_dim)
        cnn_embedding: Optional[torch.Tensor] = None,  # (B, cnn_embed_dim) or None
    ) -> HybridOutput:
        cfg = self.config

        # Verify shapes
        B = daytrade_features.shape[0]
        assert daytrade_features.shape == (B, cfg.daytrade_feature_dim), \
            f"daytrade_features shape mismatch: {tuple(daytrade_features.shape)}"
        assert ssl_embedding.shape == (B, cfg.ssl_embed_dim), \
            f"ssl_embedding shape mismatch: {tuple(ssl_embedding.shape)}"

        # CNN embedding optional
        if cfg.cnn_embed_dim > 0:
            if cnn_embedding is None:
                # Use zeros — the model can learn to ignore this channel during
                # CNN warm-up. This is the documented graceful-fallback behavior.
                cnn_embedding = torch.zeros(
                    B, cfg.cnn_embed_dim, dtype=daytrade_features.dtype,
                    device=daytrade_features.device,
                )
            else:
                assert cnn_embedding.shape == (B, cfg.cnn_embed_dim), \
                    f"cnn_embedding shape mismatch: {tuple(cnn_embedding.shape)}"
            fused = torch.cat([daytrade_features, ssl_embedding, cnn_embedding], dim=-1)
        else:
            fused = torch.cat([daytrade_features, ssl_embedding], dim=-1)

        h = self.backbone(fused)

        return HybridOutput(
            event_logit=self.event_head(h).squeeze(-1) if self.event_head is not None else None,
            direction_logits=self.direction_head(h) if self.direction_head is not None else None,
            confidence_logit=self.confidence_head(h).squeeze(-1) if self.confidence_head is not None else None,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_hybrid_loss(
    outputs: HybridOutput, targets: HybridTargets, config: HybridConfig,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-head losses. Skips heads with target=None."""
    losses: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}

    if outputs.event_logit is not None and targets.event_flag is not None:
        l = F.binary_cross_entropy_with_logits(
            outputs.event_logit, targets.event_flag.float(), reduction=reduction,
        )
        losses["event"] = l
        weighted["event"] = config.w_event * l

    if outputs.direction_logits is not None and targets.direction is not None:
        l = F.cross_entropy(
            outputs.direction_logits, targets.direction.long(), reduction=reduction,
        )
        losses["direction"] = l
        weighted["direction"] = config.w_direction * l

    if outputs.confidence_logit is not None and targets.confidence is not None:
        l = F.binary_cross_entropy_with_logits(
            outputs.confidence_logit, targets.confidence.float(), reduction=reduction,
        )
        losses["confidence"] = l
        weighted["confidence"] = config.w_confidence * l

    if not weighted:
        # No targets matched any head — return zero loss with the right device
        ref = (outputs.event_logit if outputs.event_logit is not None
               else outputs.direction_logits if outputs.direction_logits is not None
               else outputs.confidence_logit)
        total = torch.tensor(0.0, device=ref.device, requires_grad=True)
    else:
        total = sum(weighted.values())

    metrics = {f"loss_{k}": float(v.detach()) for k, v in losses.items()}
    metrics["loss_total_hybrid"] = float(total.detach()) if total.requires_grad else float(total)
    return total, metrics
