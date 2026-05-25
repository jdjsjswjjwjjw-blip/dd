"""
modules/deep_lob/order_embedder.py
───────────────────────────────────
Order-level embedding layer.

Converts raw order features (7-dim) to dense embeddings (embed_dim).

Architecture:
    Each order has 7 features:
        [0] side       — discrete (B/S)              → 8-dim embedding
        [1] type       — discrete (T/A/C/M)          → 8-dim embedding
        [2] log_size   — continuous                  → 4-dim linear
        [3] price_dist — continuous (ticks)          → 4-dim linear
        [4] time_off   — continuous (ms)             → 4-dim linear + sinusoidal
        [5] venue      — discrete (0-7)              → 2-dim embedding (small)
        [6] has_id     — binary indicator            → 2-dim linear

    Concat → projection → embed_dim (default 32)

Includes layer normalization + dropout للـ regularization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import OrderEmbedderConfig


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal encoding للـ time offset (similar to positional encoding).

    Captures multi-scale temporal patterns:
        - millisecond-scale (HFT)
        - second-scale (institutional)
        - minute-scale (slow flows)
    """

    def __init__(self, dim: int = 8, max_time_ms: float = 60_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEncoding dim must be even, got {dim}")
        self.dim = dim
        self.max_time = max_time_ms

        # Multiple frequency scales (logarithmic spread)
        freqs = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            -(math.log(max_time_ms) / dim)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, time_ms: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        time_ms : (..., ) float tensor

        Returns
        -------
        encoding : (..., dim) float tensor
        """
        # (..., 1) * (dim/2,) → (..., dim/2)
        angles = time_ms.unsqueeze(-1) * self.freqs
        # interleave sin and cos
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        # interleave: [sin_0, cos_0, sin_1, cos_1, ...]
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)
        return encoding.flatten(start_dim=-2)


class OrderEmbedder(nn.Module):
    """Embeds order feature vectors to dense embeddings.

    Architecture:
        7 input features → per-feature encoders → concat → projection → embed_dim

    Per-feature encoding choices are critical:
        - Discrete features use nn.Embedding (learnable lookup)
        - Continuous features use linear + activation
        - Time uses sinusoidal encoding for multi-scale awareness
    """

    def __init__(self, config: OrderEmbedderConfig):
        super().__init__()
        self.config = config
        D = config.embed_dim

        # Sub-dimension allocation
        D_side = max(4, D // 8)         # 4 default
        D_type = max(4, D // 8)         # 4 default
        D_size = max(4, D // 8)         # 4 default
        D_price = max(4, D // 8)        # 4 default
        D_time = max(8, D // 4)         # 8 default (more for time)
        D_venue = max(2, D // 16)       # 2 default
        D_misc = max(2, D // 16)        # 2 default

        # Discrete embeddings
        self.side_embed = nn.Embedding(config.side_vocab_size, D_side)
        self.type_embed = nn.Embedding(config.type_vocab_size, D_type)
        self.venue_embed = nn.Embedding(config.venue_vocab_size, D_venue)

        # Continuous projections
        self.size_proj = nn.Linear(1, D_size)
        self.price_proj = nn.Linear(1, D_price)
        self.misc_proj = nn.Linear(1, D_misc)  # has_id

        # Time: sinusoidal + linear projection
        self.time_encoding = SinusoidalTimeEncoding(dim=D_time)

        # Concatenated dim
        concat_dim = D_side + D_type + D_size + D_price + D_time + D_venue + D_misc

        # Final projection to embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """Custom initialization للـ stable training."""
        nn.init.normal_(self.side_embed.weight, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.venue_embed.weight, std=0.02)
        for proj in [self.size_proj, self.price_proj, self.misc_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        order_features: torch.Tensor,
        order_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        order_features : (B, N, 7) — batch, n_orders, 7 features
            [side, type, log_size, price_dist, time_off_ms, venue, has_id]
        order_mask : (B, N) bool   — True for valid orders, False for padding

        Returns
        -------
        embeddings : (B, N, embed_dim)
        """
        # NaN/Inf sanitization: NaN-cast-to-long produces large garbage ints
        # on CUDA, then clamp(0, V-1) silently maps to class 0 ("BUY").
        # Replace NaN/Inf with 0 BEFORE the cast so corruption is visible
        # (corrupted rows become deterministic padding, not a fake "BUY").
        order_features = torch.nan_to_num(order_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Split features
        sides = order_features[..., 0].long().clamp(0, self.config.side_vocab_size - 1)
        types = order_features[..., 1].long().clamp(0, self.config.type_vocab_size - 1)
        log_sizes = order_features[..., 2:3]            # (B, N, 1)
        price_dists = order_features[..., 3:4]          # (B, N, 1)
        time_offs = order_features[..., 4]              # (B, N)
        venues = order_features[..., 5].long().clamp(0, self.config.venue_vocab_size - 1)
        has_ids = order_features[..., 6:7]              # (B, N, 1)

        # Embeddings
        e_side = self.side_embed(sides)                 # (B, N, D_side)
        e_type = self.type_embed(types)                 # (B, N, D_type)
        e_venue = self.venue_embed(venues)              # (B, N, D_venue)

        # Continuous projections
        e_size = self.size_proj(log_sizes * self.config.size_log_scale)
        e_price = self.price_proj(price_dists)
        e_misc = self.misc_proj(has_ids)

        # Time encoding (multi-scale sinusoidal)
        e_time = self.time_encoding(time_offs / self.config.time_scale_ms)

        # Concatenate
        concat = torch.cat(
            [e_side, e_type, e_size, e_price, e_time, e_venue, e_misc],
            dim=-1,
        )

        # Project to embed_dim
        embeddings = self.output_proj(concat)

        # Zero out padding positions (defense-in-depth)
        if order_mask is not None:
            embeddings = embeddings * order_mask.unsqueeze(-1).float()

        return embeddings

    @property
    def output_dim(self) -> int:
        return self.config.embed_dim
