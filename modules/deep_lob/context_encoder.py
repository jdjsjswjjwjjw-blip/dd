"""
modules/deep_lob/context_encoder.py
────────────────────────────────────
Encodes the 138 existing features (regime, simulators, Bell pairs, GHZ, ...)
into a context vector that fuses with the LOB representation.

Architecture:
    Optional per-group encoders → concat → MLP → output_dim

Groups (auto-detected by column prefix):
    - regime_*       (regime labels, regime_session interaction)
    - sim_*          (V19.2 simulators, 18 features)
    - wall_*         (wall depth, 8 features)
    - iceberg_*      (iceberg detection, 5 features)
    - bp_*           (Bell pairs, 9 features)
    - ghz_*          (GHZ states, 5 features)
    - zone_*, *_event (session mapping)
    - other          (residual features)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ContextEncoderConfig


class ContextEncoder(nn.Module):
    """Encodes the 138 context features into a context vector."""

    def __init__(self, config: ContextEncoderConfig):
        super().__init__()
        self.config = config

        if config.use_per_group_encoding:
            # Per-group encoders (assumes feature groups by index)
            # For simplicity, we use a single MLP with group-aware structure
            self.encoder = nn.Sequential(
                nn.Linear(config.n_input_features, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )
        else:
            self.encoder = nn.Linear(config.n_input_features, config.output_dim)

        # Input normalization (handle outliers)
        self.input_ln = nn.LayerNorm(config.n_input_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context : (B, n_input_features)

        Returns
        -------
        encoded : (B, output_dim)
        """
        # Handle NaN/Inf defensively
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip extreme values
        context = torch.clamp(context, min=-1e6, max=1e6)
        # Normalize
        context = self.input_ln(context)
        return self.encoder(context)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: book ↔ context.

    Book representation attends to context, and vice versa.
    Result: fused representation that captures both.
    """

    def __init__(
        self,
        book_dim: int,
        context_dim: int,
        output_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project both to common dim
        common_dim = max(book_dim, context_dim)
        if common_dim % n_heads != 0:
            common_dim = ((common_dim // n_heads) + 1) * n_heads

        self.book_proj = nn.Linear(book_dim, common_dim)
        self.context_proj = nn.Linear(context_dim, common_dim)

        # Cross-attention: book ↔ context (treated as sequence of length 1+1)
        self.attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ln_book = nn.LayerNorm(common_dim)
        self.ln_context = nn.LayerNorm(common_dim)

    def forward(
        self,
        book_emb: torch.Tensor,
        context_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        book_emb : (B, book_dim)
        context_emb : (B, context_dim)

        Returns
        -------
        fused : (B, output_dim)
        """
        B = book_emb.size(0)

        # Project to common dim, add sequence dim
        book_seq = self.ln_book(self.book_proj(book_emb)).unsqueeze(1)        # (B, 1, D)
        context_seq = self.ln_context(self.context_proj(context_emb)).unsqueeze(1)  # (B, 1, D)

        # Cross-attention: book queries context
        book_attended, _ = self.attn(
            query=book_seq,
            key=context_seq,
            value=context_seq,
        )
        # Cross-attention: context queries book
        context_attended, _ = self.attn(
            query=context_seq,
            key=book_seq,
            value=book_seq,
        )

        # Fuse: book + context_attended-on-book + book_attended-on-context
        fused_seq = (book_seq + book_attended + context_attended) / 3.0  # (B, 1, D)
        fused = fused_seq.squeeze(1)  # (B, D)

        return self.output_proj(fused)
