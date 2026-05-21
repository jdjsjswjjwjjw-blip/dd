"""
modules/deep_lob/transformer_blocks.py
───────────────────────────────────────
Transformer building blocks للـ order-level attention.

Components:
    - LearnedPositionalEmbedding: learned position embeddings up to max_orders
    - SinusoidalPositionalEmbedding: classical Vaswani-style
    - LOBTransformerLayer: multi-head self-attention + FFN (pre-norm)
    - LOBTransformerEncoder: stack of layers

Design choices (informed by recent transformer research):
    - Pre-LayerNorm (more stable than post-LN, Xiong et al. 2020)
    - GELU activation (Hendrycks & Gimpel 2016)
    - Padding-aware attention (proper masking)
    - Optional flash-attention support (when available)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


# ════════════════════════════════════════════════════════════════════════════
# Positional encodings
# ════════════════════════════════════════════════════════════════════════════


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (BERT-style).

    Pros: adapts to data
    Cons: bounded by max_orders, doesn't extrapolate
    """

    def __init__(self, max_positions: int, embed_dim: int, init_std: float = 0.02):
        super().__init__()
        self.embed = nn.Embedding(max_positions, embed_dim)
        nn.init.normal_(self.embed.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, D)
        Returns: x + positional_embeddings
        """
        B, N, D = x.shape
        positions = torch.arange(N, device=x.device).expand(B, N)
        return x + self.embed(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (Vaswani 2017).

    Pros: extrapolates, no learnable params
    Cons: fixed shape
    """

    def __init__(self, max_positions: int, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        pe = torch.zeros(max_positions, embed_dim)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_positions, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ════════════════════════════════════════════════════════════════════════════
# Transformer layer (pre-norm)
# ════════════════════════════════════════════════════════════════════════════


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention مع proper padding masking.

    Standard scaled dot-product attention:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim must be divisible by n_heads")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # For attention weight extraction (pattern discovery)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D)
        key_padding_mask : (B, N) bool — True for VALID positions
        return_attention : if True, store attention weights for inspection

        Returns
        -------
        output : (B, N, D)
        """
        B, N, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)

        # Mask padding tokens
        if key_padding_mask is not None:
            # Mask is True for valid; invert to mask out invalid
            invalid_mask = ~key_padding_mask
            # (B, 1, 1, N) → broadcast to (B, n_heads, N, N)
            invalid_mask = invalid_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(invalid_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        if return_attention:
            # Store mean across heads للـ inspection
            self._last_attn_weights = attn.detach().mean(dim=1)  # (B, N, N)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.dropout(out)

        return out

    def get_last_attention(self) -> Optional[torch.Tensor]:
        """Returns last computed attention weights (for pattern discovery)."""
        return self._last_attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class LOBTransformerLayer(nn.Module):
    """Single transformer layer (pre-norm + residual)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.feedforward_dim,
            dropout=config.dropout,
            activation=config.activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        # Pre-norm: x = x + attn(ln(x)); x = x + ffn(ln(x))
        x = x + self.attn(self.ln1(x), key_padding_mask, return_attention)
        x = x + self.ffn(self.ln2(x))
        return x


# ════════════════════════════════════════════════════════════════════════════
# Encoder (stack)
# ════════════════════════════════════════════════════════════════════════════


class LOBTransformerEncoder(nn.Module):
    """Stack of transformer layers + positional encoding + final layer norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Positional encoding
        if config.positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(
                max_positions=config.max_orders_per_bar,
                embed_dim=config.embed_dim,
                init_std=config.init_std,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEmbedding(
                max_positions=config.max_orders_per_bar,
                embed_dim=config.embed_dim,
            )

        # Layers
        self.layers = nn.ModuleList([
            LOBTransformerLayer(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(config.embed_dim)

        # Input dropout
        self.input_dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """
        Parameters
        ----------
        x : (B, N, embed_dim) order embeddings
        key_padding_mask : (B, N) bool — True for valid
        return_attentions : if True, return list of attention weights per layer

        Returns
        -------
        output : (B, N, embed_dim)
        attentions : list of (B, N, N) tensors, one per layer (or None)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.input_dropout(x)

        attentions = [] if return_attentions else None

        for layer in self.layers:
            x = layer(x, key_padding_mask, return_attention=return_attentions)
            if return_attentions:
                attentions.append(layer.attn.get_last_attention())

        x = self.ln_final(x)

        # Zero out padding (clean output)
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1).float()

        return x, attentions

    def count_parameters(self) -> dict[str, int]:
        """Parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "n_layers": self.config.n_layers,
            "embed_dim": self.config.embed_dim,
        }
