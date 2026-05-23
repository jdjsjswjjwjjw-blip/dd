"""
modules/price_cycle/cycle_encoder.py
─────────────────────────────────────
Temporal Convolutional Network (TCN) encoder for macro-scale price cycles.

Why TCN (not LSTM/Transformer) for the cycle backbone?
    - CAUSAL by construction (dilated causal convolutions) → zero look-ahead
    - Large receptive field via exponential dilation (6 levels → ~127 bars)
    - Parallelizable → faster training than recurrent nets
    - Stable gradients (residual connections + weight norm)

Reference:
    Bai, Kolter, Koltun (2018): "An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling"

Multi-resolution extension:
    Process the sequence at multiple resolutions (1×, 4×, 16× downsampled)
    → captures both fine swings and broad cycles simultaneously.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CycleEncoderConfig


# ════════════════════════════════════════════════════════════════════════════
# Causal convolution building block
# ════════════════════════════════════════════════════════════════════════════


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding only.

    Output at time t depends ONLY on inputs at times <= t.
    This GUARANTEES no look-ahead leakage.
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, dilation: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T) → (B, C_out, T)."""
        # Left-pad only (causal)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual TCN block: 2× (CausalConv + WeightNorm + GELU + Dropout)."""

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, dilation: int, dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation).conv
        )
        self.pad1 = (kernel_size - 1) * dilation

        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation).conv
        )
        self.pad2 = (kernel_size - 1) * dilation

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(1, out_channels)  # = LayerNorm over channels
        self.norm2 = nn.GroupNorm(1, out_channels)

        # Residual projection if channel count changes
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T) → (B, C_out, T)."""
        residual = x if self.downsample is None else self.downsample(x)

        # Conv 1
        out = F.pad(x, (self.pad1, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Conv 2
        out = F.pad(out, (self.pad2, 0))
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + residual


class TCNEncoder(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""

    def __init__(
        self, n_input: int, n_channels: int,
        kernel_size: int, n_levels: int, dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for level in range(n_levels):
            dilation = 2 ** level
            in_ch = n_input if level == 0 else n_channels
            layers.append(TemporalBlock(
                in_ch, n_channels, kernel_size, dilation, dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n_input, T) → (B, n_channels, T)."""
        return self.network(x)


# ════════════════════════════════════════════════════════════════════════════
# Multi-resolution cycle encoder
# ════════════════════════════════════════════════════════════════════════════


class CycleEncoder(nn.Module):
    """Multi-resolution TCN encoder for macro price cycles.

    Processes the bar sequence at multiple resolutions:
        1×  → fine detail (swings)
        4×  → medium cycles
        16× → broad market cycles

    Each resolution gets its own TCN; outputs are fused.
    """

    def __init__(self, config: CycleEncoderConfig):
        super().__init__()
        self.config = config

        if config.use_multi_resolution:
            self.resolutions = config.resolution_factors
        else:
            self.resolutions = (1,)

        # One TCN per resolution
        self.encoders = nn.ModuleDict({
            f"res_{r}": TCNEncoder(
                n_input=config.n_input_features,
                n_channels=config.n_channels,
                kernel_size=config.kernel_size,
                n_levels=config.n_levels,
                dropout=config.dropout,
            )
            for r in self.resolutions
        })

        # Fusion of resolution outputs → embed_dim
        fused_dim = config.n_channels * len(self.resolutions)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Causal downsample: average-pool by factor.

        x : (B, C, T) → (B, C, T//factor)
        """
        if factor == 1:
            return x
        # Causal: pad at front so each pooled bin uses only past
        T = x.size(-1)
        pad = (factor - T % factor) % factor
        if pad:
            x = F.pad(x, (pad, 0))
        return F.avg_pool1d(x, kernel_size=factor, stride=factor)

    def forward(self, bar_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        bar_features : (B, T, n_input_features)

        Returns
        -------
        cycle_embedding : (B, embed_dim) — final-bar embedding
        """
        B, T, F_in = bar_features.shape
        # → (B, C, T) for conv
        x = bar_features.transpose(1, 2)

        resolution_outputs = []
        for r in self.resolutions:
            x_r = self._downsample(x, r)
            encoded = self.encoders[f"res_{r}"](x_r)   # (B, n_channels, T_r)
            # Take last timestep (causal → most recent info)
            last = encoded[:, :, -1]                    # (B, n_channels)
            resolution_outputs.append(last)

        fused = torch.cat(resolution_outputs, dim=-1)   # (B, n_channels × n_res)
        embedding = self.fusion(fused)                  # (B, embed_dim)
        return embedding

    def forward_sequence(self, bar_features: torch.Tensor) -> torch.Tensor:
        """Like forward() but returns per-bar embeddings (for sequence tasks).

        Returns
        -------
        (B, T, embed_dim) — embedding at each bar (resolution 1× only)
        """
        B, T, _ = bar_features.shape
        x = bar_features.transpose(1, 2)
        encoded = self.encoders[f"res_{self.resolutions[0]}"](x)  # (B, C, T)
        encoded = encoded.transpose(1, 2)  # (B, T, C)
        # project each timestep
        if self.config.n_channels != self.config.embed_dim:
            # pad with zeros from other resolutions (use res-1 only here)
            pad = torch.zeros(
                B, T, self.config.n_channels * (len(self.resolutions) - 1),
                device=encoded.device, dtype=encoded.dtype,
            )
            encoded = torch.cat([encoded, pad], dim=-1)
        return self.fusion(encoded)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
