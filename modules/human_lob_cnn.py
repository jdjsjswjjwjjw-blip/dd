"""
modules/human_lob_cnn.py
═══════════════════════════════════════════════════════════════════════════════
Human-style LOB CNN — reads the order book the way a trader reads it.

The LOB tensor is shaped (B, C, T, P) — channels × time × price-level — which
maps naturally to a 2D image. A human trader's eye scans:
  • Vertically (across price levels) → walls, gaps, imbalance
  • Horizontally (across time) → pressure, refills, sweeps
  • Both together → absorption, accumulation, breakout setups

This CNN encodes those biases with carefully-shaped kernels:
  • Layer 1 (3, 3)  — local pattern (one snapshot × 3 levels)
  • Layer 2 (5, 3)  — short temporal pattern (5 snapshots × 3 levels)
  • Layer 3 (3, 5)  — local price range (3 snapshots × 5 levels)
  • Layer 4 (3, 3)  — global integration

Output: 32-dim embedding ready to concatenate with other features in the
hybrid model.

Designed to plug into the existing `HierarchicalLOBTransformer` as the CNN
branch, or stand alone for ablation studies.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HumanLOBCNNConfig:
    """Config for HumanLOBCNN. Defaults match the v2 channel layout (13 ch)
    and the standard lookback window (T=50, P=20)."""
    in_channels: int = 13          # matches N_LOB_CHANNELS_V2
    in_time: int = 50              # T snapshots per bar
    in_levels: int = 20            # P=20 levels (10 bid + 10 ask)
    embed_dim: int = 32            # output embedding size
    dropout: float = 0.2
    use_batch_norm: bool = True


class HumanLOBCNN(nn.Module):
    """4-layer CNN that produces a 32-dim embedding from a (T, P, C) LOB
    tensor.

    Input: x of shape (B, C, T, P) — note the channel-first ordering required
    by PyTorch conv layers. The caller is responsible for permuting from the
    natural (B, T, P, C) format produced by `build_lob_tensor_v2_for_bar`:

        x_chw = lob_tensor.permute(0, 3, 1, 2)  # (B, C, T, P)

    Output: (B, embed_dim)
    """

    def __init__(self, cfg: HumanLOBCNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or HumanLOBCNNConfig()

        norm = (nn.BatchNorm2d if self.cfg.use_batch_norm else nn.Identity)

        # ── Layer 1: local pattern (1 snapshot × 3 levels neighborhood) ──
        # Kernel (3, 3) with padding=1 → preserves (T, P)
        self.conv1 = nn.Conv2d(self.cfg.in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = norm(16) if self.cfg.use_batch_norm else nn.Identity()

        # ── Layer 2: short temporal pattern (5 snapshots × 3 levels) ──
        # Kernel (5, 3), padding (2, 1) → preserves (T, P)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1))
        self.bn2 = norm(32) if self.cfg.use_batch_norm else nn.Identity()
        # Halve T to capture longer-range temporal patterns
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # (T → T/2)

        # ── Layer 3: price-axis range (3 snapshots × 5 levels) ──
        # Kernel (3, 5), padding (1, 2) → preserves shape
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn3 = norm(64) if self.cfg.use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # (T/2 → T/4, P → P/2)

        # ── Layer 4: global integration ──
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = norm(128) if self.cfg.use_batch_norm else nn.Identity()

        # ── Global pool to (B, 128, 1, 1) → flatten → Linear → embedding ──
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.fc = nn.Linear(128, self.cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T, P) — channel-first.
        Returns:
            (B, embed_dim)
        """
        # Layer 1
        x = F.gelu(self.bn1(self.conv1(x)))

        # Layer 2
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Layer 3
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        # Layer 4
        x = F.gelu(self.bn4(self.conv4(x)))

        # Global pool + project
        x = self.global_pool(x).flatten(1)  # (B, 128)
        x = self.dropout(x)
        x = self.fc(x)                       # (B, embed_dim)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_human_lob_cnn(
    in_channels: int = 13,
    embed_dim: int = 32,
    in_time: int = 50,
    in_levels: int = 20,
    dropout: float = 0.2,
    use_batch_norm: bool = True,
) -> HumanLOBCNN:
    """Factory with the most common settings."""
    return HumanLOBCNN(HumanLOBCNNConfig(
        in_channels=in_channels,
        in_time=in_time,
        in_levels=in_levels,
        embed_dim=embed_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    ))
