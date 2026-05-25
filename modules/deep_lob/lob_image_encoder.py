"""
modules/deep_lob/lob_image_encoder.py
─────────────────────────────────────
2D CNN encoder for the 9-channel LOB tensor.

Input:  (B, T=50, P=20, C=9) — time × price-level × channels
Output: (B, lob_embed_dim) — single embedding per bar

Architecture (DeepLOB-inspired but compact):

    Permute   : (B, T, P, C) → (B, C, T, P)
    Conv2d    : C → 32, kernel (3, 3), padding (1, 1)  [time × price local]
    BatchNorm + GELU
    Conv2d    : 32 → 64, kernel (3, 3), padding (1, 1)
    BatchNorm + GELU
    MaxPool2d : kernel (2, 2)        — reduce both T and P
    Conv2d    : 64 → 96, kernel (3, 3), padding (1, 1)
    BatchNorm + GELU
    AdaptiveAvgPool2d : (1, 1)
    Flatten → Linear → lob_embed_dim
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LOBImageEncoderConfig:
    in_channels: int = 9
    lob_embed_dim: int = 32
    hidden_channels: tuple[int, int, int] = (32, 64, 96)
    dropout: float = 0.1


class LOBImageEncoder(nn.Module):
    """2D CNN on (T, P, C) LOB tensors → single bar embedding."""

    def __init__(self, config: LOBImageEncoderConfig):
        super().__init__()
        self.config = config
        c0, c1, c2 = config.hidden_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(config.in_channels, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(c2, config.lob_embed_dim),
            nn.LayerNorm(config.lob_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, lob_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lob_tensor : (B, T, P, C) float

        Returns
        -------
        embedding : (B, lob_embed_dim)
        """
        # Defensive sanitize (model must not NaN-propagate)
        lob = torch.nan_to_num(lob_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        # Channel mismatch — explicit guard with actionable message
        if lob.shape[-1] != self.config.in_channels:
            raise ValueError(
                f"LOB tensor has {lob.shape[-1]} channels (shape={tuple(lob.shape)}), "
                f"encoder expects {self.config.in_channels}. "
                f"Likely cause: parquet was built before the 9-channel upgrade. "
                f"Re-run prepare_day_trading on the cleaned MBO/MBP."
            )
        # (B, T, P, C) → (B, C, T, P)
        x = lob.permute(0, 3, 1, 2).contiguous()
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.global_pool(x).flatten(1)   # (B, c2)
        return self.proj(x)

    @property
    def output_dim(self) -> int:
        return self.config.lob_embed_dim
