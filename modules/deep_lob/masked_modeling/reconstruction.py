"""
modules.deep_lob.masked_modeling.reconstruction
═══════════════════════════════════════════════════════════════════════════════
Reconstruction head + loss for the masked-modeling auxiliary task.

ReconstructionHead is a small MLP that maps encoder outputs back to the
original feature space. By design it's lightweight — if the head is too
powerful, gradient flows away from the encoder and the auxiliary task
stops contributing useful representation learning.

compute_recon_loss respects:
  1. The mask itself (loss only on positions we asked to reconstruct).
  2. The padding mask (no loss on padded positions even if the mask
     RNG happened to flag them — defensive).
  3. Per-feature scale via optional `feature_weights` so that, e.g.,
     the price dimension doesn't dominate the imbalance dimension when
     features are on different scales.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    """Small MLP that reconstructs the original order features.

    A two-layer MLP with GELU + LayerNorm. Keeping it shallow is a
    deliberate choice: a deep head would let the model "memorize" the
    reconstruction without forcing the encoder to learn good
    representations.
    """

    def __init__(
        self,
        encoder_dim: int,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """encoded: (..., encoder_dim) → (..., feature_dim).

        The leading dimensions are preserved — caller can pass
        (B, T, N, encoder_dim) and get (B, T, N, feature_dim) back.
        """
        x = self.fc1(encoded)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


def compute_recon_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
    order_masks: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position MSE on masked-AND-valid positions only.

    Parameters
    ──────────
    reconstructed   (B, T, N, D)  model output (same D as original)
    original        (B, T, N, D)  ground-truth features
    mask            (B, T, N)     bool — True where we asked to reconstruct
    order_masks     (B, T, N)     bool — True where the order is real
    feature_weights (D,) optional — per-feature loss multiplier; defaults
                                    to uniform 1.0 across all D features

    Returns
    ───────
    scalar tensor — average squared error over masked & valid positions,
    weighted by `feature_weights`. Returns 0.0 when no valid masked
    positions exist (avoids div-by-zero).
    """
    if reconstructed.shape != original.shape:
        raise ValueError(
            f"shape mismatch: reconstructed {tuple(reconstructed.shape)} "
            f"vs original {tuple(original.shape)}"
        )
    if mask.shape != order_masks.shape:
        raise ValueError(
            f"shape mismatch: mask {tuple(mask.shape)} "
            f"vs order_masks {tuple(order_masks.shape)}"
        )
    if reconstructed.shape[:3] != mask.shape:
        raise ValueError(
            f"shape mismatch: reconstructed lead {tuple(reconstructed.shape[:3])} "
            f"vs mask {tuple(mask.shape)}"
        )

    # Effective loss mask: requested-to-reconstruct AND valid order
    loss_mask = (mask & order_masks).to(reconstructed.dtype)  # (B, T, N)
    n_valid = loss_mask.sum()
    if n_valid.item() < 1.0:
        return reconstructed.new_zeros(())

    # Per-element squared error
    sq_err = (reconstructed - original) ** 2          # (B, T, N, D)

    # Optional per-feature weighting (default uniform)
    if feature_weights is not None:
        D = sq_err.shape[-1]
        if feature_weights.shape != (D,):
            raise ValueError(
                f"feature_weights shape must be ({D},), "
                f"got {tuple(feature_weights.shape)}"
            )
        w = feature_weights.to(sq_err.dtype).to(sq_err.device)
        sq_err = sq_err * w.view(1, 1, 1, D)

    # Average over feature dimension first, then masked-mean over (B, T, N)
    per_position = sq_err.mean(dim=-1)               # (B, T, N)
    return (per_position * loss_mask).sum() / n_valid
