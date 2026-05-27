"""
modules/trading_intel/anti_collapse/simplex_etf.py
═══════════════════════════════════════════════════════════════════════════════
Fixed Simplex ETF (Equiangular Tight Frame) Classifier — anti-collapse head.

Replaces a standard nn.Linear classification head whose weights are FREE
to collapse (predicting majority class on imbalanced data — exactly what
happened to our direction head: 100% UP on validation).

Theory (Papyan, Han, Donoho 2020, "Neural Collapse"):
  At terminal training, the last-layer classifier weights and the class
  means converge to a Simplex ETF — K vectors in K-1 dimensional space,
  equiangular and pointing to the K vertices of a regular simplex.

  Key insight: if we FREEZE the classifier weights as a Simplex ETF from
  the start (no gradients), the encoder is FORCED to produce class-
  separable features, and the optimal-shortcut majority predictor
  becomes mathematically impossible.

Math:
  • For K classes in D >= K-1 dimensions, the Simplex ETF matrix M ∈ R^{D×K}
    satisfies M^T M = (K / (K-1)) (I_K − (1/K) 1 1^T)
  • Cosine angle between any two class vectors = -1/(K-1)
  • Each vector has unit norm: ||M_:, k|| = 1

Loss: Dot-Regression Loss (Yang et al. 2022, AllNC)
  L = sum_k (z·m_k − y_k)^2
  where y_k = 1 if k is the true class, else 0, and z is the (normalized)
  feature vector. Unlike CE, this loss has NO shortcut to majority class —
  the model must produce a feature aligned with the correct ETF vertex.

Reference:
  • Papyan, V., Han, X., Donoho, D. (2020). Prevalence of neural collapse
    during the terminal phase of deep learning training. PNAS.
  • Yang, Y. et al. (2022). Inducing neural collapse in imbalanced
    learning: do we really need a learnable classifier at the end of
    deep neural network? (AllNC). NeurIPS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_simplex_etf(num_classes: int, feature_dim: int) -> torch.Tensor:
    """Construct a fixed Simplex ETF matrix M ∈ R^{D×K} with K columns,
    each unit-norm, equiangular at cos = -1/(K-1).

    Implementation:
      1. Build the K×K ETF Gram-target G = (K/(K-1)) (I − (1/K) 1 1^T)
      2. Pick a random orthonormal frame U ∈ R^{D×K} (D >= K-1)
      3. Project onto ETF: M = U @ sqrt(G_eigendecomp)

    Equivalent and simpler: just sample K equiangular vectors in R^D
    via the closed-form construction below.
    """
    if feature_dim < num_classes - 1:
        raise ValueError(
            f"feature_dim={feature_dim} must be >= num_classes-1={num_classes - 1} "
            f"for Simplex ETF to exist"
        )

    K = num_classes
    # Construct K equiangular unit vectors in R^{K-1}
    # Standard construction: e_k − (1/K) 1, normalized
    # then embed into R^D via random orthonormal basis
    I = torch.eye(K)
    ones = torch.ones(K, K) / K
    P = I - ones                            # K×K projection to ETF subspace
    # P has rank K-1; its columns are NOT unit norm yet
    # Compute eigendecomposition: P = V Λ V^T, rank = K-1
    eigvals, eigvecs = torch.linalg.eigh(P)
    # Keep the K-1 nonzero eigenvalues
    # eigvals returned in ascending order
    pos_mask = eigvals > 1e-6
    L = eigvecs[:, pos_mask] * eigvals[pos_mask].clamp(min=1e-9).sqrt()
    # L is K×(K-1) with rows being the K simplex vertices in R^{K-1}
    # Normalize each row to unit length
    L = L / L.norm(dim=1, keepdim=True).clamp(min=1e-9)

    # Now embed into R^D via a random orthonormal frame
    if feature_dim == K - 1:
        M = L.T  # shape (K-1, K)
    else:
        # Random orthonormal frame Q ∈ R^{D × (K-1)}
        rng = torch.Generator().manual_seed(0)  # deterministic frame
        rand = torch.randn(feature_dim, K - 1, generator=rng)
        Q, _ = torch.linalg.qr(rand)  # D × (K-1) orthonormal
        M = Q @ L.T                    # D × K
    # Final unit-normalization per column (cleanup numerical drift)
    M = M / M.norm(dim=0, keepdim=True).clamp(min=1e-9)
    return M.contiguous()


@dataclass
class SimplexETFConfig:
    feature_dim: int = 64           # backbone embedding dim
    num_classes: int = 3            # 3-way direction (LONG/SHORT/NEUTRAL)
    normalize_features: bool = True # cosine-style — feature·anchor inner product


class SimplexETFClassifier(nn.Module):
    """A drop-in replacement for nn.Linear(feature_dim, num_classes) whose
    weight matrix is fixed to a Simplex ETF (not learnable).

    Args:
      feature_dim:        D — must be >= num_classes - 1
      num_classes:        K
      normalize_features: if True, L2-normalize the input features
                          before computing logits = features @ M.

    Use with DotRegressionLoss for guaranteed neural collapse.
    """

    def __init__(self, config: SimplexETFConfig | None = None):
        super().__init__()
        self.config = config or SimplexETFConfig()
        cfg = self.config
        M = _build_simplex_etf(cfg.num_classes, cfg.feature_dim)
        # Register as a buffer (saved with state_dict but NOT a Parameter,
        # so no gradients flow into it)
        self.register_buffer("etf_anchors", M)  # (D, K)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits = features @ etf_anchors.

        features: (B, D)
        returns:  (B, K)
        """
        if self.config.normalize_features:
            features = F.normalize(features, dim=-1)
        return features @ self.etf_anchors

    def num_parameters(self) -> int:
        # ETF anchors are buffers, not parameters — this classifier has 0
        # trainable params. The encoder must learn to align features with
        # the fixed anchors.
        return 0


def dot_regression_loss(
    features: torch.Tensor,        # (B, D)
    targets: torch.Tensor,         # (B,) int class labels in [0, K)
    etf_anchors: torch.Tensor,     # (D, K) the same M used in classifier
    *,
    ignore_index: int = -1,
    normalize_features: bool = True,
) -> torch.Tensor:
    """Dot-Regression Loss (Yang et al. 2022).

    For each sample:
      • Look up the target's anchor vector m_y ∈ R^D
      • Compute the dot product s = feat · m_y
      • Loss = (s - 1)^2 (penalize features that don't align with their anchor)

    Implicitly penalizes alignment with WRONG anchors too because the
    feature can only have a high inner product with one anchor at a time
    (anchors are orthogonal-ish: cos = -1/(K-1)).

    Returns scalar mean loss. Skips rows where targets == ignore_index.
    """
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return torch.zeros((), device=features.device, requires_grad=True)
    feats = features[valid_mask]
    tgt = targets[valid_mask].long()
    if normalize_features:
        feats = F.normalize(feats, dim=-1)
    # Gather the anchor for each target
    # etf_anchors is (D, K); we want (B, D) for each row
    target_anchors = etf_anchors.T[tgt]          # (B, D)
    similarity = (feats * target_anchors).sum(dim=-1)  # (B,)
    return ((similarity - 1.0) ** 2).mean()
