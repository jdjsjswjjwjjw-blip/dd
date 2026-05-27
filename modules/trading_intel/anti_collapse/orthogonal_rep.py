"""
modules/trading_intel/anti_collapse/orthogonal_rep.py
═══════════════════════════════════════════════════════════════════════════════
Orthogonal Representation Learning — anti-overlap penalty.

Addresses Mechanism #5 (Information Overlap with Rules): in the 6-month run,
SSL embeddings were rediscovering the same microstructure features that
day_trade rules already use (OBI, CVD, Hawkes, Kyle lambda). Result: SSL
on non-event bars = 52.4% (random) — no independent value.

Solution (Chinese OMoE school):
  Force the SSL embedding to be ORTHOGONAL to the rule features. This is a
  Gram-Schmidt projection at training time: penalize the encoder when its
  embedding has high cosine similarity with any rule feature vector.

Math:
  Let Z ∈ R^{B×D_emb} be the batch embedding, F ∈ R^{B×D_feat} the rule
  features for the same batch. After standardization:
    cosine_matrix = Z_norm @ F_norm.T / B   ∈ R^{D_emb × D_feat}

  The "rule-overlap" penalty is the squared Frobenius norm of cosine_matrix:
    L_orth = (1 / D_emb*D_feat) * ||Z_norm^T @ F_norm||_F^2 / B^2

  This pushes the encoder to find features ORTHOGONAL to the rules' span.

  At inference, the encoder still produces embeddings of full dimension D_emb,
  but they're guaranteed (during training) to live in the orthogonal
  complement of the rule subspace.

Reference:
  • Chinese OMoE (Orthogonal Mixture-of-Experts) literature
  • Stiefel manifold projection during training (Wang et al. 2020)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OrthogonalityPenaltyConfig:
    """How strongly to enforce orthogonality between SSL embedding and rules."""
    weight: float = 0.5            # multiplier on the orthogonality loss
    feature_eps: float = 1e-6      # numerical stability when normalizing
    detach_features: bool = True   # gradients should NOT flow into the rule features


def orthogonality_penalty(
    embedding: torch.Tensor,           # (B, D_emb)  — SSL/CNN embedding
    rule_features: torch.Tensor,       # (B, D_feat) — hand-crafted day_trade features
    config: OrthogonalityPenaltyConfig | None = None,
) -> torch.Tensor:
    """Compute the orthogonality penalty between embedding and rule features.

    Both inputs are mean-centered and scaled to unit Frobenius norm per
    batch before computing the cross-correlation. Returns a scalar in
    approximately [0, 1] that we want to drive to 0.
    """
    cfg = config or OrthogonalityPenaltyConfig()
    if cfg.detach_features:
        rule_features = rule_features.detach()

    # Standardize (z-score) per feature
    z_emb = embedding - embedding.mean(dim=0, keepdim=True)
    z_emb = z_emb / (z_emb.std(dim=0, keepdim=True).clamp(min=cfg.feature_eps))

    z_feat = rule_features - rule_features.mean(dim=0, keepdim=True)
    z_feat = z_feat / (z_feat.std(dim=0, keepdim=True).clamp(min=cfg.feature_eps))

    # Normalize batch dimension: ||Z||_F per column = sqrt(B)
    # After standardization, each column already has Var=1 so ||col||^2 = B
    # → cross-correlation = z_emb.T @ z_feat / B
    B = embedding.shape[0]
    cross_corr = (z_emb.T @ z_feat) / max(B, 1)    # (D_emb, D_feat)

    # Penalty = squared Frobenius norm of the cross-correlation matrix
    # Normalized by the matrix size so it stays in a reasonable range
    penalty = (cross_corr ** 2).sum() / cross_corr.numel()
    return cfg.weight * penalty


def gram_schmidt_project_out(
    embedding: torch.Tensor,        # (B, D_emb)
    rule_features: torch.Tensor,    # (B, D_feat)
    *,
    feature_eps: float = 1e-6,
) -> torch.Tensor:
    """Project the embedding ONTO the orthogonal complement of the rule
    features. Use this for INFERENCE-time hard orthogonalization (returns a
    new embedding that has zero linear correlation with rule features).

    For training, prefer `orthogonality_penalty` as a soft constraint —
    hard projection during training kills gradient flow through the encoder.

    Algorithm:
      1. Center and standardize rule features → F̃
      2. Compute the orthogonal projection matrix:
           P = I − F̃ (F̃^T F̃)^{-1} F̃^T
      3. Apply Z' = P @ Z
    """
    Z = embedding
    F_ = rule_features.detach()

    # Standardize rule features
    F_mean = F_.mean(dim=0, keepdim=True)
    F_std = F_.std(dim=0, keepdim=True).clamp(min=feature_eps)
    F_norm = (F_ - F_mean) / F_std            # (B, D_feat)

    # Build the projection in feature space: B x B is too large for big batches.
    # Cheaper formulation: project each column of Z by removing its
    # projection onto each column of F_norm.

    # Treat features as a (B, D_feat) matrix. The "residual" Z' is:
    #   Z' = Z − F_norm @ (F_norm^+ @ Z)
    # where F_norm^+ is the pseudoinverse.
    # Use a regularized pseudoinverse to avoid blow-up when D_feat > B.
    G = F_norm.T @ F_norm + feature_eps * torch.eye(
        F_norm.shape[1], device=F_norm.device,
    )
    F_pinv = torch.linalg.solve(G, F_norm.T)  # (D_feat, B)
    coeffs = F_pinv @ Z                       # (D_feat, D_emb)
    Z_residual = Z - F_norm @ coeffs          # (B, D_emb)
    return Z_residual


class OrthogonalRepresentationModule(nn.Module):
    """Convenience wrapper to add to a training loop.

    Usage:
        ortho = OrthogonalRepresentationModule(weight=0.5)
        loss_main = compute_main_loss(...)
        loss_ortho = ortho(embedding, rule_features)
        loss_total = loss_main + loss_ortho
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.config = OrthogonalityPenaltyConfig(weight=weight)

    def forward(
        self, embedding: torch.Tensor, rule_features: torch.Tensor,
    ) -> torch.Tensor:
        return orthogonality_penalty(embedding, rule_features, self.config)
