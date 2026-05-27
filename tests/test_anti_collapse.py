"""Tests for anti-collapse modules (Simplex ETF + Orthogonal Rep)."""
from __future__ import annotations

import math

import pytest
import torch

from modules.trading_intel.anti_collapse import (
    OrthogonalRepresentationModule,
    SimplexETFClassifier,
    SimplexETFConfig,
    dot_regression_loss,
    gram_schmidt_project_out,
    orthogonality_penalty,
)
from modules.trading_intel.anti_collapse.simplex_etf import _build_simplex_etf


# ════════════════════════════════════════════════════════════════════
# Simplex ETF construction
# ════════════════════════════════════════════════════════════════════
class TestSimplexETFConstruction:
    def test_columns_unit_norm(self):
        M = _build_simplex_etf(num_classes=3, feature_dim=4)
        norms = M.norm(dim=0)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_equiangular_inner_products(self):
        """Any pair of distinct anchors has cosine = -1/(K-1)."""
        K = 4
        M = _build_simplex_etf(num_classes=K, feature_dim=8)
        gram = M.T @ M
        # Diagonal = 1
        diag = torch.diagonal(gram)
        assert torch.allclose(diag, torch.ones(K), atol=1e-5)
        # Off-diagonal = -1/(K-1)
        expected_off_diag = -1.0 / (K - 1)
        for i in range(K):
            for j in range(K):
                if i != j:
                    assert abs(gram[i, j].item() - expected_off_diag) < 1e-4, (
                        f"gram[{i},{j}] = {gram[i, j].item():.4f}, "
                        f"expected {expected_off_diag:.4f}"
                    )

    def test_feature_dim_too_small_raises(self):
        # Need feature_dim >= num_classes - 1
        with pytest.raises(ValueError, match="must be >="):
            _build_simplex_etf(num_classes=10, feature_dim=5)

    def test_minimum_feature_dim_works(self):
        # Exactly K-1 should work
        M = _build_simplex_etf(num_classes=4, feature_dim=3)
        assert M.shape == (3, 4)

    def test_higher_feature_dim_works(self):
        # D much bigger than K
        M = _build_simplex_etf(num_classes=3, feature_dim=128)
        assert M.shape == (128, 3)


# ════════════════════════════════════════════════════════════════════
# Simplex ETF Classifier
# ════════════════════════════════════════════════════════════════════
class TestSimplexETFClassifier:
    def test_forward_shape(self):
        clf = SimplexETFClassifier(SimplexETFConfig(feature_dim=64, num_classes=3))
        feats = torch.randn(8, 64)
        logits = clf(feats)
        assert logits.shape == (8, 3)

    def test_zero_learnable_parameters(self):
        """The ETF anchors are buffers, not parameters. The classifier
        has zero trainable params — encoder bears the entire learning burden."""
        clf = SimplexETFClassifier(SimplexETFConfig())
        assert clf.num_parameters() == 0
        learnable = [p for p in clf.parameters() if p.requires_grad]
        assert learnable == []

    def test_buffer_persists_via_state_dict(self):
        clf1 = SimplexETFClassifier(SimplexETFConfig(feature_dim=8, num_classes=3))
        state = clf1.state_dict()
        # ETF anchors should be in state_dict (as a buffer)
        assert any("etf_anchors" in k for k in state.keys())
        # Recreate and load
        clf2 = SimplexETFClassifier(SimplexETFConfig(feature_dim=8, num_classes=3))
        clf2.load_state_dict(state)
        assert torch.allclose(clf1.etf_anchors, clf2.etf_anchors)


# ════════════════════════════════════════════════════════════════════
# Dot-Regression Loss + class-collapse resistance
# ════════════════════════════════════════════════════════════════════
class TestDotRegressionLoss:
    def test_perfect_alignment_zero_loss(self):
        """If features ARE the anchor for the true class, loss = 0."""
        clf = SimplexETFClassifier(SimplexETFConfig(feature_dim=4, num_classes=3))
        # Take anchor for class 0 as our features
        feats = clf.etf_anchors[:, 0].unsqueeze(0)   # (1, 4)
        targets = torch.tensor([0])
        loss = dot_regression_loss(
            feats, targets, clf.etf_anchors, normalize_features=False,
        )
        assert loss.item() < 1e-6

    def test_wrong_alignment_high_loss(self):
        """If features point AWAY from anchor, loss is high."""
        clf = SimplexETFClassifier(SimplexETFConfig(feature_dim=4, num_classes=3))
        # Features = anti-anchor for class 0
        feats = -clf.etf_anchors[:, 0].unsqueeze(0)
        targets = torch.tensor([0])
        loss = dot_regression_loss(
            feats, targets, clf.etf_anchors, normalize_features=False,
        )
        # similarity = -1, (-1 - 1)^2 = 4
        assert loss.item() == pytest.approx(4.0, abs=1e-4)

    def test_ignore_index_excludes_invalid(self):
        clf = SimplexETFClassifier(SimplexETFConfig(feature_dim=4, num_classes=3))
        feats = torch.randn(4, 4)
        targets = torch.tensor([0, -1, 1, -1])  # 2 valid rows
        loss = dot_regression_loss(feats, targets, clf.etf_anchors, ignore_index=-1)
        assert torch.isfinite(loss)
        # All invalid → zero
        targets_all_invalid = torch.full((4,), -1)
        loss_zero = dot_regression_loss(
            feats, targets_all_invalid, clf.etf_anchors, ignore_index=-1,
        )
        assert loss_zero.item() == 0.0

    def test_resists_majority_collapse(self):
        """Crucial property: training with Dot-Regression on imbalanced data
        cannot collapse to a constant predictor.

        Setup: 90% class 0, 10% class 1. A trainable encoder learns features.
        Run a few gradient steps and verify that features for class 1
        are aligned with anchor 1, not with anchor 0 (despite imbalance).
        """
        torch.manual_seed(0)
        D, K = 8, 3
        cfg = SimplexETFConfig(feature_dim=D, num_classes=K)
        clf = SimplexETFClassifier(cfg)

        # Tiny "encoder": single linear that maps a 1-d signal to D-d feat.
        # The encoder must learn class-separable features even under imbalance.
        encoder = torch.nn.Linear(2, D)
        optim = torch.optim.Adam(encoder.parameters(), lr=1e-2)

        # Imbalanced data: 200 samples, 90% class 0, 10% class 1
        # Class 0 input centered at (1, 0); class 1 at (-1, 0); class 2 at (0, 1)
        n_per_class = [180, 15, 5]
        Xs, Ys = [], []
        for cls_id, n in enumerate(n_per_class):
            mean = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])[cls_id]
            Xs.append(mean + 0.1 * torch.randn(n, 2))
            Ys.append(torch.full((n,), cls_id))
        X = torch.cat(Xs)
        Y = torch.cat(Ys)

        for step in range(200):
            feats = encoder(X)
            loss = dot_regression_loss(feats, Y, clf.etf_anchors)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # After training, check accuracy via the classifier
        with torch.no_grad():
            logits = clf(encoder(X))
            preds = logits.argmax(dim=-1)
            acc = (preds == Y).float().mean().item()
        # Crucial: accuracy on the MINORITY classes must be > 0
        # If it had collapsed to majority, minority acc = 0
        min_class_mask = Y != 0
        min_class_acc = (preds[min_class_mask] == Y[min_class_mask]).float().mean().item()
        assert acc > 0.85, f"overall acc {acc:.3f} too low"
        assert min_class_acc > 0.5, (
            f"minority-class accuracy {min_class_acc:.3f} suggests "
            f"majority collapse — Simplex ETF guarantee violated"
        )


# ════════════════════════════════════════════════════════════════════
# Orthogonal Representation
# ════════════════════════════════════════════════════════════════════
class TestOrthogonalityPenalty:
    def test_orthogonal_inputs_low_penalty(self):
        """If embedding is INDEPENDENT of rule features, penalty should be small."""
        torch.manual_seed(0)
        B = 200
        emb = torch.randn(B, 32)
        rules = torch.randn(B, 16)   # uncorrelated by construction
        p = orthogonality_penalty(emb, rules)
        # Should be small (some noise from finite-batch correlation)
        assert p.item() < 0.5, f"penalty {p.item():.4f} too high for orthogonal inputs"

    def test_perfectly_correlated_high_penalty(self):
        """If embedding is a copy of rule features, penalty should be high."""
        torch.manual_seed(0)
        B = 200
        rules = torch.randn(B, 8)
        emb = torch.cat([rules, rules], dim=-1)   # 16-dim emb = duplicated rules
        p1 = orthogonality_penalty(emb, rules)

        # Compare against truly random
        emb_rand = torch.randn(B, 16)
        p2 = orthogonality_penalty(emb_rand, rules)

        assert p1.item() > p2.item() * 5, (
            f"correlated penalty {p1.item():.4f} not >> "
            f"uncorrelated penalty {p2.item():.4f}"
        )

    def test_gradients_flow_into_embedding_only(self):
        """The penalty should only update the encoder, not the rule features."""
        emb = torch.randn(64, 16, requires_grad=True)
        rules = torch.randn(64, 8, requires_grad=True)
        p = orthogonality_penalty(emb, rules)   # detach_features=True by default
        p.backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum().item() > 0
        # Rule features should have NO gradient (they were detached)
        assert rules.grad is None or rules.grad.abs().sum().item() == 0

    def test_module_callable_form(self):
        mod = OrthogonalRepresentationModule(weight=0.7)
        emb = torch.randn(32, 16)
        rules = torch.randn(32, 8)
        loss = mod(emb, rules)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestGramSchmidtProjectOut:
    def test_projection_removes_correlation(self):
        torch.manual_seed(0)
        B, D_emb, D_feat = 200, 16, 8
        rules = torch.randn(B, D_feat)
        # Embedding is partially aligned with rules
        emb = torch.cat([rules, torch.randn(B, D_emb - D_feat)], dim=-1)
        # Center for fair correlation measurement
        emb_c = emb - emb.mean(0, keepdim=True)
        rules_c = rules - rules.mean(0, keepdim=True)
        corr_before = (emb_c.T @ rules_c).abs().sum().item()

        # Project out
        emb_proj = gram_schmidt_project_out(emb, rules)
        emb_proj_c = emb_proj - emb_proj.mean(0, keepdim=True)
        corr_after = (emb_proj_c.T @ rules_c).abs().sum().item()

        # After projection, correlation should drop sharply
        assert corr_after < 0.01 * corr_before, (
            f"projection ineffective: corr {corr_before:.2f} → {corr_after:.2f}"
        )

    def test_output_shape_preserved(self):
        emb = torch.randn(50, 32)
        rules = torch.randn(50, 12)
        out = gram_schmidt_project_out(emb, rules)
        assert out.shape == emb.shape
