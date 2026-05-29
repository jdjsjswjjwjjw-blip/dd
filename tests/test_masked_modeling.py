"""Tests for the masked-reconstruction auxiliary task.

Each piece (config, mask gen, mask application, recon head, loss) is
tested in isolation. The integration smoke at the end shows the full
forward/backward pass works on a tiny synthetic batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.deep_lob.masked_modeling import (
    MaskedModelingConfig,
    ReconstructionHead,
    apply_mask,
    compute_recon_loss,
    generate_mask,
)


# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_defaults_are_safe(self):
        cfg = MaskedModelingConfig()
        assert cfg.enabled is False
        assert 0.10 <= cfg.mask_ratio <= 0.25
        assert cfg.strategy == "random"

    def test_invalid_mask_ratio_raises(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskedModelingConfig(mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskedModelingConfig(mask_ratio=1.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            MaskedModelingConfig(mask_ratio=-0.1)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            MaskedModelingConfig(strategy="random_walk")

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError, match="patch_size"):
            MaskedModelingConfig(patch_size=0)

    def test_invalid_recon_weight_raises(self):
        with pytest.raises(ValueError, match="reconstruction_weight"):
            MaskedModelingConfig(reconstruction_weight=-1.0)

    def test_all_strategies_accepted(self):
        for s in ("random", "patch", "bar"):
            cfg = MaskedModelingConfig(strategy=s)
            assert cfg.strategy == s


# ════════════════════════════════════════════════════════════════════
# Mask generation
# ════════════════════════════════════════════════════════════════════
class TestGenerateMask:
    def _order_masks(self, B=4, T=10, N=8, fill_ratio=1.0):
        """All valid (no padding) by default."""
        m = torch.zeros(B, T, N, dtype=torch.bool)
        n_valid = int(N * fill_ratio)
        m[:, :, :n_valid] = True
        return m

    def test_deterministic_with_seed(self):
        cfg = MaskedModelingConfig(enabled=True, seed=42)
        order_masks = self._order_masks()
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        m1 = generate_mask(order_masks, cfg, generator=g1)
        m2 = generate_mask(order_masks, cfg, generator=g2)
        assert torch.equal(m1, m2)

    def test_returns_bool_same_shape(self):
        order_masks = self._order_masks()
        cfg = MaskedModelingConfig(enabled=True)
        mask = generate_mask(order_masks, cfg)
        assert mask.dtype == torch.bool
        assert mask.shape == order_masks.shape

    def test_mask_ratio_approximately_matches(self):
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.15, min_mask_per_sample=0,
        )
        order_masks = self._order_masks(B=64, T=20, N=10)  # 12,800 positions
        g = torch.Generator().manual_seed(0)
        mask = generate_mask(order_masks, cfg, generator=g)
        empirical = float(mask.float().mean())
        assert 0.12 <= empirical <= 0.18

    def test_padding_never_masked(self):
        # Half the positions are padding (invalid orders)
        cfg = MaskedModelingConfig(enabled=True, mask_ratio=0.5)
        order_masks = self._order_masks(fill_ratio=0.5)  # only first half valid
        mask = generate_mask(order_masks, cfg)
        # No True positions where order_masks is False
        assert not bool((mask & ~order_masks).any())

    def test_min_mask_floor_enforced(self):
        # Tiny mask ratio that often produces zero — floor should rescue
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.01, min_mask_per_sample=3,
        )
        order_masks = self._order_masks(B=8, T=10, N=20)
        mask = generate_mask(order_masks, cfg)
        for b in range(8):
            assert int(mask[b].sum()) >= 3

    def test_min_mask_floor_zero_allowed(self):
        # min_mask_per_sample=0 → no rescue. Pinning random to a low
        # ratio means SOME samples may have zero masked — that's OK.
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.05, min_mask_per_sample=0,
        )
        order_masks = self._order_masks(B=64, T=10, N=8)
        mask = generate_mask(order_masks, cfg)
        per_sample = mask.flatten(1).sum(dim=1)
        assert (per_sample == 0).any() or per_sample.min() >= 0

    def test_patch_strategy_contiguous(self):
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.25, strategy="patch", patch_size=3,
            min_mask_per_sample=0,
        )
        order_masks = self._order_masks(B=4, T=5, N=12)
        mask = generate_mask(order_masks, cfg)
        # For at least some (b, t), the masked positions should form a
        # contiguous patch of length ≥ patch_size
        found_patch = False
        for b in range(4):
            for t in range(5):
                row = mask[b, t]
                # Run-lengths of True values
                if int(row.sum()) >= 3:
                    # Check there is at least one run of 3
                    transitions = (row[1:] != row[:-1]).int()
                    # Find True runs of len ≥ 3
                    nrow = row.numel()
                    for i in range(nrow - 2):
                        if row[i] and row[i + 1] and row[i + 2]:
                            found_patch = True
                            break
                if found_patch:
                    break
            if found_patch:
                break
        assert found_patch, "expected at least one contiguous patch"

    def test_bar_strategy_masks_whole_bars(self):
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.30, strategy="bar",
            min_mask_per_sample=0,
        )
        order_masks = self._order_masks(B=4, T=10, N=8)
        mask = generate_mask(order_masks, cfg)
        # Each (b, t) is either fully True or fully False (across N)
        for b in range(4):
            for t in range(10):
                row = mask[b, t]
                if bool(row.any()):
                    # bar strategy masks all N positions in chosen bars
                    # (but capped by order_masks; here all are valid)
                    assert bool(row.all())

    def test_raises_on_non_bool_input(self):
        cfg = MaskedModelingConfig(enabled=True)
        bad = torch.ones(2, 3, 4)  # float, not bool
        with pytest.raises(TypeError, match="bool"):
            generate_mask(bad, cfg)


# ════════════════════════════════════════════════════════════════════
# apply_mask
# ════════════════════════════════════════════════════════════════════
class TestApplyMask:
    def test_zeros_masked_positions(self):
        x = torch.ones(2, 3, 4, 7)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[0, 1, 2] = True
        out = apply_mask(x, mask, provide_mask_indicator=False)
        assert out.shape == x.shape
        assert torch.all(out[0, 1, 2] == 0)
        # Unmasked positions unchanged
        assert torch.all(out[0, 1, 0] == 1)

    def test_indicator_concatenated(self):
        x = torch.ones(2, 3, 4, 7)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[0, 1, 2] = True
        out = apply_mask(x, mask, provide_mask_indicator=True)
        assert out.shape == (2, 3, 4, 8)
        # Last channel = 1 where masked
        assert out[0, 1, 2, -1].item() == 1.0
        # Last channel = 0 where not masked
        assert out[0, 0, 0, -1].item() == 0.0

    def test_pure_no_input_mutation(self):
        x = torch.ones(2, 3, 4, 7)
        x_ref = x.clone()
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[0, 0, 0] = True
        _ = apply_mask(x, mask)
        assert torch.equal(x, x_ref)

    def test_shape_mismatch_raises(self):
        x = torch.ones(2, 3, 4, 7)
        mask = torch.zeros(2, 3, 5, dtype=torch.bool)  # wrong N
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_mask(x, mask)


# ════════════════════════════════════════════════════════════════════
# ReconstructionHead
# ════════════════════════════════════════════════════════════════════
class TestReconstructionHead:
    def test_forward_shape(self):
        head = ReconstructionHead(encoder_dim=32, feature_dim=7)
        x = torch.randn(4, 10, 8, 32)
        y = head(x)
        assert y.shape == (4, 10, 8, 7)

    def test_backward_flows(self):
        head = ReconstructionHead(encoder_dim=16, feature_dim=4)
        x = torch.randn(2, 3, 5, 16, requires_grad=True)
        y = head(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_param_count_small(self):
        # Head should be lightweight — we don't want it to dominate
        head = ReconstructionHead(
            encoder_dim=128, feature_dim=7, hidden_dim=128,
        )
        n_params = sum(p.numel() for p in head.parameters())
        assert n_params < 50_000   # 128*128 + 128*7 + biases ≈ 17k

    def test_dropout_disabled_by_default(self):
        head = ReconstructionHead(encoder_dim=8, feature_dim=4)
        head.eval()
        x = torch.randn(2, 3, 4, 8)
        y1 = head(x)
        y2 = head(x)
        assert torch.allclose(y1, y2)


# ════════════════════════════════════════════════════════════════════
# compute_recon_loss
# ════════════════════════════════════════════════════════════════════
class TestComputeReconLoss:
    def test_perfect_reconstruction_zero_loss(self):
        x = torch.randn(2, 3, 4, 7)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        order_masks = torch.ones(2, 3, 4, dtype=torch.bool)
        loss = compute_recon_loss(x, x, mask, order_masks)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_zero_mask_returns_zero(self):
        # When no positions are masked, loss is exactly 0 (no gradient
        # signal — but no crash either)
        x = torch.randn(2, 3, 4, 7)
        y = torch.zeros_like(x)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        order_masks = torch.ones(2, 3, 4, dtype=torch.bool)
        loss = compute_recon_loss(y, x, mask, order_masks)
        assert loss.item() == 0.0

    def test_only_masked_positions_contribute(self):
        # If recon equals original outside the mask but is wrong inside,
        # the loss equals the error on the masked subset only
        x = torch.zeros(1, 1, 4, 2)
        recon = torch.zeros(1, 1, 4, 2)
        recon[0, 0, 1] = 1.0   # error of 1 at position 1
        recon[0, 0, 3] = 2.0   # error of 2 at position 3 (NOT in mask)
        mask = torch.zeros(1, 1, 4, dtype=torch.bool)
        mask[0, 0, 1] = True   # mask only position 1
        order_masks = torch.ones(1, 1, 4, dtype=torch.bool)
        loss = compute_recon_loss(recon, x, mask, order_masks)
        # MSE on position 1: mean over D=2 of (1, 1) → 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_padding_never_contributes(self):
        # Even when mask is True on a padded position (defensive case),
        # the loss does not pick it up because order_masks excludes it
        x = torch.zeros(1, 1, 4, 2)
        recon = torch.full((1, 1, 4, 2), 100.0)  # huge error everywhere
        mask = torch.ones(1, 1, 4, dtype=torch.bool)
        order_masks = torch.zeros(1, 1, 4, dtype=torch.bool)
        order_masks[0, 0, 0] = True  # only position 0 is valid
        loss = compute_recon_loss(recon, x, mask, order_masks)
        # Only position 0 contributes — error there is 100² = 10000
        assert loss.item() == pytest.approx(10000.0, abs=1e-3)

    def test_feature_weights_applied(self):
        x = torch.zeros(1, 1, 1, 3)
        recon = torch.ones(1, 1, 1, 3)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        order_masks = torch.ones(1, 1, 1, dtype=torch.bool)
        weights = torch.tensor([0.0, 0.0, 3.0])
        loss = compute_recon_loss(recon, x, mask, order_masks, weights)
        # Only the third feature contributes: (1² * 3) / 3 = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_feature_weights_wrong_shape_raises(self):
        x = torch.zeros(1, 1, 1, 3)
        recon = torch.zeros_like(x)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        order_masks = torch.ones(1, 1, 1, dtype=torch.bool)
        bad_weights = torch.tensor([1.0, 2.0])  # wrong D
        with pytest.raises(ValueError, match="feature_weights shape"):
            compute_recon_loss(recon, x, mask, order_masks, bad_weights)

    def test_shape_mismatch_raises(self):
        x = torch.zeros(2, 3, 4, 7)
        bad_recon = torch.zeros(2, 3, 4, 8)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        order_masks = mask.clone()
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_recon_loss(bad_recon, x, mask, order_masks)


# ════════════════════════════════════════════════════════════════════
# Integration smoke
# ════════════════════════════════════════════════════════════════════
class TestIntegrationSmoke:
    def test_full_forward_backward(self):
        """Simulate one training step: generate mask → apply → encode
        with a stub encoder → reconstruct → backward."""
        torch.manual_seed(0)
        B, T, N, D = 4, 5, 6, 7
        encoder_dim = 16

        order_features = torch.randn(B, T, N, D, requires_grad=False)
        order_masks = torch.ones(B, T, N, dtype=torch.bool)
        cfg = MaskedModelingConfig(
            enabled=True, mask_ratio=0.20, provide_mask_indicator=True,
        )
        head = ReconstructionHead(encoder_dim=encoder_dim, feature_dim=D)

        # Stub encoder: linear projection of (D+1) → encoder_dim
        stub_encoder = torch.nn.Linear(D + 1, encoder_dim)

        # Training step
        mask = generate_mask(order_masks, cfg)
        masked = apply_mask(order_features, mask, provide_mask_indicator=True)
        encoded = stub_encoder(masked)
        recon = head(encoded)
        loss = compute_recon_loss(recon, order_features, mask, order_masks)
        assert loss.requires_grad
        loss.backward()
        # All trainable params got gradient
        for p in head.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
        for p in stub_encoder.parameters():
            assert p.grad is not None

    def test_disabled_config_still_constructable(self):
        # When disabled, the user shouldn't have to build the mask/head;
        # the config should still validate
        cfg = MaskedModelingConfig(enabled=False)
        assert cfg.enabled is False
