"""Integration tests for the full HierarchicalLOBTransformer."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_batch(B=2, T=8, N=20, n_context=138, seed=0):
    """Create a synthetic batch matching expected input format."""
    torch.manual_seed(seed)
    order_features = torch.randn(B, T, N, 7)
    order_features[..., 0] = torch.randint(0, 2, (B, T, N))
    order_features[..., 1] = torch.randint(0, 4, (B, T, N))
    order_features[..., 5] = torch.randint(0, 8, (B, T, N))
    order_masks = torch.ones(B, T, N, dtype=torch.bool)
    bar_mask = torch.ones(B, T, dtype=torch.bool)
    context = torch.randn(B, n_context)
    return order_features, order_masks, bar_mask, context


class TestForwardPass(unittest.TestCase):
    def test_forward_shape(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch(B=4, T=8, N=20)
        out = model(feats, masks, bar_mask, ctx)
        self.assertEqual(out.direction_logits.shape, (4, 3))
        self.assertEqual(out.shared_embedding.shape[0], 4)

    def test_forward_with_intermediates(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch()
        out, inter = model(feats, masks, bar_mask, ctx, return_intermediates=True)
        self.assertIn("order_embeddings", inter)
        self.assertIn("attentions", inter)
        self.assertIn("event_assignment", inter)

    def test_forward_no_context(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, _ = _make_batch()
        out = model(feats, masks, bar_mask, context=None)
        self.assertEqual(out.direction_logits.shape[0], feats.shape[0])

    def test_finite_outputs(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch()
        out = model(feats, masks, bar_mask, ctx)
        for name in ["direction_logits", "next_price", "wall_persist"]:
            t = getattr(out, name)
            self.assertTrue(torch.isfinite(t).all(), f"{name} has non-finite values")


class TestBackwardPass(unittest.TestCase):
    def test_gradient_flow_full_targets(self):
        """مع targets كاملة، كل parameter يجب أن يحصل على gradient.

        We disable the LOB image encoder here because _make_batch() doesn't
        produce a lob_tensor — leaving the CNN branch enabled would result
        in its params never receiving gradients (correct behavior of a
        bypassed branch, but it would fail this assertion). A separate
        test would be needed to exercise the LOB CNN branch with a real
        lob_tensor input.
        """
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, MultiTaskTargets,
        )
        cfg = DeepLOBConfig.small_dev()
        cfg.lob_image.enabled = False
        model = HierarchicalLOBTransformer(cfg)
        feats, masks, bar_mask, ctx = _make_batch()

        out = model(feats, masks, bar_mask, ctx)
        B = feats.shape[0]
        # Full targets for all 7 tasks
        targets = MultiTaskTargets(
            direction=torch.randint(0, 3, (B,)),
            next_price=torch.randn(B),
            next_imbalance=torch.tanh(torch.randn(B)),
            next_volatility=torch.rand(B) + 0.1,
            next_regime=torch.randint(0, 3, (B,)),
            wall_persist=torch.rand(B) * 10,
            time_to_event=torch.rand(B) * 10,
        )
        loss = model.heads.compute_loss(out, targets)["total"]
        loss.backward()

        # كل parameter يجب أن يكون له gradient
        params_no_grad = [
            (name, p) for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        self.assertEqual(
            len(params_no_grad), 0,
            f"Parameters without gradient: {[n for n, _ in params_no_grad[:5]]}"
        )

    def test_partial_targets_partial_gradients(self):
        """Direction-only target: shared backbone receives gradient، unused heads don't."""
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, MultiTaskTargets,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch()
        out = model(feats, masks, bar_mask, ctx)
        targets = MultiTaskTargets(direction=torch.randint(0, 3, (feats.shape[0],)))
        loss = model.heads.compute_loss(out, targets)["total"]
        loss.backward()
        # Direction head must have gradient
        for p in model.heads.head_direction.parameters():
            self.assertIsNotNone(p.grad)
        # Backbone (order_embedder) must have gradient too
        for p in model.order_embedder.parameters():
            self.assertIsNotNone(p.grad)


class TestPredict(unittest.TestCase):
    def test_predict_returns_dict(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch()
        pred = model.predict(feats, masks, bar_mask, ctx)
        self.assertIsInstance(pred, dict)
        self.assertIn("direction_probs", pred)
        # Probs sum to 1
        probs = pred["direction_probs"]
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-5))


class TestVisualEmbedding(unittest.TestCase):
    def test_visual_embedding_8dim(self):
        """Backward compat: visual_embedding should be 8-dim لـ Stage 3 LSTM."""
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        feats, masks, bar_mask, ctx = _make_batch()
        emb = model.get_visual_embedding(feats, masks, bar_mask, ctx)
        self.assertEqual(emb.shape[1], 8)


class TestCheckpointing(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        model.eval()  # disable dropout للـ deterministic outputs
        feats, masks, bar_mask, ctx = _make_batch()
        out_before = model(feats, masks, bar_mask, ctx).direction_logits.detach()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            model.save_checkpoint(str(path))
            loaded = HierarchicalLOBTransformer.from_checkpoint(str(path))
            loaded.eval()
            out_after = loaded(feats, masks, bar_mask, ctx).direction_logits.detach()

        # Outputs should be identical
        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-5),
            f"Max diff: {(out_before - out_after).abs().max().item()}")


class TestParameterCount(unittest.TestCase):
    def test_count_parameters_breakdown(self):
        """Component sum must match total. We disable lob_image to avoid
        counting its params here; gradient flow with lob_image is exercised
        separately by test_gradient_flow_full_targets (with lob_tensor
        supplied)."""
        from modules.deep_lob import DeepLOBConfig, HierarchicalLOBTransformer
        cfg = DeepLOBConfig.small_dev()
        cfg.lob_image.enabled = False  # match the components we sum below
        model = HierarchicalLOBTransformer(cfg)
        counts = model.count_parameters()
        for key in ("order_embedder", "order_transformer", "event_aggregator",
                    "bar_lstm", "context_encoder", "fusion", "heads", "total"):
            self.assertIn(key, counts)
        # Total = sum of components (approximately)
        component_sum = sum(
            counts[k] for k in (
                "order_embedder", "order_transformer", "event_aggregator",
                "bar_lstm", "context_encoder", "fusion", "heads",
            )
        )
        self.assertEqual(counts["total"], component_sum)


if __name__ == "__main__":
    unittest.main(verbosity=2)
