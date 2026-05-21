"""Unit tests for individual model components."""

from __future__ import annotations

import os
import sys
import unittest

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_order_batch(B=2, N=20, seed=0):
    torch.manual_seed(seed)
    feats = torch.randn(B, N, 7)
    feats[..., 0] = torch.randint(0, 2, (B, N))
    feats[..., 1] = torch.randint(0, 4, (B, N))
    feats[..., 5] = torch.randint(0, 8, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)
    return feats, mask


class TestSinusoidalTimeEncoding(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob.order_embedder import SinusoidalTimeEncoding
        enc = SinusoidalTimeEncoding(dim=8)
        x = torch.tensor([0.0, 1.0, 100.0, 1000.0])
        out = enc(x)
        self.assertEqual(out.shape, (4, 8))

    def test_invalid_odd_dim(self):
        from modules.deep_lob.order_embedder import SinusoidalTimeEncoding
        with self.assertRaises(ValueError):
            SinusoidalTimeEncoding(dim=7)

    def test_bounded(self):
        """sin/cos always in [-1, 1]."""
        from modules.deep_lob.order_embedder import SinusoidalTimeEncoding
        enc = SinusoidalTimeEncoding(dim=8)
        out = enc(torch.tensor([1.0, 1e5, 1e9]))
        self.assertTrue((out.abs() <= 1.0 + 1e-6).all())


class TestOrderEmbedder(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob import OrderEmbedder, OrderEmbedderConfig
        embedder = OrderEmbedder(OrderEmbedderConfig(embed_dim=32))
        feats, mask = _make_order_batch()
        out = embedder(feats, mask)
        self.assertEqual(out.shape, (2, 20, 32))

    def test_masked_positions_zero(self):
        from modules.deep_lob import OrderEmbedder, OrderEmbedderConfig
        embedder = OrderEmbedder(OrderEmbedderConfig(embed_dim=32))
        feats, mask = _make_order_batch()
        mask[:, 10:] = False  # mask out last 10 orders
        out = embedder(feats, mask)
        # masked positions should be zero
        self.assertTrue((out[:, 10:].abs() < 1e-6).all())


class TestTransformer(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob import (
            LOBTransformerEncoder, TransformerConfig,
        )
        config = TransformerConfig(embed_dim=32, n_heads=4, n_layers=2)
        enc = LOBTransformerEncoder(config)
        x = torch.randn(2, 20, 32)
        out, attns = enc(x, return_attentions=False)
        self.assertEqual(out.shape, (2, 20, 32))
        self.assertIsNone(attns)

    def test_attention_extraction(self):
        from modules.deep_lob import (
            LOBTransformerEncoder, TransformerConfig,
        )
        config = TransformerConfig(embed_dim=32, n_heads=4, n_layers=2)
        enc = LOBTransformerEncoder(config)
        x = torch.randn(2, 20, 32)
        out, attns = enc(x, return_attentions=True)
        self.assertEqual(len(attns), 2)  # one per layer
        self.assertEqual(attns[0].shape, (2, 20, 20))

    def test_masking_zeros_padded_output(self):
        from modules.deep_lob import (
            LOBTransformerEncoder, TransformerConfig,
        )
        config = TransformerConfig(embed_dim=32, n_heads=4, n_layers=2)
        enc = LOBTransformerEncoder(config)
        x = torch.randn(2, 20, 32)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 15:] = False
        out, _ = enc(x, key_padding_mask=mask)
        self.assertTrue((out[:, 15:].abs() < 1e-6).all())

    def test_parameter_count(self):
        from modules.deep_lob import (
            LOBTransformerEncoder, TransformerConfig,
        )
        config = TransformerConfig(embed_dim=32, n_heads=4, n_layers=2)
        enc = LOBTransformerEncoder(config)
        counts = enc.count_parameters()
        self.assertGreater(counts["total"], 1000)
        self.assertEqual(counts["trainable"], counts["total"])


class TestEventAggregator(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob import EventAggregator, EventAggregatorConfig
        agg = EventAggregator(EventAggregatorConfig(n_event_types=5, event_dim=32), order_embed_dim=32)
        order_emb = torch.randn(2, 20, 32)
        mask = torch.ones(2, 20, dtype=torch.bool)
        out = agg(order_emb, mask)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_assignment_extraction(self):
        from modules.deep_lob import EventAggregator, EventAggregatorConfig
        agg = EventAggregator(EventAggregatorConfig(n_event_types=4), order_embed_dim=32)
        order_emb = torch.randn(2, 20, 32)
        mask = torch.ones(2, 20, dtype=torch.bool)
        _ = agg(order_emb, mask, return_assignment=True)
        assignment = agg.get_last_assignment()
        self.assertEqual(assignment.shape, (2, 4, 20))
        # Each row sums to 1 (softmax)
        sums = assignment.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


class TestBarLSTM(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob import BarLevelLSTM, BarLSTMConfig
        config = BarLSTMConfig(input_dim=32, hidden_dim=64, n_layers=1, sequence_length=10)
        lstm = BarLevelLSTM(config)
        events = torch.randn(2, 10, 5, 32)  # (B, T, E, D)
        seq_out, final = lstm(events)
        self.assertEqual(seq_out.shape, (2, 10, 64))
        self.assertEqual(final.shape, (2, 64))


class TestContextEncoder(unittest.TestCase):
    def test_shape(self):
        from modules.deep_lob import ContextEncoder, ContextEncoderConfig
        enc = ContextEncoder(ContextEncoderConfig(n_input_features=138, output_dim=32))
        ctx = torch.randn(2, 138)
        out = enc(ctx)
        self.assertEqual(out.shape, (2, 32))

    def test_nan_handling(self):
        from modules.deep_lob import ContextEncoder, ContextEncoderConfig
        enc = ContextEncoder(ContextEncoderConfig(n_input_features=138, output_dim=32))
        ctx = torch.randn(2, 138)
        ctx[0, 5] = float("nan")
        ctx[1, 10] = float("inf")
        out = enc(ctx)
        self.assertTrue(torch.isfinite(out).all())


class TestMultiTaskHeads(unittest.TestCase):
    def test_output_shape(self):
        from modules.deep_lob import MultiTaskHeads, MultiTaskHeadsConfig
        heads = MultiTaskHeads(MultiTaskHeadsConfig(shared_dim=64))
        shared = torch.randn(2, 64)
        out = heads(shared)
        self.assertEqual(out.direction_logits.shape, (2, 3))
        self.assertEqual(out.next_price.shape, (2,))
        self.assertEqual(out.wall_persist.shape, (2,))
        # Volatility and wall_persist must be positive
        self.assertTrue((out.next_volatility >= 0).all())
        self.assertTrue((out.wall_persist >= 0).all())

    def test_loss_computation(self):
        from modules.deep_lob import MultiTaskHeads, MultiTaskHeadsConfig, MultiTaskTargets
        heads = MultiTaskHeads(MultiTaskHeadsConfig(shared_dim=64))
        shared = torch.randn(2, 64)
        outputs = heads(shared)
        targets = MultiTaskTargets(
            direction=torch.tensor([0, 1]),
            next_price=torch.randn(2),
            next_volatility=torch.rand(2) + 0.1,
        )
        losses = heads.compute_loss(outputs, targets)
        self.assertIn("total", losses)
        self.assertIn("direction", losses)
        self.assertIn("next_price", losses)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_partial_targets_supported(self):
        from modules.deep_lob import MultiTaskHeads, MultiTaskHeadsConfig, MultiTaskTargets
        heads = MultiTaskHeads(MultiTaskHeadsConfig(shared_dim=64))
        shared = torch.randn(2, 64)
        outputs = heads(shared)
        # Only direction target
        targets = MultiTaskTargets(direction=torch.tensor([0, 1]))
        losses = heads.compute_loss(outputs, targets)
        self.assertIn("direction", losses)
        self.assertNotIn("next_price", losses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
