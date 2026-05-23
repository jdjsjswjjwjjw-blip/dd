"""Unit tests for config dataclasses."""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestOrderEmbedderConfig(unittest.TestCase):
    def test_default_valid(self):
        from modules.deep_lob import OrderEmbedderConfig
        c = OrderEmbedderConfig()
        self.assertEqual(c.embed_dim, 32)

    def test_invalid_embed_dim_raises(self):
        from modules.deep_lob import OrderEmbedderConfig
        with self.assertRaises(ValueError):
            OrderEmbedderConfig(embed_dim=4)  # too small
        with self.assertRaises(ValueError):
            OrderEmbedderConfig(embed_dim=33)  # not divisible by 4


class TestTransformerConfig(unittest.TestCase):
    def test_default_valid(self):
        from modules.deep_lob import TransformerConfig
        c = TransformerConfig()
        self.assertEqual(c.embed_dim, 32)
        self.assertEqual(c.n_heads, 4)
        self.assertEqual(c.embed_dim % c.n_heads, 0)

    def test_invalid_heads_raises(self):
        from modules.deep_lob import TransformerConfig
        with self.assertRaises(ValueError):
            TransformerConfig(embed_dim=32, n_heads=5)  # 32 % 5 != 0

    def test_invalid_n_layers(self):
        from modules.deep_lob import TransformerConfig
        with self.assertRaises(ValueError):
            TransformerConfig(n_layers=0)
        with self.assertRaises(ValueError):
            TransformerConfig(n_layers=25)

    def test_invalid_activation(self):
        from modules.deep_lob import TransformerConfig
        with self.assertRaises(ValueError):
            TransformerConfig(activation="tanh")


class TestDeepLOBConfig(unittest.TestCase):
    def test_default_consistency(self):
        from modules.deep_lob import DeepLOBConfig
        c = DeepLOBConfig()
        # embed_dim consistency
        self.assertEqual(c.order_embedder.embed_dim, c.transformer.embed_dim)
        # event/bar dim consistency
        self.assertEqual(c.event_aggregator.event_dim, c.bar_lstm.input_dim)

    def test_small_dev_smaller_than_default(self):
        from modules.deep_lob import DeepLOBConfig
        default = DeepLOBConfig()
        small = DeepLOBConfig.small_dev()
        self.assertLess(small.transformer.embed_dim, default.transformer.embed_dim)
        self.assertLess(small.transformer.n_layers, default.transformer.n_layers)

    def test_serializable(self):
        from modules.deep_lob import DeepLOBConfig
        c = DeepLOBConfig()
        d = c.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("order_embedder", d)
        self.assertIn("multi_task_heads", d)


class TestTrainingConfig(unittest.TestCase):
    def test_default_valid(self):
        from modules.deep_lob import TrainingConfig
        c = TrainingConfig()
        self.assertEqual(c.optimizer, "adamw")

    def test_invalid_optimizer(self):
        from modules.deep_lob import TrainingConfig
        with self.assertRaises(ValueError):
            TrainingConfig(optimizer="rmsprop")

    def test_invalid_lr(self):
        from modules.deep_lob import TrainingConfig
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=-0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
