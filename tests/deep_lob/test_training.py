"""Tests for the training loop."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_batch_dict(B=2, T=8, N=20, n_context=138):
    torch.manual_seed(0)
    feats = torch.randn(B, T, N, 7)
    feats[..., 0] = torch.randint(0, 2, (B, T, N))
    feats[..., 1] = torch.randint(0, 4, (B, T, N))
    feats[..., 5] = torch.randint(0, 8, (B, T, N))
    masks = torch.ones(B, T, N, dtype=torch.bool)
    bar_mask = torch.ones(B, T, dtype=torch.bool)
    ctx = torch.randn(B, n_context)
    return {
        "order_features": feats,
        "order_masks": masks,
        "bar_mask": bar_mask,
        "context": ctx,
    }


def _make_targets(B=2):
    from modules.deep_lob import MultiTaskTargets
    return MultiTaskTargets(
        direction=torch.randint(0, 3, (B,)),
        next_price=torch.randn(B),
        next_volatility=torch.rand(B) + 0.1,
        wall_persist=torch.rand(B) * 5,
    )


class TestTrainerInitialization(unittest.TestCase):
    def test_create_trainer(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(
            learning_rate=1e-4, max_steps=10, use_amp=False,
            warmup_steps=2, batch_size=2,
        )
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(
                model, config, output_dir=td, device="cpu",
            )
            self.assertIsNotNone(trainer.optimizer)
            self.assertEqual(trainer.device.type, "cpu")


class TestTrainStep(unittest.TestCase):
    def test_single_step(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False, learning_rate=1e-3)
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(
                model, config, output_dir=td, device="cpu",
            )
            metric = trainer.train_step_batch(
                _make_batch_dict(), _make_targets(), step=0, epoch=0,
            )
            self.assertGreater(metric.train_loss, 0)
            self.assertIn("direction", metric.train_losses_per_task)

    def test_loss_decreases_over_steps(self):
        """Sanity: loss should decrease on a small fixed batch."""
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        torch.manual_seed(42)
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False, learning_rate=1e-2, scheduler="constant")
        batch = _make_batch_dict()
        targets = _make_targets()

        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(
                model, config, output_dir=td, device="cpu",
            )
            losses = []
            for step in range(20):
                m = trainer.train_step_batch(batch, targets, step=step)
                losses.append(m.train_loss)
        # Loss should decrease (at least 30%)
        self.assertLess(losses[-1], losses[0] * 0.7,
            f"Loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}")


class TestValidation(unittest.TestCase):
    def test_validate_loop(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False)
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(model, config, output_dir=td, device="cpu")
            val_batches = [(_make_batch_dict(), _make_targets()) for _ in range(3)]
            metrics = trainer.validate(val_batches)
            self.assertIn("val_loss", metrics)
            self.assertGreater(metrics["val_loss"], 0)


class TestCheckpointing(unittest.TestCase):
    def test_save_checkpoint(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False, keep_top_k_checkpoints=2)
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(model, config, output_dir=td, device="cpu")
            trainer.save_checkpoint(step=10, score=0.5, tag="test")
            files = list(os.listdir(td))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("ckpt_step"))

    def test_top_k_keeping(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False, keep_top_k_checkpoints=2)
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(model, config, output_dir=td, device="cpu")
            # Save 5 checkpoints, only top-2 (lowest score) survive
            for step, score in [(1, 0.5), (2, 0.3), (3, 0.8), (4, 0.2), (5, 0.9)]:
                trainer.save_checkpoint(step=step, score=score)
            files = list(os.listdir(td))
            self.assertEqual(len(files), 2)


class TestEarlyStopping(unittest.TestCase):
    def test_early_stops_after_patience(self):
        from modules.deep_lob import (
            DeepLOBConfig, HierarchicalLOBTransformer, TrainingConfig,
            HierarchicalLOBTrainer,
        )
        model = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        config = TrainingConfig(use_amp=False, early_stopping_patience=3)
        with tempfile.TemporaryDirectory() as td:
            trainer = HierarchicalLOBTrainer(model, config, output_dir=td, device="cpu")
            self.assertFalse(trainer.early_stopping_check(1.0))   # new best
            self.assertFalse(trainer.early_stopping_check(1.1))   # no improvement (1/3)
            self.assertFalse(trainer.early_stopping_check(1.2))   # 2/3
            should_stop = trainer.early_stopping_check(1.1)       # 3/3 → stop
            self.assertTrue(should_stop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
