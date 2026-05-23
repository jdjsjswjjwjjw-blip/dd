"""Tests for CycleEncoder, PriceCycleModel, Multi-Scale Fusion."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestCausalConv(unittest.TestCase):
    def test_causal_no_lookahead(self):
        """Causal conv: output[t] must not depend on input[>t]."""
        from modules.price_cycle import CausalConv1d
        conv = CausalConv1d(2, 4, kernel_size=3, dilation=1)
        conv.eval()
        x = torch.randn(1, 2, 20)
        with torch.no_grad():
            out_full = conv(x)
            # Modify only the LAST timestep of input
            x_mod = x.clone()
            x_mod[:, :, -1] += 100.0
            out_mod = conv(x_mod)
        # Output at t < last must be UNCHANGED
        self.assertTrue(
            torch.allclose(out_full[:, :, :-1], out_mod[:, :, :-1], atol=1e-5),
            "Causal conv leaked future information!",
        )


class TestCycleEncoder(unittest.TestCase):
    def test_forward_shape(self):
        from modules.price_cycle import CycleEncoder, CycleEncoderConfig
        config = CycleEncoderConfig(
            n_input_features=16, n_channels=16, n_levels=3, embed_dim=32,
            resolution_factors=(1, 4),
        )
        enc = CycleEncoder(config)
        x = torch.randn(4, 100, 16)
        out = enc(x)
        self.assertEqual(out.shape, (4, 32))

    def test_receptive_field_property(self):
        from modules.price_cycle import CycleEncoderConfig
        config = CycleEncoderConfig(kernel_size=3, n_levels=6)
        # 1 + 2*(3-1)*(2^6 - 1) = 1 + 4*63 = 253
        self.assertEqual(config.receptive_field, 253)

    def test_forward_sequence(self):
        from modules.price_cycle import CycleEncoder, CycleEncoderConfig
        config = CycleEncoderConfig(
            n_input_features=16, n_channels=16, n_levels=2, embed_dim=32,
            resolution_factors=(1,),
        )
        enc = CycleEncoder(config)
        x = torch.randn(2, 50, 16)
        seq = enc.forward_sequence(x)
        self.assertEqual(seq.shape, (2, 50, 32))


class TestPriceCycleModel(unittest.TestCase):
    def test_forward_shapes(self):
        from modules.price_cycle import PriceCycleModel, PriceCycleConfig
        model = PriceCycleModel(PriceCycleConfig.small_dev())
        x = torch.randn(4, 100, 16)
        out = model(x)
        self.assertEqual(out.phase_logits.shape, (4, 4))
        self.assertEqual(out.maturity_logits.shape, (4, 3))
        self.assertEqual(out.swing_logits.shape, (4, 3))
        self.assertEqual(out.reversal_proximity.shape, (4,))
        self.assertEqual(out.cycle_position.shape, (4,))

    def test_outputs_valid_ranges(self):
        from modules.price_cycle import PriceCycleModel, PriceCycleConfig
        model = PriceCycleModel(PriceCycleConfig.small_dev())
        out = model(torch.randn(4, 100, 16))
        # cycle_position in [0,1]
        self.assertTrue((out.cycle_position >= 0).all())
        self.assertTrue((out.cycle_position <= 1).all())
        # reversal_proximity >= 0
        self.assertTrue((out.reversal_proximity >= 0).all())
        # phase probs sum to 1
        self.assertTrue(torch.allclose(
            out.phase_probs().sum(-1), torch.ones(4), atol=1e-5,
        ))

    def test_loss_computation(self):
        from modules.price_cycle import PriceCycleModel, PriceCycleConfig, CycleTargets
        model = PriceCycleModel(PriceCycleConfig.small_dev())
        out = model(torch.randn(4, 100, 16))
        targets = CycleTargets(
            phase=torch.randint(0, 4, (4,)),
            maturity=torch.randint(0, 3, (4,)),
            swing_direction=torch.randint(0, 3, (4,)),
            reversal_proximity=torch.rand(4) * 10,
            cycle_position=torch.rand(4),
        )
        losses = model.heads.compute_loss(out, targets)
        self.assertIn("total", losses)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_gradient_flow(self):
        from modules.price_cycle import PriceCycleModel, PriceCycleConfig, CycleTargets
        model = PriceCycleModel(PriceCycleConfig.small_dev())
        out = model(torch.randn(4, 100, 16))
        targets = CycleTargets(
            phase=torch.randint(0, 4, (4,)),
            maturity=torch.randint(0, 3, (4,)),
            swing_direction=torch.randint(0, 3, (4,)),
            reversal_proximity=torch.rand(4) * 10,
            cycle_position=torch.rand(4),
        )
        loss = model.heads.compute_loss(out, targets)["total"]
        loss.backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        self.assertEqual(len(no_grad), 0, f"No gradient: {no_grad[:3]}")

    def test_checkpoint_roundtrip(self):
        from modules.price_cycle import PriceCycleModel, PriceCycleConfig
        model = PriceCycleModel(PriceCycleConfig.small_dev())
        model.eval()
        x = torch.randn(2, 100, 16)
        before = model(x).phase_logits.detach()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cycle.pt"
            model.save_checkpoint(str(path))
            loaded = PriceCycleModel.from_checkpoint(str(path))
            loaded.eval()
            after = loaded(x).phase_logits.detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-5))


class TestMultiScaleFusion(unittest.TestCase):
    def test_cross_attention_fusion(self):
        from modules.price_cycle import MultiScaleFusion, MultiScaleFusionConfig
        config = MultiScaleFusionConfig(
            micro_dim=32, macro_dim=32, fusion_dim=32,
            fusion_method="cross_attention",
        )
        fusion = MultiScaleFusion(config)
        out = fusion(torch.randn(4, 32), torch.randn(4, 32))
        self.assertEqual(out.decision_logits.shape, (4, 3))
        self.assertEqual(out.alignment_score.shape, (4,))

    def test_gating_fusion(self):
        from modules.price_cycle import MultiScaleFusion, MultiScaleFusionConfig
        config = MultiScaleFusionConfig(
            micro_dim=32, macro_dim=32, fusion_dim=32, fusion_method="gating",
        )
        fusion = MultiScaleFusion(config)
        out = fusion(torch.randn(4, 32), torch.randn(4, 32))
        # gating: micro_gate + macro_gate = 1
        self.assertTrue(torch.allclose(
            out.micro_gate + out.macro_gate, torch.ones(4), atol=1e-5,
        ))

    def test_concat_fusion(self):
        from modules.price_cycle import MultiScaleFusion, MultiScaleFusionConfig
        config = MultiScaleFusionConfig(
            micro_dim=32, macro_dim=32, fusion_dim=32, fusion_method="concat",
        )
        fusion = MultiScaleFusion(config)
        out = fusion(torch.randn(4, 32), torch.randn(4, 32))
        self.assertEqual(out.decision_logits.shape, (4, 3))

    def test_alignment_score_range(self):
        from modules.price_cycle import MultiScaleFusion, MultiScaleFusionConfig
        config = MultiScaleFusionConfig(micro_dim=32, macro_dim=32, fusion_dim=32)
        fusion = MultiScaleFusion(config)
        out = fusion(torch.randn(8, 32), torch.randn(8, 32))
        # cosine similarity ∈ [-1, 1]
        self.assertTrue((out.alignment_score >= -1.01).all())
        self.assertTrue((out.alignment_score <= 1.01).all())

    def test_fusion_loss(self):
        from modules.price_cycle import MultiScaleFusion, MultiScaleFusionConfig
        config = MultiScaleFusionConfig(micro_dim=32, macro_dim=32, fusion_dim=32)
        fusion = MultiScaleFusion(config)
        out = fusion(torch.randn(4, 32), torch.randn(4, 32))
        losses = fusion.compute_loss(out, torch.randint(0, 3, (4,)))
        self.assertIn("decision", losses)
        self.assertTrue(torch.isfinite(losses["total"]))


class TestMultiScaleSystemIntegration(unittest.TestCase):
    """Integration: LOB Transformer (micro) + Price Cycle (macro) + Fusion."""

    def test_full_system_decide(self):
        from modules.deep_lob import HierarchicalLOBTransformer, DeepLOBConfig
        from modules.price_cycle import (
            PriceCycleModel, PriceCycleConfig,
            MultiScaleFusion, MultiScaleFusionConfig,
            MultiScaleTradingSystem,
        )

        lob = HierarchicalLOBTransformer(DeepLOBConfig.small_dev())
        cycle = PriceCycleModel(PriceCycleConfig.small_dev())
        fusion = MultiScaleFusion(MultiScaleFusionConfig(
            micro_dim=32, macro_dim=32, fusion_dim=32,
        ))
        system = MultiScaleTradingSystem(lob, cycle, fusion)

        # LOB inputs
        B, T, N = 2, 8, 20
        order_features = torch.randn(B, T, N, 7)
        order_features[..., 0] = torch.randint(0, 2, (B, T, N))
        order_features[..., 1] = torch.randint(0, 4, (B, T, N))
        order_features[..., 5] = torch.randint(0, 8, (B, T, N))
        order_masks = torch.ones(B, T, N, dtype=torch.bool)
        # Cycle inputs
        bar_features = torch.randn(B, 100, 16)

        result = system.decide(
            order_features, order_masks, bar_features,
        )
        self.assertEqual(result["decision_probs"].shape, (B, 3))
        self.assertIn("micro_direction", result)
        self.assertIn("alignment_score", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
