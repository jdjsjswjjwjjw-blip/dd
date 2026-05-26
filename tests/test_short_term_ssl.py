"""Tests for short-term SSL heads + target builders."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.deep_lob.short_term_heads import (
    ShortTermHeads,
    ShortTermHeadsConfig,
    ShortTermOutput,
    ShortTermTargets,
    compute_short_term_loss,
)
from modules.deep_lob.short_term_targets import (
    build_all_short_term_targets,
    build_next_gap_fill_targets,
    build_next_imbalance_shift_targets,
    build_next_liquidity_sweep_targets,
    build_next_micro_target_targets,
    build_next_wall_break_targets,
)


# ════════════════════════════════════════════════════════════════════
# Short-term heads — NN module
# ════════════════════════════════════════════════════════════════════
class TestShortTermHeads:
    def test_forward_shapes(self):
        cfg = ShortTermHeadsConfig(shared_dim=64)
        model = ShortTermHeads(cfg)
        emb = torch.randn(8, 64)
        out = model(emb)
        assert isinstance(out, ShortTermOutput)
        assert out.wall_break_logits.shape == (8, 3)
        assert out.imbalance_shift_logit.shape == (8,)
        assert out.micro_target_logits.shape == (8, 3)
        assert out.liquidity_sweep_logit.shape == (8,)
        assert out.gap_fill_logit.shape == (8,)

    def test_probability_methods(self):
        cfg = ShortTermHeadsConfig(shared_dim=32)
        model = ShortTermHeads(cfg)
        emb = torch.randn(4, 32)
        out = model(emb)
        # Multiclass probs sum to ~1
        wb_probs = out.wall_break_probs()
        assert torch.allclose(wb_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        mt_probs = out.micro_target_probs()
        assert torch.allclose(mt_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        # Binary probs in [0, 1]
        is_prob = out.imbalance_shift_prob()
        assert (is_prob >= 0).all() and (is_prob <= 1).all()

    def test_backward_pass(self):
        cfg = ShortTermHeadsConfig(shared_dim=64)
        model = ShortTermHeads(cfg)
        emb = torch.randn(8, 64, requires_grad=True)
        out = model(emb)
        targets = ShortTermTargets(
            wall_break=torch.tensor([0, 1, 2, 1, 0, 1, 2, 1]),
            imbalance_shift=torch.tensor([0., 1., 0., 1., 0., 1., 1., 0.]),
            micro_target=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
            liquidity_sweep=torch.tensor([0., 1., 0., 0., 1., 0., 1., 0.]),
            gap_fill=torch.tensor([1., 0., 0., 1., 0., 1., 0., 1.]),
        )
        total, metrics = compute_short_term_loss(out, targets, cfg)
        total.backward()
        # Some gradient must reach emb
        assert emb.grad is not None
        assert emb.grad.abs().sum().item() > 0
        # All metrics present
        for k in ("loss_wall_break", "loss_imbalance_shift", "loss_micro_target",
                  "loss_liquidity_sweep", "loss_gap_fill", "loss_total_short_term"):
            assert k in metrics

    def test_partial_targets(self):
        """If some targets are None, those losses should be skipped."""
        cfg = ShortTermHeadsConfig(shared_dim=32)
        model = ShortTermHeads(cfg)
        emb = torch.randn(4, 32)
        out = model(emb)
        targets = ShortTermTargets(
            wall_break=torch.tensor([0, 1, 2, 1]),
            # other 4 are None
        )
        total, metrics = compute_short_term_loss(out, targets, cfg)
        assert "loss_wall_break" in metrics
        assert "loss_imbalance_shift" not in metrics
        assert total.item() > 0

    def test_param_count(self):
        cfg = ShortTermHeadsConfig(shared_dim=64)
        model = ShortTermHeads(cfg)
        n = model.num_parameters()
        # 5 heads × ~2k params each ≈ 10k
        assert 5_000 < n < 30_000, f"unexpected param count: {n}"


# ════════════════════════════════════════════════════════════════════
# Wall-break targets
# ════════════════════════════════════════════════════════════════════
class TestWallBreakTargets:
    def _synth(self, N=20, T=10, P=20):
        rng = np.random.RandomState(0)
        raw = rng.rand(N, T, P) * 50
        # Make sure walls exist by injecting big sizes in some snapshots
        raw[5, -1, 12] = 500   # ask wall (idx 12, ask side >= 10)
        raw[10, -1, 3] = 500   # bid wall (idx 3, bid side < 10)
        return raw

    def test_shape(self):
        raw = self._synth()
        high = np.full(20, 1.30)
        low = np.full(20, 1.29)
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (20, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (20, 1))
        sb = np.zeros(20, dtype=bool)
        label, valid = build_next_wall_break_targets(
            raw, high, low, bid_prices, ask_prices, sb, forward_bars=2,
        )
        assert label.shape == (20,)
        assert valid.shape == (20,)
        assert label.dtype == np.int64
        assert valid.dtype == bool

    def test_ask_wall_broken_upward(self):
        N, T, P = 5, 5, 20
        raw = np.full((N, T, P), 10.0)
        # ask wall at idx 12 in bar 1
        raw[1, -1, 12] = 1000
        # Simulate high price moving past the wall in bar 2-3
        high = np.array([1.3000, 1.3010, 1.3050, 1.3060, 1.3070])
        low = np.array([1.2990, 1.3000, 1.3005, 1.3010, 1.3020])
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (N, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (N, 1))
        # ask price at idx 12 = ask_prices[..., 12-10] = ask_prices[..., 2] = 1.303
        sb = np.zeros(N, dtype=bool)
        label, valid = build_next_wall_break_targets(
            raw, high, low, bid_prices, ask_prices, sb, forward_bars=2,
        )
        # Bar 1: ask wall at 1.303; forward high in bars 2-3 = max(1.305, 1.306) = 1.306 > 1.303
        assert valid[1]
        assert label[1] == 2  # upward break

    def test_no_wall_no_label(self):
        N, T, P = 5, 5, 20
        raw = np.full((N, T, P), 10.0)  # uniform, no walls
        high = np.full(N, 1.30)
        low = np.full(N, 1.29)
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (N, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (N, 1))
        sb = np.zeros(N, dtype=bool)
        label, valid = build_next_wall_break_targets(
            raw, high, low, bid_prices, ask_prices, sb,
        )
        # Uniform raw_depth → no walls (3× median = 3× same value, no level > 3×)
        assert valid.sum() == 0

    def test_session_break_blocks_label(self):
        N, T, P = 5, 5, 20
        raw = np.full((N, T, P), 10.0)
        raw[1, -1, 12] = 1000
        high = np.full(N, 1.305)
        low = np.full(N, 1.295)
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (N, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (N, 1))
        sb = np.zeros(N, dtype=bool)
        sb[2] = True  # break right after bar 1
        label, valid = build_next_wall_break_targets(
            raw, high, low, bid_prices, ask_prices, sb, forward_bars=2,
        )
        assert not valid[1]


# ════════════════════════════════════════════════════════════════════
# Imbalance-shift targets
# ════════════════════════════════════════════════════════════════════
class TestImbalanceShiftTargets:
    def test_shift_detected(self):
        # OBI: +0.3, +0.2, -0.4, ...  → at i=0, shift in window
        obi = np.array([0.3, 0.2, -0.4, -0.5, 0.1])
        sb = np.zeros(5, dtype=bool)
        label, valid = build_next_imbalance_shift_targets(
            obi, sb, forward_bars=2, min_magnitude=0.10,
        )
        assert valid[0]
        assert label[0] == 1  # sign flipped at i=2

    def test_no_shift_when_below_magnitude(self):
        # OBI = noise around zero
        obi = np.array([0.05, -0.03, 0.04, -0.02])
        sb = np.zeros(4, dtype=bool)
        label, valid = build_next_imbalance_shift_targets(
            obi, sb, forward_bars=2, min_magnitude=0.10,
        )
        # all below magnitude → all invalid
        assert valid.sum() == 0

    def test_strong_continuation(self):
        # OBI stays positive — no shift
        obi = np.array([0.5, 0.4, 0.3, 0.45, 0.5])
        sb = np.zeros(5, dtype=bool)
        label, valid = build_next_imbalance_shift_targets(obi, sb)
        assert valid[0]
        assert label[0] == 0  # no flip


# ════════════════════════════════════════════════════════════════════
# Micro-target targets
# ════════════════════════════════════════════════════════════════════
class TestMicroTargetTargets:
    def test_1r_hit_long(self):
        close = np.array([1.30, 1.301, 1.305, 1.310, 1.312])
        high = close + 0.001
        low = close - 0.001
        atr = np.full(5, 0.005)  # 50 pips ATR
        bias = np.array([1, 1, 1, 1, 1])  # all LONG
        sb = np.zeros(5, dtype=bool)
        label, valid = build_next_micro_target_targets(
            close, high, low, atr, bias, sb,
            forward_bars=4, target_R_low=0.5, target_R_high=1.0,
        )
        # bar 0: entry=1.30, 1R = 1.30 + 0.005 = 1.305
        # fwd high in (0, 4] = max(1.302, 1.306, 1.311, 1.313) = 1.313
        # move = 1.313 - 1.30 = 0.013 / 0.005 = 2.6R → label 2
        assert valid[0]
        assert label[0] == 2

    def test_no_target_when_no_bias(self):
        close = np.array([1.30, 1.301, 1.302])
        high = close + 0.0005
        low = close - 0.0005
        atr = np.full(3, 0.005)
        bias = np.zeros(3, dtype=int)  # no direction
        sb = np.zeros(3, dtype=bool)
        label, valid = build_next_micro_target_targets(
            close, high, low, atr, bias, sb,
        )
        assert valid.sum() == 0


# ════════════════════════════════════════════════════════════════════
# Liquidity-sweep targets
# ════════════════════════════════════════════════════════════════════
class TestLiquiditySweepTargets:
    def test_sweep_detected(self):
        # Stable trade volume, then a 5× spike
        vol = np.array([10.0] * 50 + [50.0] + [10.0] * 5)  # spike at idx 50
        sb = np.zeros(len(vol), dtype=bool)
        label, valid = build_next_liquidity_sweep_targets(
            vol, sb, forward_bars=2, sweep_multiplier=3.0, rolling_window=20,
        )
        # bar 48 looks forward to bars 49-50; bar 50 = 5× median → sweep
        assert valid[48]
        assert label[48] == 1

    def test_no_sweep_with_stable_volume(self):
        vol = np.full(30, 10.0)
        sb = np.zeros(30, dtype=bool)
        label, valid = build_next_liquidity_sweep_targets(
            vol, sb, forward_bars=2, sweep_multiplier=3.0, rolling_window=10,
        )
        assert label.sum() == 0


# ════════════════════════════════════════════════════════════════════
# Gap-fill targets
# ════════════════════════════════════════════════════════════════════
class TestGapFillTargets:
    def test_no_gap_no_valid(self):
        N, T, P = 5, 3, 20
        # No gaps — uniform liquidity
        raw = np.full((N, T, P), 10.0)
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (N, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (N, 1))
        high = np.full(N, 1.31)
        low = np.full(N, 1.29)
        sb = np.zeros(N, dtype=bool)
        label, valid = build_next_gap_fill_targets(
            raw, bid_prices, ask_prices, high, low, sb,
        )
        assert valid.sum() == 0

    def test_gap_touched(self):
        N, T, P = 4, 3, 20
        raw = np.full((N, T, P), 10.0)
        # Inject a gap at level 3 (bid side) — zero with non-zero neighbors
        raw[0, -1, 3] = 0
        bid_prices = np.tile(np.linspace(1.299, 1.290, 10), (N, 1))
        ask_prices = np.tile(np.linspace(1.301, 1.310, 10), (N, 1))
        # gap is at bid_prices[0, half-1-3] = bid_prices[0, 6] = ?
        # linspace(1.299, 1.290, 10) → idx 6 = 1.293
        # set forward low to reach 1.293
        high = np.array([1.300, 1.295, 1.294, 1.300])
        low = np.array([1.299, 1.294, 1.292, 1.295])  # low at bar 2 = 1.292 < 1.293
        sb = np.zeros(N, dtype=bool)
        label, valid = build_next_gap_fill_targets(
            raw, bid_prices, ask_prices, high, low, sb, forward_bars=2,
        )
        assert valid[0]
        assert label[0] == 1  # gap touched


# ════════════════════════════════════════════════════════════════════
# Bundle builder
# ════════════════════════════════════════════════════════════════════
class TestBundle:
    def test_all_five_returned(self):
        N, T, P = 20, 5, 20
        rng = np.random.RandomState(0)
        raw = rng.rand(N, T, P) * 50
        raw[5, -1, 12] = 500  # plant a wall
        out = build_all_short_term_targets(
            close=np.full(N, 1.30),
            high=np.full(N, 1.305),
            low=np.full(N, 1.295),
            atr=np.full(N, 0.005),
            obi_signed=rng.uniform(-0.5, 0.5, N),
            trade_volume=rng.rand(N) * 100,
            bias_direction=rng.choice([-1, 0, 1], N),
            is_session_break=np.zeros(N, dtype=bool),
            raw_depth_seq=raw,
            bid_prices=np.tile(np.linspace(1.299, 1.290, 10), (N, 1)),
            ask_prices=np.tile(np.linspace(1.301, 1.310, 10), (N, 1)),
        )
        for k in ("wall_break", "imbalance_shift", "micro_target",
                  "liquidity_sweep", "gap_fill"):
            assert k in out, f"missing {k}"
            assert "label" in out[k]
            assert "valid" in out[k]
            assert out[k]["label"].shape == (N,)
            assert out[k]["valid"].shape == (N,)
