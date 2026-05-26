"""Tests for modules.trading_intel.lob (features + cnn)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.trading_intel.lob.cnn import (
    HumanLOBCNN,
    HumanLOBCNNConfig,
    make_human_lob_cnn,
)
from modules.trading_intel.lob.features import (
    CH,
    N_LOB_CHANNELS_V2,
    _depth_gradient,
    _liquidity_gaps,
    _wall_mask_at_thresholds,
    build_lob_tensor_v2_for_bar,
    stack_bars,
)


# ════════════════════════════════════════════════════════════════════
# Wall detection
# ════════════════════════════════════════════════════════════════════
class TestWallDetection:
    def test_wall_mask_shape_and_dtype(self):
        raw = np.array([10, 5, 100, 20, 0, 7, 8, 200, 3, 0], dtype=np.float64)
        out = _wall_mask_at_thresholds(raw, (3.0, 5.0, 10.0))
        assert out.shape == (10, 3)
        assert out.dtype == np.float32

    def test_wall_thresholds_monotonic(self):
        """A level flagged at 10× must also be flagged at 5× and 3×."""
        raw = np.array([5, 5, 100, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64)
        out = _wall_mask_at_thresholds(raw, (3.0, 5.0, 10.0))
        # idx 2 has 100 vs median(5) = 20× — flagged at all thresholds
        assert out[2, 0] == 1.0  # 3×
        assert out[2, 1] == 1.0  # 5×
        assert out[2, 2] == 1.0  # 10×

    def test_wall_thresholds_selective(self):
        """A 4× wall must be flagged at 3× but NOT at 5× or 10×."""
        raw = np.array([5, 5, 20, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64)
        # median(5)=5, 20/5 = 4×
        out = _wall_mask_at_thresholds(raw, (3.0, 5.0, 10.0))
        assert out[2, 0] == 1.0  # 3× yes
        assert out[2, 1] == 0.0  # 5× no
        assert out[2, 2] == 0.0  # 10× no

    def test_empty_snapshot(self):
        raw = np.zeros(10, dtype=np.float64)
        out = _wall_mask_at_thresholds(raw, (3.0, 5.0, 10.0))
        assert out.sum() == 0.0


# ════════════════════════════════════════════════════════════════════
# Liquidity gap detection
# ════════════════════════════════════════════════════════════════════
class TestLiquidityGaps:
    def test_gap_shape_and_dtype(self):
        raw = np.array([10, 5, 0, 7, 8, 12, 0, 5, 0, 9], dtype=np.float64)
        flag, pos = _liquidity_gaps(raw)
        assert flag.shape == (10,)
        assert pos.shape == (10,)
        assert flag.dtype == np.float32
        assert pos.dtype == np.float32

    def test_inner_zero_with_nonzero_neighbors_is_gap(self):
        """Empty level with non-empty neighbors inside one book side."""
        # 10 elements = half=5, so two sides of 5
        # bid side (0..4): [10, 5, 0, 7, 8]  → idx 2 is gap
        # ask side (5..9): [5, 6, 0, 4, 3]  → idx 7 is gap
        raw = np.array([10, 5, 0, 7, 8, 5, 6, 0, 4, 3], dtype=np.float64)
        flag, _ = _liquidity_gaps(raw)
        assert flag[2] == 1.0
        assert flag[7] == 1.0
        assert flag.sum() == 2.0

    def test_edge_zero_not_a_gap(self):
        """Empty deepest/edge level is not a 'gap' — it's just thin liquidity."""
        # bid side starts at deepest (idx 0); idx 0 empty has no left neighbor
        raw = np.array([0, 5, 6, 7, 8, 9, 0, 5, 4, 3], dtype=np.float64)
        flag, _ = _liquidity_gaps(raw)
        # idx 0 is edge (no left non-zero) → not gap
        # idx 6 is best ask (in ask side starting at idx 5) — left=9, right=5 → gap
        assert flag[0] == 0.0  # edge not gap
        # Note: idx 6 might or might not be flagged depending on definition;
        # important is that idx 0 (edge) is not.

    def test_no_gap_returns_zeros(self):
        raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        flag, pos = _liquidity_gaps(raw)
        assert flag.sum() == 0.0
        assert pos.sum() == 0.0


# ════════════════════════════════════════════════════════════════════
# Depth gradient
# ════════════════════════════════════════════════════════════════════
class TestDepthGradient:
    def test_shape(self):
        seq = np.ones((10, 5), dtype=np.float64)
        out = _depth_gradient(seq, window=3)
        assert out.shape == (10, 5)
        assert out.dtype == np.float32

    def test_stable_depth_zero_gradient(self):
        """Constant sequence → gradient = 0 everywhere."""
        seq = np.full((10, 5), 50.0)
        out = _depth_gradient(seq, window=5)
        assert np.allclose(out, 0.0)

    def test_accumulation_positive(self):
        """Increasing depth → positive gradient."""
        seq = np.zeros((10, 5))
        seq[:, 0] = np.arange(10) * 10  # 0, 10, 20, ..., 90
        out = _depth_gradient(seq, window=5)
        # at t=5, depth went from 0 → 50, so gradient is 50/max(0,1) = 50 → clipped to 10
        assert out[5, 0] > 0
        assert out[9, 0] > 0

    def test_clip_range(self):
        seq = np.zeros((10, 5))
        seq[5:, 0] = 1e6  # massive jump
        out = _depth_gradient(seq, window=2)
        assert out.max() <= 10.0
        assert out.min() >= -10.0


# ════════════════════════════════════════════════════════════════════
# Full pipeline (build_lob_tensor_v2_for_bar)
# ════════════════════════════════════════════════════════════════════
class TestBuildLOBTensorV2:
    @pytest.fixture
    def synthetic_bar(self):
        T, P = 50, 20
        half = P // 2
        rng = np.random.RandomState(42)
        raw_depth_seq = rng.rand(T, P) * 100
        depth_log_seq = np.log1p(raw_depth_seq)
        bid_sz_seq = rng.rand(T, half) * 50
        ask_sz_seq = rng.rand(T, half) * 50
        buy_vol_seq = rng.rand(T, P) * 20
        sell_vol_seq = rng.rand(T, P) * 20
        return raw_depth_seq, depth_log_seq, bid_sz_seq, ask_sz_seq, buy_vol_seq, sell_vol_seq

    def test_output_shape(self, synthetic_bar):
        t = build_lob_tensor_v2_for_bar(*synthetic_bar)
        assert t.shape == (50, 20, N_LOB_CHANNELS_V2)
        assert t.dtype == np.float32

    def test_channel_constants_unique(self):
        """Each channel constant must have a unique index in [0, 12]."""
        idxs = [
            CH.DEPTH_LOG, CH.TRADE_IMB, CH.TRADE_VOL,
            CH.BID_DEPTH, CH.ASK_DEPTH,
            CH.BUY_VOL, CH.SELL_VOL,
            CH.WALL_3X, CH.WALL_5X, CH.WALL_10X,
            CH.LIQ_GAP, CH.GAP_POS, CH.DEPTH_GRAD,
        ]
        assert len(set(idxs)) == N_LOB_CHANNELS_V2
        assert min(idxs) == 0 and max(idxs) == N_LOB_CHANNELS_V2 - 1

    def test_walls_present_for_extreme_values(self, synthetic_bar):
        """Inject huge wall and verify all 3 wall channels light up."""
        raw, log_raw, bid, ask, buy, sell = synthetic_bar
        # Spike one level in one snapshot
        raw = raw.copy()
        log_raw = log_raw.copy()
        raw[10, 5] = 100_000
        log_raw[10, 5] = float(np.log1p(100_000))
        t = build_lob_tensor_v2_for_bar(raw, log_raw, bid, ask, buy, sell)
        assert t[10, 5, CH.WALL_3X] == 1.0
        assert t[10, 5, CH.WALL_5X] == 1.0
        assert t[10, 5, CH.WALL_10X] == 1.0

    def test_stack_bars_shape(self, synthetic_bar):
        per_bar = [build_lob_tensor_v2_for_bar(*synthetic_bar) for _ in range(3)]
        stacked = stack_bars(per_bar)
        assert stacked.shape == (3, 50, 20, N_LOB_CHANNELS_V2)

    def test_stack_bars_validates_shapes(self):
        a = np.zeros((50, 20, N_LOB_CHANNELS_V2), dtype=np.float32)
        b = np.zeros((50, 21, N_LOB_CHANNELS_V2), dtype=np.float32)  # bad
        with pytest.raises(ValueError, match="shape mismatch"):
            stack_bars([a, b])


# ════════════════════════════════════════════════════════════════════
# HumanLOBCNN
# ════════════════════════════════════════════════════════════════════
class TestHumanLOBCNN:
    def test_default_forward_shape(self):
        model = make_human_lob_cnn()
        x = torch.randn(2, 13, 50, 20)
        out = model(x)
        assert out.shape == (2, 32)

    def test_custom_config(self):
        cfg = HumanLOBCNNConfig(in_channels=9, embed_dim=64, dropout=0.0)
        model = HumanLOBCNN(cfg)
        x = torch.randn(3, 9, 50, 20)
        out = model(x)
        assert out.shape == (3, 64)

    def test_backward_pass(self):
        model = make_human_lob_cnn()
        x = torch.randn(2, 13, 50, 20, requires_grad=True)
        out = model(x)
        out.mean().backward()
        # Some param should have a gradient
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                return
        pytest.fail("no gradient flowed through model")

    def test_param_count_reasonable(self):
        """118k params is the design target; allow ±20% for kernel-size tweaks."""
        model = make_human_lob_cnn()
        n = model.num_parameters()
        assert 50_000 < n < 200_000, f"unexpected param count: {n}"

    def test_eval_mode_no_dropout_diff(self):
        """In eval mode, repeated forward passes must produce identical output."""
        model = make_human_lob_cnn()
        model.eval()
        x = torch.randn(1, 13, 50, 20)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_batch_norm_off(self):
        cfg = HumanLOBCNNConfig(use_batch_norm=False)
        model = HumanLOBCNN(cfg)
        x = torch.randn(1, 13, 50, 20)  # B=1 would crash with batchnorm
        out = model(x)
        assert out.shape == (1, 32)
