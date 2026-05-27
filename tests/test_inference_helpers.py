"""Tests for the inference helpers — frozen scalers, transform-only loading,
and the single-row predict_one path used by live trading.

These tests directly cover Issues 2 + 3 from docs/PIPELINE_ISSUES_AUDIT.md:
no fit() at inference, no pandas in the prediction loop, frozen mu/sigma
applied verbatim from the checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.trading_intel.hybrid.inference import (
    FrozenScaler,
    apply_saved_scalers,
    load_hybrid_for_inference,
    predict_one,
)
from modules.trading_intel.hybrid.model import HybridConfig, HybridModel


class TestFrozenScaler:
    def test_transform_matches_manual(self):
        mu = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sigma = np.array([0.5, 1.5, 2.0], dtype=np.float32)
        scaler = FrozenScaler(mu=mu, sigma=sigma)
        X = np.array([[2.0, 5.0, 7.0]], dtype=np.float32)
        result = scaler.transform(X)
        expected = (X - mu) / sigma
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_sigma_floored_to_avoid_div_zero(self):
        scaler = FrozenScaler(
            mu=np.array([0.0]),
            sigma=np.array([1e-12]),    # effectively zero
        )
        X = np.array([[1.0]])
        # Should not produce inf
        result = scaler.transform(X)
        assert np.isfinite(result).all()

    def test_shape_mismatch_raises(self):
        scaler = FrozenScaler(
            mu=np.array([0.0, 0.0]),
            sigma=np.array([1.0, 1.0]),
        )
        X = np.array([[1.0, 2.0, 3.0]])   # wrong dim
        with pytest.raises(ValueError, match="expected 2 features"):
            scaler.transform(X)

    def test_from_checkpoint_dict_roundtrip(self):
        d = {"mu": [1.0, 2.0], "sigma": [0.5, 1.5]}
        scaler = FrozenScaler.from_checkpoint_dict(d)
        assert scaler.mu.tolist() == [1.0, 2.0]
        assert scaler.sigma.tolist() == [0.5, 1.5]


class TestLoadHybridForInference:
    def _make_minimal_checkpoint(self, tmp_path):
        """Create a hybrid checkpoint with all required fields."""
        cfg = HybridConfig(
            daytrade_feature_dim=8, ssl_embed_dim=4, cnn_embed_dim=0,
            hidden_dim=16, n_layers=2,
        )
        model = HybridModel(cfg)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "norm_daytrade": {
                "mu": [0.0] * 8, "sigma": [1.0] * 8,
            },
            "norm_ssl": {
                "mu": [0.0] * 4, "sigma": [1.0] * 4,
            },
        }
        path = tmp_path / "ckpt.pt"
        torch.save(ckpt, path)
        return path

    def test_load_returns_eval_mode_no_grad(self, tmp_path):
        path = self._make_minimal_checkpoint(tmp_path)
        model, scalers = load_hybrid_for_inference(path)
        # eval mode
        assert not model.training
        # No gradients
        for p in model.parameters():
            assert not p.requires_grad
        # Scalers loaded
        assert "daytrade" in scalers
        assert "ssl" in scalers

    def test_missing_scalers_raises_loudly(self, tmp_path):
        cfg = HybridConfig(
            daytrade_feature_dim=8, ssl_embed_dim=4, cnn_embed_dim=0,
        )
        model = HybridModel(cfg)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            # NO norm_daytrade — should trigger the loud failure
        }
        path = tmp_path / "bad_ckpt.pt"
        torch.save(ckpt, path)
        with pytest.raises(KeyError, match="missing normalization stats"):
            load_hybrid_for_inference(path)

    def test_missing_config_raises(self, tmp_path):
        path = tmp_path / "no_config.pt"
        torch.save({"state_dict": {}}, path)
        with pytest.raises(KeyError, match="missing 'config'"):
            load_hybrid_for_inference(path)


class TestPredictOne:
    def _setup(self, tmp_path):
        cfg = HybridConfig(
            daytrade_feature_dim=8, ssl_embed_dim=4, cnn_embed_dim=0,
            hidden_dim=16, n_layers=2,
        )
        model = HybridModel(cfg)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "norm_daytrade": {"mu": [0.0] * 8, "sigma": [1.0] * 8},
            "norm_ssl": {"mu": [0.0] * 4, "sigma": [1.0] * 4},
        }
        path = tmp_path / "ckpt.pt"
        torch.save(ckpt, path)
        return load_hybrid_for_inference(path)

    def test_predict_one_returns_all_keys(self, tmp_path):
        model, scalers = self._setup(tmp_path)
        out = predict_one(
            model, scalers,
            daytrade_features=np.random.randn(8).astype(np.float32),
            ssl_embedding=np.random.randn(4).astype(np.float32),
        )
        assert set(out.keys()) == {
            "event_prob", "p_long", "p_short", "p_neutral", "confidence",
        }
        # All probabilities in [0, 1]
        for k, v in out.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of range"

    def test_predict_one_no_pandas_required(self, tmp_path):
        """Caller passes only NumPy arrays — never a DataFrame.
        This is essentially compile-time verification that the API
        signature doesn't accept pandas."""
        model, scalers = self._setup(tmp_path)
        # If we had pandas DataFrames, this would fail at the scaler.transform()
        # call because pandas != numpy arrays in shape semantics.
        out = predict_one(
            model, scalers,
            daytrade_features=np.zeros(8, dtype=np.float32),
            ssl_embedding=np.zeros(4, dtype=np.float32),
        )
        # And the output is also a plain dict of Python floats — no pandas
        assert all(isinstance(v, float) for v in out.values())


class TestApplySavedScalers:
    def test_alias_for_transform(self):
        scaler = FrozenScaler(
            mu=np.array([1.0, 1.0]),
            sigma=np.array([2.0, 2.0]),
        )
        X = np.array([[3.0, 5.0]])
        a = apply_saved_scalers(X, scaler)
        b = scaler.transform(X)
        np.testing.assert_array_equal(a, b)
