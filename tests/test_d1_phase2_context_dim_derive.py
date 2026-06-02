"""D1 Phase 2 — the context dim is DERIVED from data, never a fixed 138.

After D1 removed the hand-built depth features, the real number of causal
context features ≠ the old hardcoded 138 (and is data-dependent). The fix:
  • the Dataset / context builder emits the REAL dim (no phantom zero-pad,
    no silent truncation),
  • every model-construction site derives n_input_features from that emitted
    dim (pretrain_lob / verify_integration / diagnose_nan),
  • config.py keeps 138 only as a runtime-overridden FALLBACK.

These tests lock that contract:
  1. `_build_context_features(target_dim=None)` emits exactly len(cols) — no pad.
  2. the explicit-cap path still pads/truncates (back-compat), proving the
     phantom pad is OPT-IN, not the default.
  3. ContextEncoder built at a DERIVED dim D accepts (B, D); both its first
     Linear and its input LayerNorm track D.
  4. TEETH: a model left at the default 138 fed a D≠138 context RAISES — so the
     derive is load-bearing, not cosmetic (a regression would crash, not lie).

(quant-rigor-guard R6: no phantom/dead features; R10: honest, reproducible
config — the dim reflects reality and is persisted with the checkpoint.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.data_loader import _build_context_features

torch = pytest.importorskip("torch")
from modules.deep_lob.config import ContextEncoderConfig, DeepLOBConfig
from modules.deep_lob.context_encoder import ContextEncoder


def _row(cols):
    return pd.Series({c: float(i + 1) for i, c in enumerate(cols)})


class TestContextBuilderHonestDim:
    def test_none_target_emits_exact_len_no_pad(self):
        cols = [f"feat_{i}" for i in range(104)]      # post-D1-ish, ≠ 138
        arr = _build_context_features(_row(cols), cols, target_dim=None)
        assert arr.shape == (104,)                    # the REAL dim, no phantom pad
        assert not np.any(arr == 0.0)                 # every slot is a real feature

    def test_missing_column_filled_but_dim_unchanged(self):
        cols = [f"feat_{i}" for i in range(10)]
        row = _row(cols[:7])                           # 3 cols absent from the row
        arr = _build_context_features(row, cols, target_dim=None)
        assert arr.shape == (10,)                      # dim follows cols, not the row
        assert (arr[7:] == 0.0).all()                  # absent → 0.0, still real slots

    def test_explicit_cap_still_pads_and_truncates(self):
        cols = [f"feat_{i}" for i in range(5)]
        padded = _build_context_features(_row(cols), cols, target_dim=8)
        assert padded.shape == (8,) and (padded[5:] == 0.0).all()   # opt-in pad
        truncated = _build_context_features(_row(cols), cols, target_dim=3)
        assert truncated.shape == (3,)                              # opt-in truncate


class TestEncoderDerivesDim:
    @pytest.mark.parametrize("D", [37, 104, 211])
    def test_encoder_accepts_derived_dim(self, D):
        enc = ContextEncoder(ContextEncoderConfig(n_input_features=D, output_dim=16))
        # both the input LayerNorm and the first Linear must track D
        assert enc.input_ln.normalized_shape == (D,)
        first_linear = enc.encoder[0] if isinstance(enc.encoder, torch.nn.Sequential) else enc.encoder
        assert first_linear.in_features == D
        out = enc(torch.randn(4, D))
        assert out.shape == (4, 16)

    def test_default_config_mismatch_is_caught(self):
        """TEETH: the OLD behavior (model fixed at 138) fed a real D≠138 context
        must RAISE — proving the derive is load-bearing. If a future regression
        re-pins the model to 138 while the data emits 104, training crashes here
        rather than silently training on 34 dead zero-pad dims."""
        D_real = 104
        enc_default = ContextEncoder(ContextEncoderConfig())   # n_input_features=138
        assert enc_default.config.n_input_features == 138
        with pytest.raises(RuntimeError):
            enc_default(torch.randn(2, D_real))                # 104 vs 138 → matmul error


class TestHierarchicalModelDerive:
    def test_full_config_derive_round_trips(self):
        """Setting config.context_encoder.n_input_features (what pretrain_lob /
        verify_integration / diagnose_nan do) builds an encoder at that dim."""
        D = 104
        cfg = DeepLOBConfig()
        assert cfg.context_encoder.n_input_features == 138    # fallback default
        cfg.context_encoder.n_input_features = D              # the DERIVE step
        enc = ContextEncoder(cfg.context_encoder)
        assert enc.input_ln.normalized_shape == (D,)
        assert enc(torch.randn(3, D)).shape[0] == 3
