"""Leakage-guard for modules/trading_intel/training/train_hybrid.py.

A full leakage sweep (this commit's study) found two forward-looking columns
that train_hybrid's substring LEAKAGE_PATTERNS missed, so _select_features
would have admitted them into the model's feature matrix X:

  • fwd_ret_clean     — (close[t+N] - close[t]) / close[t], a forward return
  • effective_horizon — = trade_duration, the bars-until-barrier-touch known
                         only after the fact

Both are now excluded. These tests lock that, prove the existing exclusions
still hold, and (R6) prove genuinely CAUSAL features are NOT over-excluded —
specifically market_state_code, which is derived from the causal regime_label
and is redundant-at-most, not a leak, so it must stay IN X.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.trading_intel.training.train_hybrid import (
    _is_leakage_col, _select_features,
)


# ── (1) TEETH: the two newly-found forward-looking leaks are excluded ──────
class TestNewlyClosedLeaks:
    def test_fwd_ret_clean_excluded(self):
        assert _is_leakage_col("fwd_ret_clean"), (
            "fwd_ret_clean is a forward return — must be excluded from X"
        )

    def test_effective_horizon_excluded(self):
        assert _is_leakage_col("effective_horizon"), (
            "effective_horizon = trade_duration (post-hoc) — must be excluded"
        )

    def test_fwd_ret_prefix_fail_closes_future_siblings(self):
        # the 'fwd_ret' substring must catch any future fwd_ret_* column
        assert _is_leakage_col("fwd_ret_5m")
        assert _is_leakage_col("fwd_ret_raw")


# ── (2) regression: every previously-excluded leak stays excluded ──────────
class TestExistingExclusionsHold:
    def test_known_leaks_still_excluded(self):
        for c in (
            "forward_return", "path_outcome", "bias_label", "label_confidence",
            "label_end_ts", "label_horizon_steps", "trade_duration",
            "soft_label_long", "soft_label_short", "neutral_reason",
            "event_label_tier", "train_event_flag", "event_flag", "event_score",
            "is_event", "event_direction", "signal_quality",
            "mfe", "mae", "stop_first_flag", "time_to_first_touch",
            "net_expectancy_proxy", "neutral_type", "tradability_label",
            "exec_label", "exec_valid", "next_price_delta", "next_price_delta_valid",
        ):
            assert _is_leakage_col(c), f"regression: {c!r} should be excluded"


# ── (3) R6: causal features must NOT be over-excluded ──────────────────────
class TestCausalFeaturesKept:
    def test_market_state_code_is_kept(self):
        """market_state_code is derived from the causal regime_label — it is
        NOT forward-looking. Excluding it would be a taxonomy error (R6, not
        R1). It must remain available to the model."""
        assert not _is_leakage_col("market_state_code"), (
            "market_state_code is causal — must not be flagged as leakage"
        )

    def test_genuine_features_not_flagged(self):
        for c in ("current_vwap", "tick_count", "atr_14", "regime_cluster",
                  "london_sess_high", "dist_to_vwap_atr", "cvd", "cvd_bar_5m",
                  "vwap_z_score", "order_flow_imbalance", "session_phase"):
            assert not _is_leakage_col(c), f"over-exclusion: {c!r} is a real feature"


# ── (4) end-to-end via _select_features over a representative frame ────────
class TestSelectFeaturesEndToEnd:
    def _frame(self):
        n = 20
        return pd.DataFrame({
            # forward / label-side (must be dropped)
            "fwd_ret_clean":      np.random.randn(n).astype(np.float32),
            "effective_horizon":  np.random.randint(1, 24, n).astype(np.int32),
            "forward_return":     np.random.randn(n).astype(np.float32),
            "bias_label":         np.random.randint(0, 3, n).astype(np.int8),
            "mfe_atr":            np.random.randn(n).astype(np.float32),
            "net_expectancy_proxy": np.random.randn(n).astype(np.float32),
            # string label-side (dropped by numeric filter regardless)
            "market_state_label": np.array(["chop"] * n, dtype=object),
            "regime_label_grouped": np.array(["rare"] * n, dtype=object),
            # genuine causal features (must be kept)
            "current_vwap":       np.random.randn(n).astype(np.float32),
            "tick_count":         np.random.randint(1, 500, n).astype(np.int32),
            "market_state_code":  np.random.randint(0, 5, n).astype(np.int8),
            "regime_cluster":     np.random.randint(0, 4, n).astype(np.int8),
        })

    def test_leaks_dropped_features_kept(self):
        feats = set(_select_features(self._frame()))
        # the two newly-closed leaks must be gone
        assert "fwd_ret_clean" not in feats
        assert "effective_horizon" not in feats
        # other known leaks gone
        for c in ("forward_return", "bias_label", "mfe_atr", "net_expectancy_proxy"):
            assert c not in feats
        # string label-side gone (numeric filter)
        assert "market_state_label" not in feats
        assert "regime_label_grouped" not in feats
        # causal features kept
        for c in ("current_vwap", "tick_count", "market_state_code", "regime_cluster"):
            assert c in feats, f"{c!r} (causal) was wrongly dropped"
