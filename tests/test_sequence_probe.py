"""Teeth tests for tools/diagnostics/sequence_probe.py.

Covers:
  T1. Sequence primitives are R1-causal (mutating bars after t does NOT change
       derived[t]).
  T2. Slope correctness on a planted linear sequence.
  T3. Consecutive-run counter resets on direction change.
  T4. Momentum semantics: feature[t] - feature[t-w].
  T5. build_sequence_features produces expected derived columns + still causal
       when given a real-shaped frame.
  T6. directional_vs_magnitude correctly classifies a planted DIRECTIONAL
       signal (IC(dir) >> IC(|r|)) and a planted VOLATILITY signal (the
       reverse).
  T7. decoder_probe.run_probe accepts df= (back-compat for sequence_probe).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.sequence_probe import (
    _rolling_slope, _rolling_momentum, _consecutive_run,
    build_sequence_features, directional_vs_magnitude,
    BASE_FEATURES_FOR_SEQUENCE, DEFAULT_WINDOWS,
)
from tools.diagnostics.decoder_probe import run_probe


# ── T1: causal — mutating future does NOT change past derived values ───────
class TestCausalSequencePrimitives:
    def test_slope_does_not_use_future(self):
        s = pd.Series(np.arange(20, dtype=np.float64))           # perfect linear: slope == 1
        slopes_a = _rolling_slope(s, w=3)
        # Mutate row 15 onward
        s_mut = s.copy()
        s_mut.iloc[15:] = 999.0
        slopes_b = _rolling_slope(s_mut, w=3)
        # Rows 0..14 must be UNCHANGED — they only saw [t-2..t]
        np.testing.assert_array_equal(slopes_a[:15], slopes_b[:15])

    def test_momentum_does_not_use_future(self):
        s = pd.Series(np.arange(20, dtype=np.float64))
        mom_a = _rolling_momentum(s, w=3)
        s_mut = s.copy()
        s_mut.iloc[15:] = 999.0
        mom_b = _rolling_momentum(s_mut, w=3)
        np.testing.assert_array_equal(mom_a[:15], mom_b[:15])

    def test_consec_run_does_not_use_future(self):
        d = np.array([1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1])
        run_a = _consecutive_run(d, +1)
        # Future flip: change last 4 — past must stay equal
        d_mut = d.copy()
        d_mut[7:] = -1
        run_b = _consecutive_run(d_mut, +1)
        np.testing.assert_array_equal(run_a[:7], run_b[:7])


# ── T2: slope correctness ──────────────────────────────────────────────────
class TestSlopeCorrectness:
    def test_planted_linear_slope(self):
        # y = 2x + 5 → slope of any 3 consecutive points = 2
        s = pd.Series(2.0 * np.arange(30) + 5.0)
        out = _rolling_slope(s, w=3)
        # First two are NaN (not enough samples)
        assert np.isnan(out[0]) and np.isnan(out[1])
        # From row 2 onward, slope = 2 exactly
        np.testing.assert_allclose(out[2:], 2.0, atol=1e-9)

    def test_planted_flat_slope_zero(self):
        s = pd.Series(np.full(20, 7.0))
        out = _rolling_slope(s, w=4)
        # All non-NaN entries must be 0
        np.testing.assert_allclose(out[3:], 0.0, atol=1e-9)


# ── T3: consecutive-run resets on direction change ─────────────────────────
class TestConsecutiveRun:
    def test_counter_resets_on_flip(self):
        # +1, +1, +1, -1, +1, +1 → up-run counter: 1, 2, 3, 0, 1, 2
        d = np.array([1, 1, 1, -1, 1, 1])
        out = _consecutive_run(d, +1)
        np.testing.assert_array_equal(out, [1, 2, 3, 0, 1, 2])

    def test_zero_direction_resets(self):
        # 0 is neither +1 nor -1 → resets the +1 counter
        d = np.array([1, 1, 0, 1, 1])
        out = _consecutive_run(d, +1)
        np.testing.assert_array_equal(out, [1, 2, 0, 1, 2])


# ── T4: momentum semantics ─────────────────────────────────────────────────
class TestMomentum:
    def test_momentum_w3(self):
        s = pd.Series([10.0, 11.0, 12.0, 15.0, 20.0, 22.0])
        out = _rolling_momentum(s, w=3)
        # rows 0..2 NaN; row 3 = s[3]-s[0] = 5; row 4 = s[4]-s[1] = 9; row 5 = s[5]-s[2] = 10
        assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
        assert out[3] == 5.0
        assert out[4] == 9.0
        assert out[5] == 10.0


# ── T5: build_sequence_features produces expected derived + causal ─────────
class TestBuildSequence:
    def _synth_df(self, n=200):
        rng = np.random.RandomState(0)
        ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        df = pd.DataFrame({"ts_event": ts, "close": close,
                           "is_session_break": np.zeros(n, dtype=bool)})
        for f in BASE_FEATURES_FOR_SEQUENCE:
            df[f] = rng.randn(n).astype(np.float32)
        return df

    def test_derived_columns_present(self):
        df = self._synth_df()
        out, derived = build_sequence_features(df)
        assert "consec_up_close" in derived
        assert "consec_down_close" in derived
        for f in BASE_FEATURES_FOR_SEQUENCE:
            for w in DEFAULT_WINDOWS:
                assert f"{f}_slope_{w}" in derived
                assert f"{f}_mom_{w}" in derived
        # All derived must exist as columns in out
        for c in derived:
            assert c in out.columns

    def test_build_is_causal_endtoend(self):
        df = self._synth_df()
        out_a, derived = build_sequence_features(df)
        # Mutate the last 30 rows of every base feature + close
        df_mut = df.copy()
        for f in BASE_FEATURES_FOR_SEQUENCE + ("close",):
            df_mut.loc[df_mut.index[-30:], f] = 999.0
        out_b, _ = build_sequence_features(df_mut)
        # First 170 rows of every derived column must be IDENTICAL
        for c in derived:
            a = out_a[c].iloc[:170].to_numpy()
            b = out_b[c].iloc[:170].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            np.testing.assert_array_equal(a[mask], b[mask], err_msg=f"{c} leaked future")


# ── T6: directional_vs_magnitude classifies correctly ──────────────────────
class TestDirectionalVsMagnitude:
    def _synth(self, n=2500, seed=0):
        rng = np.random.RandomState(seed)
        ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        df = pd.DataFrame({"ts_event": ts, "close": close,
                           "is_session_break": np.zeros(n, dtype=bool)})
        return df

    def test_planted_directional_signal(self):
        """feature ≈ sign of next bar's return diluted with noise so |IC|
        stays in the MODERATE/STRONG band (not SUSPECT)."""
        df = self._synth(seed=1)
        ret1 = pd.Series(df["close"]).pct_change(1).shift(-1).fillna(0).to_numpy()
        # heavy noise dilution: weak directional signal that lands in MODERATE
        rng = np.random.RandomState(2)
        df["F"] = (np.sign(ret1) * 0.3 + rng.randn(len(df)) * 3.0).astype(np.float32)
        results = directional_vs_magnitude(df, ["F"], horizons=(1, 6))
        r = results[0]
        h1 = r["per_horizon"]["1"]
        # directional IC must be in MODERATE/STRONG band — not SUSPECT (too strong)
        assert 0.05 <= h1["abs_dir"] < 0.30, h1
        # magnitude IC must be far below directional → DIRECTIONAL verdict
        assert h1["abs_dir"] > h1["abs_mag"] * 0.5, h1
        assert h1["verdict"] == "DIRECTIONAL", h1

    def test_planted_volatility_signal(self):
        """feature = |next bar return| → strong IC(|r|), near-zero IC(dir)."""
        df = self._synth(seed=3)
        ret1 = pd.Series(df["close"]).pct_change(1).shift(-1).fillna(0).to_numpy()
        df["F"] = np.abs(ret1).astype(np.float32)
        results = directional_vs_magnitude(df, ["F"], horizons=(1, 6))
        r = results[0]
        h1 = r["per_horizon"]["1"]
        assert h1["abs_mag"] > 0.30, h1
        # directional IC should be much smaller than magnitude IC
        assert h1["abs_dir"] < h1["abs_mag"] * 0.5, h1
        assert h1["verdict"] == "VOLATILITY_ONLY", h1


# ── T7: decoder_probe.run_probe accepts df= (back-compat for sequence_probe)
class TestRunProbeAcceptsDf:
    def test_run_probe_with_df_no_path(self, tmp_path):
        rng = np.random.RandomState(0)
        n = 5000
        ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        df = pd.DataFrame({
            "ts_event": ts, "close": close,
            "is_session_break": np.zeros(n, dtype=bool),
            "bias_label": np.full(n, 2, dtype=np.int8),
            "F1": rng.randn(n).astype(np.float32),
        })
        summary = run_probe(
            features_parquet=None, output_dir=tmp_path / "out",
            df=df, feature_names=("F1",),
            horizons=(1, 6), k_folds=3, n_null=20,
        )
        assert summary["n_rows"] == n
        assert summary["features_parquet"] == "(in-memory df)"
        assert "stage_1_univariate" in summary
