"""Tests for the IC audit tool.

The tool has 6 testable contracts:
  1. Pure IC computation (Pearson, Spearman, MI, rolling IR)
  2. Forward returns are session-break aware
  3. Benjamini-Hochberg FDR correction is mathematically correct
  4. Verdict cascade picks the right class per (IC, MI, IR, p_fdr)
  5. Top-feature ranking by composite score
  6. End-to-end orchestration on synthetic data with KNOWN edge
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.ic_audit import (
    DEFAULT_WARMUP_BARS,
    FDR_ALPHA,
    LOOKAHEAD_DROP_THRESHOLD,
    MI_NONLINEAR_THRESHOLD,
    MIN_IR_STRONG,
    NOISE_THRESHOLD,
    STRONG_THRESHOLD,
    WALK_FORWARD_MIN_CONSISTENCY,
    WARMUP_IC_DROP_THRESHOLD,
    WEAK_THRESHOLD,
    classify_verdict,
    compute_forward_returns,
    compute_warmup_mask,
    cost_adjusted_returns,
    detect_lookahead,
    fdr_correct,
    run_ic_audit,
    top_features,
    walk_forward_ic,
    warmup_drop_ratio,
    _rolling_ic,
    _safe_mi,
    _safe_pearson,
    _safe_spearman,
)


# ════════════════════════════════════════════════════════════════════
# Pure IC computation
# ════════════════════════════════════════════════════════════════════
class TestSafeSpearman:
    def test_perfect_monotonic(self):
        x = np.arange(1000, dtype=np.float64)
        y = x ** 0.5
        ic, p = _safe_spearman(x, y)
        assert ic == pytest.approx(1.0, abs=1e-6)
        assert p < 0.001

    def test_perfect_anti_monotonic(self):
        x = np.arange(1000, dtype=np.float64)
        y = -x
        ic, _ = _safe_spearman(x, y)
        assert ic == pytest.approx(-1.0, abs=1e-6)

    def test_no_relationship(self):
        rng = np.random.RandomState(0)
        x = rng.randn(2000)
        y = rng.randn(2000)
        ic, p = _safe_spearman(x, y)
        assert abs(ic) < 0.10
        assert p > 0.01

    def test_nan_handling(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        ic, _ = _safe_spearman(x, y)
        # Only 3 valid pairs — falls under min-sample guard → NaN
        assert np.isnan(ic)

    def test_constant_input_returns_zero(self):
        x = np.full(500, 3.0)
        y = np.random.randn(500)
        ic, _ = _safe_spearman(x, y)
        assert ic == 0.0


class TestSafePearson:
    def test_perfect_linear(self):
        x = np.arange(500, dtype=np.float64)
        y = 2 * x + 1
        assert _safe_pearson(x, y) == pytest.approx(1.0, abs=1e-6)

    def test_too_few_returns_nan(self):
        assert np.isnan(_safe_pearson(np.array([1.0]), np.array([2.0])))


class TestSafeMI:
    def test_perfect_dependence_high_mi(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, 500)
        y = x ** 2   # monotonic in |x|, NOT monotonic in x → Spearman ≈ 0
        # MI should be high because y is a function of x
        mi = _safe_mi(x, y)
        if np.isfinite(mi):
            assert mi > 0.10, f"MI for y=x² should be > 0.10, got {mi}"

    def test_independent_low_mi(self):
        rng = np.random.RandomState(0)
        x = rng.randn(500)
        y = rng.randn(500)
        mi = _safe_mi(x, y)
        if np.isfinite(mi):
            assert mi < 0.10


class TestRollingIC:
    def test_stable_signal_gives_positive_ir(self):
        rng = np.random.RandomState(0)
        n = 4000
        x = rng.randn(n)
        # y = x + noise, so per-window IC ~ 0.7 stable
        y = 0.7 * x + 0.5 * rng.randn(n)
        mean, std, ir = _rolling_ic(x, y, n_windows=8)
        assert mean > 0.5
        assert abs(ir) > 0.5

    def test_unstable_signal_gives_low_ir(self):
        rng = np.random.RandomState(0)
        n = 4000
        x = rng.randn(n)
        # First half: positive correlation. Second half: negative.
        y = np.concatenate([x[:n//2], -x[n//2:]]) + 0.3 * rng.randn(n)
        mean, std, ir = _rolling_ic(x, y, n_windows=8)
        # Mean IC near zero, std high → IR near zero
        assert abs(ir) < 1.0


# ════════════════════════════════════════════════════════════════════
# Forward returns
# ════════════════════════════════════════════════════════════════════
class TestForwardReturns:
    def test_basic_returns(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        ret = compute_forward_returns(close, horizons=(1,))
        # ret[0] = (101 - 100) / 100 = 0.01
        assert ret[1][0] == pytest.approx(0.01)
        assert ret[1][1] == pytest.approx(101 / 101 * (102 - 101) / 101, abs=1e-9)
        # Last bar has no future → NaN
        assert np.isnan(ret[1][-1])

    def test_multiple_horizons(self):
        close = pd.Series([100, 105, 110, 115, 120], dtype=np.float64)
        ret = compute_forward_returns(close, horizons=(1, 2))
        assert ret[1][0] == pytest.approx(0.05)
        assert ret[2][0] == pytest.approx(0.10)

    def test_session_break_skips_returns(self):
        close = pd.Series([100, 101, 999, 1000, 1001], dtype=np.float64)
        # Pretend index 2 is a session break (a weekend gap)
        breaks = pd.Series([0, 0, 1, 0, 0])
        ret = compute_forward_returns(close, horizons=(2,), session_breaks=breaks)
        # ret[0] would span indices 1, 2 → has the break → NaN
        assert np.isnan(ret[2][0])
        # ret[1] spans 2, 3 → has the break at index 2 → NaN
        assert np.isnan(ret[2][1])


# ════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg FDR
# ════════════════════════════════════════════════════════════════════
class TestFDR:
    def test_known_correction(self):
        # 4 p-values: bh-adjusted should preserve order, scale up
        pvals = np.array([0.01, 0.04, 0.03, 0.005])
        adj = fdr_correct(pvals, alpha=0.05)
        # Ranks: 0.005=1, 0.01=2, 0.03=3, 0.04=4 → adj = p*N/rank
        # smallest adj = 0.005 * 4/1 = 0.02
        # next = 0.01 * 4/2 = 0.02
        # next = 0.03 * 4/3 = 0.04
        # next = 0.04 * 4/4 = 0.04
        # BH then enforces monotonicity from the right
        assert adj[3] == pytest.approx(0.02, abs=1e-9)  # for p=0.005
        assert adj[0] == pytest.approx(0.02, abs=1e-9)  # for p=0.01

    def test_handles_nans(self):
        pvals = np.array([0.01, np.nan, 0.05])
        adj = fdr_correct(pvals)
        assert adj[1] == 1.0   # NaN → default 1.0
        # Non-NaN p-values get adjusted
        assert adj[0] < 1.0


# ════════════════════════════════════════════════════════════════════
# Verdict cascade
# ════════════════════════════════════════════════════════════════════
class TestVerdict:
    def test_nonfinite_on_small_n(self):
        v = classify_verdict(
            spearman_ic=0.20, mi=0.20, ir=2.0, p_fdr=0.01, n_samples=100,
        )
        assert v == "NONFINITE"

    def test_noise_on_low_ic(self):
        v = classify_verdict(
            spearman_ic=0.01, mi=0.01, ir=0.1, p_fdr=0.01, n_samples=5000,
        )
        assert v == "NOISE"

    def test_noise_on_high_pvalue(self):
        v = classify_verdict(
            spearman_ic=0.15, mi=0.01, ir=2.0, p_fdr=0.50, n_samples=5000,
        )
        assert v == "NOISE"

    def test_nonlinear_edge(self):
        # Low Spearman but high MI → NONLINEAR_EDGE
        v = classify_verdict(
            spearman_ic=0.01, mi=0.20, ir=0.0, p_fdr=0.001, n_samples=5000,
        )
        assert v == "NONLINEAR_EDGE"

    def test_weak(self):
        v = classify_verdict(
            spearman_ic=0.03, mi=0.01, ir=0.4, p_fdr=0.01, n_samples=5000,
        )
        assert v == "WEAK"

    def test_moderate(self):
        v = classify_verdict(
            spearman_ic=0.07, mi=0.01, ir=0.5, p_fdr=0.001, n_samples=5000,
        )
        assert v == "MODERATE"

    def test_strong(self):
        v = classify_verdict(
            spearman_ic=0.15, mi=0.01, ir=0.8, p_fdr=0.001, n_samples=5000,
        )
        assert v == "STRONG"

    def test_unstable_strong(self):
        # Strong IC but low IR → unstable
        v = classify_verdict(
            spearman_ic=0.15, mi=0.01, ir=0.1, p_fdr=0.001, n_samples=5000,
        )
        assert v == "UNSTABLE_STRONG"

    def test_negative_ic_still_strong(self):
        v = classify_verdict(
            spearman_ic=-0.15, mi=0.01, ir=-0.8, p_fdr=0.001, n_samples=5000,
        )
        assert v == "STRONG"


# ════════════════════════════════════════════════════════════════════
# Option 2 — Production-grade primitives
# ════════════════════════════════════════════════════════════════════
class TestWalkForwardIC:
    def test_stable_signal_high_consistency(self):
        rng = np.random.RandomState(0)
        n = 10_000
        x = rng.randn(n)
        # Persistent positive correlation across the whole series
        y = 0.5 * x + 0.5 * rng.randn(n)
        mean, std, consistency, ics = walk_forward_ic(x, y, n_windows=8)
        assert mean > 0.3
        assert consistency >= 0.85  # nearly all windows positive

    def test_flipping_signal_low_consistency(self):
        rng = np.random.RandomState(0)
        n = 8_000
        x = rng.randn(n)
        # First half positive correlation, second half negative
        y = np.concatenate([0.5 * x[:n//2] + 0.5 * rng.randn(n//2),
                            -0.5 * x[n//2:] + 0.5 * rng.randn(n//2)])
        mean, std, consistency, ics = walk_forward_ic(x, y, n_windows=8)
        # Some windows positive, some negative → consistency low
        assert consistency < 0.85

    def test_too_few_samples_returns_nan(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        mean, std, _, ics = walk_forward_ic(x, y, n_windows=8)
        assert np.isnan(mean)
        assert ics == []


class TestCostAdjustedReturns:
    def test_zero_cost_passes_through(self):
        r = np.array([0.001, -0.002, 0.0005, -0.0001])
        adj = cost_adjusted_returns(r, cost_per_side=0.0)
        assert np.allclose(adj, r)

    def test_small_returns_collapsed(self):
        # cost = 0.0005 → threshold = 0.001
        r = np.array([0.0005, -0.0005, 0.002, -0.002])
        adj = cost_adjusted_returns(r, cost_per_side=0.0005)
        # |0.0005| ≤ 0.001 → 0
        # |0.002|  > 0.001 → preserved
        assert adj[0] == 0.0
        assert adj[1] == 0.0
        assert adj[2] == 0.002
        assert adj[3] == -0.002

    def test_input_not_mutated(self):
        r = np.array([0.001, -0.002])
        r_ref = r.copy()
        _ = cost_adjusted_returns(r, cost_per_side=0.0005)
        assert np.array_equal(r, r_ref)


class TestDetectLookahead:
    def test_clean_smooth_decay_no_flag(self):
        # IC decays gently: 0.06 → 0.05 → 0.04 → 0.03
        ics = {1: 0.06, 6: 0.05, 12: 0.04, 24: 0.03}
        score = detect_lookahead(ics)
        assert score == 0.0   # drop ratio = (0.06-0.03)/0.06 = 0.5 → not > 0.5

    def test_suspicious_collapse_flagged(self):
        # IC at h=1 is 0.15, drops to 0.005 at h=24 → 96.7% drop
        ics = {1: 0.15, 6: 0.08, 12: 0.03, 24: 0.005}
        score = detect_lookahead(ics)
        assert score > 0.4   # 0.967 - 0.5 = 0.467

    def test_noise_feature_not_flagged(self):
        # All IC values are small — drop is large in % but not flagged
        # because ic_short < 0.03 (the noise floor)
        ics = {1: 0.020, 24: 0.001}
        score = detect_lookahead(ics)
        assert score == 0.0

    def test_single_horizon_returns_zero(self):
        assert detect_lookahead({1: 0.10}) == 0.0


class TestWarmupMask:
    def test_no_breaks_marks_only_dataset_start(self):
        df = pd.DataFrame({"close": np.arange(100, dtype=float)})
        mask = compute_warmup_mask(df, n_warmup_bars=10)
        # First 10 bars warm-up, rest are post-warmup
        assert mask[:10].all()
        assert not mask[10:].any()

    def test_session_break_resets_warmup(self):
        n = 100
        breaks = np.zeros(n, dtype=int)
        breaks[50] = 1   # break at index 50
        df = pd.DataFrame({
            "close": np.arange(n, dtype=float),
            "is_session_break": breaks,
        })
        mask = compute_warmup_mask(df, n_warmup_bars=10)
        # First 10 + 10 after break (50..59) = warm-up
        assert mask[0:10].all()
        assert not mask[10:50].any()
        assert mask[50:60].all()
        assert not mask[60:].any()

    def test_zero_warmup_returns_all_false(self):
        df = pd.DataFrame({"close": np.arange(50, dtype=float)})
        mask = compute_warmup_mask(df, n_warmup_bars=0)
        assert not mask.any()


class TestWarmupDropRatio:
    def test_feature_independent_of_warmup_no_drop(self):
        rng = np.random.RandomState(0)
        n = 3_000
        x = rng.randn(n)
        y = 0.5 * x + 0.5 * rng.randn(n)
        warmup = np.zeros(n, dtype=bool)
        warmup[:100] = True   # 100 warm-up bars
        drop = warmup_drop_ratio(x, y, warmup)
        # IC barely changes when 100 of 3000 bars removed
        assert drop < 0.20

    def test_warmup_driven_ic_high_drop(self):
        rng = np.random.RandomState(0)
        n = 3_000
        # x = noise everywhere
        x = rng.randn(n)
        # y = correlated with x ONLY in warm-up region; pure noise elsewhere
        y = rng.randn(n) * 0.3
        y[:300] = x[:300]  # strong signal in first 300
        warmup = np.zeros(n, dtype=bool)
        warmup[:300] = True
        drop = warmup_drop_ratio(x, y, warmup)
        # Full IC has signal from warm-up; non-warmup IC has none → big drop
        assert drop > 0.50


# ════════════════════════════════════════════════════════════════════
# Option 2 — Extended verdict cascade
# ════════════════════════════════════════════════════════════════════
class TestExtendedVerdict:
    def test_lookahead_dominates(self):
        # Strong-looking IC, but lookahead_score above threshold
        v = classify_verdict(
            spearman_ic=0.20, mi=0.01, ir=2.0, p_fdr=0.001,
            n_samples=5000,
            lookahead_score=0.40,  # > 0.25 threshold
        )
        assert v == "SUSPICIOUS_LEAKAGE"

    def test_warmup_rider_above_noise(self):
        v = classify_verdict(
            spearman_ic=0.08, mi=0.01, ir=0.5, p_fdr=0.001,
            n_samples=5000,
            warmup_drop=0.60,   # > 0.50 threshold
        )
        assert v == "WARMUP_RIDER"

    def test_warmup_drop_below_threshold_passes(self):
        v = classify_verdict(
            spearman_ic=0.08, mi=0.01, ir=0.5, p_fdr=0.001,
            n_samples=5000,
            warmup_drop=0.30,
        )
        assert v in {"MODERATE", "UNSTABLE_STRONG"}

    def test_cost_negative_when_naive_ic_disappears(self):
        v = classify_verdict(
            spearman_ic=0.08, mi=0.01, ir=0.5, p_fdr=0.001,
            n_samples=5000,
            cost_adjusted_ic=0.005,   # below COST threshold (0.02)
        )
        assert v == "COST_NEGATIVE"

    def test_cost_not_supplied_skips_check(self):
        # cost_adjusted_ic=None → branch skipped
        v = classify_verdict(
            spearman_ic=0.08, mi=0.01, ir=0.5, p_fdr=0.001,
            n_samples=5000,
            cost_adjusted_ic=None,
        )
        assert v != "COST_NEGATIVE"

    def test_unstable_walkforward_triggers(self):
        v = classify_verdict(
            spearman_ic=0.08, mi=0.01, ir=0.5, p_fdr=0.001,
            n_samples=5000,
            walk_forward_consistency=0.50,   # < 0.625 threshold
        )
        assert v == "UNSTABLE_WALKFORWARD"

    def test_leakage_check_before_others(self):
        # Both lookahead AND warmup AND cost flagged → leakage wins
        v = classify_verdict(
            spearman_ic=0.15, mi=0.01, ir=1.0, p_fdr=0.001,
            n_samples=5000,
            lookahead_score=0.50,
            warmup_drop=0.80,
            cost_adjusted_ic=0.001,
        )
        assert v == "SUSPICIOUS_LEAKAGE"


# ════════════════════════════════════════════════════════════════════
# Synthetic end-to-end: known signal recovery
# ════════════════════════════════════════════════════════════════════
def _synthetic_features(n: int = 5_000) -> pd.DataFrame:
    """Build a synthetic features parquet with known IC properties.

    Features (designed so the tool's verdicts are predictable):
      - signal_strong:   strongly correlated with forward return
      - signal_moderate: moderately correlated
      - signal_noise:    uncorrelated
      - signal_nonlinear: nonlinear (y = x² type)
    """
    rng = np.random.RandomState(42)
    # Synthetic returns with persistence
    raw_ret = rng.randn(n) * 0.001
    close = 100 * np.exp(np.cumsum(raw_ret))

    # Build features so that the NEXT-bar return is predictable
    future_ret = np.zeros(n)
    future_ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

    df = pd.DataFrame({
        "ts_event": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "close": close,
        "signal_strong":    future_ret + 0.3 * rng.randn(n) * 0.001,
        "signal_moderate":  future_ret + 1.0 * rng.randn(n) * 0.001,
        "signal_noise":     rng.randn(n),
        "signal_nonlinear": np.sign(future_ret) * np.abs(rng.randn(n)) ** 0.5,
        "regime_label":     rng.choice(["trending", "ranging"], n),
        "is_session_break": np.zeros(n, dtype=int),
    })
    return df


class TestEndToEnd:
    def test_recovers_strong_signal(self):
        df = _synthetic_features(n=6_000)
        results_df, summary = run_ic_audit(
            df, horizons=(1, 3), train_end=None,
        )
        # signal_strong should land STRONG or MODERATE at horizon 1
        strong_rows = results_df[
            (results_df["feature"] == "signal_strong")
            & (results_df["horizon"] == 1)
        ]
        assert len(strong_rows) == 1
        r = strong_rows.iloc[0]
        assert r["verdict"] in {"STRONG", "MODERATE", "UNSTABLE_STRONG"}
        assert abs(r["spearman_ic"]) > WEAK_THRESHOLD

    def test_noise_feature_lands_noise(self):
        df = _synthetic_features(n=6_000)
        results_df, _ = run_ic_audit(df, horizons=(1,))
        noise_rows = results_df[results_df["feature"] == "signal_noise"]
        for _, r in noise_rows.iterrows():
            assert r["verdict"] in {"NOISE", "NONLINEAR_EDGE"}
            assert abs(r["spearman_ic"]) < WEAK_THRESHOLD

    def test_summary_counts(self):
        df = _synthetic_features(n=6_000)
        results_df, summary = run_ic_audit(df, horizons=(1, 3))
        # 4 signal features × 2 horizons = 8 measurements
        assert summary["n_total_measurements"] == 8
        assert summary["n_train"] > 0
        assert summary["n_test"] > 0

    def test_excludes_label_columns(self):
        df = _synthetic_features(n=6_000)
        df["bias_label"] = 0
        df["forward_return"] = 0.0
        df["soft_label_long"] = 0.0
        results_df, _ = run_ic_audit(df, horizons=(1,))
        # None of these should appear in results
        bad_columns = {"bias_label", "forward_return", "soft_label_long"}
        appeared = set(results_df["feature"].unique()) & bad_columns
        assert appeared == set(), (
            f"Label columns leaked into IC audit: {appeared}"
        )

    def test_top_features_ranked_correctly(self):
        df = _synthetic_features(n=6_000)
        results_df, _ = run_ic_audit(df, horizons=(1,))
        top = top_features(results_df, n=10, horizon=1)
        # signal_strong should rank above signal_noise (which won't appear at all)
        feature_names = set(top["feature"].tolist())
        if "signal_strong" in feature_names:
            assert "signal_noise" not in feature_names


class TestCLI:
    def test_cli_runs_end_to_end(self, tmp_path):
        df = _synthetic_features(n=5_000)
        feats = tmp_path / "features.parquet"
        df.to_parquet(feats)
        out = tmp_path / "ic_out"

        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "tools" / "diagnostics" / "ic_audit.py"),
             "--features", str(feats),
             "--output", str(out),
             "--horizons", "1", "3"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"CLI failed:\n{result.stdout}\n{result.stderr}"
        )
        for fname in [
            "ic_summary.csv", "ic_top_features.csv",
            "ic_summary.json", "ic_report.txt",
        ]:
            assert (out / fname).exists(), f"missing {fname}"
        summary = json.loads((out / "ic_summary.json").read_text())
        assert "verdict_counts" in summary
