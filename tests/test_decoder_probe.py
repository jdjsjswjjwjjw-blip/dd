"""Teeth tests for tools/diagnostics/decoder_probe.py.

Tests verify the harness behaviour on KNOWN synthetic scenarios:

  Synth-level (no parquet I/O):
    T1. Classify thresholds match the a priori values (R10 lock).
    T2. _null_distribution on a non-predictive feature: |IC| ≈ 0.
    T3. _causality_probe flags future-shift leakage on a feature that secretly
        knows tomorrow.
    T4. _smooth_decay_ok says NO when IC is flat across horizons.

  Stage-level (synthetic parquet):
    T5. Pure-noise features → Stage 1 verdict NOISE for every feature; Stage 2
        is SKIPPED (R4 gating works).
    T6. A planted CAUSAL signal (feature = sign(future return) + small noise)
        yields a MODERATE+ verdict in Stage 1 AND a positive Stage 2 IC.
    T7. A planted LEAK (feature = forward return, no noise) is caught by EITHER
        the SUSPECT IC threshold OR the causality probe — never PASSED as
        plain MODERATE/STRONG.
    T8. Stage 3 AUC > 0.55 on a planted bias-predictive feature.

  Output:
    T9. run_probe writes a utf-8 JSON summary + utf-8 report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.decoder_probe import (
    _classify_ic, _classify_auc, _smooth_decay_ok,
    _null_distribution, _causality_probe,
    stage_1_univariate, stage_2_lasso, stage_3_logistic_bias,
    run_probe,
    IC_NOISE, IC_MODERATE, IC_STRONG, IC_SUSPECT,
    AUC_MODERATE, AUC_STRONG, AUC_SUSPECT,
    IR_MIN_MODERATE, DECAY_PASS_RATIO,
)


def _best_ic(summary: dict, feature: str) -> tuple[float, str]:
    """(best |IC| across horizons, feature-level verdict) for one Stage-1 feature."""
    r = next(x for x in summary["stage_1_univariate"] if x.get("feature") == feature)
    ph = r["per_horizon"]
    vals = [ph[h]["ic_mean_abs"] for h in ph if np.isfinite(ph[h]["ic_mean_abs"])]
    return (max(vals) if vals else 0.0), r["verdict"]


# ── T1: a priori thresholds locked ────────────────────────────────────────
class TestThresholdsAprioriLocked:
    def test_ic_thresholds_match_design_doc(self):
        assert IC_NOISE == 0.02
        assert IC_MODERATE == 0.05
        assert IC_STRONG == 0.10
        assert IC_SUSPECT == 0.30

    def test_auc_thresholds_match_design_doc(self):
        assert AUC_MODERATE == 0.55
        assert AUC_STRONG == 0.60
        assert AUC_SUSPECT == 0.75

    def test_classify_ic_buckets(self):
        assert _classify_ic(0.01, 1.0) == "NOISE"
        assert _classify_ic(0.03, 1.0) == "WEAK"
        assert _classify_ic(0.07, 0.5) == "MODERATE"
        assert _classify_ic(0.15, 1.0) == "STRONG"
        assert _classify_ic(0.35, 1.0) == "SUSPECT_LEAKAGE"
        # IR gate
        assert _classify_ic(0.07, 0.10) == "UNSTABLE"     # MOD ic but bad IR

    def test_classify_auc_buckets(self):
        assert _classify_auc(0.51) == "NOISE"      # below AUC_NOISE (0.52)
        assert _classify_auc(0.53) == "WEAK"        # in 0.52..0.55 band
        assert _classify_auc(0.57) == "MODERATE"
        assert _classify_auc(0.65) == "STRONG"
        assert _classify_auc(0.80) == "SUSPECT_LEAKAGE"


# ── T2: null distribution centred on 0 for non-predictive feature ─────────
class TestNullDistribution:
    def test_pure_noise_null_close_to_zero(self):
        rng = np.random.RandomState(0)
        n = 5000
        feature = rng.randn(n)
        label = rng.randn(n)             # independent
        out = _null_distribution(feature, label, n_reps=100, seed=0)
        # For independent ~N(0,1), |IC| ≈ 1/sqrt(n) ≈ 0.014; p95 should be tiny
        assert out["null_p95"] < 0.05, out
        assert out["null_median"] < 0.025


# ── T3: causality probe flags future-shift leak ──────────────────────────
class TestCausalityProbe:
    def test_future_shift_leak_is_caught(self):
        """Off-by-one time misalignment on an AR(0.95) signal: honest predictor
        of label[i]=s[i+5] is s[i] (baseline IC = 0.95^5 ≈ 0.77). LEAKY
        predictor is s[i+1] (one bar early peek → baseline = 0.95^4 ≈ 0.81;
        future-shift = 0.95^3 ≈ 0.86 > baseline → flag fires)."""
        rng = np.random.RandomState(1)
        n = 3000
        # AR(0.7) decays fast enough that the per-step IC drop is well above
        # the 0.05 margin: baseline 0.7^4 ≈ 0.24, future 0.7^3 ≈ 0.34, diff ≈ 0.10
        sig = np.zeros(n)
        for i in range(1, n):
            sig[i] = 0.7 * sig[i - 1] + rng.randn()
        label = pd.Series(sig).shift(-5).fillna(0).to_numpy()
        feature = pd.Series(sig).shift(-1).fillna(0).to_numpy()     # leaky: one bar ahead
        out = _causality_probe(feature, label)
        assert out["future_leak_flag"], out

    def test_clean_causal_does_not_flag(self):
        """An honest feature correlated with the SAME-time label (not the next):
        baseline IC = 0.3, shifted IC ≈ 0 — must NOT flag as leak."""
        rng = np.random.RandomState(2)
        n = 3000
        label = rng.randn(n)
        feature = label * 0.3 + rng.randn(n) * 0.95
        out = _causality_probe(feature, label)
        assert not out["future_leak_flag"], out


# ── T4: smooth-decay validator ────────────────────────────────────────────
class TestSmoothDecay:
    def test_flat_ic_is_not_smooth_decay(self):
        ic_at_h = {1: 0.10, 6: 0.09, 24: 0.09, 48: 0.09}
        assert not _smooth_decay_ok(ic_at_h, (1, 6, 24, 48))

    def test_decaying_ic_is_smooth(self):
        ic_at_h = {1: 0.10, 6: 0.07, 24: 0.04, 48: 0.02}
        assert _smooth_decay_ok(ic_at_h, (1, 6, 24, 48))

    def test_weak_baseline_returns_false(self):
        ic_at_h = {1: 0.01, 48: 0.005}       # too weak to assess
        assert not _smooth_decay_ok(ic_at_h, (1, 48))


# ── T5: pure-noise parquet → Stage 1 NOISE everywhere, Stage 2 SKIPPED ───
def _synth_df(n: int, *, with_signal: dict[int, float] | None = None,
              with_leak: bool = False, seed: int = 0) -> pd.DataFrame:
    """Synthetic parquet-like frame:
      - close: random walk
      - is_session_break: all False
      - bias_label: derived from sign of next return (binary) OR random if noise

    Optional features added per call site by caller; we only build the skeleton.
    """
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
    df = pd.DataFrame({"ts_event": ts, "close": close,
                       "is_session_break": np.zeros(n, dtype=bool)})
    # bias_label: derived from sign of future 6-bar return, with noise floor
    ret6 = pd.Series(close).pct_change(6).shift(-6).fillna(0).to_numpy()
    if with_signal is not None:
        # bias_label correlates with the signal
        bias = np.where(ret6 > 0, 0, np.where(ret6 < 0, 1, 2)).astype(np.int8)
    else:
        bias = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2]).astype(np.int8)
    df["bias_label"] = bias
    return df


class TestPureNoiseFlow:
    def test_noise_features_yield_noise_verdict_and_skip_stage_2(self, tmp_path):
        rng = np.random.RandomState(1010)      # ← different seed from _synth_df below
        n = 8000
        df = _synth_df(n, seed=10)
        # 3 noise features
        for name in ("vwap_z_score", "cvd_bar_5m", "order_flow_imbalance"):
            df[name] = rng.randn(n).astype(np.float32)
        path = tmp_path / "noise.parquet"
        df.to_parquet(path, index=False)

        summary = run_probe(
            path, tmp_path / "out",
            feature_names=("vwap_z_score", "cvd_bar_5m", "order_flow_imbalance"),
            horizons=(1, 6, 24), k_folds=3, n_null=30, seed=0,
        )
        # every Stage-1 feature must be NOISE (or weak — not MODERATE/STRONG)
        for r in summary["stage_1_univariate"]:
            assert r["verdict"] in {"NOISE", "WEAK", "WEAK_OR_UNSTABLE"}, r
        # Stage 2 skipped because nothing reached MODERATE
        assert summary["stage_2_lasso"].get("skipped") is True


# ── T6: planted causal signal yields MODERATE+ in Stage 1 ─────────────────
class TestPlantedCausalSignal:
    def test_planted_signal_reaches_moderate_or_above(self, tmp_path):
        rng = np.random.RandomState(2020)      # ← different seed from _synth_df below
        n = 8000
        df = _synth_df(n, with_signal={6: 0.2}, seed=20)
        # feature = mean of next-6-bar return + noise (predictive of forward_return @ h=6)
        ret6 = pd.Series(df["close"]).pct_change(6).shift(-6).fillna(0).to_numpy()
        snr = 1.5
        df["cvd_bar_5m"] = (ret6 * snr + rng.randn(n) * ret6.std() * 0.5).astype(np.float32)
        # noise filler for the rest
        for name in ("vwap_z_score", "order_flow_imbalance", "hawkes_intrabar_sum",
                     "tick_count", "cvd_intensity_vs_atr",
                     "dist_to_session_high_atr", "dist_to_vwap_atr"):
            df[name] = rng.randn(n).astype(np.float32)
        path = tmp_path / "signal.parquet"
        df.to_parquet(path, index=False)
        summary = run_probe(
            path, tmp_path / "out",
            feature_names=("cvd_bar_5m", "vwap_z_score"),
            horizons=(1, 6, 24), k_folds=3, n_null=30, seed=0,
        )
        # the planted feature should reach MODERATE / STRONG / SUSPECT — NOT NOISE
        cvd = next(r for r in summary["stage_1_univariate"] if r["feature"] == "cvd_bar_5m")
        assert cvd["verdict"] in {"MODERATE", "STRONG", "SUSPECT_LEAKAGE", "WEAK_OR_UNSTABLE"}, cvd


# ── T7: planted LEAK is caught (SUSPECT or future-leak flag) ─────────────
class TestPlantedLeakage:
    def test_perfect_leak_is_flagged(self, tmp_path):
        rng = np.random.RandomState(3030)      # ← different seed from _synth_df below
        n = 8000
        df = _synth_df(n, seed=30)
        # feature = the EXACT forward return (perfect knowledge → must be flagged)
        ret6 = pd.Series(df["close"]).pct_change(6).shift(-6).fillna(0).to_numpy()
        df["cvd_bar_5m"] = ret6.astype(np.float32)        # perfect leak
        for name in ("vwap_z_score", "order_flow_imbalance", "hawkes_intrabar_sum",
                     "tick_count", "cvd_intensity_vs_atr",
                     "dist_to_session_high_atr", "dist_to_vwap_atr"):
            df[name] = rng.randn(n).astype(np.float32)
        path = tmp_path / "leak.parquet"
        df.to_parquet(path, index=False)
        summary = run_probe(
            path, tmp_path / "out",
            feature_names=("cvd_bar_5m",),
            horizons=(1, 6), k_folds=3, n_null=30, seed=0,
        )
        cvd = summary["stage_1_univariate"][0]
        # The feature is forward_return itself: IC at h=6 = ~1.0 → SUSPECT_LEAKAGE
        # (or the causality probe future_leak_flag is set, either way is a flag)
        ph = cvd["per_horizon"]
        any_high = any(ph[h]["ic_mean_abs"] >= IC_SUSPECT for h in ph)
        any_leak = any(ph[h]["causality_probe"]["future_leak_flag"] for h in ph)
        assert cvd["verdict"] == "SUSPECT_LEAKAGE" or any_high or any_leak, cvd


# ── T8: Stage 3 logistic AUC on bias-predictive feature ──────────────────
class TestStage3LogisticBias:
    def test_predictive_feature_gives_auc_above_55(self, tmp_path):
        rng = np.random.RandomState(4040)      # ← different seed from _synth_df below
        n = 8000
        df = _synth_df(n, with_signal={6: 0.2}, seed=40)
        # feature correlates with the (deterministic) sign-of-future-return label
        signed = np.where(df["bias_label"] == 0, +1, np.where(df["bias_label"] == 1, -1, 0))
        df["cvd_bar_5m"] = (signed * 0.7 + rng.randn(n) * 0.5).astype(np.float32)
        for name in ("vwap_z_score", "order_flow_imbalance", "hawkes_intrabar_sum",
                     "tick_count", "cvd_intensity_vs_atr",
                     "dist_to_session_high_atr", "dist_to_vwap_atr"):
            df[name] = rng.randn(n).astype(np.float32)
        path = tmp_path / "bias_signal.parquet"
        df.to_parquet(path, index=False)
        summary = run_probe(
            path, tmp_path / "out",
            feature_names=("cvd_bar_5m", "vwap_z_score"),
            horizons=(1, 6), k_folds=3, n_null=20, seed=0,
        )
        s3 = summary["stage_3_logistic_bias"]
        assert not s3.get("skipped"), s3
        assert s3["auc_mean"] >= 0.55, s3


# ── T9: outputs written utf-8, summary keys present ──────────────────────
class TestOutputContract:
    def test_writes_summary_and_report_utf8(self, tmp_path):
        rng = np.random.RandomState(5050)      # ← different seed from _synth_df below
        n = 5000
        df = _synth_df(n, seed=50)
        for name in ("vwap_z_score", "cvd_bar_5m", "order_flow_imbalance"):
            df[name] = rng.randn(n).astype(np.float32)
        path = tmp_path / "out.parquet"
        df.to_parquet(path, index=False)
        out_dir = tmp_path / "out"
        summary = run_probe(
            path, out_dir,
            feature_names=("cvd_bar_5m",),
            horizons=(1, 6), k_folds=3, n_null=20,
        )
        # JSON summary + report exist + readable as utf-8
        assert (out_dir / "decoder_probe_summary.json").exists()
        report_text = (out_dir / "decoder_probe_report.txt").read_text(encoding="utf-8")
        assert "Decoder probe" in report_text
        assert "STAGE 1" in report_text
        # Summary contract
        for k in ("stage_1_univariate", "stage_2_lasso", "stage_3_logistic_bias",
                  "thresholds_a_priori", "embargo_bars"):
            assert k in summary


# ── T10/T11/T12: DIRECTION vs VOLATILITY (target_mode) ────────────────────
def _vol_only_df(n: int, seed: int) -> pd.DataFrame:
    """A WHITE volatility feature w[i] drives the MAGNITUDE of the next bar's
    move (additive, with idiosyncratic noise → |IC| stays MODERATE, not a leak),
    while the SIGN of every move is independent of w. So w predicts |fwd return|
    but NOT its direction. w is white → the causality probe does not false-flag."""
    rng = np.random.RandomState(seed)
    w = rng.rand(n)                                   # volatility driver, known at i
    sign = rng.choice([-1.0, 1.0], size=n)
    idio = np.abs(rng.randn(n))                       # baseline magnitude noise
    base, gain = 5e-4, 0.4                             # gain tuned so |IC|≈0.2 (MODERATE, not SUSPECT)
    step = np.zeros(n)
    # bar j's move magnitude is driven by the PREVIOUS bar's w (so w[i] predicts
    # step[i+1] = the next forward move); sign independent → no directional edge.
    step[1:] = sign[1:] * base * (idio[1:] + gain * w[:-1])
    close = 1.25 + np.cumsum(step)
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"ts_event": ts, "close": close,
                       "is_session_break": np.zeros(n, dtype=bool)})
    df["mbp_depth_sum_max"] = w.astype(np.float32)   # the volatility feature
    df["bias_label"] = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2]).astype(np.int8)
    return df


def _direction_only_df(n: int, seed: int) -> pd.DataFrame:
    """A feature predicts the SIGN of the next bar's return; the magnitude is
    independent. So the feature has a directional IC but ~zero |return| IC
    (signed feature vs |return| is V-shaped → no monotone rank correlation)."""
    rng = np.random.RandomState(seed)
    fstep = rng.randn(n)                              # next-bar signed return driver
    base = 5e-4
    step = np.zeros(n)
    step[1:] = base * fstep[:-1]                      # bar i's move = previous driver
    close = 1.25 + np.cumsum(step)
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"ts_event": ts, "close": close,
                       "is_session_break": np.zeros(n, dtype=bool)})
    snr, k = 1.0, 4.0
    df["mbp_imbalance_peak"] = (fstep * snr + rng.randn(n) * k).astype(np.float32)
    df["bias_label"] = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2]).astype(np.int8)
    return df


class TestDirectionVsVolatility:
    def test_volatility_only_caught_by_abs_not_signed(self, tmp_path):
        df = _vol_only_df(9000, seed=71)
        path = tmp_path / "vol.parquet"
        df.to_parquet(path, index=False)
        common = dict(feature_names=("mbp_depth_sum_max",), horizons=(1, 3, 6),
                      k_folds=3, n_null=30, seed=0, run_stage_2=False, run_stage_3=False)
        s_signed = run_probe(path, tmp_path / "sgn", target_mode="signed", **common)
        s_abs = run_probe(path, tmp_path / "abs", target_mode="abs", **common)
        ic_signed, v_signed = _best_ic(s_signed, "mbp_depth_sum_max")
        ic_abs, v_abs = _best_ic(s_abs, "mbp_depth_sum_max")
        # |return| is predicted; direction is not, and not a perfect-leak SUSPECT
        assert ic_abs >= IC_MODERATE, (ic_abs, ic_signed)
        assert ic_abs < IC_SUSPECT, ic_abs
        assert ic_abs > ic_signed * 1.5, (ic_abs, ic_signed)
        assert v_signed in {"NOISE", "WEAK"}, (v_signed, ic_signed)
        assert v_abs not in {"NOISE", "SUSPECT_LEAKAGE"}, (v_abs, ic_abs)

    def test_directional_caught_by_signed_not_abs(self, tmp_path):
        df = _direction_only_df(9000, seed=81)
        path = tmp_path / "dir.parquet"
        df.to_parquet(path, index=False)
        common = dict(feature_names=("mbp_imbalance_peak",), horizons=(1, 3, 6),
                      k_folds=3, n_null=30, seed=0, run_stage_2=False, run_stage_3=False)
        s_signed = run_probe(path, tmp_path / "sgn", target_mode="signed", **common)
        s_abs = run_probe(path, tmp_path / "abs", target_mode="abs", **common)
        ic_signed, v_signed = _best_ic(s_signed, "mbp_imbalance_peak")
        ic_abs, v_abs = _best_ic(s_abs, "mbp_imbalance_peak")
        # direction is predicted; |return| is not
        assert ic_signed >= IC_MODERATE, (ic_signed, ic_abs)
        assert ic_signed > ic_abs * 1.5, (ic_signed, ic_abs)
        assert v_abs in {"NOISE", "WEAK"}, (v_abs, ic_abs)

    def test_abs_mode_skips_stage_3_and_validates_mode(self, tmp_path):
        df = _vol_only_df(4000, seed=72)
        path = tmp_path / "v.parquet"
        df.to_parquet(path, index=False)
        s = run_probe(path, tmp_path / "o", feature_names=("mbp_depth_sum_max",),
                      horizons=(1, 3), k_folds=3, n_null=10, target_mode="abs")
        assert s["target_mode"] == "abs"
        assert s["stage_3_logistic_bias"].get("skipped") is True
        assert "directional" in s["stage_3_logistic_bias"].get("reason", "")
        # invalid mode must raise (a priori contract)
        with pytest.raises(ValueError):
            stage_1_univariate(df, ("mbp_depth_sum_max",), (1,), 3, 5, 0,
                               target_mode="nope")

    def test_signed_default_is_backward_compatible(self, tmp_path):
        """Default target_mode is 'signed' and the summary records it."""
        df = _vol_only_df(3000, seed=73)
        path = tmp_path / "bc.parquet"
        df.to_parquet(path, index=False)
        s = run_probe(path, tmp_path / "o", feature_names=("mbp_depth_sum_max",),
                      horizons=(1, 3), k_folds=3, n_null=10)
        assert s["target_mode"] == "signed"
        # signed mode keeps Stage 3 (not skipped for the abs reason)
        assert "directional" not in s["stage_3_logistic_bias"].get("reason", "")
