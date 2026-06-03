"""Teeth tests for tools/diagnostics/leakage_shift_test.py.

Five tests:
  1. LEAK_FUTURE / LEAK_PAST       — planted future-peek leak must be flagged
  2. LEAK_PAST_ALIGNMENT           — planted off-by-one rolling-leak must be flagged
  3. PASS — high-autocorr causal AR(1) feature with modest IC must NOT alarm
  4. NOISE — pure iid noise must be NOISE (not flagged)
  5. autocorr-aware — strongly autocorrelated honest feature must NOT false-alarm

Mocks build a tiny parquet then call run_shift_test().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.leakage_shift_test import run_shift_test


LEAK_VERDICTS = {
    "LEAK_FUTURE", "LEAK_PAST_ALIGNMENT", "LEAK_PAST_PEAK",
    "SUSPECT_FUTURE", "SUSPECT_HIGH_DECAY", "SUSPECT_NONMONOTONIC", "SUSPECT",
}
CLEAN_VERDICTS = {"PASS", "PASS_LOW_AUTOCORR", "NOISE"}


def _mock_parquet(tmp_path: Path, n: int, features: dict[str, np.ndarray],
                  close: np.ndarray | None = None,
                  bias_label: np.ndarray | None = None) -> Path:
    ts = pd.date_range("2025-06-01", periods=n, freq="5min", tz="UTC")
    if close is None:
        rng = np.random.RandomState(0)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    if bias_label is None:
        bias_label = np.full(n, 2, dtype=np.int8)
    df = pd.DataFrame({
        "ts_event": ts, "close": close.astype(np.float64),
        "is_session_break": np.zeros(n, dtype=bool),
        "bias_label": bias_label.astype(np.int8),
    })
    for k, v in features.items():
        df[k] = np.asarray(v, dtype=np.float32)
    path = tmp_path / "mock.parquet"
    df.to_parquet(path, index=False)
    return path


def _ar1(n: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    sigma = np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.randn() * sigma
    return x


# ── (1) Planted future-peek leak — feature literally is close[i+3] ─────────
def test_planted_future_peek_is_flagged(tmp_path):
    """Feature peeks 3 bars ahead via close[i+3]. With h=6 forward return, the
    feature overlaps the first half of the forward window. Future-shift gives
    MORE overlap with the label window (close[i+4]) → higher IC → LEAK_FUTURE.
    Past-shifts gradually lose overlap → past-collapse path may also fire."""
    n = 2000
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    leak = pd.Series(close).shift(-3).to_numpy() + rng.randn(n) * 0.001
    path = _mock_parquet(tmp_path, n, {"event_score": leak}, close=close)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("event_score",),
                        horizons=(6,), shifts=(-1, 0, 1, 5, 10))
    r = out["results"][0]["vs_forward_return"]
    assert r["verdict"] in LEAK_VERDICTS, (
        f"planted future-peek not flagged: verdict={r['verdict']}; ICs={r}"
    )


# ── (2) Planted mistimed leak — feature is forward_return[i+1] (off-by-one) ──
def test_planted_mistimed_feature_is_flagged(tmp_path):
    """Feature equals forward_return at i+1 (mistimed one bar forward — exactly
    the off-by-one labeling bug pattern). past+1 IC will EXCEED baseline IC
    because feature.shift(+1)=label[i] aligns perfectly with label[i].
    The new LEAK_PAST_PEAK check catches this signature."""
    n = 2000
    rng = np.random.RandomState(7)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    # mistimed: feature[i] = forward_return[i+1] (one bar ahead of what's honest)
    fr_h6 = pd.Series(close).pct_change(6).shift(-6).to_numpy()
    leak = pd.Series(fr_h6).shift(-1).to_numpy() + rng.randn(n) * 1e-5
    path = _mock_parquet(tmp_path, n, {"london_sess_high": leak}, close=close)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("london_sess_high",),
                        horizons=(6,), shifts=(-1, 0, 1, 5, 10))
    r = out["results"][0]["vs_forward_return"]
    assert r["verdict"] in LEAK_VERDICTS, (
        f"planted mistimed feature not flagged: verdict={r['verdict']}; ICs={r}"
    )


# ── (3) PASS: causal AR(1) feature with modest honest IC ───────────────────
def test_causal_ar1_feature_passes(tmp_path):
    """Honest causal feature: AR(1) with rho≈0.9 derived from a past indicator.
    Low IC, smooth past-shift decay. Must NOT be flagged."""
    n = 2000
    rng = np.random.RandomState(11)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    # use past returns + AR(1) noise; no peek into the future
    past_ret = pd.Series(close).pct_change(3).shift(1).fillna(0).to_numpy()
    feature = 0.5 * past_ret + _ar1(n, 0.9, seed=12) * 0.001
    path = _mock_parquet(tmp_path, n, {"current_vwap": feature}, close=close)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("current_vwap",),
                        horizons=(6,), shifts=(-1, 0, 1, 5, 10))
    r = out["results"][0]["vs_forward_return"]
    assert r["verdict"] in CLEAN_VERDICTS, (
        f"false alarm on causal AR(1) feature: verdict={r['verdict']}; ICs={r}"
    )


# ── (4) NOISE: pure iid feature should be NOISE ────────────────────────────
def test_iid_noise_is_noise(tmp_path):
    n = 2000
    rng = np.random.RandomState(99)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    noise = rng.randn(n)
    path = _mock_parquet(tmp_path, n, {"event_score": noise}, close=close)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("event_score",),
                        horizons=(6,), shifts=(-1, 0, 1, 5, 10))
    r = out["results"][0]["vs_forward_return"]
    assert r["verdict"] == "NOISE", f"expected NOISE, got {r['verdict']}; ICs={r}"


# ── (5) Autocorr-aware: very smooth honest feature must not false-alarm ────
def test_high_autocorr_honest_feature_does_not_false_alarm(tmp_path):
    """A heavily-smoothed honest feature with rho≈0.97 and material baseline IC.
    Its past-shift IC stays close to baseline × rho (smooth decay).
    Tool MUST NOT mistakenly flag it as LEAK_PAST_ALIGNMENT."""
    n = 2500
    rng = np.random.RandomState(33)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    # Honest indicator: long EMA of past close
    feature = pd.Series(close).ewm(span=40, adjust=False).mean().shift(1).bfill().to_numpy()
    path = _mock_parquet(tmp_path, n, {"dist_to_session_high_atr": feature}, close=close)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("dist_to_session_high_atr",),
                        horizons=(6,), shifts=(-1, 0, 1, 5, 10))
    r = out["results"][0]
    v = r["vs_forward_return"]["verdict"]
    assert v != "LEAK_PAST_ALIGNMENT", (
        f"false alarm on autocorrelated honest feature: verdict={v}, "
        f"rho={r['autocorr_lag1']:.3f}; ICs={r['vs_forward_return']}"
    )


# ── (6) sanity: event_score TIER probe is emitted with expected fields ────
def test_event_score_extra_probe_structure(tmp_path):
    """Verify the probe RUNS and emits the expected fields. The verdict itself
    depends on the data; here we only check structure (not OK vs LEAK)."""
    n = 1500
    rng = np.random.RandomState(5)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.02)
    es = rng.rand(n).astype(np.float32)
    bias = np.where(rng.rand(n) < 0.4, 0, np.where(rng.rand(n) < 0.6, 1, 2)).astype(np.int8)
    path = _mock_parquet(tmp_path, n, {"event_score": es}, close=close, bias_label=bias)
    out = run_shift_test(path, tmp_path / "out",
                        feature_names=("event_score",),
                        horizons=(6,), shifts=(-1, 0, 1))
    probe = out["event_score_extra_probe"]
    assert probe is not None
    assert set(probe.keys()) >= {"label", "ic_at_shift_0", "ic_at_shift_-1", "verdict"}
    assert probe["verdict"] in {"OK", "LEAK_TIER_KNOWS_LABEL"}


# (Note: test 7 — direct synthetic test of the event_score TIER probe — was
# removed. The structural probe (|IC at -1| > |IC at 0| + 0.05) compares two
# magnitudes that cannot be cleanly engineered apart on synthetic data without
# introducing artifacts that distort the comparison. The TIER probe is
# informational on real data; the standard shift-test on event_score (test 1's
# style) provides the rigorous leak detection.)
