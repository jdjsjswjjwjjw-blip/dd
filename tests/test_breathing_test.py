"""Teeth tests for tools/diagnostics/breathing_test.py (v2).

v2 judges the absorb_z TAIL via excess-over-normal (objective anchor), then
asks whether tail firings are STRUCTURED (session-concentrated + clustered with
overnight/session gaps removed). Firing rate is context only (it is a
mechanical function of the threshold = circular if used as a verdict).

Tests build synthetic detector outputs (absorb_z + ts) directly and exercise
the stat/verdict functions:
  1. EXCESS detects a planted fat tail; ≈1 on pure N(0,1) noise.
  2. pure-noise z → SIGNAL_LIKELY_ABSENT (tail no fatter than normal).
  3. gap-aware CV² < raw CV² when overnight gaps are present (the v1 1633 fix).
  4. planted fat tail concentrated in NY hours + clustered → STRUCTURED;
     planted fat tail spread flat across hours + unclustered → DIFFUSE_TAIL.
  5. per-bar metric uses fire FRACTION (not any()) — does not saturate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.breathing_test import (
    _compute_stats, _overall_verdict, _threshold_stats, _gap_aware_cv2,
    EXCESS_TAIL_ANCHOR,
)


def _detector_out(z: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal AbsorptionDetector.compute() output for the stats functions."""
    n = len(z)
    return pd.DataFrame({
        "ts_event": ts,
        "price": np.full(n, 1.25),
        "cvd": np.zeros(n),
        "absorption_intensity": np.zeros(n),
        "absorb_z": z.astype(np.float64),
        "fired": z > 1.0,
    })


def _excess_at(stats: dict, t: float) -> float:
    for s in stats["sweep"]:
        if abs(s["threshold"] - t) < 1e-9:
            return s["excess_over_normal"]
    raise KeyError(t)


# ── (1) excess-over-normal: detects fat tail, ≈1 on noise ──────────────────
class TestExcessOverNormal:
    def test_pure_noise_excess_near_one(self):
        rng = np.random.RandomState(0)
        n = 200_000
        z = rng.randn(n)                       # pure N(0,1)
        ts = pd.date_range("2025-06-01", periods=n, freq="1s", tz="UTC")
        stats = _compute_stats(_detector_out(z, ts))
        # at z>2.0 and z>2.5 the excess must be close to 1 (no fat tail)
        assert _excess_at(stats, 2.0) < 1.6, stats["sweep"]
        assert _excess_at(stats, 2.5) < 1.8

    def test_planted_fat_tail_excess_large(self):
        rng = np.random.RandomState(1)
        n = 200_000
        z = rng.randn(n)
        # plant a heavy tail: 3% of points drawn from N(5,1)
        k = int(0.03 * n)
        idx = rng.choice(n, k, replace=False)
        z[idx] = rng.randn(k) + 5.0
        ts = pd.date_range("2025-06-01", periods=n, freq="1s", tz="UTC")
        stats = _compute_stats(_detector_out(z, ts))
        assert _excess_at(stats, 3.0) >= EXCESS_TAIL_ANCHOR, (
            f"excess at z>3 should be >= anchor on a planted fat tail; {stats['sweep']}"
        )


# ── (2) pure noise → SIGNAL_LIKELY_ABSENT ──────────────────────────────────
class TestNoiseIsAbsent:
    def test_pure_noise_overall_absent(self):
        rng = np.random.RandomState(2)
        n = 200_000
        z = rng.randn(n)
        ts = pd.date_range("2025-06-01", periods=n, freq="1s", tz="UTC")
        ov = _overall_verdict(_compute_stats(_detector_out(z, ts)))
        assert ov["verdict"] == "SIGNAL_LIKELY_ABSENT", ov


# ── (3) gap-aware CV² < raw CV² with overnight gaps ────────────────────────
class TestGapAwareCV2:
    def test_gap_removal_reduces_cv2(self):
        # firings: tight bursts within each of 3 days, huge overnight gaps between
        base = pd.Timestamp("2025-06-02 08:00", tz="UTC")
        ts_list = []
        for day in range(3):
            day0 = base + pd.Timedelta(days=day)
            # a burst of 80 firings ~1s apart
            ts_list += [day0 + pd.Timedelta(seconds=i) for i in range(80)]
        fire_ns = np.array([t.value for t in ts_list], dtype=np.int64)
        cv2_raw, cv2_gap, n_within = _gap_aware_cv2(fire_ns, gap_seconds=3600.0)
        assert np.isfinite(cv2_raw) and np.isfinite(cv2_gap)
        assert cv2_gap < cv2_raw, (
            f"gap-aware CV² ({cv2_gap:.2f}) must drop below raw ({cv2_raw:.2f}) "
            f"once overnight gaps are removed"
        )


# ── (4) STRUCTURED vs DIFFUSE on planted fat tails ─────────────────────────
def _frame_with_tail(hours_for_tail, seed, n_days=8):
    """Build z over n_days of 24h 1-min ticks. Background N(0,1); a fat tail
    (z~N(5,1)) planted only in the given UTC hours, in tight clusters."""
    rng = np.random.RandomState(seed)
    n = n_days * 24 * 60
    ts = pd.date_range("2025-06-02 00:00", periods=n, freq="1min", tz="UTC")
    z = rng.randn(n)
    hours = ts.hour.to_numpy()
    in_hours = np.isin(hours, hours_for_tail)
    # cluster: only fire on a subset (every other minute) within those hours
    cluster = in_hours & (np.arange(n) % 2 == 0)
    z[cluster] = rng.randn(int(cluster.sum())) + 5.0
    return z, ts


class TestStructuredVsDiffuse:
    def test_ny_concentrated_clustered_is_structured(self):
        # fat tail only in NY/overlap hours (13-19 UTC), clustered
        z, ts = _frame_with_tail([13, 14, 15, 16, 17, 18, 19], seed=3)
        ov = _overall_verdict(_compute_stats(_detector_out(z, ts)))
        assert ov["verdict"] in {"STRUCTURED", "PARTIAL_STRUCTURE"}, ov
        assert ov["session_concentrated"] is True, ov

    def test_flat_uniform_tail_is_diffuse(self):
        # fat tail spread across ALL 24 hours (no session preference)
        z, ts = _frame_with_tail(list(range(24)), seed=4)
        ov = _overall_verdict(_compute_stats(_detector_out(z, ts)))
        # tail IS fat (anchor found) but not concentrated → not STRUCTURED
        assert ov["anchor_threshold"] is not None
        assert ov["verdict"] != "STRUCTURED", ov
        assert ov["session_concentrated"] is False, ov


# ── (5) per-bar uses fire FRACTION, not any() ──────────────────────────────
class TestPerBarFraction:
    def test_per_bar_is_fraction_not_saturated(self):
        """At a HIGH per-tick rate, the old any()-per-bar saturated to ~1.0.
        The fraction metric must report the actual within-bar fire fraction,
        well below 1.0 when only part of each bar fires."""
        rng = np.random.RandomState(5)
        n = 60_000
        ts = pd.date_range("2025-06-02 08:00", periods=n, freq="1s", tz="UTC")
        z = rng.randn(n)
        # make ~30% of ticks strong (z>2) — any()-per-bar would saturate near 1
        idx = rng.choice(n, int(0.30 * n), replace=False)
        z[idx] = 3.0
        stats = _compute_stats(_detector_out(z, ts))
        s2 = next(s for s in stats["sweep"] if abs(s["threshold"] - 2.0) < 1e-9)
        frac = s2["per_bar_median_fire_frac"]
        assert 0.0 < frac < 0.9, (
            f"per-bar median fire fraction should reflect the ~30% rate, "
            f"not saturate; got {frac}"
        )
