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
    _bar_tick_counts, _thin_tick_mask, _run_pipeline,
    EXCESS_TAIL_ANCHOR, WARMUP_TICKS,
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


# ══════════════════════════════════════════════════════════════════════════
# v2 + thin-bar filter (5 teeth: artifact-detect, signal-preserve, p25, N=0,
# always-show-both)
# ══════════════════════════════════════════════════════════════════════════

def _detected_with_density(
    asia_ticks_per_bar: int, ny_ticks_per_bar: int, other_ticks_per_bar: int,
    asia_z_mean: float, ny_z_mean: float, other_z_mean: float,
    n_days: int = 6, seed: int = 0,
) -> pd.DataFrame:
    """Build a detector-output frame where tick density and absorb_z mean are
    controlled per hour-band. Density is set per 5-min bar."""
    rng = np.random.RandomState(seed)
    base = pd.Timestamp("2025-06-02 00:00", tz="UTC")
    rows: list[dict] = []
    for day in range(n_days):
        day0 = base + pd.Timedelta(days=day)
        for hour in range(24):
            if 0 <= hour < 6:
                tpb, zm = asia_ticks_per_bar, asia_z_mean
            elif 13 <= hour < 19:
                tpb, zm = ny_ticks_per_bar, ny_z_mean
            else:
                tpb, zm = other_ticks_per_bar, other_z_mean
            if tpb <= 0:
                continue
            for bar_i in range(12):                              # 12 × 5min per hour
                bar_start = day0 + pd.Timedelta(hours=hour, minutes=bar_i * 5)
                spacing_s = 300.0 / tpb
                for k in range(tpb):
                    ts = bar_start + pd.Timedelta(seconds=k * spacing_s)
                    z = float(rng.randn() + zm)
                    rows.append({"ts_event": ts, "absorb_z": z})
    df = pd.DataFrame(rows).sort_values("ts_event").reset_index(drop=True)
    df["price"] = 1.25
    df["cvd"] = 0.0
    df["absorption_intensity"] = 0.0
    df["fired"] = df["absorb_z"] > 1.0
    # Add a few warmup rows so WARMUP_TICKS drop doesn't kill the test data
    pad = pd.DataFrame({
        "ts_event": pd.date_range("2025-06-01 23:00", periods=WARMUP_TICKS, freq="1s", tz="UTC"),
        "price": 1.25, "cvd": 0.0, "absorption_intensity": 0.0,
        "absorb_z": np.zeros(WARMUP_TICKS), "fired": False,
    })
    return pd.concat([pad, df], ignore_index=True)


# ── (T1) TEETH: thin-bar artifact tail revealed by filter ──────────────────
class TestThinBarArtifactRevealed:
    def test_filter_reveals_thin_bar_artifact(self):
        """Plant a fat tail ONLY in thin asia bars (3 ticks/5min) — exactly the
        AII-formula artifact pattern. NY/other have dense bars but only noise.

          unfiltered: tail is concentrated in asia → sess a/q < 1 (anti-pattern)
          filtered (N=20): asia thin bars dropped → tail vanishes → ABSENT

        Confirms the filter exposes the artifact, not real institutional flow."""
        det = _detected_with_density(
            asia_ticks_per_bar=3,  ny_ticks_per_bar=40, other_ticks_per_bar=15,
            asia_z_mean=4.5,       ny_z_mean=0.0,      other_z_mean=0.0,
            n_days=6, seed=1,
        )
        res = _run_pipeline(det, bar_freq="5min", min_ticks_per_bar=20)

        # Unfiltered: a tail exists (asia spikes) but it's anti-concentrated
        unf_ov = res["unfiltered"]["overall"]
        assert unf_ov.get("anchor_threshold") is not None, "tail should be found unfiltered"
        unf_sess = unf_ov.get("anchor_session_ratio")
        assert isinstance(unf_sess, float) and unf_sess < 1.5, (
            f"unfiltered anchor sess a/q should reveal anti-concentration; got {unf_sess}"
        )

        # Filtered: only dense bars (NY/other ≥ 20 ticks per 5min) survive.
        # In this synthetic NY/other are noise → tail vanishes → ABSENT.
        flt_ov = res["filtered"]["overall"]
        assert flt_ov["verdict"] == "SIGNAL_LIKELY_ABSENT", (
            f"filtered tail should vanish on planted thin-bar artifact; got {flt_ov}"
        )


# ── (T2) TEETH counter: filter must NOT kill a real institutional signal ───
class TestRealSignalPreserved:
    def test_filter_preserves_dense_ny_tail(self):
        """Plant a fat tail in DENSE NY bars (≥30 ticks/5min) with noise elsewhere.
        Both unfiltered and filtered should find the tail; filtered must not
        regress to ABSENT/DIFFUSE on a genuine concentrated signal."""
        det = _detected_with_density(
            asia_ticks_per_bar=8,  ny_ticks_per_bar=40, other_ticks_per_bar=15,
            asia_z_mean=0.0,       ny_z_mean=2.5,      other_z_mean=0.0,
            n_days=8, seed=2,
        )
        res = _run_pipeline(det, bar_freq="5min", min_ticks_per_bar=20)
        unf_ov = res["unfiltered"]["overall"]
        flt_ov = res["filtered"]["overall"]
        assert unf_ov["verdict"] != "SIGNAL_LIKELY_ABSENT", unf_ov
        assert flt_ov["verdict"] != "SIGNAL_LIKELY_ABSENT", (
            f"filter must not erase a real NY-dense signal; got {flt_ov}"
        )
        # the tail must remain found in the filtered branch
        assert flt_ov.get("anchor_threshold") is not None


# ── (T3) p25 suggestion is empirical, not targeted ─────────────────────────
class TestSuggestedN_p25:
    def test_suggested_p25_matches_numpy(self):
        # Mix of dense/thin bars across the data; the suggestion must equal
        # numpy.percentile of the actual ticks-per-bar distribution.
        det = _detected_with_density(
            asia_ticks_per_bar=5, ny_ticks_per_bar=40, other_ticks_per_bar=15,
            asia_z_mean=0, ny_z_mean=0, other_z_mean=0, n_days=4, seed=3,
        )
        res = _run_pipeline(det, bar_freq="5min", min_ticks_per_bar=0)
        suggested = res["ticks_per_bar_suggested_n_p25"]
        # Recompute independently from the post-warmup ticks
        settled = det.iloc[WARMUP_TICKS:].reset_index(drop=True)
        counts = _bar_tick_counts(settled, "5min").to_numpy()
        expected = int(round(float(np.percentile(counts, 25))))
        assert suggested == expected, (
            f"p25 suggestion {suggested} must equal numpy p25 {expected} of the actual "
            "ticks-per-bar distribution (R10: derived, not targeted)"
        )


# ── (T4) N=0 byte-equality with v2 ─────────────────────────────────────────
class TestNZeroByteEqualV2:
    def test_min_ticks_zero_matches_v2(self):
        """min_ticks_per_bar=0 must produce stats identical to the legacy v2
        _compute_stats path. Guards every pre-existing v2 test."""
        rng = np.random.RandomState(4)
        n = 200_000
        z = rng.randn(n)
        ts = pd.date_range("2025-06-02 00:00", periods=n, freq="1s", tz="UTC")
        det = _detector_out(z, ts)
        # legacy path
        legacy = _compute_stats(det)
        legacy_ov = _overall_verdict(legacy)
        # new path
        res = _run_pipeline(det, bar_freq="5min", min_ticks_per_bar=0)
        new_stats = res["unfiltered"]["stats"]
        new_ov = res["unfiltered"]["overall"]
        assert "filtered" not in res, "filtered branch must not run when N=0"
        # sweep equality field-by-field
        assert len(legacy["sweep"]) == len(new_stats["sweep"])
        for a, b in zip(legacy["sweep"], new_stats["sweep"]):
            for k in ("threshold", "firing_rate", "excess_over_normal",
                      "session_active_quiet", "cv2_raw", "cv2_gap_aware",
                      "per_bar_median_fire_frac", "n_firings"):
                va, vb = a[k], b[k]
                if isinstance(va, float) and isinstance(vb, float):
                    if np.isnan(va) and np.isnan(vb):
                        continue
                assert va == vb, f"v2 byte-equality broken at threshold {a['threshold']} key {k}"
        assert legacy_ov["verdict"] == new_ov["verdict"]


# ── (T5) Filtered path always reports BOTH sections (R10: no hiding) ───────
class TestAlwaysShowBoth:
    def test_pipeline_always_emits_unfiltered_section_under_filter(self):
        det = _detected_with_density(
            asia_ticks_per_bar=3, ny_ticks_per_bar=40, other_ticks_per_bar=15,
            asia_z_mean=4.5, ny_z_mean=0, other_z_mean=0, n_days=4, seed=5,
        )
        res = _run_pipeline(det, bar_freq="5min", min_ticks_per_bar=20)
        # both sections present with the expected keys
        assert "unfiltered" in res and "filtered" in res
        for branch in ("unfiltered", "filtered"):
            assert "overall" in res[branch] and "stats" in res[branch]
            assert "sweep" in res[branch]["stats"]
            assert "absorb_z_percentiles" in res[branch]["stats"]
        # filtered section also records the bar bookkeeping
        f = res["filtered"]
        assert {"n_bars_total", "n_bars_kept", "fraction_bars_kept", "n_ticks_kept"} <= f.keys()
        assert 0 < f["n_bars_kept"] < f["n_bars_total"]


# ── (T6) thin_tick_mask correctness ────────────────────────────────────────
class TestThinTickMask:
    def test_mask_keeps_only_dense_bars(self):
        det = _detected_with_density(
            asia_ticks_per_bar=4, ny_ticks_per_bar=30, other_ticks_per_bar=10,
            asia_z_mean=0, ny_z_mean=0, other_z_mean=0, n_days=2, seed=6,
        )
        settled = det.iloc[WARMUP_TICKS:].reset_index(drop=True)
        mask = _thin_tick_mask(settled, min_ticks=20, bar_freq="5min")
        # all kept ticks must belong to bars with ≥20 ticks
        kept_bar_id = (pd.to_datetime(settled.loc[mask, "ts_event"], utc=True)
                       .dt.floor("5min").value_counts())
        assert (kept_bar_id >= 20).all(), kept_bar_id.min()
        # all dropped ticks belong to thin bars
        if (~mask).any():
            dropped_bar_id = (pd.to_datetime(settled.loc[~mask, "ts_event"], utc=True)
                              .dt.floor("5min").value_counts())
            counts = _bar_tick_counts(settled, "5min")
            for bar, _ in dropped_bar_id.items():
                assert counts.loc[bar] < 20
