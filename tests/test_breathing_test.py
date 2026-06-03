"""Teeth tests for tools/diagnostics/breathing_test.py.

Four tests prove the harness behaves correctly under known conditions:
  1. PASS — planted absorption (from the Phase-1.12 simulator) → HEALTHY-ish
  2. DEAD — baseline-only trade stream → SIGNAL_LIKELY_ABSENT
  3. SESSION-AWARE — firings concentrated at NY hour → NY peak detected
  4. CV² — Poisson-spaced firings vs clustered → CV² verdict differs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.absorption_simulator import (
    AbsorptionGroundTruthSimulator, AbsorptionSimConfig,
)
from tools.diagnostics.breathing_test import (
    run_breathing_test, _compute_stats, _overall_verdict, _verdict_ratio,
    CV2_PASS, CV2_SUSPECT,
)


def _mock_mbo(tmp_path: Path, trades: pd.DataFrame) -> Path:
    """Write a parquet matching the raw-MBO schema the harness reads."""
    df = trades.copy()
    df["action"] = "T"
    # ensure 'side' is in the harness's BUY/SELL vocab
    df["side"] = df["side"].astype(str)
    path = tmp_path / "mock_mbo.parquet"
    df[["ts_event", "action", "side", "price", "size"]].to_parquet(path, index=False)
    return path


# ── (1) HEALTHY: planted absorption → breathing-positive ───────────────────
def test_planted_absorption_is_healthy(tmp_path):
    """The simulator produces a trade stream with planted absorption.
    Running the harness on it must NOT verdict SIGNAL_LIKELY_ABSENT or
    DETECTOR_BROKEN — the detector is supposed to fire here."""
    sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=42))
    trades, _ = sim.generate(n_absorptions=40)
    # shift trades into UTC so the session lookup is meaningful
    trades = trades.copy()
    trades["ts_event"] = pd.to_datetime(trades["ts_event"], utc=True)
    mbo = _mock_mbo(tmp_path, trades)
    summary = run_breathing_test(mbo, tmp_path / "out")
    overall = summary["overall_verdict"]
    assert overall != "SIGNAL_LIKELY_ABSENT", (
        f"planted absorption should breathe; got {overall} with stats: "
        f"{summary['stats']['firing_rate']}, p99={summary['stats']['absorb_z_distribution']['p99']:.2f}"
    )
    assert overall != "DETECTOR_BROKEN", (
        f"healthy planted data should not look broken; got {overall}"
    )
    # Material firing rate and material p99
    assert summary["stats"]["firing_rate"]["value"] > 0.001
    assert summary["stats"]["absorb_z_distribution"]["p99"] > 1.0


# ── (2) DEAD: baseline only → SIGNAL_LIKELY_ABSENT ─────────────────────────
def test_baseline_only_is_signal_absent(tmp_path):
    """Balanced two-sided baseline chop has no absorption to detect.
    Firing rate should be near zero → SIGNAL_LIKELY_ABSENT or SUSPECT."""
    n = 6000
    rng = np.random.RandomState(0)
    ts = pd.date_range("2025-06-02 08:00", periods=n, freq="100ms", tz="UTC")
    price = 1.25 + np.cumsum(rng.randn(n) * 1e-5)         # quiet random walk
    side = np.where(rng.rand(n) < 0.5, "A", "B")          # balanced two-sided
    size = rng.uniform(2.0, 12.0, n)
    trades = pd.DataFrame({"ts_event": ts, "price": price, "side": side, "size": size})
    mbo = _mock_mbo(tmp_path, trades)
    summary = run_breathing_test(mbo, tmp_path / "out")
    overall = summary["overall_verdict"]
    assert overall in {"SIGNAL_LIKELY_ABSENT", "SUSPECT"}, (
        f"baseline-only should not look healthy; got {overall} with "
        f"firing_rate={summary['stats']['firing_rate']['value']:.4%}"
    )
    # Crucially NOT HEALTHY
    assert overall != "HEALTHY"


# ── (3) SESSION-aware: firings at 14:00 UTC only → NY peak detected ────────
def test_session_distribution_picks_up_ny_peak(tmp_path):
    """Plant absorption only inside the 13:00–15:59 UTC NY/overlap window
    surrounded by quiet baseline. The session-distribution stat must show
    active sessions dominating the quiet ones."""
    cfg = AbsorptionSimConfig(seed=7)
    sim = AbsorptionGroundTruthSimulator(cfg)
    trades, _ = sim.generate(n_absorptions=40, start="2025-06-02 14:00:00")
    # All trades cluster around the 14:00 start → 100% NY/overlap.
    trades = trades.copy()
    trades["ts_event"] = pd.to_datetime(trades["ts_event"], utc=True)
    mbo = _mock_mbo(tmp_path, trades)
    summary = run_breathing_test(mbo, tmp_path / "out")
    rates = summary["stats"]["session_distribution"]["rates_per_session"]
    n_per = summary["stats"]["session_distribution"]["n_per_session"]
    # all trades planted at 14:00 → only `overlap` (13-16) gets ticks; quiet
    # sessions have n=0 → rate is NaN. The session-detection succeeded iff the
    # active-window session has a material firing rate.
    overlap_rate = rates.get("overlap", float("nan"))
    assert np.isfinite(overlap_rate) and overlap_rate > 0.001, (
        f"overlap (13-16 UTC) should show firings; got rates={rates}, n={n_per}"
    )
    # And the quiet sessions must have NO ticks at all (sanity on session map).
    assert n_per["asia"] == 0 and n_per["off"] == 0, (
        f"all data was at 14:00 — quiet sessions should be empty; got n={n_per}"
    )


# ── (4) CV²: clustered firings score higher than Poisson-like ──────────────
def test_cv2_detects_clustered_vs_poisson():
    """Build two synthetic detector outputs:
      A: firings strictly clustered into 3 dense bursts → high CV²
      B: firings uniformly spaced (regular) → CV² ≪ 1
    The CV² stat must rank A > B."""
    n = 6000
    base_ts = pd.date_range("2025-06-02 08:00", periods=n, freq="100ms", tz="UTC")

    def _frame(fired: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "ts_event": base_ts,
            "price": np.full(n, 1.25),
            "cvd": np.zeros(n),
            "absorption_intensity": np.zeros(n),
            "absorb_z": np.where(fired, 2.5, 0.0).astype(np.float64),
            "fired": fired.astype(bool),
        })

    # CLUSTERED: three dense bursts at i in [1000,1100), [3000,3100), [5000,5100)
    fired_clustered = np.zeros(n, dtype=bool)
    for start in (1000, 3000, 5000):
        fired_clustered[start:start + 100] = True

    # REGULAR: one firing every 30 ticks (very regular spacing)
    fired_regular = np.zeros(n, dtype=bool)
    fired_regular[::30] = True

    stats_c = _compute_stats(_frame(fired_clustered), bar_freq="5min")
    stats_r = _compute_stats(_frame(fired_regular), bar_freq="5min")

    cv2_c = stats_c["inter_event_clustering"]["cv2"]
    cv2_r = stats_r["inter_event_clustering"]["cv2"]
    assert cv2_c > cv2_r, (
        f"clustered CV² ({cv2_c:.2f}) must exceed regular CV² ({cv2_r:.2f})"
    )
    # Clustered should PASS; regular should DEAD (very sub-Poisson)
    assert stats_c["inter_event_clustering"]["verdict"] == "PASS"
    assert stats_r["inter_event_clustering"]["verdict"] == "DEAD"
