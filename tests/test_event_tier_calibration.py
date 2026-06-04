"""#1 — event_score tier thresholds recalibrated for the structure_compass
PRODUCT formula.

The old thresholds (STRONG>=0.70, MID>=0.50) were calibrated for the legacy
microstructure event_score (a sum-of-binaries over [0,1]). In structure_compass
mode event_score = active × level_prox × compass_mag (a PRODUCT) whose realistic
range is far smaller — on 6B June 2025 the is_event rows had p33=0.105, p66=0.263,
max=0.651. So STRONG=0.70 sat ABOVE the max → every event was WEAK (628/684) and
the TP/SL/horizon tier system was effectively dead.

Recalibrated as FROZEN constants (not runtime percentiles — that would make a
bar's tier depend on later bars' scores = R1 leakage) from the empirical
distribution: STRONG≈p66=0.25, MID≈p33=0.10.

Tests lock:
  • the runtime consistency guard (MID_MIN <= STRONG_MIN),
  • reachability — STRONG_MIN is below the empirical compass max (regression
    guard against reverting to an unreachable threshold),
  • TEETH: driving the real label_by_outcome over event_scores spanning the
    empirical distribution now yields ALL THREE tiers (weak/mid/strong),
    proving the tier system is active again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P
from regime_config import (
    EVENT_LABEL_SCORE_STRONG_MIN, EVENT_LABEL_SCORE_MID_MIN,
)

# The empirical compass event_score ceiling observed on 6B June 2025 (is_event rows).
EMPIRICAL_COMPASS_MAX = 0.651


class TestThresholdConstants:
    def test_consistency_mid_le_strong(self):
        """Matches the runtime guard in label_by_outcome (raises if violated)."""
        assert EVENT_LABEL_SCORE_MID_MIN <= EVENT_LABEL_SCORE_STRONG_MIN

    def test_strong_is_reachable(self):
        """STRONG must sit BELOW the empirical compass max, else strong=0 always
        (the bug #1 fixes). Guards against reverting to the old 0.70."""
        assert EVENT_LABEL_SCORE_STRONG_MIN < EMPIRICAL_COMPASS_MAX, (
            f"STRONG_MIN={EVENT_LABEL_SCORE_STRONG_MIN} >= empirical max "
            f"{EMPIRICAL_COMPASS_MAX} → strong tier unreachable"
        )

    def test_mid_above_noise(self):
        """MID must be > 0 so the weak/mid boundary is meaningful."""
        assert EVENT_LABEL_SCORE_MID_MIN > 0.0


def _events_frame(event_scores: np.ndarray) -> pd.DataFrame:
    """Minimal bars frame of is_event rows with the given event_score values,
    arranged so label_by_outcome can run its tier + barrier scan."""
    n = len(event_scores) + 60          # padding so forward windows exist
    ts = pd.date_range("2025-06-02 08:00", periods=n, freq="5min", tz="UTC")
    rng = np.random.RandomState(0)
    close = 1.2500 + np.cumsum(rng.randn(n) * 0.0002)
    high = close + 0.0003
    low = close - 0.0003
    es = np.zeros(n, dtype=np.float64)
    is_ev = np.zeros(n, dtype=np.int8)
    # place the event rows in the first len(event_scores) bars
    es[: len(event_scores)] = event_scores
    is_ev[: len(event_scores)] = 1
    return pd.DataFrame({
        "ts_event": ts,
        "open": close, "high": high, "low": low, "close": close,
        "atr_14": np.full(n, 0.0010),
        "regime_label": np.array(["ranging"] * n, dtype=object),
        "is_event": is_ev,
        "event_score": es,
        "is_london": np.ones(n, dtype=np.int8),
    })


class TestTierSystemActive:
    def test_all_three_tiers_appear_on_empirical_distribution(self):
        """TEETH: with the recalibrated thresholds, event_scores spanning the
        real 6B distribution must produce weak (0), mid (1) AND strong (2)
        tiers. Under the old 0.70/0.50 thresholds every row was weak."""
        # span the empirical distribution: below p33, between p33-p66, above p66
        scores = np.array([
            0.02, 0.05, 0.08,           # < MID (0.10) → weak
            0.12, 0.18, 0.22,           # MID..STRONG → mid
            0.30, 0.45, 0.60,           # >= STRONG (0.25) → strong
        ])
        out = P.label_by_outcome(_events_frame(scores), use_event_score_tier_labels=True)
        tiers = out.loc[out["is_event"] == 1, "event_label_tier"].to_numpy()
        present = set(int(t) for t in tiers)
        assert present == {0, 1, 2}, (
            f"tier system not fully active: present tiers={present} "
            f"(expected weak=0, mid=1, strong=2)"
        )

    def test_old_strong_threshold_was_unreachable(self):
        """Counter-check on the EXACT bug: the old STRONG threshold (0.70) sat
        above the empirical compass max (0.651), so the strong tier was
        impossible — every event was weak or mid (the run showed strong=0,
        mid=56, weak=628). The recalibrated STRONG (0.25) is reachable, so the
        same distribution now yields strong events too."""
        # representative scores across the empirical compass range [0, 0.651]
        scores = np.array([0.02, 0.12, 0.30, 0.45, 0.60])
        s_hi_old = 0.70
        old_strong = (scores >= s_hi_old)
        assert not old_strong.any(), (
            "with the old 0.70 threshold no event reaches strong — that is the bug #1 fixes"
        )
        new_strong = (scores >= EVENT_LABEL_SCORE_STRONG_MIN)
        assert new_strong.any(), (
            "recalibrated STRONG must make the strong tier reachable on the empirical range"
        )
