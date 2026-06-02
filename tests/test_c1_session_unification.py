"""C1 fix (option B1) — session-boundary unification on session_features.

Locks three guarantees so the C1 contradiction cannot silently return:
  1. SINGLE SOURCE OF TRUTH: seasonal_map + prepare_day_trading's daytrade_default
     both derive from session_features.SESSION_WINDOWS (no 4th definition).
  2. B1 EDGE PRESERVATION: london_sess_high/low use the 07:00-12:00 morning
     window (LONDON_LEVEL_WINDOW), deliberately decoupled from the unified
     07:00-16:00 is_london flag — so the validated "London top" edge (IC≈-0.11)
     is byte-identical to before. After noon the level is the frozen morning
     extreme, NOT the full-session high.
  3. Dead `london_ny` profile entry is gone.

quant-rigor-guard RULE 1 (consistent session frame across features) + RULE 8
(timezone/session consistency) + RULE 6 (do not break the validated edge).
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P
from modules import seasonal_map as SM
from modules.session_features import SESSION_WINDOWS


class TestSingleSourceOfTruth:
    def test_seasonal_map_derives_from_reference(self):
        assert (SM.LONDON_OPEN_HOUR, SM.LONDON_CLOSE_HOUR) == SESSION_WINDOWS["london"]
        assert (SM.NY_OPEN_HOUR, SM.NY_CLOSE_HOUR) == SESSION_WINDOWS["ny"]

    def test_prepare_daytrade_derives_from_reference(self):
        sess = P.SESSION_PROFILES["daytrade_default"]
        # hour-based reference rendered to hh:mm
        assert sess["london"] == ("07:00", "16:00") == (
            f"{SESSION_WINDOWS['london'][0]:02d}:00", f"{SESSION_WINDOWS['london'][1]:02d}:00")
        assert sess["ny"] == ("13:00", "22:00")
        assert sess["asia"] == ("00:00", "08:00")
        assert sess["overlap"] == ("13:00", "16:00")

    def test_no_fourth_definition(self):
        # the three live definitions agree on london + ny
        assert P.SESSIONS["london"] == ("07:00", "16:00")
        assert (SM.LONDON_OPEN_HOUR, SM.LONDON_CLOSE_HOUR) == (7, 16)
        assert SESSION_WINDOWS["london"] == (7, 16)


class TestDeadProfileRemoved:
    def test_london_ny_gone(self):
        for prof in P.SESSION_PROFILES.values():
            assert "london_ny" not in prof, "dead london_ny entry must be removed"
        assert "london_ny" not in P.SESSIONS


class TestB1EdgePreserved:
    def test_london_level_window_is_morning(self):
        assert P.LONDON_LEVEL_WINDOW == ("07:00", "12:00")

    def test_london_sess_high_uses_07_12_not_full_session(self):
        """The decisive B1 check: after 12:00 (still inside the unified 07-16
        is_london flag), london_sess_high must equal the FROZEN MORNING high,
        not the full-session high — i.e. the validated feature is unchanged."""
        ts = pd.date_range("2025-04-07 06:00", "2025-04-07 17:00", freq="5min", tz="UTC")
        rng = np.random.RandomState(0)
        close = 100 + np.cumsum(rng.randn(len(ts)) * 0.1)
        df = pd.DataFrame({
            "ts_event": ts, "high": close + 0.2, "low": close - 0.2,
            "close": close, "atr_14": 0.5,
        })
        out = P.add_london_session_running_levels(df.copy())

        tt = pd.to_datetime(df["ts_event"]).dt.time.to_numpy()
        in_morning = np.array([time(7, 0) <= x < time(12, 0) for x in tt])
        in_full = np.array([time(7, 0) <= x < time(16, 0) for x in tt])
        h = df["high"].to_numpy()
        morning_high = h[in_morning].max()
        full_high = h[in_full].max()

        assert full_high > morning_high + 1e-6, "test frame must have a higher afternoon high"
        last = float(out["london_sess_high"].iloc[-1])   # 16:55 — well after noon
        assert abs(last - morning_high) < 1e-9, "london_sess_high must be the frozen morning high (B1)"
        assert abs(last - full_high) > 1e-6, "must NOT be the full-session high (that would be B2)"


class TestFlagUnified:
    def test_is_london_covers_full_session(self):
        """is_london now marks the full 07:00-16:00 session (unified), so a
        14:00 bar (london afternoon) is flagged — under the old 07-12 it wasn't."""
        ls, le = P.SESSIONS["london"]
        m = P._session_time_mask(pd.Series([time(14, 0)]), ls, le)
        assert bool(m.iloc[0]) is True, "14:00 must be inside the unified london session"
