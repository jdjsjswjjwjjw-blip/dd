"""Tests for Phase 1 / Commit A2 — event_direction veto default inversion.

The old kwarg `ignore_event_direction_veto: bool = False` defaulted to
veto-ON, even though the code's own diagnostic showed the veto drops 60%+
of true directional signal (CVD+OBI+Kalman voting is ~70% wrong on real
data). A2 renames + inverts the default to `apply_event_direction_veto:
bool = False` so veto-OFF is the safe, documented default. The veto must
remain available as an explicit opt-in for experiments.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as pdt


def _bars(closes, *, atr=0.001, event_direction=0):
    n = len(closes)
    c = np.asarray(closes, dtype=np.float64)
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "ts_event": ts, "open": c, "high": c, "low": c, "close": c,
        "atr_14": np.full(n, atr),
        "is_event": np.ones(n, dtype=np.int8),
        "event_score": np.zeros(n),
        "kalman_direction": np.zeros(n, dtype=np.int8),
        "event_direction": np.full(n, event_direction, dtype=np.int8),
        "is_london": np.ones(n, dtype=bool),
        "is_overlap": np.zeros(n, dtype=bool),
        "regime_label": np.array(["ranging"] * n),
        "session": np.array(["london"] * n),
    })


class TestA2Signature:
    def test_new_kwarg_present(self):
        sig = inspect.signature(pdt.label_by_outcome).parameters
        assert "apply_event_direction_veto" in sig

    def test_old_kwarg_gone(self):
        sig = inspect.signature(pdt.label_by_outcome).parameters
        assert "ignore_event_direction_veto" not in sig, (
            "A2 regressed — the double-negative old kwarg is back"
        )

    def test_default_is_false(self):
        sig = inspect.signature(pdt.label_by_outcome).parameters
        assert sig["apply_event_direction_veto"].default is False, (
            "A2 regressed — veto must be OFF by default"
        )


class TestA2VetoBehaviour:
    """Pair-test: same downward path under bullish event_direction.
    Veto-OFF (default): allowed to be SHORT. Veto-ON (opt-in): blocked.
    """

    def test_veto_off_default_allows_short_under_bull_event(self):
        # Bullish event_direction (=+1) on every bar. The downward price
        # path should still be labeled SHORT because the veto is off.
        df = _bars([1.0, 0.9990, 0.9980, 0.9970, 0.9965, 0.9965],
                   event_direction=+1)
        out = pdt.label_by_outcome(df, default_tp_mult=1.5, default_sl_mult=1.0)
        assert int(out["bias_label"].iloc[0]) == 1     # SHORT
        assert int(out["path_outcome"].iloc[0]) == 1   # short_tp

    def test_veto_on_blocks_short_under_bull_event(self):
        # Same path, veto explicitly turned ON via opt-in kwarg
        df = _bars([1.0, 0.9990, 0.9980, 0.9970, 0.9965, 0.9965],
                   event_direction=+1)
        out = pdt.label_by_outcome(
            df, default_tp_mult=1.5, default_sl_mult=1.0,
            apply_event_direction_veto=True,
        )
        # Bullish event_dir → allow_short=False → short_hit suppressed,
        # so the row cannot be short_tp (path_outcome=1).
        assert int(out["path_outcome"].iloc[0]) != 1
        assert int(out["bias_label"].iloc[0]) != 1

    def test_veto_off_default_allows_long_under_bear_event(self):
        # Symmetric mirror — upward path under bearish event_direction
        df = _bars([1.0, 1.0010, 1.0020, 1.0030, 1.0035, 1.0035],
                   event_direction=-1)
        out = pdt.label_by_outcome(df, default_tp_mult=1.5, default_sl_mult=1.0)
        assert int(out["bias_label"].iloc[0]) == 0     # LONG
        assert int(out["path_outcome"].iloc[0]) == 0   # long_tp


class TestA2RefineryAndCli:
    """Lock in that the rename propagated through every layer."""

    def test_refinery_signature_renamed(self):
        sig = inspect.signature(pdt.run_day_trading_refinery).parameters
        assert "apply_event_direction_veto" in sig
        assert "ignore_event_direction_veto" not in sig

    def test_no_stale_string_in_source(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        # Old snake-case kwarg name and old CLI flag must be gone
        assert "ignore_event_direction_veto" not in src
        assert "--ignore-event-direction-veto" not in src
        # New opt-in markers must be present
        assert "apply_event_direction_veto" in src
        assert "--apply-event-direction-veto" in src
