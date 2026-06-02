"""C2 fix (part ب) — consumer-side LOB↔parquet alignment check (data_loader).

The SSL consumer pairs tensor[i] ↔ parquet.iloc[i] POSITIONALLY and previously
checked LENGTH ONLY (it ignored the saved lob_ts) — the real point of danger:
a reused/regenerated/reordered .npy would mispair every LOB window with the
wrong bar's label. SSLDataset._verify_lob_alignment now loads lob_ts and
asserts tensor_ts == parquet ts_event off-by-zero, with a SAFE FALLBACK to
length-only when the timestamps sidecar is absent (foundation phase; a TODO
marks the future hard-fail).

These tests exercise the method in isolation via a tiny fake-self carrying only
`.df` (the method reads nothing else), so no heavy SSLDataset construction is
needed. quant-rigor-guard RULE 1 + RULE 8.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.data_loader import SSLDataset

_verify = SSLDataset._verify_lob_alignment   # unbound; call with a fake self


def _frame(n=20):
    ts = pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"ts_event": ts})


def _save_ts(path, ts_series):
    """Save like prepare_day_trading: int64 ns-since-epoch (UTC)."""
    ns = pd.to_datetime(ts_series, utc=True).to_numpy("datetime64[ns]").astype("int64")
    np.save(path, ns)
    return path


class TestAlignedPasses:
    def test_matching_timestamps_pass(self, tmp_path):
        df = _frame()
        p = _save_ts(tmp_path / "lob_ts.npy", df["ts_event"])
        _verify(types.SimpleNamespace(df=df), str(p))   # must not raise


class TestTeeth:
    def test_reordered_timestamps_raise(self, tmp_path):
        """Same set, wrong ORDER → must refuse with the clear message."""
        df = _frame()
        reversed_ts = df["ts_event"].iloc[::-1].reset_index(drop=True)
        p = _save_ts(tmp_path / "lob_ts.npy", reversed_ts)
        with pytest.raises(ValueError, match="MISALIGNMENT"):
            _verify(types.SimpleNamespace(df=df), str(p))

    def test_length_mismatch_raises(self, tmp_path):
        df = _frame(20)
        p = _save_ts(tmp_path / "lob_ts.npy", _frame(15)["ts_event"])
        with pytest.raises(ValueError, match="length"):
            _verify(types.SimpleNamespace(df=df), str(p))

    def test_message_explains_what_why_how(self, tmp_path):
        df = _frame()
        p = _save_ts(tmp_path / "lob_ts.npy", df["ts_event"].iloc[::-1].reset_index(drop=True))
        with pytest.raises(ValueError) as exc:
            _verify(types.SimpleNamespace(df=df), str(p))
        msg = str(exc.value)
        assert "WHY THIS MATTERS" in msg and "HOW TO FIX" in msg
        assert "prepare_day_trading.py" in msg   # tells the user how to rebuild


class TestBackwardCompatSafeFallback:
    def test_none_path_does_not_break(self, capsys):
        df = _frame()
        _verify(types.SimpleNamespace(df=df), None)        # no raise
        assert "LENGTH ONLY" in capsys.readouterr().out     # warned, didn't fail

    def test_missing_file_does_not_break(self, tmp_path, capsys):
        df = _frame()
        _verify(types.SimpleNamespace(df=df), str(tmp_path / "does_not_exist.npy"))
        assert "LENGTH ONLY" in capsys.readouterr().out

    def test_fallback_warns_about_future_hardfail(self, capsys):
        df = _frame()
        _verify(types.SimpleNamespace(df=df), None)
        assert "TODO" in capsys.readouterr().out            # the tightening reminder is visible
