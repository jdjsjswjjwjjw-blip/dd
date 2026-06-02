"""C2 fix (part أ) — depth↔candle↔label alignment guard for LOB tensors.

The LOB tensor is built in the builder's INTERNAL ts_event sort order and then
assigned POSITIONALLY onto df_final / saved beside the parquet; the SSL consumer
pairs tensor[i] ↔ parquet.iloc[i] by position. A silent reorder would mispair
every LOB window with a neighbouring bar's label — the worst leak in the
late-fusion merge. These tests lock:

  1. the alignment guard `_assert_lob_tensor_aligned` (passes aligned, RAISES on
     reorder / length mismatch — the teeth),
  2. the builder actually emits tensor_ts == bars['ts_event'] off-by-zero,
  3. the per-bar snapshot window is causal — a future bar's ticks never change
     an earlier bar's tensor row.

quant-rigor-guard RULE 1 (causality / no future leak in the merge) + RULE 8
(timestamp consistency).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P


# ── 1. the alignment guard (teeth) ──────────────────────────────────────────
class TestAlignmentGuard:
    def _bars(self, n=10):
        ts = pd.date_range("2025-04-07 08:00", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({"ts_event": ts})

    def test_passes_when_aligned(self):
        bars = self._bars()
        tns_ts = pd.to_datetime(bars["ts_event"]).to_numpy("datetime64[ns]")
        P._assert_lob_tensor_aligned(tns_ts, bars, where="t")  # must not raise

    def test_raises_on_reorder(self):
        """The decisive teeth: same timestamps, wrong ORDER → must raise."""
        bars = self._bars()
        tns_ts = pd.to_datetime(bars["ts_event"]).to_numpy("datetime64[ns]")
        shuffled = bars.iloc[::-1].reset_index(drop=True)   # reversed order, same set
        with pytest.raises(ValueError, match="C2 LOB alignment"):
            P._assert_lob_tensor_aligned(tns_ts, shuffled, where="t")

    def test_raises_on_length_mismatch(self):
        bars = self._bars(10)
        tns_ts = pd.to_datetime(self._bars(8)["ts_event"]).to_numpy("datetime64[ns]")
        with pytest.raises(ValueError, match="C2 LOB alignment"):
            P._assert_lob_tensor_aligned(tns_ts, bars, where="t")


# ── synthetic MBP/MBO/bars for the builder-level tests ──────────────────────
def _synth_lob_inputs(n_bars=12, freq="5min", levels=10, seed=0):
    rng = np.random.RandomState(seed)
    bar_ts = pd.date_range("2025-04-07 08:00", periods=n_bars, freq=freq, tz="UTC")
    bars = pd.DataFrame({"ts_event": bar_ts, "close": 100.0})
    mbp_rows, mbo_rows = [], []
    for bi, t0 in enumerate(bar_ts):
        for k in range(3):                       # 3 snapshots / bar
            ts = t0 + pd.Timedelta(seconds=20 * (k + 1))
            mid = 100.0 + bi * 0.1 + rng.randn() * 0.02
            row = {"ts_event": ts}
            for L in range(levels):
                row[f"bid_px_{L:02d}"] = mid - 0.01 * (L + 1)
                row[f"ask_px_{L:02d}"] = mid + 0.01 * (L + 1)
                row[f"bid_sz_{L:02d}"] = float(rng.randint(1, 50))
                row[f"ask_sz_{L:02d}"] = float(rng.randint(1, 50))
            mbp_rows.append(row)
            mbo_rows.append({
                "ts_event": ts, "action": "T",
                "side": "A" if rng.rand() < 0.5 else "B",
                "price": mid, "size": float(rng.randint(1, 10)),
            })
    return bars, pd.DataFrame(mbp_rows), pd.DataFrame(mbo_rows)


# ── 2. builder emits off-by-zero tensor_ts ──────────────────────────────────
class TestBuilderOffByZero:
    def test_from_mbp_tensor_ts_equals_bars(self):
        bars, mbp, mbo = _synth_lob_inputs(n_bars=12)
        tensors, tensor_ts, _ = P.build_rolling_lob_tensors_from_mbp(
            mbo, mbp, bars, freq="5min", lookback_bars=5, levels=10,
        )
        assert tensors.shape[0] == len(bars)
        bar_ts = pd.to_datetime(bars["ts_event"]).to_numpy("datetime64[ns]")
        tns_ts = np.asarray(tensor_ts, dtype="datetime64[ns]")
        assert np.array_equal(tns_ts, bar_ts), "tensor_ts must equal bars ts_event off-by-zero"
        # and the production guard agrees
        P._assert_lob_tensor_aligned(tensor_ts, bars, where="test")


# ── 3. causal window: a future bar's ticks never change an earlier row ──────
class TestCausalWindow:
    def test_future_tick_does_not_affect_earlier_bar(self):
        bars, mbp, mbo = _synth_lob_inputs(n_bars=12, seed=1)
        base, _, _ = P.build_rolling_lob_tensors_from_mbp(
            mbo, mbp, bars, freq="5min", lookback_bars=5, levels=10,
        )
        # perturb bar index 9's snapshots with extreme values
        t9 = bars["ts_event"].iloc[9]
        in_b9 = (mbp["ts_event"] >= t9) & (mbp["ts_event"] < t9 + pd.Timedelta("5min"))
        mbp_p = mbp.copy()
        for c in [c for c in mbp.columns if c.startswith(("bid_sz", "ask_sz"))]:
            mbp_p.loc[in_b9, c] = 1e6                    # absurd depth in bar 9
        in_b9_mbo = (mbo["ts_event"] >= t9) & (mbo["ts_event"] < t9 + pd.Timedelta("5min"))
        mbo_p = mbo.copy()
        mbo_p.loc[in_b9_mbo, "size"] = 1e6

        pert, _, _ = P.build_rolling_lob_tensors_from_mbp(
            mbo_p, mbp_p, bars, freq="5min", lookback_bars=5, levels=10,
        )
        # bar 4's window is bars [0..4] → must be byte-identical despite bar-9 perturbation
        assert np.allclose(base[4], pert[4], equal_nan=True), (
            "future bar-9 ticks leaked into bar-4 tensor row — non-causal window"
        )
        # sanity: bar 9 itself DID change (perturbation is real)
        assert not np.allclose(base[9], pert[9], equal_nan=True), "perturbation had no effect?"
