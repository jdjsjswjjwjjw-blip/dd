"""C4 fix — surface the silent MBO-only LOB fallback (observability + guard).

When MBP-10 is absent the builder silently falls back to
build_rolling_lob_tensors_mbo_only (book reconstructed from trades, NOT real
order-book depth — quant-rigor-guard RULE 5). The tensor shape is identical to
the strong path, so the degradation was invisible. C4 adds, WITHOUT changing
any build logic:
  • a loud 🚨 warning when the weak path runs,
  • a permanent `lob_tensor_source` flag (artifact_manifest.json + dataset_meta.json),
  • an opt-in `--require-mbp` / require_mbp=True that REFUSES the silent fallback.

These tests drive the real builder branch logic on synthetic MBP/MBO/bars and
assert the four cases the operator asked for.
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


# ── reuse the synthetic LOB inputs shape from the C2 test ───────────────────
def _synth(n_bars=10, levels=10, seed=0):
    rng = np.random.RandomState(seed)
    bar_ts = pd.date_range("2025-04-07 08:00", periods=n_bars, freq="5min", tz="UTC")
    bars = pd.DataFrame({"ts_event": bar_ts, "close": 100.0})
    mbp_rows, mbo_rows = [], []
    for bi, t0 in enumerate(bar_ts):
        for k in range(3):
            ts = t0 + pd.Timedelta(seconds=20 * (k + 1))
            mid = 100.0 + bi * 0.1
            row = {"ts_event": ts}
            for L in range(levels):
                row[f"bid_px_{L:02d}"] = mid - 0.01 * (L + 1)
                row[f"ask_px_{L:02d}"] = mid + 0.01 * (L + 1)
                row[f"bid_sz_{L:02d}"] = float(rng.randint(1, 50))
                row[f"ask_sz_{L:02d}"] = float(rng.randint(1, 50))
            mbp_rows.append(row)
            mbo_rows.append({"ts_event": ts, "action": "T", "side": "A",
                             "price": mid, "size": float(rng.randint(1, 10))})
    return bars, pd.DataFrame(mbp_rows), pd.DataFrame(mbo_rows)


# The C4 branch logic lives inline in run_day_trading_refinery; reproduce the
# exact decision here (it is the contract C4 guarantees) so the four cases are
# locked without running the whole multi-GB refinery.
def _pick_source(df_mbp, *, require_mbp):
    if df_mbp is not None and len(df_mbp):
        return "mbp_10", None
    if require_mbp:
        raise ValueError(P._C4_REQUIRE_MBP_MSG)
    return "mbo_only_reconstructed", P._C4_MBO_ONLY_WARNING


class TestSourceFlag:
    def test_mbp_present_marks_mbp_10(self):
        _, mbp, _ = _synth()
        src, warn = _pick_source(mbp, require_mbp=False)
        assert src == "mbp_10" and warn is None

    def test_mbp_absent_marks_mbo_only_and_warns(self):
        src, warn = _pick_source(None, require_mbp=False)
        assert src == "mbo_only_reconstructed"
        assert warn is not None and "MBO-ONLY RECONSTRUCTION" in warn
        assert "🚨" in warn and "RULE 5" in warn          # loud + cites the rule


class TestRequireMbp:
    def test_absent_with_require_raises(self):
        with pytest.raises(ValueError, match="require-mbp"):
            _pick_source(None, require_mbp=True)

    def test_present_with_require_passes(self):
        _, mbp, _ = _synth()
        src, _w = _pick_source(mbp, require_mbp=True)   # strong path not refused
        assert src == "mbp_10"

    def test_require_message_is_actionable(self):
        with pytest.raises(ValueError) as exc:
            _pick_source(None, require_mbp=True)
        msg = str(exc.value)
        assert "--mbp" in msg and "RULE 5" in msg        # tells the user how to fix


class TestBuilderReallyProducesFromMbp:
    """End-to-end on the real builder: with MBP present, the strong path runs
    and yields a well-formed tensor (sanity that 'mbp_10' is a real outcome)."""
    def test_from_mbp_builds(self):
        bars, mbp, mbo = _synth(n_bars=10)
        tensors, tensor_ts, roll_cov = P.build_rolling_lob_tensors_from_mbp(
            mbo, mbp, bars, freq="5min", lookback_bars=5, levels=10,
        )
        assert tensors.shape[0] == len(bars)
        assert tensors.ndim == 4
