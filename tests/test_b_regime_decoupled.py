"""B — regime مفصول عن MBP (post-D1 architectural correctness).

The pre-B `assign_regime_label` gated `low_liquidity` on `mbp_bar_coverage<0.30`,
which (1) re-coupled day_trade labels to the depth path D1 had decoupled,
(2) measured MBP-snapshot density not market liquidity, (3) broke
reproducibility — labels differed depending on whether MBP was provided.

B replaces it with a CAUSAL RELATIVE measure: tick activity (tick_count,
fallback volume) below a TRAILING rolling median (same window pattern as
atr_med/cvd_ref in the same function). Tests lock the contract:
  • TEETH (the bug we're fixing): with mbp_bar_coverage=0 and varied
    tick_count, the OLD logic produced 100% low_liquidity; the NEW one
    DISTRIBUTES across regimes.
  • REPRODUCIBILITY (R10): same tick/atr — regime is byte-identical
    whether MBP coverage is 0 or 1. The label no longer depends on MBP.
  • SENSITIVITY: low_liquidity rises when tick activity drops.
  • CAUSALITY (R1): a future tick-count spike does NOT change a past bar's
    label (truncate-invariance — the rolling median sees only the past).
  • SCHEMA: still produces the 4 known labels + the documented int cluster
    mapping; downstream REGIME_TP_SL / regime_label_grouped consumers unaffected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import prepare_day_trading as P


def _frame(n=200, tick_counts=None, seed=0):
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2025-06-01 08:00", periods=n, freq="5min", tz="UTC")
    tc = (np.full(n, 50.0) if tick_counts is None else np.asarray(tick_counts, float))
    return pd.DataFrame({
        "ts_event": ts,
        "atr_14":           rng.uniform(0.8, 1.2, n),
        "hawkes_intensity": rng.uniform(0.0, 1.0, n),
        "bar_cvd_delta":    rng.uniform(-1, 1, n),
        "trend_strength":   rng.uniform(0.0, 0.4, n),
        "tick_count":       tc,
        "volume":           tc,  # proxy when tick_count absent
        "mbp_bar_coverage": np.zeros(n, dtype=np.float32),  # the prodtest scenario
    })


class TestTeethTheBugWeFixed:
    def test_no_mbp_no_longer_collapses_to_low_liquidity(self):
        """The exact prodtest scenario: --no_lob, no MBP → mbp_bar_coverage=0.
        Pre-B: 100% low_liquidity. Post-B: distributes across regimes."""
        df = _frame()
        out = P.assign_regime_label(df)
        labels = out["regime_label"].astype(str)
        # After the warm-up window (first ~min_periods rows) regime should NOT
        # be saturated by low_liquidity. The pre-B bug produced 100%.
        post_warmup = labels.iloc[20:]
        low_liq_rate = float((post_warmup == "low_liquidity").mean())
        assert low_liq_rate < 0.90, (
            f"regime collapsed to low_liquidity at {low_liq_rate:.0%} — "
            f"the bug B is supposed to fix is still present"
        )
        # And at least one non-low_liquidity label must appear.
        assert (post_warmup != "low_liquidity").any()


class TestReproducibilityR10:
    def test_regime_identical_with_and_without_mbp(self):
        """The decisive R10 test: regime must be byte-identical whether MBP
        coverage is 0 (no MBP passed) or 1 (MBP passed). Pre-B these diverged
        because the label depended on mbp_bar_coverage."""
        df_no_mbp = _frame(seed=7)
        df_with_mbp = df_no_mbp.copy()
        df_with_mbp["mbp_bar_coverage"] = np.float32(1.0)  # "MBP present"
        out_a = P.assign_regime_label(df_no_mbp)
        out_b = P.assign_regime_label(df_with_mbp)
        assert (out_a["regime_label"].to_numpy() == out_b["regime_label"].to_numpy()).all()
        assert (out_a["regime_cluster"].to_numpy() == out_b["regime_cluster"].to_numpy()).all()


class TestSensitivity:
    def test_low_liquidity_rises_when_tick_activity_drops(self):
        n = 200
        tc_normal = np.full(n, 100.0)
        tc_drops = tc_normal.copy()
        tc_drops[120:160] = 5.0                          # a quiet stretch (relative to past)
        df_n = _frame(n=n, tick_counts=tc_normal, seed=1)
        df_d = _frame(n=n, tick_counts=tc_drops,  seed=1)
        rate_n = float((P.assign_regime_label(df_n)["regime_label"] == "low_liquidity").mean())
        rate_d = float((P.assign_regime_label(df_d)["regime_label"] == "low_liquidity").mean())
        assert rate_d > rate_n, (
            f"low_liquidity should rise on a quiet stretch: normal={rate_n:.0%} drops={rate_d:.0%}"
        )

    def test_volume_fallback_when_tick_count_absent(self):
        df = _frame()
        df = df.drop(columns=["tick_count"])             # only volume available
        out = P.assign_regime_label(df)
        # still produces labels (no crash) and is not 100% low_liquidity
        labels = out["regime_label"].astype(str).iloc[20:]
        assert (labels != "low_liquidity").any()


class TestCausalityR1:
    def test_future_spike_does_not_change_past_label(self):
        """Truncate-invariance: classifying bar i must not depend on bars > i.
        A future activity spike inserted AFTER bar 100 must not flip bar 100's
        label — the rolling median window is strictly past."""
        n = 200
        tc = np.full(n, 50.0)
        # version A: ends as-is
        df_a = _frame(n=n, tick_counts=tc.copy(), seed=2)
        # version B: a huge future spike at bar 150
        tc_b = tc.copy(); tc_b[150:] = 5000.0
        df_b = _frame(n=n, tick_counts=tc_b, seed=2)
        out_a = P.assign_regime_label(df_a)
        out_b = P.assign_regime_label(df_b)
        # bars [0..100] must be byte-identical
        assert (out_a["regime_label"].iloc[:100].to_numpy()
                == out_b["regime_label"].iloc[:100].to_numpy()).all(), (
            "regime at bar i depends on bars > i — causality violated"
        )


class TestSchemaUnchanged:
    def test_labels_and_cluster_mapping_intact(self):
        df = _frame(seed=3)
        out = P.assign_regime_label(df)
        assert set(out["regime_label"].astype(str).unique()) <= {
            "trending", "ranging", "volatile", "low_liquidity"
        }
        # the cluster ints downstream code (REGIME_TP_SL keys, regime_cluster
        # consumers) reads must stay stable.
        m = {"trending": 0, "ranging": 1, "volatile": 2, "low_liquidity": 3}
        sub = out[["regime_label", "regime_cluster"]].drop_duplicates()
        for _, r in sub.iterrows():
            assert int(r["regime_cluster"]) == m[str(r["regime_label"])]
