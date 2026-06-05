"""(B) Depth-features export — separate parquet, D1-respecting, bar-aligned.

The mbp_* book features (queue imbalance, walls, depth, spread, slopes) are
computed by enrich_bars_with_intrabar_mbp but dropped from the day_trade parquet
by finalize (D1 keeps day_trade depth-free). --export-depth-features writes them
to a SEPARATE depth_features<tag>.parquet, aligned to the main parquet by
ts_event.

These tests run the FULL refinery on tiny synthetic MBO + MBP-10 and assert:
  T1. with --export-depth-features + MBP → depth_features parquet exists, has
      mbp_* columns, one row per main-parquet bar, aligned 1:1 by ts_event.
  T2. without the flag → no depth parquet (default off, day_trade unchanged).
  T3. the main day_trade parquet does NOT contain mbp_* depth columns either way
      (D1 separation preserved).
  T4. depth columns carry real (non-constant) values on a book with structure.
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


def _synth_mbo(n_days=2, base="2025-04-07 08:00", seed=0):
    """Trade-action MBO ticks across n_days, ~1 tick / 2s."""
    rng = np.random.RandomState(seed)
    rows = []
    t0 = pd.Timestamp(base, tz="UTC")
    per_day = 6 * 60 * 30          # 6h × 30 ticks/min ≈ dense enough for 5min bars
    price = 1.2500
    for d in range(n_days):
        day0 = t0 + pd.Timedelta(days=d)
        for k in range(per_day):
            price += rng.randn() * 1e-4
            ts = day0 + pd.Timedelta(seconds=k * 2)
            side = "A" if rng.rand() < 0.5 else "B"
            rows.append({"ts_event": ts, "action": "T", "side": side,
                         "price": round(price, 5), "size": float(rng.randint(1, 10))})
    return pd.DataFrame(rows)


def _synth_mbp(mbo, levels=10, seed=1):
    """MBP-10 snapshots aligned to the MBO timestamps, with STRUCTURED depth
    (varying imbalance + a wall) so the mbp_* features are non-constant."""
    rng = np.random.RandomState(seed)
    n = len(mbo)
    out = {"ts_event": mbo["ts_event"].to_numpy(), "symbol": "6BM5"}
    mid = pd.to_numeric(mbo["price"]).to_numpy()
    for i in range(levels):
        out[f"bid_px_{i:02d}"] = mid - 0.0001 * (i + 1)
        out[f"ask_px_{i:02d}"] = mid + 0.0001 * (i + 1)
        # structured sizes: bid heavier on some ticks (imbalance), L0 wall
        base = rng.randint(1, 40, n).astype(float)
        wall = np.where(np.arange(n) % 7 == 0, 200.0, 0.0) if i == 0 else 0.0
        out[f"bid_sz_{i:02d}"] = base + wall
        out[f"ask_sz_{i:02d}"] = rng.randint(1, 40, n).astype(float)
    return pd.DataFrame(out)


def _run(tmp_path, mbo, mbp, *, export_depth, tag):
    mbo_path = tmp_path / f"mbo_{tag}.parquet"
    mbp_path = tmp_path / f"mbp_{tag}.parquet"
    mbo.to_parquet(mbo_path, index=False)
    if mbp is not None:
        mbp.to_parquet(mbp_path, index=False)
    out_dir = tmp_path / f"out_{tag}"
    P.run_day_trading_refinery(
        mbo_dir=str(mbo_path),
        mbp_path=str(mbp_path) if mbp is not None else None,
        output_dir=str(out_dir),
        freq="5min", horizon_bars=6, build_lob_tensors=False,
        event_gate_mode="structure_compass", artifact_tag=tag,
        n_workers=1, add_cycle_features_flag=False, add_seasonal_features_flag=False,
        export_depth_features=export_depth,
    )
    return out_dir


class TestDepthExport:
    def test_depth_parquet_written_and_aligned(self, tmp_path):
        mbo = _synth_mbo(seed=2)
        mbp = _synth_mbp(mbo, seed=3)
        out_dir = _run(tmp_path, mbo, mbp, export_depth=True, tag="dx")

        main = out_dir / "day_trading_features_dx.parquet"
        depth = out_dir / "depth_features_dx.parquet"
        assert main.exists(), "main parquet missing"
        assert depth.exists(), "depth parquet not written despite --export-depth-features + MBP"

        df_main = pd.read_parquet(main)
        df_depth = pd.read_parquet(depth)
        # one depth row per main bar, aligned by ts_event
        assert len(df_depth) == len(df_main), (len(df_depth), len(df_main))
        m_ts = pd.to_datetime(df_main["ts_event"], utc=True).to_numpy()
        d_ts = pd.to_datetime(df_depth["ts_event"], utc=True).to_numpy()
        np.testing.assert_array_equal(m_ts, d_ts)
        # has the mbp_* depth columns
        mbp_cols = [c for c in df_depth.columns if c.startswith("mbp_")]
        assert any("imbalance" in c for c in mbp_cols), mbp_cols
        assert any("wall" in c for c in mbp_cols), mbp_cols
        assert any("depth" in c for c in mbp_cols), mbp_cols

    def test_no_flag_no_depth_parquet(self, tmp_path):
        mbo = _synth_mbo(seed=4)
        mbp = _synth_mbp(mbo, seed=5)
        out_dir = _run(tmp_path, mbo, mbp, export_depth=False, tag="nodx")
        assert (out_dir / "day_trading_features_nodx.parquet").exists()
        assert not (out_dir / "depth_features_nodx.parquet").exists(), \
            "depth parquet written without the flag"

    def test_main_parquet_has_no_mbp_depth_cols(self, tmp_path):
        """D1 separation preserved: the day_trade parquet never carries the
        mbp_* book features, with or without the depth-export flag."""
        mbo = _synth_mbo(seed=6)
        mbp = _synth_mbp(mbo, seed=7)
        out_dir = _run(tmp_path, mbo, mbp, export_depth=True, tag="sep")
        df_main = pd.read_parquet(out_dir / "day_trading_features_sep.parquet")
        depth_like = [c for c in df_main.columns
                      if c.startswith("mbp_") and c not in ("mbp_bar_coverage", "mbp_roll_lob_coverage")]
        assert depth_like == [], f"depth cols leaked into day_trade parquet: {depth_like}"

    def test_depth_values_are_non_constant(self, tmp_path):
        """A structured book must yield varying queue-imbalance / wall features —
        not all zeros (which would mean the capture grabbed empty columns)."""
        mbo = _synth_mbo(seed=8)
        mbp = _synth_mbp(mbo, seed=9)
        out_dir = _run(tmp_path, mbo, mbp, export_depth=True, tag="val")
        df_depth = pd.read_parquet(out_dir / "depth_features_val.parquet")
        imb = [c for c in df_depth.columns if "imbalance" in c]
        assert imb, "no imbalance column"
        # at least one imbalance feature must vary across bars
        assert any(pd.to_numeric(df_depth[c], errors="coerce").std() > 0 for c in imb), \
            "all imbalance features are constant — capture or computation broke"
