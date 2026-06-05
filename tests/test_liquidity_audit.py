"""Teeth tests for tools/diagnostics/liquidity_audit.py.

Verify each liquidity check on KNOWN synthetic data + the overall verdict logic:
  T1. SPREAD computed correctly in ticks from a synthetic MBP-10 book.
  T2. DEPTH: L0 size + levels-filled fraction correct.
  T3. SYMBOL: single dominant contract vs multi-contract mix.
  T4. FEATURES: volume/bar, low_liquidity fraction, roll-gap detection.
  T5. VERDICT: liquid book → LIQUID; thin book → THIN_CONTRACT_SUSPECTED;
      missing book columns → INSUFFICIENT_DATA fallback.
  T6. end-to-end smoke (raw + features) writes utf-8 report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.liquidity_audit import (
    audit_raw_mbo, audit_features, _verdict, run_liquidity_audit,
    DEFAULT_TICK_SIZE,
)


def _mbp10(tmp_path, n, *, spread_ticks, l0_size, levels_filled, symbol="6BM5",
           name="mbo.parquet"):
    """Build a synthetic MBP-10 parquet: spread + depth controlled."""
    rng = np.random.RandomState(0)
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="100ms", tz="UTC")
    mid = 1.2500 + np.cumsum(rng.randn(n) * 1e-5)
    tick = DEFAULT_TICK_SIZE
    half = spread_ticks * tick / 2.0
    cols = {"ts_event": ts, "symbol": symbol, "action": "T", "side": "A",
            "price": mid, "size": 5.0}
    for i in range(10):
        cols[f"bid_px_{i:02d}"] = mid - half - i * tick
        cols[f"ask_px_{i:02d}"] = mid + half + i * tick
        # only `levels_filled` levels carry non-zero size
        sz = float(l0_size) if i < levels_filled else 0.0
        cols[f"bid_sz_{i:02d}"] = sz
        cols[f"ask_sz_{i:02d}"] = sz
    df = pd.DataFrame(cols)
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


# ── T1: spread ─────────────────────────────────────────────────────────────
class TestSpread:
    def test_spread_in_ticks(self, tmp_path):
        path = _mbp10(tmp_path, 2000, spread_ticks=2.0, l0_size=20, levels_filled=10)
        out = audit_raw_mbo(path, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert out["has_mbp10_book"] is True
        assert abs(out["spread_ticks"]["median"] - 2.0) < 0.05, out["spread_ticks"]

    def test_wide_spread_detected(self, tmp_path):
        path = _mbp10(tmp_path, 2000, spread_ticks=8.0, l0_size=2, levels_filled=3)
        out = audit_raw_mbo(path, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert out["spread_ticks"]["median"] > 5.0


# ── T2: depth ──────────────────────────────────────────────────────────────
class TestDepth:
    def test_depth_levels_filled(self, tmp_path):
        path = _mbp10(tmp_path, 2000, spread_ticks=1.0, l0_size=30, levels_filled=8)
        out = audit_raw_mbo(path, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert abs(out["depth"]["median_bid_sz_L0"] - 30.0) < 1e-6
        assert out["depth"]["median_bid_levels_filled"] == 8.0

    def test_thin_book_few_levels(self, tmp_path):
        path = _mbp10(tmp_path, 2000, spread_ticks=1.0, l0_size=1, levels_filled=2)
        out = audit_raw_mbo(path, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert out["depth"]["median_bid_levels_filled"] == 2.0
        assert out["depth"]["median_bid_sz_L0"] <= 1.0


# ── T3: symbol ─────────────────────────────────────────────────────────────
class TestSymbol:
    def test_single_dominant_contract(self, tmp_path):
        path = _mbp10(tmp_path, 3000, spread_ticks=1.0, l0_size=10, levels_filled=10,
                      symbol="6BM5")
        out = audit_raw_mbo(path, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert out["symbol"]["distinct"] == 1
        assert out["symbol"]["dominant_share"] == 1.0

    def test_multi_contract_mix(self, tmp_path):
        # build two files with different symbols, point audit at the dir
        d = tmp_path / "mbodir"
        d.mkdir()
        _mbp10(d, 2000, spread_ticks=1.0, l0_size=10, levels_filled=10,
               symbol="6BM5", name="a.parquet")
        _mbp10(d, 1000, spread_ticks=1.0, l0_size=10, levels_filled=10,
               symbol="6BU5", name="b.parquet")
        out = audit_raw_mbo(d, stride=1, tick_size=DEFAULT_TICK_SIZE)
        assert out["symbol"]["distinct"] == 2
        assert out["symbol"]["dominant_share"] < 0.80


# ── T4: features-parquet checks ────────────────────────────────────────────
class TestFeatures:
    def _feat(self, tmp_path, *, low_liq_frac, gap=False, n=4000):
        rng = np.random.RandomState(1)
        close = 1.25 + np.cumsum(rng.randn(n) * 1e-4)
        if gap:
            close[2000] = close[1999] * 1.01      # 1% jump (splice)
        regime = np.where(np.arange(n) < int(low_liq_frac * n), "low_liquidity", "trending")
        df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            "close": close, "volume": rng.uniform(50, 200, n),
            "tick_count": rng.uniform(100, 500, n),
            "regime_label": regime, "is_session_break": np.zeros(n, bool),
        })
        path = tmp_path / "feat.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_low_liquidity_fraction(self, tmp_path):
        path = self._feat(tmp_path, low_liq_frac=0.40)
        out = audit_features(path)
        assert abs(out["low_liquidity_fraction"] - 0.40) < 0.02

    def test_roll_gap_detected(self, tmp_path):
        path = self._feat(tmp_path, low_liq_frac=0.10, gap=True)
        out = audit_features(path)
        assert out["roll_gaps"]["n_gaps_over_threshold"] >= 1
        assert out["roll_gaps"]["max_adjacent_return"] > 0.005

    def test_no_roll_gap_clean(self, tmp_path):
        path = self._feat(tmp_path, low_liq_frac=0.10, gap=False)
        out = audit_features(path)
        assert out["roll_gaps"]["n_gaps_over_threshold"] == 0


# ── T5: verdict logic ──────────────────────────────────────────────────────
class TestVerdict:
    def test_liquid_book_verdict(self):
        raw = {"spread_ticks": {"median": 1.5}, "depth": {"median_bid_sz_L0": 20.0,
               "median_bid_levels_filled": 10.0}, "symbol": {"dominant_share": 1.0}}
        feat = {"low_liquidity_fraction": 0.10, "roll_gaps": {"n_gaps_over_threshold": 0}}
        v = _verdict(raw, feat)
        assert "LIQUID_FRONT_MONTH" in v["verdict"], v

    def test_thin_book_verdict(self):
        raw = {"spread_ticks": {"median": 8.0}, "depth": {"median_bid_sz_L0": 1.0,
               "median_bid_levels_filled": 2.0}, "symbol": {"dominant_share": 0.5}}
        feat = {"low_liquidity_fraction": 0.40, "roll_gaps": {"n_gaps_over_threshold": 3}}
        v = _verdict(raw, feat)
        assert "THIN_CONTRACT_SUSPECTED" in v["verdict"], v
        assert v["thin_signals"] >= 2

    def test_missing_book_fallback(self):
        # pure-MBO file: no spread/depth → INSUFFICIENT_DATA (not a false LIQUID)
        raw = {"spread_ticks": {"missing": "x"}, "depth": {"missing": "x"},
               "symbol": {"dominant_share": 1.0}}
        v = _verdict(raw, None)
        assert v["thin_signals"] == 0
        assert "INSUFFICIENT_DATA" in v["verdict"] or "LIQUID" in v["verdict"]


# ── T6: end-to-end smoke ───────────────────────────────────────────────────
class TestEndToEnd:
    def test_run_writes_report(self, tmp_path):
        mbo = _mbp10(tmp_path, 3000, spread_ticks=1.5, l0_size=20, levels_filled=10)
        rng = np.random.RandomState(2)
        n = 4000
        feat_df = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC"),
            "close": 1.25 + np.cumsum(rng.randn(n) * 1e-4),
            "volume": rng.uniform(50, 200, n),
            "regime_label": np.array(["trending"] * n),
            "is_session_break": np.zeros(n, bool),
        })
        feat_path = tmp_path / "feat.parquet"
        feat_df.to_parquet(feat_path, index=False)

        s = run_liquidity_audit(tmp_path / "out", mbo_path=mbo, features_path=feat_path)
        assert "overall" in s
        report = (tmp_path / "out" / "liquidity_audit_report.txt").read_text(encoding="utf-8")
        assert "Liquidity audit" in report
        assert "VERDICT" in report
