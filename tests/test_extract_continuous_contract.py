"""Tests for tools/extract_continuous_contract.py — smart contract-rollover
extractor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.extract_continuous_contract import (
    CONTRACT_RE,
    _contract_expiry_key,
    _parse_contract,
    compute_offset_for_rollover,
    detect_rollover_day,
    extract_continuous,
)


# ════════════════════════════════════════════════════════════════════
# Contract parsing
# ════════════════════════════════════════════════════════════════════
class TestParseContract:
    def test_6bh5(self):
        result = _parse_contract("6BH5")
        assert result == ("6B", "H", 2025)

    def test_6bm25(self):
        result = _parse_contract("6BM25")
        assert result == ("6B", "M", 2025)

    def test_esz4(self):
        result = _parse_contract("ESZ4")
        assert result == ("ES", "Z", 2024)

    def test_unparseable_returns_none(self):
        assert _parse_contract("NOTACONTRACT") is None
        assert _parse_contract("") is None
        assert _parse_contract(None) is None

    def test_lowercase_uppercased(self):
        # Should still parse if input is lowercase
        result = _parse_contract("6bh5")
        assert result == ("6B", "H", 2025)


class TestContractExpiryKey:
    def test_h_before_m_same_year(self):
        # 6BH5 (March 2025) before 6BM5 (June 2025)
        assert _contract_expiry_key("6BH5") < _contract_expiry_key("6BM5")

    def test_z_year_before_h_next(self):
        # 6BZ4 (Dec 2024) before 6BH5 (Mar 2025)
        assert _contract_expiry_key("6BZ4") < _contract_expiry_key("6BH5")


# ════════════════════════════════════════════════════════════════════
# Rollover detection
# ════════════════════════════════════════════════════════════════════
class TestDetectRolloverDay:
    def _make_df(self, near="6BH5", far="6BM5"):
        """Synthetic month: days 1-10 near dominates, days 11-20 far dominates."""
        rows = []
        for d in range(1, 21):
            day_str = f"2025-03-{d:02d}"
            # Near volume drops over the month
            near_vol = max(1, 100 - d * 5)
            # Far volume rises
            far_vol = d * 5
            for _ in range(near_vol):
                rows.append({"ts_event": day_str, "symbol": near, "size": 1, "price": 1.30 + d * 0.0001})
            for _ in range(far_vol):
                rows.append({"ts_event": day_str, "symbol": far, "size": 1, "price": 1.32 + d * 0.0001})
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        return df

    def test_detects_crossover(self):
        df = self._make_df()
        # Day where far_vol > near_vol: when d*5 > 100 - d*5 → d > 10
        rollover = detect_rollover_day(df, "ts_event", "symbol", "6BH5", "6BM5")
        # First day where far > near should be d=11
        assert rollover is not None
        # Compare as date strings to avoid TZ subtleties
        assert str(rollover.date()) == "2025-03-11"

    def test_no_crossover_returns_none(self):
        # Make a df where near always dominates
        rows = []
        for d in range(1, 11):
            for _ in range(100):
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BH5",
                             "size": 1, "price": 1.30})
            for _ in range(5):
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BM5",
                             "size": 1, "price": 1.32})
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        result = detect_rollover_day(df, "ts_event", "symbol", "6BH5", "6BM5")
        assert result is None

    def test_missing_contract_returns_none(self):
        df = pd.DataFrame({
            "ts_event": pd.to_datetime(["2025-03-01"], utc=True),
            "symbol": ["6BH5"],
            "size": [10],
        })
        assert detect_rollover_day(df, "ts_event", "symbol", "6BH5", "6BM5") is None


# ════════════════════════════════════════════════════════════════════
# Offset computation
# ════════════════════════════════════════════════════════════════════
class TestComputeOffset:
    def test_offset_uses_pre_rollover_window(self):
        rng = np.random.RandomState(0)
        rows = []
        # 5 days before rollover; near at 1.32, far at 1.30
        for d in range(7, 12):
            for _ in range(50):
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BH5",
                             "size": 1, "price": 1.32 + rng.randn() * 0.0001})
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BM5",
                             "size": 1, "price": 1.30 + rng.randn() * 0.0001})
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        rollover = pd.Timestamp("2025-03-12", tz="UTC")
        offset = compute_offset_for_rollover(
            df, "ts_event", "symbol", "6BH5", "6BM5", rollover, window_days=3,
        )
        # Expect ~+0.02 (near is 0.02 above far)
        assert abs(offset - 0.02) < 0.005


# ════════════════════════════════════════════════════════════════════
# Full extraction
# ════════════════════════════════════════════════════════════════════
class TestExtractContinuous:
    def _synthetic_month(self):
        """6BH5 dominates days 1-10, 6BM5 dominates days 11-20.
        6BH5 trades at ~1.32, 6BM5 trades at ~1.30 (carry of ~+0.02)."""
        rng = np.random.RandomState(42)
        rows = []
        for d in range(1, 21):
            day_str = f"2025-03-{d:02d}"
            near_vol = max(1, 100 - d * 5)
            far_vol = d * 5
            for _ in range(near_vol):
                rows.append({
                    "ts_event": day_str, "symbol": "6BH5",
                    "size": 1, "price": 1.32 + rng.randn() * 0.0001,
                })
            for _ in range(far_vol):
                rows.append({
                    "ts_event": day_str, "symbol": "6BM5",
                    "size": 1, "price": 1.30 + rng.randn() * 0.0001,
                })
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        return df

    def test_output_continuous_no_jump(self):
        df = self._synthetic_month()
        out, log = extract_continuous(df, root="6B", adjust_prices=True)
        # Output should have rows from both contracts
        assert log["mode"] == "continuous_with_rollover"
        assert log["near"] == "6BH5"
        assert log["far"] == "6BM5"
        assert log["near_rows"] > 0
        assert log["far_rows"] > 0
        # After adjustment, the near (pre-rollover) prices should ~match
        # far (post-rollover) prices — both around 1.30
        ts_rollover = pd.Timestamp(log["rollover_day"])
        pre = out[out["ts_event"] < ts_rollover]
        post = out[out["ts_event"] >= ts_rollover]
        pre_mean = pre["price"].mean()
        post_mean = post["price"].mean()
        assert abs(pre_mean - post_mean) < 0.01, \
            f"pre-rollover mean {pre_mean:.4f} != post {post_mean:.4f}"

    def test_no_adjustment_preserves_gap(self):
        df = self._synthetic_month()
        out, log = extract_continuous(df, root="6B", adjust_prices=False)
        assert log["applied_offset"] == 0.0
        ts_rollover = pd.Timestamp(log["rollover_day"])
        pre = out[out["ts_event"] < ts_rollover]
        post = out[out["ts_event"] >= ts_rollover]
        # WITHOUT adjustment, the gap should be visible (~0.02)
        gap = pre["price"].mean() - post["price"].mean()
        assert abs(gap - 0.02) < 0.005

    def test_single_contract_passthrough(self):
        """If only one contract is in the file, output is just that contract."""
        rng = np.random.RandomState(0)
        rows = [{
            "ts_event": f"2025-03-{d:02d}", "symbol": "6BH5",
            "size": 1, "price": 1.32 + rng.randn() * 1e-4,
        } for d in range(1, 11) for _ in range(20)]
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        out, log = extract_continuous(df, root="6B")
        assert log["mode"] == "single_contract"
        assert log["chosen"] == "6BH5"
        assert (out["symbol"] == "6BH5").all()

    def test_manual_rollover_day(self):
        df = self._synthetic_month()
        manual_day = pd.Timestamp("2025-03-08", tz="UTC")
        out, log = extract_continuous(df, root="6B", rollover_day=manual_day)
        assert str(pd.Timestamp(log["rollover_day"]).date()) == "2025-03-08"

    def test_no_crossover_picks_dominant(self):
        # Make near dominate the whole month
        rows = []
        for d in range(1, 11):
            for _ in range(100):
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BH5",
                             "size": 1, "price": 1.30})
            for _ in range(5):
                rows.append({"ts_event": f"2025-03-{d:02d}", "symbol": "6BM5",
                             "size": 1, "price": 1.32})
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        out, log = extract_continuous(df, root="6B")
        assert log["mode"] == "no_rollover_picked_dominant"
        assert log["chosen"] == "6BH5"


# ════════════════════════════════════════════════════════════════════
# CLI smoke (writes parquet then reads it back)
# ════════════════════════════════════════════════════════════════════
class TestCLISmoke:
    def test_full_roundtrip(self, tmp_path):
        rng = np.random.RandomState(0)
        rows = []
        for d in range(1, 21):
            ds = f"2025-03-{d:02d}"
            near_vol = max(1, 100 - d * 5)
            far_vol = d * 5
            for _ in range(near_vol):
                rows.append({"ts_event": ds, "symbol": "6BH5",
                             "size": 1, "price": 1.32 + rng.randn() * 1e-4})
            for _ in range(far_vol):
                rows.append({"ts_event": ds, "symbol": "6BM5",
                             "size": 1, "price": 1.30 + rng.randn() * 1e-4})
        df = pd.DataFrame(rows)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

        in_path = tmp_path / "input.parquet"
        out_path = tmp_path / "output.parquet"
        log_path = out_path.with_suffix(out_path.suffix + ".log.json")
        df.to_parquet(in_path, index=False)

        # Invoke via subprocess
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "extract_continuous_contract.py"),
             str(in_path), "--root", "6B", "--output", str(out_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert out_path.exists()
        assert log_path.exists()
        out_df = pd.read_parquet(out_path)
        assert len(out_df) > 0
        with open(log_path) as f:
            log = json.load(f)
        assert log["mode"] == "continuous_with_rollover"
