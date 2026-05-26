"""
tests/test_sprint9_audit_ops.py — Sprint 9 audit + operational tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_df_with_dead_cols(n: int = 200, n_dead: int = 5, n_weak: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    data = {
        f"good_{i}": rng.randn(n) for i in range(10)
    }
    for i in range(n_dead):
        data[f"dead_{i}"] = np.zeros(n)
    for i in range(n_weak):
        data[f"weak_{i}"] = rng.randn(n) * 1e-5
    return pd.DataFrame(data)


class TestDeadFeaturesAudit(unittest.TestCase):
    def test_audit_detects_dead_cols(self):
        from tools.dead_features_audit import audit_features

        df = _make_df_with_dead_cols(n=200, n_dead=5, n_weak=3)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "features.parquet"
            df.to_parquet(path)
            result = audit_features(path)

        self.assertEqual(result["dead"]["count"], 5)
        self.assertEqual(result["weak"]["count"], 3)
        for c in result["dead"]["list"]:
            self.assertTrue(c.startswith("dead_"))

    def test_audit_empty_dataframe(self):
        from tools.dead_features_audit import audit_features
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.parquet"
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_parquet(path)
            result = audit_features(path)
        self.assertEqual(result["dead"]["count"], 0)
        self.assertEqual(result["total_columns"], 1)

    def test_audit_cli(self):
        df = _make_df_with_dead_cols(n=100, n_dead=2, n_weak=1)
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "in.parquet"
            out_path = Path(td) / "out.json"
            df.to_parquet(in_path)

            res = subprocess.run(
                [sys.executable, "tools/dead_features_audit.py",
                 "--input", str(in_path), "--output", str(out_path), "--quiet"],
                cwd=_ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(res.returncode, 0, f"stderr: {res.stderr[:500]}")
            self.assertTrue(out_path.exists())
            audit = json.loads(out_path.read_text())
            self.assertEqual(audit["dead"]["count"], 2)


class TestBacktestSmokeHarness(unittest.TestCase):
    def test_synthetic_data_generation(self):
        from tools.backtest_smoke import make_synthetic_features
        df = make_synthetic_features(n_rows=200, seed=0)
        self.assertEqual(len(df), 200)
        self.assertIn("ts_event", df.columns)
        self.assertIn("price", df.columns)
        # MBP-10 columns
        self.assertIn("bid_sz_00", df.columns)
        self.assertIn("ask_sz_09", df.columns)

    def test_smoke_script_help(self):
        res = subprocess.run(
            [sys.executable, "tools/backtest_smoke.py", "--help"],
            cwd=_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--n-rows", res.stdout)


@unittest.skip("tools/paper_dry_run.py archived (depends on legacy V19 paper_v19)")
class TestPaperDryRun(unittest.TestCase):
    def test_synthetic_bars(self):
        from tools.paper_dry_run import synthetic_bars
        rows = synthetic_bars(n_bars=50)
        self.assertEqual(len(rows), 50)
        self.assertIn("ts_event", rows[0])
        self.assertIn("depth_pressure", rows[0])
        self.assertIn("informed_prob", rows[0])

    def test_run_dry_paper(self):
        from tools.paper_dry_run import synthetic_bars, run_dry_paper
        rows = synthetic_bars(n_bars=30)
        with tempfile.TemporaryDirectory() as td:
            result = run_dry_paper(rows, alpha_set_path=None, output_dir=td)
            self.assertEqual(result["n_bars"], 30)
            self.assertGreater(sum(result["actions"].values()), 0)
            self.assertTrue(Path(result["log_path"]).exists())


class TestOperationalGuide(unittest.TestCase):
    def test_guide_exists(self):
        path = Path(_ROOT) / "docs" / "OPERATIONAL_GUIDE.md"
        self.assertTrue(path.exists(), "OPERATIONAL_GUIDE.md مفقود")
        content = path.read_text()
        # المراحل من التقرير
        for phase in ["Phase 1", "Phase 2", "Phase 3", "Phase 4",
                      "Phase 5", "Phase 6", "Phase A", "Phase B",
                      "Phase C", "Phase D"]:
            self.assertIn(phase, content, f"{phase} مفقود من الـ guide")

    def test_guide_lists_sprints(self):
        content = (Path(_ROOT) / "docs" / "OPERATIONAL_GUIDE.md").read_text()
        # Sprints 1-9
        for i in range(1, 10):
            self.assertIn(f"Sprint {i}", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
