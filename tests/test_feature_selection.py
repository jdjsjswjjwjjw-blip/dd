"""Tests for the feature-selection helpers + their integration with the
enhanced walk-forward fold runner.

Each test verifies a specific invariant of the drop-from-audit path:
  • Audit JSON is parsed safely
  • Drop list is honored verbatim — no extra columns removed
  • Stale-audit detection raises early
  • Integration: passing --drop-features-from-audit to the enhanced
    runner actually reduces the feature dimension at training time
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.trading_intel.training.feature_selection import (
    AuditFileError,
    apply_drop_list,
    load_drop_list_from_audit,
    validate_audit_against_features,
)

ENHANCED_RUNNER = REPO_ROOT / "tools" / "run_walk_forward_fold_enhanced.py"
AUDIT_TOOL = REPO_ROOT / "tools" / "audit_feature_redundancy.py"


# ════════════════════════════════════════════════════════════════════
# load_drop_list_from_audit
# ════════════════════════════════════════════════════════════════════
class TestLoadDropList:
    def test_valid_audit_returns_set(self, tmp_path):
        p = tmp_path / "audit.json"
        p.write_text(json.dumps({"drop_list": ["a", "b", "c"]}))
        drops = load_drop_list_from_audit(p)
        assert drops == {"a", "b", "c"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AuditFileError, match="not found"):
            load_drop_list_from_audit(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(AuditFileError, match="not valid JSON"):
            load_drop_list_from_audit(p)

    def test_missing_drop_list_key_raises(self, tmp_path):
        p = tmp_path / "no_key.json"
        p.write_text(json.dumps({"other_key": []}))
        with pytest.raises(AuditFileError, match="no 'drop_list'"):
            load_drop_list_from_audit(p)

    def test_drop_list_must_be_list(self, tmp_path):
        p = tmp_path / "wrong_type.json"
        p.write_text(json.dumps({"drop_list": "not a list"}))
        with pytest.raises(AuditFileError, match="not a list"):
            load_drop_list_from_audit(p)


# ════════════════════════════════════════════════════════════════════
# apply_drop_list
# ════════════════════════════════════════════════════════════════════
class TestApplyDropList:
    def test_removes_only_listed(self):
        kept, dropped = apply_drop_list(["a", "b", "c", "d"], {"b", "d"})
        assert kept == ["a", "c"]
        assert dropped == ["b", "d"]

    def test_drop_set_extra_items_ignored(self):
        """If drop_set contains features that aren't in the input, they're
        silently ignored — they show up in `validate_audit_against_features`
        but don't cause apply_drop_list itself to fail."""
        kept, dropped = apply_drop_list(["a", "b"], {"b", "ghost"})
        assert kept == ["a"]
        assert dropped == ["b"]   # only the actually-present drop

    def test_empty_drop_set_keeps_all(self):
        kept, dropped = apply_drop_list(["a", "b"], set())
        assert kept == ["a", "b"]
        assert dropped == []

    def test_drop_all_returns_empty(self):
        kept, dropped = apply_drop_list(["a", "b"], {"a", "b"})
        assert kept == []
        assert dropped == ["a", "b"]


# ════════════════════════════════════════════════════════════════════
# validate_audit_against_features
# ════════════════════════════════════════════════════════════════════
class TestValidateAudit:
    def test_perfect_match_not_stale(self, tmp_path):
        p = tmp_path / "audit.json"
        p.write_text(json.dumps({"drop_list": ["a", "b"]}))
        diag = validate_audit_against_features(p, ["a", "b", "c", "d"])
        assert diag["n_drop_actually_in_features"] == 2
        assert diag["n_in_audit_but_missing_from_features"] == 0
        assert diag["stale_audit"] is False

    def test_majority_missing_is_stale(self, tmp_path):
        p = tmp_path / "audit.json"
        # 4 drops, 3 missing → strictly > half (which is // 2 = 2) → stale
        p.write_text(json.dumps({
            "drop_list": ["a", "ghost1", "ghost2", "ghost3"],
        }))
        diag = validate_audit_against_features(p, ["a", "b", "c"])
        assert diag["stale_audit"] is True
        assert "ghost1" in diag["missing_from_features"]
        assert "ghost2" in diag["missing_from_features"]
        assert "ghost3" in diag["missing_from_features"]

    def test_exactly_half_missing_not_stale(self, tmp_path):
        p = tmp_path / "audit.json"
        # 4 drops, 2 missing → exactly half, NOT stale (boundary case)
        p.write_text(json.dumps({"drop_list": ["a", "b", "ghost1", "ghost2"]}))
        diag = validate_audit_against_features(p, ["a", "b", "c"])
        assert diag["stale_audit"] is False


# ════════════════════════════════════════════════════════════════════
# Integration: enhanced runner with --drop-features-from-audit
# ════════════════════════════════════════════════════════════════════
class TestEnhancedRunnerWithDropAudit:
    def _make_dataset_with_known_redundancy(self, tmp_path: Path):
        """Synthetic monthly dataset (Jan-Aug 2022) with:
          • 4 highly-correlated CVD-like features
          • 4 independent generic features
        Total feature parquet has 8 columns + labels.
        """
        ts = pd.date_range("2022-01-01", "2022-08-31 23:00:00", freq="1h", tz=None)
        n = len(ts)
        rng = np.random.RandomState(7)
        cvd_base = rng.randn(n)
        target_ret = rng.randn(n).astype("float32") * 0.005
        df = pd.DataFrame({
            "ts_event": ts,
            "close": 1.30 + rng.randn(n) * 0.005,
            "atr_14": np.full(n, 0.005, dtype="float32"),
            "event_flag": rng.choice([0, 1], n, p=[0.93, 0.07]).astype("int8"),
            "event_direction": rng.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]).astype("int8"),
            "path_outcome": rng.choice([0, 1, 2, 3, 4], n,
                                        p=[0.32, 0.32, 0.10, 0.10, 0.16]).astype("int8"),
            "target_ret_24": target_ret,
            # 4 redundant CVD-like features
            "cvd": cvd_base,
            "cvd_slope": cvd_base + 0.02 * rng.randn(n),
            "cvd_momentum": cvd_base + 0.02 * rng.randn(n),
            "session_cvd": cvd_base + 0.02 * rng.randn(n),
            # 4 independent
            "feat_a": rng.randn(n).astype("float32"),
            "feat_b": rng.randn(n).astype("float32"),
            "feat_c": rng.randn(n).astype("float32"),
            "feat_d": rng.randn(n).astype("float32"),
        })
        features_path = tmp_path / "features.parquet"
        df.to_parquet(features_path, index=False)
        emb = rng.randn(n, 8).astype("float32")
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, emb)
        valid_path = tmp_path / "embedding_valid.npy"
        np.save(valid_path, np.ones(n, dtype=bool))
        return features_path, emb_path, valid_path

    def _run_audit(self, features_path: Path, tmp_path: Path) -> Path:
        """Run the auditor on the synthetic parquet, return summary JSON path."""
        out = tmp_path / "audit_out"
        result = subprocess.run(
            [sys.executable, str(AUDIT_TOOL),
             "--features", str(features_path),
             "--output", str(out),
             "--threshold", "0.9"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        summary = out / "redundancy_summary.json"
        assert summary.exists()
        return summary

    def _run_fold(self, tmp_path: Path, extra_args: list[str], suffix: str) -> dict:
        feat, emb, valid = self._make_dataset_with_known_redundancy(tmp_path)
        out_dir = tmp_path / f"fold_{suffix}"
        cmd = [
            sys.executable, str(ENHANCED_RUNNER),
            "--fold-id", suffix,
            "--combined-features", str(feat),
            "--ssl-embeddings", str(emb),
            "--embedding-valid", str(valid),
            "--train-start", "2022-01", "--train-end", "2022-05",
            "--test-start", "2022-06", "--test-end", "2022-07",
            "--output-dir", str(out_dir),
            "--epochs", "3", "--hidden-dim", "32", "--batch-size", "64",
        ] + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"fold failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        m = json.loads((out_dir / "metrics.json").read_text())
        return m

    def test_baseline_fold_uses_all_features(self, tmp_path):
        m = self._run_fold(tmp_path, extra_args=[], suffix="baseline")
        # Numeric non-leakage cols: close, atr_14, 4 CVD, 4 feat = 10
        assert m["n_features_after_dedup"] == 10

    def test_drop_audit_reduces_feature_count(self, tmp_path):
        feat, _, _ = self._make_dataset_with_known_redundancy(tmp_path)
        audit_summary = self._run_audit(feat, tmp_path)
        # Re-create the dataset since `_make_dataset_with_known_redundancy`
        # writes fresh files at the SAME paths each call — that's fine
        m = self._run_fold(
            tmp_path,
            extra_args=["--drop-features-from-audit", str(audit_summary)],
            suffix="dedup",
        )
        # CVD family has 4 members; auditor keeps 1, drops 3
        # → 10 - 3 = 7 features after dedup
        assert m["n_features_after_dedup"] == 7
        # Diagnostics are recorded in metrics.json
        assert "feature_audit" in m
        assert m["feature_audit"]["n_drop_actually_in_features"] == 3
        assert m["feature_audit"]["stale_audit"] is False
        assert m["config"]["drop_features_from_audit"] == str(audit_summary)
