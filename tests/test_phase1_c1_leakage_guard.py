"""Tests for Phase 1 / Commit C1 — SSL feature-vector leakage guard.

After B1/B2 the parquet ships THREE classes of forward-looking columns:
  - bias_label / path_outcome / forward_return  (old, already guarded)
  - mfe / mae / stop_first_flag / time_to_first_touch / net_expectancy_proxy
    (multi-task diagnostics — previously NOT guarded; the `mfe_`/`mae_`
    prefix only catches `mfe_atr`/`mae_atr`, not the bare names)
  - exec_label / exec_path / exec_valid + next_price_delta(_valid)
    (B1/B2 dual-target heads — brand new)

Any of these appearing in the SSL feature vector would let the backbone
read its own training target — catastrophic label leakage. C1 extends the
data_loader's `_is_leakage_column` so all three classes are excluded by
exact name AND by future-extension prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from self_supervised.data_loader import (
    _is_leakage_column,
    _LEAKAGE_COLS_EXACT,
    _LEAKAGE_PREFIXES,
)


class TestNewTargetColumnsExcluded:
    """B1/B2 columns must be excluded by name (defense layer 1)."""

    def test_exec_label_excluded(self):
        assert _is_leakage_column("exec_label")

    def test_exec_path_excluded(self):
        assert _is_leakage_column("exec_path")

    def test_exec_valid_excluded(self):
        assert _is_leakage_column("exec_valid")

    def test_next_price_delta_excluded(self):
        assert _is_leakage_column("next_price_delta")

    def test_next_price_delta_valid_excluded(self):
        assert _is_leakage_column("next_price_delta_valid")


class TestPreviouslyLeakingMultitaskColumnsNowExcluded:
    """Multi-task diagnostics that were NOT guarded before C1."""

    def test_bare_mfe_excluded(self):
        # `mfe_` prefix already caught `mfe_atr`, but bare `mfe` slipped through
        assert _is_leakage_column("mfe")

    def test_bare_mae_excluded(self):
        assert _is_leakage_column("mae")

    def test_stop_first_flag_excluded(self):
        assert _is_leakage_column("stop_first_flag")

    def test_time_to_first_touch_excluded(self):
        assert _is_leakage_column("time_to_first_touch")

    def test_net_expectancy_proxy_excluded(self):
        assert _is_leakage_column("net_expectancy_proxy")


class TestFutureExtensionPrefixes:
    """Fail-closed: any new column matching the family prefix is excluded."""

    def test_unknown_exec_column_excluded_by_prefix(self):
        assert _is_leakage_column("exec_confidence")  # hypothetical sibling
        assert _is_leakage_column("exec_quality_score")

    def test_unknown_next_price_column_excluded_by_prefix(self):
        assert _is_leakage_column("next_price_horizon")  # hypothetical
        assert _is_leakage_column("next_price_delta_v2")


class TestExistingGuardsStillHold:
    """Regression: the old guards still work."""

    def test_bias_label_still_excluded(self):
        assert _is_leakage_column("bias_label")

    def test_forward_return_still_excluded(self):
        assert _is_leakage_column("forward_return")

    def test_mfe_atr_still_excluded(self):
        assert _is_leakage_column("mfe_atr")  # via `mfe_` prefix

    def test_real_feature_columns_NOT_excluded(self):
        # Sanity: actual features must NOT be flagged
        for col in (
            "close", "high", "low", "open", "atr_14", "obi_net",
            "cvd_cumulative", "kalman_position", "rsi_14",
        ):
            assert not _is_leakage_column(col), (
                f"feature column wrongly flagged as leakage: {col!r}"
            )


class TestSetShape:
    """Lock in the structural invariants the guard depends on."""

    def test_exec_prefix_present(self):
        assert "exec_" in _LEAKAGE_PREFIXES

    def test_next_price_prefix_present(self):
        assert "next_price_" in _LEAKAGE_PREFIXES

    def test_new_exact_cols_present(self):
        for c in ("exec_label", "exec_path", "exec_valid",
                  "next_price_delta", "next_price_delta_valid",
                  "mfe", "mae", "stop_first_flag",
                  "time_to_first_touch", "net_expectancy_proxy"):
            assert c in _LEAKAGE_COLS_EXACT, f"missing from exact set: {c!r}"
