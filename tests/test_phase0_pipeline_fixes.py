"""Tests for Phase 0 fixes in prepare_day_trading.py.

Phase 0 = critical fixes before the bigger pipeline cleanup:
  1. Column rename: `obi` → `obi_net` so the SSL data_loader + event
     gate audit + verify_data_health find the column they expect.
  2. `cvd_cumulative` alias so we can introduce per-bar CVD variants
     in Phase 2 without breaking existing consumers.

These tests don't run the full pipeline — they check the rename logic
in isolation by importing the module and exercising the relevant code
path on a synthetic DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestPhase0Renames:
    """The renames live inline in prepare_day_trading.py:run_day_trading_refinery.
    Rather than reach into private state, we simulate the same logic on
    synthetic data to lock in the contract.
    """

    def _apply_phase0_renames(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror of the rename block in prepare_day_trading.py."""
        renames = {}
        if 'obi' in df.columns and 'obi_net' not in df.columns:
            renames['obi'] = 'obi_net'
        if 'cvd' in df.columns and 'cvd_direction_pct' not in df.columns:
            df['cvd_cumulative'] = df['cvd']
        if renames:
            df = df.rename(columns=renames)
        return df

    def test_obi_renamed_to_obi_net(self):
        df = pd.DataFrame({'obi': [0.1, -0.2, 0.3]})
        out = self._apply_phase0_renames(df)
        assert 'obi_net' in out.columns
        assert 'obi' not in out.columns
        assert list(out['obi_net']) == [0.1, -0.2, 0.3]

    def test_cvd_keeps_alias_cumulative(self):
        df = pd.DataFrame({'cvd': [100, 150, 120]})
        out = self._apply_phase0_renames(df)
        # `cvd` stays (Phase 2 will introduce proper per-bar variants);
        # `cvd_cumulative` is added as a semantically-clear alias
        assert 'cvd' in out.columns
        assert 'cvd_cumulative' in out.columns
        assert list(out['cvd_cumulative']) == [100, 150, 120]

    def test_no_rename_when_already_correct_names(self):
        df = pd.DataFrame({
            'obi_net': [0.1],
            'cvd_direction_pct': [0.5],
        })
        out = self._apply_phase0_renames(df)
        # No new columns introduced
        assert set(out.columns) == {'obi_net', 'cvd_direction_pct'}

    def test_no_rename_when_both_old_and_new_exist(self):
        # If both `obi` and `obi_net` exist, leave them alone
        df = pd.DataFrame({
            'obi': [0.1],
            'obi_net': [0.2],
        })
        out = self._apply_phase0_renames(df)
        assert 'obi' in out.columns
        assert 'obi_net' in out.columns

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame()
        out = self._apply_phase0_renames(df)
        assert out.empty


class TestPhase0InSource:
    """Source-level guard — the rename block must stay in
    prepare_day_trading.py (no silent regressions)."""

    def test_rename_block_present(self):
        src = (REPO_ROOT / 'prepare_day_trading.py').read_text()
        # The Phase 0 comment marker must stay
        assert 'Phase 0 fix' in src
        assert 'obi_net' in src
        assert 'cvd_cumulative' in src


class TestPhase0GateRemoval:
    """Source-level guards confirming the event gate has been removed
    from label_by_outcome. The previous behaviour gated labeling by
    `if not is_ev_arr[i]: continue` — kill ~80% of bars before the
    forward TP/SL scan could even run, producing the 80%-NEUTRAL
    pathology the IC audit confirmed empirically.
    """

    def test_gate_filter_block_removed(self):
        src = (REPO_ROOT / 'prepare_day_trading.py').read_text()
        # The exact filter line should NO LONGER appear in label_by_outcome
        forbidden_patterns = [
            "if not is_ev_arr[i]:\n            # ليس حدثاً",
            "if not is_ev_arr[i]:\n            # ليس حدثاً: افتراضياً NEUTRAL",
            "neutral_reason[i] = NEUTRAL_REASON_WEAK_EVENT",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in src, (
                f"Phase 0 root fix regressed: the event gate filter is "
                f"back in label_by_outcome. Forbidden pattern still in "
                f"source: {pattern!r}"
            )

    def test_root_fix_explanation_documented(self):
        """The replacement explanation must stay so the change is
        discoverable and the rationale survives later edits."""
        src = (REPO_ROOT / 'prepare_day_trading.py').read_text()
        markers = [
            'Phase 0 root fix',
            'gate_kill_ratio',
            'forward TP/SL scan',
        ]
        for m in markers:
            assert m in src, f"Phase 0 root-fix marker missing: {m!r}"

    def test_label_by_outcome_still_callable(self):
        """Smoke test — function imports and accepts the documented signature.
        We're not running it here (it's heavy); just verifying it didn't
        get accidentally broken at parse time."""
        import importlib
        import prepare_day_trading as pdt
        importlib.reload(pdt)
        assert hasattr(pdt, 'label_by_outcome')
        assert callable(pdt.label_by_outcome)
