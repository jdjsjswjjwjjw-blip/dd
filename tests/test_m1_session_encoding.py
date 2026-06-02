"""M1 (Phase 0) — exclusive per-session encoding in ic_audit.

The `--per-session` IC breakdown used to encode the session as
`is_london*1 + is_ny*2 + is_overlap*3`. After C1 the flags overlap by
construction (overlap ⊂ london ∩ ny: london 07-16, ny 13-22, overlap 13-16),
so the additive sum produced ambiguous codes {0,1,2,6} that mis-attributed
each feature's per-session IC. `exclusive_session_label` replaces it with a
priority encoding (overlap > ny > london; rest = 'other'), so every bar lands
in exactly one clean, human-readable group. These tests lock that contract and
prove the collision is gone.

(quant-rigor-guard RULE 6: correct grouping for the diagnostic. The overall /
per-event IC verdicts do NOT use this and are unaffected.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.ic_audit import exclusive_session_label


def _frame_from_hours(hours):
    """Build a bars frame with C1 session flags from a list of UTC hours."""
    h = np.array(hours)
    return pd.DataFrame({
        "is_london": ((h >= 7) & (h < 16)).astype(int),
        "is_ny": ((h >= 13) & (h < 22)).astype(int),
        "is_overlap": ((h >= 13) & (h < 16)).astype(int),
        "_hour": h,
    })


class TestExclusiveEncoding:
    def test_each_bar_exactly_one_known_label(self):
        df = _frame_from_hours(list(range(24)))
        sess = exclusive_session_label(df)
        assert set(np.unique(sess)) <= {"london", "ny", "overlap", "other"}
        assert len(sess) == len(df)

    def test_priority_assignment_by_hour(self):
        # representative hours hitting each region
        df = _frame_from_hours([3, 9, 14, 18, 23])
        sess = exclusive_session_label(df)
        expected = ["other",     # 03:00 — off/asia
                    "london",    # 09:00 — london only (07-13)
                    "overlap",   # 14:00 — london∩ny∩overlap → overlap wins
                    "ny",        # 18:00 — ny only (16-22)
                    "other"]     # 23:00 — off
        assert list(sess) == expected, f"got {list(sess)}"

    def test_overlap_bars_are_overlap_not_a_sum_code(self):
        """The collision fix: a 14:00 bar (london=ny=overlap=1) must be the
        single label 'overlap', NOT the old additive code 6."""
        df = _frame_from_hours([14])
        assert exclusive_session_label(df)[0] == "overlap"

    def test_no_ambiguous_numeric_codes(self):
        """No bar should carry a meaningless additive code; the label set is the
        clean string vocabulary only (regression guard for the old {0,1,2,6})."""
        df = _frame_from_hours(list(range(24)) * 3)
        labels = set(np.unique(exclusive_session_label(df)))
        assert labels.issubset({"london", "ny", "overlap", "other"})
        assert "overlap" in labels and "london" in labels and "ny" in labels

    def test_missing_optional_flags_default_safe(self):
        # only is_london present (is_ny / is_overlap absent) → no crash
        df = pd.DataFrame({"is_london": [1, 0, 1]})
        sess = exclusive_session_label(df)
        assert list(sess) == ["london", "other", "london"]
