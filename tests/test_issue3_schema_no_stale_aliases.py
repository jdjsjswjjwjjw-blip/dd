"""Issue 3 — schema validator is consistent with the post-D1 export contract.

D1 Phase 1 removed `obi_net` / `cvd_cumulative` from ORIGINAL_FEATURES (they
were depth-coupled duplicates of `obi` / `cvd`). The validator in
modules/dataset_schema.py was left expecting them, so every refinery run
since D1 emitted two false-positive 'issue(s)' on the post-write
schema_validate() call. Hygiene per quant-rigor-guard R6.

Tests lock the contract:
  • a D1-shaped frame (cvd present, no stale aliases) passes — no false
    obi_net / cvd_cumulative complaint.
  • TEETH (the validator still has bite): a frame missing `cvd` IS
    reported, and missing OHLCV is still caught — so the fix removes the
    stale checks without weakening the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.dataset_schema import REQUIRED_OHLCV, REQUIRED_REGIME, validate


def _d1_frame(n=10):
    """Frame matching the post-D1 export contract: cvd present, no obi_net /
    cvd_cumulative aliases."""
    return pd.DataFrame({
        "ts_event":     pd.date_range("2025-06-01", periods=n, freq="5min", tz="UTC"),
        "open":         np.full(n, 100.0),
        "high":         np.full(n, 100.5),
        "low":          np.full(n,  99.5),
        "close":        np.full(n, 100.2),
        "volume":       np.full(n,  50.0),
        "regime_label": np.array(["trending"] * n, dtype=object),
        "atr_14":       np.full(n,  1.0),
        "is_session_break": np.zeros(n, dtype=bool),
        "cvd":          np.zeros(n, dtype=np.float64),
    })


class TestNoFalsePositiveOnD1Frame:
    def test_post_d1_frame_passes_clean(self):
        errs = validate(_d1_frame())
        assert errs == [], f"unexpected errors on a clean D1 frame: {errs}"

    def test_no_complaint_about_dropped_aliases(self):
        """Even with neither alias present, the validator must NOT complain —
        D1 removed them by design."""
        errs = validate(_d1_frame())
        joined = " | ".join(errs)
        assert "obi_net" not in joined
        assert "cvd_cumulative" not in joined


class TestTeethValidatorStillBites:
    def test_missing_cvd_is_reported(self):
        df = _d1_frame().drop(columns=["cvd"])
        errs = validate(df)
        assert any("cvd" in e for e in errs), (
            f"validator must still flag missing cvd; got {errs}"
        )

    def test_missing_ohlcv_still_caught(self):
        df = _d1_frame().drop(columns=["close"])
        errs = validate(df)
        assert any("close" in e for e in errs)

    def test_missing_regime_still_caught(self):
        df = _d1_frame().drop(columns=["regime_label"])
        errs = validate(df)
        assert any("regime_label" in e for e in errs)
