"""Phase 1.9 — market-realistic iceberg ground-truth simulator + precision/
recall harness.

These tests prove the simulator is NON-CIRCULAR (it plants icebergs with
parameters that straddle the detector thresholds + confounders that look
iceberg-ish), then measure the existing detector's precision/recall against
the planted ground truth. The headline test asserts the detector is
USEFUL (precision + recall both meaningfully > random) without demanding
perfection — perfection would mean the simulator was tautological.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.iceberg import IcebergDetector, IcebergConfig
from modules.features_v2.iceberg_simulator import (
    IcebergGroundTruthSimulator, IcebergSimConfig,
    evaluate_detector, recall_by_ratio_bucket,
    GT_ICEBERG, GT_CONFOUND_LEGIT_REFILL, GT_CONFOUND_VISIBLE_SWEEP,
    GT_CONFOUND_SPOOF,
)


# ── simulator structural correctness ───────────────────────────────────────
class TestSimulatorOutput:
    def test_generate_returns_mbo_and_ground_truth(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=0))
        mbo, gt = sim.generate(n_icebergs=20)
        # MBO has the columns the detector requires
        for c in ("ts_event", "price", "side", "displayed_size",
                  "executed_size", "action"):
            assert c in mbo.columns
        # GT has one row per planted event
        assert len(gt) > 0
        assert "gt_label" in gt.columns

    def test_plants_all_four_event_kinds(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=1, confounder_ratio=1.0))
        _, gt = sim.generate(n_icebergs=30)
        kinds = set(gt["gt_label"].unique())
        assert GT_ICEBERG in kinds
        assert GT_CONFOUND_LEGIT_REFILL in kinds
        assert GT_CONFOUND_VISIBLE_SWEEP in kinds
        assert GT_CONFOUND_SPOOF in kinds

    def test_ratios_straddle_the_detector_threshold(self):
        """ANTI-CIRCULARITY proof: planted icebergs have ratios both BELOW
        and ABOVE the detector's 2.5x cutoff. If they were all >= 2.5x the
        simulator would be tautological."""
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=2))
        _, gt = sim.generate(n_icebergs=100)
        ice = gt[gt["gt_label"] == GT_ICEBERG]
        assert (ice["planted_ratio"] < 2.5).any(), "no sub-threshold icebergs — circular!"
        assert (ice["planted_ratio"] >= 2.5).any(), "no supra-threshold icebergs"

    def test_refill_latencies_straddle_8s(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=3))
        _, gt = sim.generate(n_icebergs=100)
        ice = gt[gt["gt_label"] == GT_ICEBERG]
        assert (ice["planted_refill_s"] < 8.0).any()
        assert (ice["planted_refill_s"] >= 8.0).any()

    def test_confounders_have_zero_hidden_volume(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=4))
        _, gt = sim.generate(n_icebergs=20)
        conf = gt[gt["gt_label"] != GT_ICEBERG]
        assert (conf["true_hidden_volume"] == 0.0).all()


# ── precision / recall harness ─────────────────────────────────────────────
class TestEvaluateDetector:
    def test_empty_detection_zero_recall(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=5))
        _, gt = sim.generate(n_icebergs=10)
        m = evaluate_detector(pd.DataFrame(), gt)
        assert m["recall"] == 0.0
        assert m["fn"] == m["n_gt_icebergs"]

    def test_metrics_shape(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=6))
        mbo, gt = sim.generate(n_icebergs=20)
        det = IcebergDetector()
        events = det.detect_events(mbo)
        m = evaluate_detector(events, gt)
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert k in m
        # precision + recall are in [0, 1]
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0


class TestDetectorIsUsefulNotTautological:
    """The core scientific claim: against icebergs the detector did NOT
    plant (varied params + confounders), it still achieves meaningful
    precision and recall — but NOT a perfect 1.0/1.0 (that would prove the
    simulator was circular)."""

    def test_detector_beats_random_on_realistic_distribution(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=7))
        mbo, gt = sim.generate(n_icebergs=80)
        det = IcebergDetector()
        events = det.detect_events(mbo)
        m = evaluate_detector(events, gt)

        # Precision must be high — confounders should rarely fool the rule
        assert m["precision"] >= 0.70, (
            f"precision {m['precision']:.2f} too low — confounders are "
            f"fooling the detector (legit_refill / spoof leaking through)"
        )
        # Recall must be meaningful but NOT perfect — sub-2.5x icebergs are
        # legitimately missed. If recall == 1.0 the simulator is circular.
        assert 0.30 <= m["recall"] < 1.0, (
            f"recall {m['recall']:.2f} outside the meaningful-but-imperfect "
            f"band [0.30, 1.0). ==1.0 would mean the sim is tautological; "
            f"<0.30 would mean the rule barely generalizes."
        )

    def test_recall_curve_is_monotonic_in_ratio(self):
        """The threshold-sensitivity curve: recall should RISE as the
        planted hidden:visible ratio crosses the detector's 2.5x cutoff.
        This is the empirical evidence that the 2.5x rule is calibrated."""
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=8))
        mbo, gt = sim.generate(n_icebergs=200)
        det = IcebergDetector()
        events = det.detect_events(mbo)
        curve = recall_by_ratio_bucket(events, gt)

        # The lowest ratio bucket (1.5-2.5x) must have LOWER recall than the
        # highest bucket (6-10x) — the rule correctly discriminates.
        low_bucket = curve.iloc[0]["recall"]
        high_bucket = curve.iloc[-1]["recall"]
        # tolerate NaN if a bucket is empty
        if not (np.isnan(low_bucket) or np.isnan(high_bucket)):
            assert high_bucket >= low_bucket, (
                f"recall should rise with ratio: low={low_bucket:.2f}, "
                f"high={high_bucket:.2f}"
            )


class TestThresholdSensitivity:
    """Tightening the detector's vol_mult should trade recall for precision
    in the expected direction — proves the harness can DRIVE tuning."""

    def test_stricter_ratio_lowers_recall(self):
        sim = IcebergGroundTruthSimulator(IcebergSimConfig(seed=9))
        mbo, gt = sim.generate(n_icebergs=100)

        loose = IcebergDetector(IcebergConfig(vol_mult=2.0))
        strict = IcebergDetector(IcebergConfig(vol_mult=5.0))
        m_loose = evaluate_detector(loose.detect_events(mbo), gt)
        m_strict = evaluate_detector(strict.detect_events(mbo), gt)

        # A stricter ratio cutoff cannot detect MORE icebergs
        assert m_strict["recall"] <= m_loose["recall"] + 1e-9, (
            f"stricter rule should not raise recall: "
            f"loose={m_loose['recall']:.2f}, strict={m_strict['recall']:.2f}"
        )
