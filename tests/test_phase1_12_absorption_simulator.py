"""Phase 1.12 — absorption ground-truth simulator + detector tests.

These tests prove the absorption signal is MEASURED, not assumed. The
centrepiece is the volume-matched-trend separation test: absorption and
trend segments carry identical one-sided volume, differing only in whether
price moved. A detector that merely fired on volume would score them
identically and FAIL test_absorption_separates_from_volume_matched_trend.
That test passing is the non-circularity guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.absorption import AbsorptionDetector, AbsorptionConfig
from modules.features_v2.absorption_simulator import (
    AbsorptionGroundTruthSimulator, AbsorptionSimConfig,
    evaluate_detector, absorption_vs_trend_separation, recall_by_ratio_bucket,
    GT_ABSORPTION, GT_CONFOUND_TREND, GT_CONFOUND_THIN, GT_BASELINE,
)


# ── detector wrapper ────────────────────────────────────────────────────────
class TestAbsorptionDetector:
    def test_compute_returns_expected_columns(self):
        trades = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=200, freq="100ms", tz="UTC"),
            "price": 1.25 + np.zeros(200),
            "size": np.full(200, 5.0),
            "side": ["A"] * 200,
        })
        out = AbsorptionDetector().compute(trades)
        for c in ("ts_event", "price", "cvd", "absorption_intensity", "absorb_z", "fired"):
            assert c in out.columns
        assert len(out) == 200

    def test_missing_column_raises(self):
        bad = pd.DataFrame({"ts_event": [], "price": [], "size": []})
        with pytest.raises(ValueError, match="column missing"):
            AbsorptionDetector().compute(bad)

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["ts_event", "price", "size", "side"])
        out = AbsorptionDetector().compute(empty)
        assert len(out) == 0

    def test_pinned_price_high_volume_fires(self):
        """Heavy one-sided flow with price pinned → AII should spike and the
        z-rule should fire on a meaningful fraction of trades (sanity that the
        wrapper reproduces the production absorption signature)."""
        n = 400
        # 200 quiet balanced trades, then 200 one-sided trades at a pinned price
        sides = (["A", "B"] * 100) + (["A"] * 200)
        prices = list(1.25 + np.random.RandomState(0).randn(200) * 0.0005) + [1.25] * 200
        trades = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=n, freq="100ms", tz="UTC"),
            "price": prices,
            "size": np.full(n, 6.0),
            "side": sides,
        })
        out = AbsorptionDetector().compute(trades)
        # In the pinned one-sided tail, AII should be elevated vs the quiet head
        head_aii = out["absorption_intensity"].iloc[50:200].median()
        tail_aii = out["absorption_intensity"].iloc[250:].median()
        assert tail_aii > head_aii, f"tail {tail_aii:.3f} should exceed head {head_aii:.3f}"
        assert out["fired"].iloc[200:].sum() > 0, "pinned absorption tail should fire"

    def test_config_threshold_respected(self):
        # A very high z_threshold should suppress almost all fires
        trades = pd.DataFrame({
            "ts_event": pd.date_range("2025-04-01", periods=300, freq="100ms", tz="UTC"),
            "price": 1.25 + np.random.RandomState(1).randn(300) * 0.0002,
            "size": np.full(300, 5.0),
            "side": ["A"] * 300,
        })
        strict = AbsorptionDetector(AbsorptionConfig(z_threshold=10.0)).compute(trades)
        loose = AbsorptionDetector(AbsorptionConfig(z_threshold=0.5)).compute(trades)
        assert int(strict["fired"].sum()) <= int(loose["fired"].sum())


# ── simulator structure ─────────────────────────────────────────────────────
class TestSimulatorStructure:
    def test_generates_trades_and_groundtruth(self):
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=1))
        trades, gt = sim.generate(n_absorptions=10)
        assert len(trades) > 0
        assert {"ts_event", "price", "size", "side"} <= set(trades.columns)
        assert {"gt_label", "start_idx", "end_idx", "planted_ratio"} <= set(gt.columns)
        # segment spans tile the trade stream contiguously
        spans = gt.sort_values("start_idx")
        assert spans["start_idx"].iloc[0] == 0
        assert spans["end_idx"].iloc[-1] == len(trades)
        prev_end = 0
        for _, s in spans.iterrows():
            assert s["start_idx"] == prev_end, "segments must be contiguous"
            prev_end = s["end_idx"]

    def test_timestamps_monotonic(self):
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=2))
        trades, _ = sim.generate(n_absorptions=8)
        ts = pd.to_datetime(trades["ts_event"])
        assert ts.is_monotonic_increasing, "ts_event must be monotonic for index alignment"

    def test_all_segment_types_present(self):
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=3))
        _, gt = sim.generate(n_absorptions=12)
        labels = set(gt["gt_label"])
        assert {GT_ABSORPTION, GT_CONFOUND_TREND, GT_CONFOUND_THIN, GT_BASELINE} <= labels

    def test_trend_is_volume_matched_to_absorption(self):
        """The control's validity hinges on this: trend and absorption draw
        volume from the SAME distribution. Their mean planted_volume should be
        statistically indistinguishable (within 25%)."""
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=4))
        _, gt = sim.generate(n_absorptions=60)
        abs_vol = gt[gt["gt_label"] == GT_ABSORPTION]["planted_volume"].mean()
        trend_vol = gt[gt["gt_label"] == GT_CONFOUND_TREND]["planted_volume"].mean()
        assert abs(abs_vol - trend_vol) / abs_vol < 0.25, (
            f"trend volume {trend_vol:.1f} not matched to absorption {abs_vol:.1f}"
        )


# ── the non-circularity centrepiece ─────────────────────────────────────────
class TestNonCircularSeparation:
    def test_absorption_separates_from_volume_matched_trend(self):
        """THE key test. Absorption and trend carry identical volume; only
        price behaviour differs. AII must rank absorption trades ABOVE trend
        trades (AUC >> 0.5). A volume-only detector would score 0.5 here."""
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=42))
        trades, gt = sim.generate(n_absorptions=30)
        out = AbsorptionDetector().compute(trades)
        sep = absorption_vs_trend_separation(out, gt)
        assert sep["auc_absorption_vs_trend"] > 0.85, (
            f"AII must separate absorption from volume-matched trend; "
            f"AUC={sep['auc_absorption_vs_trend']:.3f}"
        )
        assert sep["median_aii_absorption"] > sep["median_aii_trend"] * 2, (
            f"absorption AII {sep['median_aii_absorption']:.2f} should dwarf "
            f"trend {sep['median_aii_trend']:.2f}"
        )

    def test_absorption_separates_from_thin(self):
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=42))
        trades, gt = sim.generate(n_absorptions=30)
        out = AbsorptionDetector().compute(trades)
        sep = absorption_vs_trend_separation(out, gt)
        assert sep["auc_absorption_vs_thin"] > 0.85


# ── precision / recall + sensitivity ────────────────────────────────────────
class TestDetectionQuality:
    def test_precision_recall_reasonable(self):
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=42))
        trades, gt = sim.generate(n_absorptions=40)
        out = AbsorptionDetector().compute(trades)
        m = evaluate_detector(out, gt)
        # Recall: catch the majority; precision: don't drown in confounders
        assert m["recall"] >= 0.55, f"recall too low: {m['recall']:.2f}"
        assert m["precision"] >= 0.65, f"precision too low: {m['precision']:.2f}"
        assert m["tp"] > 0 and m["n_absorptions"] == 40

    def test_recall_rises_with_absorption_strength(self):
        """Sensitivity curve: recall in the HIGHEST volume/excursion bucket
        must exceed the LOWEST. Strongly-pinned absorption is easier to
        detect than weakly-pinned (which blends toward trend)."""
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=42))
        trades, gt = sim.generate(n_absorptions=80)
        out = AbsorptionDetector().compute(trades)
        curve = recall_by_ratio_bucket(out, gt)
        valid = curve.dropna(subset=["recall"])
        assert len(valid) >= 2
        assert valid["recall"].iloc[-1] >= valid["recall"].iloc[0], (
            f"recall should rise with ratio:\n{curve.to_string(index=False)}"
        )

    def test_baseline_quieter_than_absorption(self):
        """The z-score reference (baseline) must fire far less often than
        genuine absorption — otherwise the threshold is meaningless."""
        sim = AbsorptionGroundTruthSimulator(AbsorptionSimConfig(seed=123))
        trades, gt = sim.generate(n_absorptions=40)
        out = AbsorptionDetector().compute(trades)
        m = evaluate_detector(out, gt)
        assert m["baseline_fire_rate"] < m["recall"], (
            f"baseline fire {m['baseline_fire_rate']:.2f} should be below "
            f"absorption recall {m['recall']:.2f}"
        )
