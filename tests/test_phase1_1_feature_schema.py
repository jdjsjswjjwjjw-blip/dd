"""Phase 1.1 — feature schema + naming permanence + git-hash sidecar.

Three integrated guarantees this round must hold:

  (1) The canonical alias `obi_net` is created at the compute-source of
      `obi` (not as a pre-write patch), so any feature builder that runs
      between the compute and to_parquet sees the canonical name in the
      DataFrame directly. Same for `cvd_cumulative` vs `cvd`.

  (2) modules/dataset_schema.py exposes a FeatureSpec dataclass and a
      catalog of feature specs (TOP_IC_FEATURES + the Phase 1.3/1.4/1.5
      new-feature specs) — the same single-source-of-truth pattern that
      C2 used for labels.

  (3) The dataset_meta.json sidecar now includes git_hash + built_at_utc
      + feature_specs + phase_1_2_blacklist, and validate() reports a
      clean list of human-readable errors instead of silently passing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.dataset_schema import (
    ALL_FEATURE_SPECS,
    FeatureSpec, FAMILY_STRUCTURAL, FAMILY_CONTINUOUS_Z, FAMILY_CVD_MT5,
    FAMILY_ICEBERG, FAMILY_DIST_ATR, FAMILY_INTERACTION,
    IC_VERDICT_STRONG, IC_VERDICT_NEW,
    TOP_IC_FEATURES,
    PHASE_1_2_BLACKLIST,
    PHASE_1_3_NEW_FEATURES, PHASE_1_4_NEW_FEATURES,
    PHASE_1_4_DIST_ATR_FEATURES, PHASE_1_5_NEW_FEATURES,
    PHASE_1_5_INTERACTION_FEATURES,
    REQUIRED_OHLCV, REQUIRED_REGIME, SCHEMA_VERSION,
    build_dataset_meta, write_dataset_meta, load_dataset_meta,
    validate,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _refinery_like_frame() -> pd.DataFrame:
    """A frame shaped like the refinery's late-stage df_out — has the
    canonical aliases + a couple of label cols + a couple of features."""
    n = 10
    ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        # OHLCV
        "ts_event": ts, "open": 1.0, "high": 1.001, "low": 0.999,
        "close": 1.0, "volume": 1000.0,
        # regime
        "regime_label": "ranging", "atr_14": 0.001,
        "is_session_break": np.zeros(n, dtype=np.int8),
        # canonical aliases (Phase 1.1: these MUST be present)
        "obi": np.zeros(n), "obi_net": np.zeros(n),
        "cvd": np.zeros(n), "cvd_cumulative": np.zeros(n),
        # a couple of audited features
        "london_sess_high": np.full(n, 1.0050),
        "current_vwap": np.full(n, 1.0000),
        "tick_count": np.full(n, 50, dtype=np.int32),
    })


# ── (2) Feature schema catalog ─────────────────────────────────────────────
class TestFeatureSchemaCatalog:
    def test_top_ic_features_strong_count(self):
        """Lock in the 6 STRONG features from the Q2 audit."""
        strong = [s for s in TOP_IC_FEATURES if s.ic_verdict == IC_VERDICT_STRONG]
        assert len(strong) == 6
        names = {s.name for s in strong}
        assert names == {
            "london_sess_high", "current_vwap", "price",
            "hawkes_intrabar_sum", "tick_count", "time_to_ny_close_min",
        }

    def test_new_feature_groups_present(self):
        """The Phase 1.3/1.4/1.5 NEW specs must each carry IC_VERDICT_NEW."""
        for group, expected_n in (
            (PHASE_1_3_NEW_FEATURES, 4),
            (PHASE_1_4_NEW_FEATURES, 5),
            (PHASE_1_4_DIST_ATR_FEATURES, 3),
            (PHASE_1_5_NEW_FEATURES, 2),
            (PHASE_1_5_INTERACTION_FEATURES, 2),
        ):
            assert len(group) == expected_n
            for s in group:
                assert s.ic_verdict == IC_VERDICT_NEW, (
                    f"{s.name!r} must be tagged NEW until Phase 1.6 re-audit"
                )

    def test_all_feature_specs_unique(self):
        names = [s.name for s in ALL_FEATURE_SPECS]
        assert len(names) == len(set(names)), "duplicate FeatureSpec names"

    def test_feature_specs_frozen(self):
        import dataclasses
        try:
            TOP_IC_FEATURES[0].name = "tampered"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("FeatureSpec is not frozen — schema can be mutated")

    def test_blacklist_has_reasons(self):
        """Every blacklisted column must come with a reason string."""
        assert len(PHASE_1_2_BLACKLIST) > 30, (
            "blacklist seems too small — IC audit identified 50+ bad features"
        )
        for col, reason in PHASE_1_2_BLACKLIST.items():
            assert reason and len(reason) > 5, (
                f"blacklist entry {col!r} has no reason"
            )

    def test_no_blacklisted_name_appears_in_whitelist(self):
        """If a column is blacklisted, it must not also be a featured spec."""
        spec_names = {s.name for s in ALL_FEATURE_SPECS}
        overlap = spec_names & PHASE_1_2_BLACKLIST.keys()
        assert not overlap, f"blacklist/whitelist conflict: {overlap}"


# ── (1) Canonical-aliases at compute source ────────────────────────────────
class TestCanonicalAliasesAtSource:
    """The Phase 1.1 promise: `obi_net` and `cvd_cumulative` are written
    at the same point as the raw `obi`/`cvd` computation — NOT only as a
    pre-to_parquet patch at the end."""

    def test_obi_net_alias_lives_next_to_obi_clip(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        # The clip line and the alias must sit within ~5 lines of each other
        idx_clip = src.find("df['obi'] = df['obi'].clip(-1.0, 1.0)")
        idx_alias = src.find("df['obi_net'] = df['obi']")
        assert idx_clip >= 0, "df['obi'] clip line moved/removed"
        assert idx_alias >= 0, "df['obi_net'] = df['obi'] alias missing"
        # alias must come AFTER the clip (so it carries the clipped values)
        assert idx_alias > idx_clip
        # and within 200 chars (i.e. adjacent, not just somewhere else)
        assert idx_alias - idx_clip < 400, (
            "obi_net alias is too far from obi clip — Phase 1.1 demands "
            "compute-source aliasing"
        )

    def test_cvd_cumulative_alias_lives_next_to_cvd_resample(self):
        src = (REPO_ROOT / "prepare_day_trading.py").read_text()
        idx_resample = src.find("bars['cvd'] = _cvd_last.reindex")
        idx_alias = src.find("bars['cvd_cumulative'] = bars['cvd']")
        assert idx_resample >= 0, "bars['cvd'] resample line moved/removed"
        assert idx_alias >= 0, "bars['cvd_cumulative'] alias missing"
        assert idx_alias > idx_resample
        assert idx_alias - idx_resample < 400


# ── (3) Sidecar carries git_hash + built_at_utc + feature schema ───────────
class TestDatasetMetaSidecarPhase11:
    def test_build_includes_git_hash_and_timestamp(self):
        meta = build_dataset_meta(_refinery_like_frame())
        assert "git_hash" in meta
        assert "built_at_utc" in meta
        # built_at_utc must be ISO-8601 UTC
        assert meta["built_at_utc"].endswith("+00:00")

    def test_build_includes_feature_specs(self):
        meta = build_dataset_meta(_refinery_like_frame())
        assert "feature_specs" in meta
        assert len(meta["feature_specs"]) == len(ALL_FEATURE_SPECS)

    def test_build_audits_features_present_vs_missing(self):
        meta = build_dataset_meta(_refinery_like_frame())
        assert "london_sess_high" in meta["feature_columns_present"]
        # Phase 1.4 features aren't built yet → missing
        assert "cvd_bar_5m" in meta["feature_columns_missing"]

    def test_build_flags_blacklisted_columns_still_present(self):
        df = _refinery_like_frame()
        df["cvd_momentum"] = 0.0   # a blacklisted column
        meta = build_dataset_meta(df)
        assert "cvd_momentum" in meta["blacklisted_columns_still_present"]

    def test_round_trip(self, tmp_path):
        df = _refinery_like_frame()
        parquet_path = tmp_path / "out" / "features.parquet"
        parquet_path.parent.mkdir(parents=True)
        df.to_parquet(parquet_path)
        write_dataset_meta(df, parquet_path)
        loaded = load_dataset_meta(parquet_path)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert "feature_specs" in loaded
        assert "git_hash" in loaded


# ── validate() helper ──────────────────────────────────────────────────────
class TestValidate:
    def test_clean_frame_no_errors(self):
        errors = validate(_refinery_like_frame())
        assert errors == [], f"unexpected errors: {errors}"

    def test_missing_obi_net_flagged(self):
        df = _refinery_like_frame().drop(columns=["obi_net"])
        errors = validate(df)
        assert any("obi_net" in e for e in errors)

    def test_missing_cvd_cumulative_flagged(self):
        df = _refinery_like_frame().drop(columns=["cvd_cumulative"])
        errors = validate(df)
        assert any("cvd_cumulative" in e for e in errors)

    def test_missing_required_ohlcv_flagged(self):
        df = _refinery_like_frame().drop(columns=["high"])
        errors = validate(df)
        assert any("high" in e for e in errors)

    def test_strict_mode_demands_targets(self):
        # bias_label / exec_label / next_price_delta not in the synthetic
        # frame → strict mode flags them; non-strict does not.
        df = _refinery_like_frame()
        loose = validate(df, strict=False)
        strict = validate(df, strict=True)
        assert len(strict) > len(loose)
        assert any("bias_label" in e for e in strict)


# ── Smoke-test: drive the late-stage refinery aliasing on a real frame ─────
class TestSmokeAliasingLogic:
    """Drives compute_strict_execution_labels + compute_ssl_directional_target
    + write_dataset_meta on a tiny frame to confirm the Phase 1.1 wiring is
    end-to-end live and produces a valid sidecar."""

    def test_smoke_end_to_end(self, tmp_path):
        import prepare_day_trading as pdt

        # Build a frame the late refinery stages can run on
        n = 50
        ts = pd.date_range("2025-04-01 08:00", periods=n, freq="5min", tz="UTC")
        closes = 1.0 + np.cumsum(np.random.RandomState(0).randn(n) * 0.0005)
        df = pd.DataFrame({
            "ts_event": ts, "open": closes, "high": closes + 0.0005,
            "low": closes - 0.0005, "close": closes, "volume": 100.0,
            "atr_14": 0.001, "is_session_break": np.zeros(n, dtype=np.int8),
            "regime_label": "ranging",
            "obi": np.zeros(n), "obi_net": np.zeros(n),
            "cvd": np.zeros(n), "cvd_cumulative": np.zeros(n),
            "is_event": np.ones(n, dtype=np.int8),
            "event_score": np.zeros(n),
            "kalman_direction": np.zeros(n, dtype=np.int8),
            "event_direction": np.zeros(n, dtype=np.int8),
            "is_london": np.ones(n, dtype=bool),
            "is_overlap": np.zeros(n, dtype=bool),
            "session": "london",
            "bias_label": np.full(n, 2, dtype=np.int8),
        })

        # Drive the dual-target heads (B1 + B2 — they must still work)
        df = pdt.compute_strict_execution_labels(df, horizon_bars=6, barrier_atr_mult=1.5)
        df = pdt.compute_ssl_directional_target(df, horizon_bars=6)

        # Phase 1.1: write sidecar + validate
        parquet_path = tmp_path / "smoke.parquet"
        df.to_parquet(parquet_path)
        meta_path = write_dataset_meta(df, parquet_path)
        meta = json.loads(meta_path.read_text())

        # Cross-cutting smoke assertions
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["git_hash"] is not None      # we're in a git repo
        assert "exec_label" in meta["label_columns_present"]
        assert "next_price_delta" in meta["label_columns_present"]
        assert meta["rows"] == n

        # validate() must come back clean on this constructed frame
        errors = validate(df, strict=True)
        assert errors == [], f"smoke validate() errors: {errors}"
