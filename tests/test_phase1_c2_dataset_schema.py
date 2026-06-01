"""Tests for Phase 1 / Commit C2 — dataset schema source of truth.

C2 introduces modules/dataset_schema.py: one place that defines every
label-side column the refinery writes — its dtype, semantic role, leakage
class, paired valid mask. A dataset_meta.json sidecar is written next to
the parquet so external consumers can introspect without importing.

These tests pin down the schema's internal consistency, the sidecar
round-trip, and the leakage-class agreement with the data_loader guard.
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
    ALL_LABEL_SCHEMAS,
    BIAS_SCHEMA, EXEC_SCHEMA, EXEC_VALID_SCHEMA,
    SSL_DIRECTIONAL_SCHEMA, SSL_VALID_SCHEMA,
    LEAKAGE_TARGET, LEAKAGE_FORWARD, LEAKAGE_NONE,
    ROLE_DIRECTIONAL_BIAS, ROLE_EXECUTION_STRICT, ROLE_SSL_CONTINUOUS,
    ROLE_DIAGNOSTIC,
    SCHEMA_VERSION, PIPELINE_CLEANUP_PHASES,
    LabelSpec,
    specs_by_role, specs_by_leakage, target_column_names,
    build_dataset_meta, write_dataset_meta, load_dataset_meta,
    assert_schema,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _frame_with_targets() -> pd.DataFrame:
    n = 5
    return pd.DataFrame({
        "bias_label": np.zeros(n, dtype=np.int8),
        "exec_label": np.zeros(n, dtype=np.int8),
        "exec_path": np.zeros(n, dtype=np.int8),
        "exec_valid": np.zeros(n, dtype=bool),
        "next_price_delta": np.zeros(n, dtype=np.float32),
        "next_price_delta_valid": np.zeros(n, dtype=bool),
        # features (not in schema)
        "close": np.ones(n, dtype=np.float64),
        "atr_14": np.ones(n, dtype=np.float64) * 0.001,
    })


# ── schema internal consistency ────────────────────────────────────────────
class TestSchemaInternalConsistency:
    def test_all_specs_have_required_fields(self):
        for s in ALL_LABEL_SCHEMAS:
            assert isinstance(s, LabelSpec)
            assert s.name and s.dtype and s.role and s.leakage_class

    def test_names_unique(self):
        names = [s.name for s in ALL_LABEL_SCHEMAS]
        assert len(names) == len(set(names)), "duplicate spec names"

    def test_valid_cols_point_to_real_specs(self):
        # If a spec declares valid_col=X, X must itself be in the schema
        names = {s.name for s in ALL_LABEL_SCHEMAS}
        for s in ALL_LABEL_SCHEMAS:
            if s.valid_col is not None:
                assert s.valid_col in names, (
                    f"{s.name} declares valid_col={s.valid_col!r} which is "
                    f"not in the schema"
                )

    def test_specs_are_frozen(self):
        # frozen dataclass — mutation should raise
        import dataclasses
        try:
            BIAS_SCHEMA.name = "tampered"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("LabelSpec is not frozen — schema can be mutated")


# ── role / leakage filters ──────────────────────────────────────────────────
class TestRoleAndLeakageFilters:
    def test_directional_bias_role(self):
        names = {s.name for s in specs_by_role(ROLE_DIRECTIONAL_BIAS)}
        assert names == {"bias_label"}

    def test_execution_strict_role(self):
        names = {s.name for s in specs_by_role(ROLE_EXECUTION_STRICT)}
        assert names == {"exec_label", "exec_path", "exec_valid"}

    def test_ssl_continuous_role(self):
        names = {s.name for s in specs_by_role(ROLE_SSL_CONTINUOUS)}
        assert names == {"next_price_delta", "next_price_delta_valid"}

    def test_target_column_names_set(self):
        targets = set(target_column_names())
        # Training targets the SSL backbone must never see as features
        assert "bias_label" in targets
        assert "exec_label" in targets
        assert "next_price_delta" in targets
        # mask columns are forward-looking but not targets themselves
        assert "exec_valid" not in targets
        assert "next_price_delta_valid" not in targets


# ── agreement with the SSL leakage guard ────────────────────────────────────
class TestAgreementWithDataLoaderGuard:
    """Every LEAKAGE_TARGET / LEAKAGE_FORWARD spec must also be flagged
    by the data_loader's _is_leakage_column. If they ever diverge, one
    or the other is wrong — fail loudly here."""

    def test_targets_all_excluded_by_data_loader(self):
        from self_supervised.data_loader import _is_leakage_column
        for s in specs_by_leakage(LEAKAGE_TARGET):
            assert _is_leakage_column(s.name), (
                f"target {s.name!r} not excluded by data_loader guard"
            )

    def test_forward_diagnostics_all_excluded_by_data_loader(self):
        from self_supervised.data_loader import _is_leakage_column
        for s in specs_by_leakage(LEAKAGE_FORWARD):
            assert _is_leakage_column(s.name), (
                f"forward diagnostic {s.name!r} not excluded by data_loader "
                f"guard"
            )


# ── sidecar build / write / load ────────────────────────────────────────────
class TestDatasetMetaSidecar:
    def test_build_includes_schema_version_and_specs(self):
        df = _frame_with_targets()
        meta = build_dataset_meta(df)
        assert meta["schema_version"] == SCHEMA_VERSION
        assert "label_specs" in meta and len(meta["label_specs"]) > 0
        # All schema names should appear in label_specs
        names = {s["name"] for s in meta["label_specs"]}
        for spec in ALL_LABEL_SCHEMAS:
            assert spec.name in names

    def test_build_audits_present_vs_missing(self):
        df = _frame_with_targets()
        meta = build_dataset_meta(df)
        # bias_label is in the frame; mfe is not
        # Phase 1.1 renamed these to disambiguate label vs feature audits
        assert "bias_label" in meta["label_columns_present"]
        assert "mfe" in meta["label_columns_missing"]

    def test_build_records_dtype_mismatch(self):
        df = _frame_with_targets()
        # tamper bias_label dtype to float64 instead of int8 → mismatch logged
        df["bias_label"] = df["bias_label"].astype(np.float64)
        meta = build_dataset_meta(df)
        mismatched = {m["column"] for m in meta["dtype_mismatches"]}
        assert "bias_label" in mismatched

    def test_write_and_load_round_trip(self, tmp_path):
        df = _frame_with_targets()
        parquet_path = tmp_path / "fold_2025Q2" / "features.parquet"
        parquet_path.parent.mkdir(parents=True)
        df.to_parquet(parquet_path)
        sidecar = write_dataset_meta(df, parquet_path, extra={"horizon_bars": 6})
        assert sidecar.exists() and sidecar.name == "dataset_meta.json"
        loaded = load_dataset_meta(parquet_path)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["extra"]["horizon_bars"] == 6
        assert "bias_label" in loaded["label_columns_present"]

    def test_load_missing_sidecar_raises(self, tmp_path):
        parquet_path = tmp_path / "features.parquet"
        parquet_path.touch()
        try:
            load_dataset_meta(parquet_path)
        except FileNotFoundError as e:
            assert "dataset_meta.json" in str(e)
            return
        raise AssertionError("load_dataset_meta should have raised")


# ── assert_schema preflight helper ──────────────────────────────────────────
class TestAssertSchema:
    def test_pass_when_role_present(self):
        df = _frame_with_targets()
        assert_schema(df, required_roles=[ROLE_DIRECTIONAL_BIAS, ROLE_EXECUTION_STRICT])

    def test_raise_when_role_missing(self):
        df = pd.DataFrame({"close": [1.0]})
        try:
            assert_schema(df, required_roles=[ROLE_EXECUTION_STRICT])
        except AssertionError as e:
            assert "exec_label" in str(e)
            return
        raise AssertionError("assert_schema should have raised")


# ── phase marker is up to date ──────────────────────────────────────────────
class TestPipelineCleanupPhases:
    def test_all_phases_documented(self):
        for ph in ("phase0", "A1", "A2", "A3", "B1", "B2", "C1", "C2"):
            assert ph in PIPELINE_CLEANUP_PHASES, f"missing phase doc: {ph}"
