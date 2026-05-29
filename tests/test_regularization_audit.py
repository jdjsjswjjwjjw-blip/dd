"""Tests for the SSL backbone dropout audit.

The audit has four behavioural contracts:
  1. Each sub-config maps to its expected audit path(s).
  2. Below-floor dropout → "fail" severity.
  3. ContextEncoder gets the stricter "warn" band between floor and
     CONTEXT_DROPOUT_RECOMMENDED.
  4. verify_dropout_floor returns (passed, violations) suitable for
     piping into a CLI smoke check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.deep_lob.config import (
    BarLSTMConfig,
    ContextEncoderConfig,
    MultiTaskHeadsConfig,
    OrderEmbedderConfig,
    TransformerConfig,
)
from modules.deep_lob.regularization_audit import (
    CONTEXT_DROPOUT_RECOMMENDED,
    MIN_DROPOUT_FLOOR,
    AuditEntry,
    audit_dropout,
    verify_dropout_floor,
)


# ════════════════════════════════════════════════════════════════════
# Thresholds — sanity
# ════════════════════════════════════════════════════════════════════
class TestThresholds:
    def test_recommended_above_floor(self):
        assert CONTEXT_DROPOUT_RECOMMENDED > MIN_DROPOUT_FLOOR

    def test_floor_in_reasonable_range(self):
        # Mild dropout, not silly low or silly high
        assert 0.05 <= MIN_DROPOUT_FLOOR <= 0.20


# ════════════════════════════════════════════════════════════════════
# Per-sub-config presence
# ════════════════════════════════════════════════════════════════════
class TestAuditPaths:
    def test_default_configs_audit_ok(self):
        # All defaults are 0.1 → exactly at the floor → pass
        entries = audit_dropout(
            order_embedder=OrderEmbedderConfig(),
            transformer=TransformerConfig(),
            bar_lstm=BarLSTMConfig(),
            context_encoder=ContextEncoderConfig(),
            multi_task_heads=MultiTaskHeadsConfig(),
        )
        # 1 (oe) + 2 (tx) + 1 (lstm) + 1 (ctx) + 1 (mth) = 6 entries
        assert len(entries) == 6
        # Context encoder at 0.10 → between floor and recommended → warn
        ctx_entries = [e for e in entries if e.path.startswith("context_encoder")]
        assert ctx_entries[0].severity in ("ok", "warn")
        # Everything else at the floor → ok
        non_ctx = [e for e in entries if not e.path.startswith("context_encoder")]
        for e in non_ctx:
            assert e.severity == "ok", f"{e.path} → {e.severity}: {e.note}"

    def test_only_supplied_configs_audited(self):
        # Pass only transformer → exactly 2 entries
        entries = audit_dropout(transformer=TransformerConfig())
        assert len(entries) == 2
        assert {e.path for e in entries} == {
            "transformer.dropout", "transformer.attention_dropout",
        }

    def test_empty_call_returns_empty(self):
        assert audit_dropout() == []


# ════════════════════════════════════════════════════════════════════
# Severity logic
# ════════════════════════════════════════════════════════════════════
class TestSeverity:
    def test_below_floor_is_fail(self):
        cfg = TransformerConfig(dropout=0.0, attention_dropout=0.05)
        entries = audit_dropout(transformer=cfg)
        for e in entries:
            assert e.severity == "fail", f"{e.path} should be fail"

    def test_above_floor_is_ok(self):
        cfg = TransformerConfig(dropout=0.15, attention_dropout=0.15)
        entries = audit_dropout(transformer=cfg)
        for e in entries:
            assert e.severity == "ok"

    def test_context_encoder_floor_above_min_is_warn(self):
        # 0.10 → exactly floor → between floor and recommended → warn
        cfg = ContextEncoderConfig(dropout=0.10)
        entries = audit_dropout(context_encoder=cfg)
        assert entries[0].severity == "warn"
        assert entries[0].recommended == CONTEXT_DROPOUT_RECOMMENDED

    def test_context_encoder_at_recommended_is_ok(self):
        cfg = ContextEncoderConfig(dropout=CONTEXT_DROPOUT_RECOMMENDED)
        entries = audit_dropout(context_encoder=cfg)
        assert entries[0].severity == "ok"

    def test_context_encoder_below_floor_is_fail(self):
        cfg = ContextEncoderConfig(dropout=0.05)
        entries = audit_dropout(context_encoder=cfg)
        assert entries[0].severity == "fail"
        assert "floor" in entries[0].note

    def test_severity_carries_useful_note(self):
        cfg = TransformerConfig(dropout=0.0)
        entries = audit_dropout(transformer=cfg)
        violation = [e for e in entries if e.severity == "fail"][0]
        assert "floor" in violation.note
        assert "memorisation risk" in violation.note


# ════════════════════════════════════════════════════════════════════
# verify_dropout_floor
# ════════════════════════════════════════════════════════════════════
class TestVerifyDropoutFloor:
    def test_default_passes(self):
        ok, violations = verify_dropout_floor(
            transformer=TransformerConfig(),
            context_encoder=ContextEncoderConfig(),
        )
        assert ok is True
        assert violations == []

    def test_below_floor_fails(self):
        ok, violations = verify_dropout_floor(
            transformer=TransformerConfig(dropout=0.05),
        )
        assert ok is False
        assert len(violations) == 1
        assert "transformer.dropout" in violations[0]
        assert "0.050" in violations[0]

    def test_multiple_failures_collected(self):
        ok, violations = verify_dropout_floor(
            transformer=TransformerConfig(dropout=0.0, attention_dropout=0.0),
            context_encoder=ContextEncoderConfig(dropout=0.0),
        )
        assert ok is False
        assert len(violations) == 3

    def test_warn_severity_does_not_block(self):
        # Context encoder at exactly the floor → warn, not fail →
        # verify still passes
        ok, violations = verify_dropout_floor(
            context_encoder=ContextEncoderConfig(dropout=MIN_DROPOUT_FLOOR),
        )
        assert ok is True


# ════════════════════════════════════════════════════════════════════
# AuditEntry dataclass
# ════════════════════════════════════════════════════════════════════
class TestAuditEntry:
    def test_to_dict_serialises(self):
        import json
        e = AuditEntry(
            path="x.y", value=0.1, recommended=0.2,
            severity="warn", note="example",
        )
        s = json.dumps(e.to_dict())
        assert "x.y" in s and "warn" in s

    def test_frozen(self):
        e = AuditEntry(path="x", value=0.1, recommended=0.2, severity="ok")
        with pytest.raises(Exception):
            e.path = "y"
