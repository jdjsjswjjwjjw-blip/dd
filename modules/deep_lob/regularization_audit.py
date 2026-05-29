"""
modules.deep_lob.regularization_audit
═══════════════════════════════════════════════════════════════════════════════
Audit + enforce dropout settings across every layer of the SSL backbone.

Why this exists
───────────────
The SSL backbone has dropout knobs in EIGHT places (order_embedder,
two transformer dropouts, bar_lstm, context_encoder, lob_image_encoder,
multi_task_heads, etc.). All default to 0.1. One careless edit that
sets `dropout=0.0` anywhere down the stack would silently kill the
regularization signal — the operator might never notice, and the
model would happily memorise the hawkes/absorption/kyle signal that
directly drives the time_to_event SSL target (the documented
correlation risk).

This module turns "is dropout enabled?" from an implicit assumption
into a unit-testable, automatable check.

Two thresholds
──────────────
MIN_DROPOUT_FLOOR             0.10 — anything below this on ANY
                                     dropout knob is a hard violation.
CONTEXT_DROPOUT_RECOMMENDED   0.20 — stricter recommendation for the
                                     ContextEncoder specifically,
                                     because that's where the 138
                                     causal context features
                                     (hawkes/absorb/kyle/regime/seasonal)
                                     enter the encoder. Memorisation
                                     risk concentrates here.

Public API
──────────
audit_dropout(config)        → list of AuditEntry per knob
verify_dropout_floor(config) → (passed: bool, violations: list[str])
"""
from __future__ import annotations

from dataclasses import dataclass

from modules.deep_lob.config import (
    BarLSTMConfig,
    ContextEncoderConfig,
    EventAggregatorConfig,
    MultiTaskHeadsConfig,
    OrderEmbedderConfig,
    TransformerConfig,
)


# ── Thresholds ──────────────────────────────────────────────────────
MIN_DROPOUT_FLOOR: float = 0.10
CONTEXT_DROPOUT_RECOMMENDED: float = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class AuditEntry:
    """One dropout knob's audit result."""
    path: str               # e.g. "context_encoder.dropout"
    value: float            # the current setting
    recommended: float      # what we think it should be
    severity: str           # "ok" | "warn" | "fail"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "value": float(self.value),
            "recommended": float(self.recommended),
            "severity": self.severity,
            "note": self.note,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Per-config-class extractors
# ══════════════════════════════════════════════════════════════════════════════
def _extract_order_embedder(cfg: OrderEmbedderConfig | None) -> list[AuditEntry]:
    if cfg is None:
        return []
    return [_classify(
        path="order_embedder.dropout",
        value=float(cfg.dropout),
        recommended=MIN_DROPOUT_FLOOR,
    )]


def _extract_transformer(cfg: TransformerConfig | None) -> list[AuditEntry]:
    if cfg is None:
        return []
    return [
        _classify(
            path="transformer.dropout",
            value=float(cfg.dropout),
            recommended=MIN_DROPOUT_FLOOR,
        ),
        _classify(
            path="transformer.attention_dropout",
            value=float(cfg.attention_dropout),
            recommended=MIN_DROPOUT_FLOOR,
        ),
    ]


def _extract_event_aggregator(
    cfg: EventAggregatorConfig | None,
) -> list[AuditEntry]:
    if cfg is None or not hasattr(cfg, "dropout"):
        return []
    return [_classify(
        path="event_aggregator.dropout",
        value=float(cfg.dropout),
        recommended=MIN_DROPOUT_FLOOR,
    )]


def _extract_bar_lstm(cfg: BarLSTMConfig | None) -> list[AuditEntry]:
    if cfg is None:
        return []
    return [_classify(
        path="bar_lstm.dropout",
        value=float(cfg.dropout),
        recommended=MIN_DROPOUT_FLOOR,
    )]


def _extract_context_encoder(
    cfg: ContextEncoderConfig | None,
) -> list[AuditEntry]:
    """ContextEncoder gets the STRICTER recommendation because that's
    where hawkes/absorption/kyle features enter — the columns most
    correlated with the time_to_event SSL target."""
    if cfg is None:
        return []
    value = float(cfg.dropout)
    if value < MIN_DROPOUT_FLOOR:
        return [_classify(
            path="context_encoder.dropout",
            value=value,
            recommended=CONTEXT_DROPOUT_RECOMMENDED,
        )]
    if value < CONTEXT_DROPOUT_RECOMMENDED:
        return [AuditEntry(
            path="context_encoder.dropout",
            value=value,
            recommended=CONTEXT_DROPOUT_RECOMMENDED,
            severity="warn",
            note=(
                f"above the {MIN_DROPOUT_FLOOR:.2f} floor but below the "
                f"{CONTEXT_DROPOUT_RECOMMENDED:.2f} recommended for the "
                f"context encoder — hawkes/absorb/kyle memorisation risk"
            ),
        )]
    return [AuditEntry(
        path="context_encoder.dropout",
        value=value,
        recommended=CONTEXT_DROPOUT_RECOMMENDED,
        severity="ok",
    )]


def _extract_multi_task_heads(
    cfg: MultiTaskHeadsConfig | None,
) -> list[AuditEntry]:
    if cfg is None:
        return []
    return [_classify(
        path="multi_task_heads.dropout",
        value=float(cfg.dropout),
        recommended=MIN_DROPOUT_FLOOR,
    )]


def _classify(path: str, value: float, recommended: float) -> AuditEntry:
    """Pick severity and note from value vs floor + recommended."""
    if value < MIN_DROPOUT_FLOOR:
        return AuditEntry(
            path=path,
            value=value,
            recommended=max(recommended, MIN_DROPOUT_FLOOR),
            severity="fail",
            note=(
                f"below the {MIN_DROPOUT_FLOOR:.2f} hard floor — "
                f"raise or memorisation risk dominates"
            ),
        )
    return AuditEntry(
        path=path, value=value, recommended=recommended,
        severity="ok",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def audit_dropout(
    *,
    order_embedder: OrderEmbedderConfig | None = None,
    transformer: TransformerConfig | None = None,
    event_aggregator: EventAggregatorConfig | None = None,
    bar_lstm: BarLSTMConfig | None = None,
    context_encoder: ContextEncoderConfig | None = None,
    multi_task_heads: MultiTaskHeadsConfig | None = None,
) -> list[AuditEntry]:
    """Inspect every dropout knob in a constellation of SSL sub-configs.

    Pass only the sub-configs you actually use. Each missing sub-config
    contributes zero entries — the caller's result list is exactly the
    set of audited paths. Pure: no side effects, no IO."""
    entries: list[AuditEntry] = []
    entries.extend(_extract_order_embedder(order_embedder))
    entries.extend(_extract_transformer(transformer))
    entries.extend(_extract_event_aggregator(event_aggregator))
    entries.extend(_extract_bar_lstm(bar_lstm))
    entries.extend(_extract_context_encoder(context_encoder))
    entries.extend(_extract_multi_task_heads(multi_task_heads))
    return entries


def verify_dropout_floor(
    *,
    order_embedder: OrderEmbedderConfig | None = None,
    transformer: TransformerConfig | None = None,
    event_aggregator: EventAggregatorConfig | None = None,
    bar_lstm: BarLSTMConfig | None = None,
    context_encoder: ContextEncoderConfig | None = None,
    multi_task_heads: MultiTaskHeadsConfig | None = None,
) -> tuple[bool, list[str]]:
    """Returns (passed, violation_messages).

    `passed` is True iff every audited dropout knob is at or above
    MIN_DROPOUT_FLOOR. Designed to plug into the smoke verifier:

        ok, violations = verify_dropout_floor(transformer=cfg.transformer, ...)
        if not ok:
            for v in violations:
                print(f"  🚨 {v}")
            sys.exit(1)
    """
    entries = audit_dropout(
        order_embedder=order_embedder,
        transformer=transformer,
        event_aggregator=event_aggregator,
        bar_lstm=bar_lstm,
        context_encoder=context_encoder,
        multi_task_heads=multi_task_heads,
    )
    failures = [
        f"{e.path} = {e.value:.3f} (floor {MIN_DROPOUT_FLOOR:.2f}) — {e.note}"
        for e in entries if e.severity == "fail"
    ]
    return len(failures) == 0, failures
