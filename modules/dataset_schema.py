"""C2 — dataset schema as a single source of truth.

The refinery's parquet ships with three classes of label-side columns:

  1. `bias_label` — directional bias target (Phase 0 + A1 + A2).
     Used by: SSL direction probe / decision policy.
  2. `exec_label`/`exec_path`/`exec_valid` — strict execution target (B1).
     Used by: the execution head, with exec_valid as the loss mask.
  3. `next_price_delta`/`next_price_delta_valid` — continuous SSL
     directional target (B2). Used by the SSL backbone, gate-free.

Plus multi-task diagnostics (mfe/mae/stop_first_flag/…) that are
forward-looking but not training targets.

Before C2, each consumer had to know these names by heart and re-derive
which were leakage and which were features. This module gives every
consumer one place to look — and ships a `dataset_meta.json` sidecar next
to the parquet so external readers can introspect without importing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd


# ── role tags (semantic, not enforced) ──────────────────────────────────────
ROLE_DIRECTIONAL_BIAS = "directional_bias"
ROLE_EXECUTION_STRICT = "execution_strict"
ROLE_SSL_CONTINUOUS = "ssl_continuous"
ROLE_DIAGNOSTIC = "diagnostic"

# ── leakage classes (matched against data_loader._is_leakage_column) ───────
LEAKAGE_TARGET = "target"        # this IS a training target → must be masked
LEAKAGE_FORWARD = "forward"      # forward-looking but not a target
LEAKAGE_NONE = "none"            # past-only / causal (a feature)


@dataclass(frozen=True)
class LabelSpec:
    """One row of the schema. Frozen so consumers can't mutate it."""
    name: str
    dtype: str                       # numpy dtype short code: 'int8'/'float32'/'bool'
    role: str                        # ROLE_* constant
    leakage_class: str               # LEAKAGE_* constant
    valid_col: str | None = None     # name of the matching *_valid mask, if any
    description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# THE SCHEMA — one place, one definition. Add a column here when the
# refinery starts writing it.
# ════════════════════════════════════════════════════════════════════════════

# Phase 0 + A1 + A2: directional bias target (label_by_outcome)
BIAS_SCHEMA: LabelSpec = LabelSpec(
    name="bias_label",
    dtype="int8",
    role=ROLE_DIRECTIONAL_BIAS,
    leakage_class=LEAKAGE_TARGET,
    valid_col=None,                  # bias is always defined; NEUTRAL (=2) is the no-trade value
    description=(
        "0=LONG, 1=SHORT, 2=NEUTRAL. Produced by label_by_outcome with the "
        "A1 symmetric single-pair barrier scan and A2 veto-off default; "
        "carries an MFE/MAE rescue on timeout so the directional bias "
        "signal is not held hostage to a TP touch."
    ),
)

# B1: strict execution target (label_engine_v2 with rescue OFF + session-break mask)
EXEC_SCHEMA: LabelSpec = LabelSpec(
    name="exec_label",
    dtype="int8",
    role=ROLE_EXECUTION_STRICT,
    leakage_class=LEAKAGE_TARGET,
    valid_col="exec_valid",
    description=(
        "0=LONG, 1=SHORT, 2=NEUTRAL. MT5-faithful: first barrier touch is "
        "frozen, otherwise NEUTRAL. The exec_valid mask is True iff a "
        "hard barrier was touched AND the forward window did not cross "
        "an is_session_break."
    ),
)
EXEC_PATH_SCHEMA: LabelSpec = LabelSpec(
    name="exec_path",
    dtype="int8",
    role=ROLE_EXECUTION_STRICT,
    leakage_class=LEAKAGE_FORWARD,   # not a target itself; diagnostic of exec_label
    description=(
        "label_engine_v2 PATH_* code that produced exec_label. Rescue "
        "codes never appear (rescue_on_timeout=False)."
    ),
)
EXEC_VALID_SCHEMA: LabelSpec = LabelSpec(
    name="exec_valid",
    dtype="bool",
    role=ROLE_EXECUTION_STRICT,
    leakage_class=LEAKAGE_FORWARD,
    description="Per-bar loss mask for the execution head (see exec_label).",
)

# B2: continuous SSL directional target (gate-free, H-bar log-return)
SSL_DIRECTIONAL_SCHEMA: LabelSpec = LabelSpec(
    name="next_price_delta",
    dtype="float32",
    role=ROLE_SSL_CONTINUOUS,
    leakage_class=LEAKAGE_TARGET,
    valid_col="next_price_delta_valid",
    description=(
        "log(close[i+H] / close[i]) over the label horizon. Gate-free: "
        "every bar gets a value (dead chop → microscopic delta). "
        "next_price_delta_valid is False at right-edge or when the "
        "window crosses a session break."
    ),
)
SSL_VALID_SCHEMA: LabelSpec = LabelSpec(
    name="next_price_delta_valid",
    dtype="bool",
    role=ROLE_SSL_CONTINUOUS,
    leakage_class=LEAKAGE_FORWARD,
    description="Per-bar mask for next_price_delta.",
)

# Multi-task diagnostics (compute_multitask_label_diagnostics, not training
# targets but forward-looking — listed here for completeness + leakage audit).
DIAGNOSTIC_SCHEMAS: tuple[LabelSpec, ...] = (
    LabelSpec("mfe", "float32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="Max favourable excursion over horizon_bars*4 window."),
    LabelSpec("mae", "float32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="Max adverse excursion over horizon_bars*4 window."),
    LabelSpec("mfe_atr", "float32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="mfe normalized by ATR."),
    LabelSpec("mae_atr", "float32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="mae normalized by ATR."),
    LabelSpec("stop_first_flag", "int8", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="1 if MAE>1ATR crossed before MFE>1ATR in the forward window."),
    LabelSpec("time_to_first_touch", "int32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="Bars to first 0.5-ATR touch in the forward window."),
    LabelSpec("net_expectancy_proxy", "float32", ROLE_DIAGNOSTIC, LEAKAGE_FORWARD,
              description="Net expected gain after spread+fees (in ATR units)."),
)

ALL_LABEL_SCHEMAS: tuple[LabelSpec, ...] = (
    BIAS_SCHEMA,
    EXEC_SCHEMA, EXEC_PATH_SCHEMA, EXEC_VALID_SCHEMA,
    SSL_DIRECTIONAL_SCHEMA, SSL_VALID_SCHEMA,
    *DIAGNOSTIC_SCHEMAS,
)

# Convenience: schema-version tag updated whenever ALL_LABEL_SCHEMAS changes.
# Bump when adding/removing/renaming entries above so manifests + sidecars
# can be matched against a known version.
SCHEMA_VERSION: str = "phase1-c2"

# Pipeline-cleanup phase markers — embed in the sidecar so consumers can
# tell which fixes were live when the parquet was produced.
PIPELINE_CLEANUP_PHASES: dict[str, str] = {
    "phase0": "event gate removed; obi→obi_net + cvd_cumulative alias",
    "A1": "symmetric single-pair barrier scan; short_tp reachable",
    "A2": "event_direction veto off by default (apply_event_direction_veto)",
    "A3": "build_day_trading_labels (dead) removed",
    "B1": "compute_strict_execution_labels (exec_*) wired",
    "B2": "compute_ssl_directional_target (next_price_delta) wired",
    "C1": "SSL leakage guard extended for new + previously-uncovered cols",
    "C2": "dataset_schema source of truth + dataset_meta.json sidecar",
}


# ── helpers ─────────────────────────────────────────────────────────────────
def specs_by_role(role: str) -> tuple[LabelSpec, ...]:
    return tuple(s for s in ALL_LABEL_SCHEMAS if s.role == role)


def specs_by_leakage(leakage_class: str) -> tuple[LabelSpec, ...]:
    return tuple(s for s in ALL_LABEL_SCHEMAS if s.leakage_class == leakage_class)


def target_column_names() -> tuple[str, ...]:
    """Names of training-target columns (LEAKAGE_TARGET). The set the SSL
    backbone, train_hybrid, and walk_forward must NEVER expose as features.
    """
    return tuple(s.name for s in ALL_LABEL_SCHEMAS if s.leakage_class == LEAKAGE_TARGET)


def build_dataset_meta(
    df: pd.DataFrame,
    *,
    extra: dict | None = None,
) -> dict:
    """Return the dict written to `dataset_meta.json` next to the parquet.

    Includes the full schema, a present/absent audit for each spec'd column
    on the actual frame, and any extra contract fields the refinery passes
    in.
    """
    columns_present = []
    columns_missing = []
    dtype_mismatches = []
    for spec in ALL_LABEL_SCHEMAS:
        if spec.name in df.columns:
            columns_present.append(spec.name)
            # Cheap dtype check (kind-level)
            try:
                want = np.dtype(spec.dtype).kind
                got = df[spec.name].dtype.kind
                if want != got:
                    dtype_mismatches.append({
                        "column": spec.name, "want": spec.dtype,
                        "got": str(df[spec.name].dtype),
                    })
            except Exception:
                pass
        else:
            columns_missing.append(spec.name)

    meta: dict = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_cleanup_phases": PIPELINE_CLEANUP_PHASES,
        "label_specs": [s.as_dict() for s in ALL_LABEL_SCHEMAS],
        "target_columns": list(target_column_names()),
        "columns_present": columns_present,
        "columns_missing": columns_missing,
        "dtype_mismatches": dtype_mismatches,
        "rows": int(len(df)),
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_dataset_meta(
    df: pd.DataFrame,
    parquet_path: str | Path,
    *,
    extra: dict | None = None,
) -> Path:
    """Write `<parquet_dir>/dataset_meta.json` next to the parquet."""
    p = Path(parquet_path)
    out_path = p.parent / "dataset_meta.json"
    meta = build_dataset_meta(df, extra=extra)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_path


def load_dataset_meta(parquet_path: str | Path) -> dict:
    """Read the sidecar next to a parquet. Raises FileNotFoundError if
    missing — callers should treat that as a contract violation."""
    p = Path(parquet_path)
    meta_path = p.parent / "dataset_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"dataset_meta.json sidecar missing next to {parquet_path}; "
            f"the parquet was likely produced before C2."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def assert_schema(df: pd.DataFrame, *, required_roles: Iterable[str] = ()) -> None:
    """Raise AssertionError if any spec for a required role is missing
    from the frame. Use in tests / preflight checks.
    """
    missing: list[str] = []
    for role in required_roles:
        for spec in specs_by_role(role):
            if spec.name not in df.columns:
                missing.append(f"{spec.name} (role={role})")
    if missing:
        raise AssertionError(
            "dataset_schema: missing required columns: " + ", ".join(missing)
        )
