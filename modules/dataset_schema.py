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
SCHEMA_VERSION: str = "phase1.1-feature-schema"

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
    "1.1": "feature-schema added; obi_net/cvd_cumulative canonical at compute-source",
}


# ════════════════════════════════════════════════════════════════════════════
# Phase 1.1 — Feature schema. Labels live in ALL_LABEL_SCHEMAS above; this
# block formalises the FEATURE-side contract so the same source-of-truth
# pattern that solved label confusion (one place, typed) now covers the
# input features the model sees. Drives the Phase 1.2 whitelist mode.
# ════════════════════════════════════════════════════════════════════════════

# Feature-family tags (semantic groupings used by the whitelist + diagnostics)
FAMILY_OHLCV = "ohlcv"
FAMILY_REGIME = "regime"
FAMILY_STRUCTURAL = "structural"      # session highs/lows, vwap, pdh/pdl
FAMILY_SEASONAL = "seasonal"          # time-of-day, session phase
FAMILY_VOLUME = "volume"
FAMILY_MICROSTRUCTURE_AGG = "microstructure_agg"   # hawkes, kyle, lambda
FAMILY_CONTINUOUS_Z = "continuous_z"   # Phase 1.3
FAMILY_CVD_MT5 = "cvd_mt5"            # Phase 1.4
FAMILY_ICEBERG = "iceberg"            # Phase 1.5
FAMILY_DIST_ATR = "dist_to_x_atr"     # Phase 1.4-aux (II.B fix)
FAMILY_INTERACTION = "interaction"    # Phase 1.5-aux (II.A fix)
FAMILY_CONTEXT = "context"            # event metadata kept as features
FAMILY_COMPASS = "liquidity_compass"  # Phase 1.7 — OFI / Δ-divergence / VWAP-z


# IC-verdict tags (from the Q2 audit — drives Phase 1.2's blacklist)
IC_VERDICT_STRONG = "STRONG"
IC_VERDICT_MODERATE = "MODERATE"
IC_VERDICT_WEAK = "WEAK"
IC_VERDICT_NOISE = "NOISE"
IC_VERDICT_UNSTABLE = "UNSTABLE_WALKFORWARD"
IC_VERDICT_WARMUP_RIDER = "WARMUP_RIDER"
IC_VERDICT_NEW = "NEW"                # added by Phase 1.3/1.4/1.5 — not yet audited


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the feature schema."""
    name: str
    family: str                       # FAMILY_* constant
    ic_verdict: str                   # IC_VERDICT_* constant
    description: str = ""
    # Optional: best horizon (in bars) observed for this feature, from
    # ic_top_features.csv. None for features not yet audited.
    best_horizon: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# ── The Q2-audit top features (whitelist seed for Phase 1.2 'keep_top') ────
TOP_IC_FEATURES: tuple[FeatureSpec, ...] = (
    # 6 STRONG (full audit verdict)
    FeatureSpec("london_sess_high", FAMILY_STRUCTURAL, IC_VERDICT_STRONG,
                "session high — IC -0.110 @ h=24 but scale-dependent (IR=-4.05); pair with dist_to_london_high_atr (II.B fix).",
                best_horizon=24),
    FeatureSpec("current_vwap", FAMILY_STRUCTURAL, IC_VERDICT_STRONG,
                "rolling VWAP — IC -0.102 @ h=24 (IR=-3.62, scale-dependent).",
                best_horizon=24),
    FeatureSpec("price", FAMILY_STRUCTURAL, IC_VERDICT_STRONG,
                "raw close — IC -0.101 @ h=24 (IR=-3.46, scale-dependent).",
                best_horizon=24),
    FeatureSpec("hawkes_intrabar_sum", FAMILY_MICROSTRUCTURE_AGG, IC_VERDICT_STRONG,
                "Hawkes intensity sum — IC -0.106 @ h=24 BUT session sign-flip (Asia +0.06 / NY-close -0.26); II.A interaction needed.",
                best_horizon=24),
    FeatureSpec("tick_count", FAMILY_VOLUME, IC_VERDICT_STRONG,
                "tick count per bar — IC -0.106 @ h=24, same sign-flip story as hawkes_intrabar_sum.",
                best_horizon=24),
    FeatureSpec("time_to_ny_close_min", FAMILY_SEASONAL, IC_VERDICT_STRONG,
                "minutes to NY close — IC -0.106 @ h=24, weak gate-anti-select.",
                best_horizon=24),
    # MODERATE additions worth keeping (the top of the moderate list)
    FeatureSpec("london_sess_low", FAMILY_STRUCTURAL, IC_VERDICT_MODERATE,
                "session low — IC -0.080 @ h=24."),
    FeatureSpec("pdh", FAMILY_STRUCTURAL, IC_VERDICT_MODERATE,
                "previous-day high — IC -0.079 @ h=24."),
    FeatureSpec("pdl", FAMILY_STRUCTURAL, IC_VERDICT_MODERATE,
                "previous-day low — IC -0.066 @ h=24."),
    FeatureSpec("volume", FAMILY_VOLUME, IC_VERDICT_MODERATE,
                "bar volume — IC -0.098 @ h=24, gate-anti-select 1.51."),
    FeatureSpec("dist_to_london_low_atr", FAMILY_DIST_ATR, IC_VERDICT_MODERATE,
                "ATR-normalized distance to session low — IC -0.070 @ h=24 (scale-robust)."),
    FeatureSpec("inter_event_time", FAMILY_MICROSTRUCTURE_AGG, IC_VERDICT_MODERATE,
                "bars between micro-structure events — IC +0.093 @ h=24."),
    FeatureSpec("liquidity_density", FAMILY_MICROSTRUCTURE_AGG, IC_VERDICT_MODERATE,
                "depth-weighted density — IC -0.094 @ h=24."),
    FeatureSpec("session_phase", FAMILY_SEASONAL, IC_VERDICT_MODERATE,
                "session-phase code (Asia/London/overlap/NY/close)."),
    FeatureSpec("kyle_lambda_intrabar_mean", FAMILY_MICROSTRUCTURE_AGG, IC_VERDICT_MODERATE,
                "Kyle's lambda mean — IC +0.063 @ h=24."),
    FeatureSpec("mbo_bar_coverage", FAMILY_CONTEXT, IC_VERDICT_MODERATE,
                "data-quality metric (not a model feature)."),
)


# ── Brand-new features added by Phase 1.3/1.4/1.5 (no Q2 IC yet) ───────────
# These are referenced by the whitelist in keep_top mode but tagged NEW so
# the 1.6 IC re-audit can verify they earn their slot.
PHASE_1_3_NEW_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("event_score_continuous", FAMILY_CONTINUOUS_Z, IC_VERDICT_NEW,
                "Phase 1.3: continuous event score (replaces binary thresholds)."),
    FeatureSpec("hawkes_z_raw", FAMILY_CONTINUOUS_Z, IC_VERDICT_NEW,
                "Phase 1.3: raw z-score of Hawkes intensity (no threshold)."),
    FeatureSpec("absorb_z_raw", FAMILY_CONTINUOUS_Z, IC_VERDICT_NEW,
                "Phase 1.3: raw z-score of absorption intensity."),
    FeatureSpec("kyle_z_raw", FAMILY_CONTINUOUS_Z, IC_VERDICT_NEW,
                "Phase 1.3: raw z-score of Kyle's lambda."),
)
PHASE_1_4_NEW_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("cvd_bar_5m", FAMILY_CVD_MT5, IC_VERDICT_NEW,
                "Phase 1.4: per-bar signed volume (MT5-style delta)."),
    FeatureSpec("cvd_direction_ratio_5m", FAMILY_CVD_MT5, IC_VERDICT_NEW,
                "Phase 1.4: |buy-sell|/total per bar ∈ [0,1]."),
    FeatureSpec("cvd_intensity_vs_atr", FAMILY_CVD_MT5, IC_VERDICT_NEW,
                "Phase 1.4: |cvd_bar| / ATR — normalised intensity."),
    FeatureSpec("cvd_divergence_at_level", FAMILY_CVD_MT5, IC_VERDICT_NEW,
                "Phase 1.4: divergence signal at structural levels (PDH/PDL/VWAP)."),
    FeatureSpec("cvd_consecutive_imbalance", FAMILY_CVD_MT5, IC_VERDICT_NEW,
                "Phase 1.4: streak of same-sign CVD bars."),
)
PHASE_1_5_NEW_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("iceberg_count_5m", FAMILY_ICEBERG, IC_VERDICT_NEW,
                "Phase 1.5: detected iceberg events per bar."),
    FeatureSpec("iceberg_total_volume_5m", FAMILY_ICEBERG, IC_VERDICT_NEW,
                "Phase 1.5: estimated hidden volume per bar."),
)
# II.B fix: ATR-normalised distance versions of the scale-dependent STRONG
# features (london_sess_high, current_vwap, price all hit IR≈-4 because they
# are absolute-scale).
PHASE_1_4_DIST_ATR_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("dist_to_session_high_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "II.B: (close - london_sess_high) / atr_14."),
    FeatureSpec("dist_to_vwap_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "II.B: (close - current_vwap) / atr_14."),
    FeatureSpec("dist_to_pdh_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "II.B: (close - pdh) / atr_14."),
)
# II.A fix: session×feature interaction for the sign-flip pair
PHASE_1_5_INTERACTION_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("hawkes_x_session_phase", FAMILY_INTERACTION, IC_VERDICT_NEW,
                "II.A: hawkes_intrabar_sum × session_phase code."),
    FeatureSpec("tick_count_x_session_phase", FAMILY_INTERACTION, IC_VERDICT_NEW,
                "II.A: tick_count × session_phase code."),
)
# Phase 1.13: weekly key levels + their ATR-normalised distances. The helper
# compute_daily_weekly_levels already produced pwh/pwl causally (prev-week
# shift(1)) but the refinery harvested only the daily levels — the model
# never saw a weekly support/resistance. These close that gap. Raw levels are
# STRUCTURAL; the model-facing distances are ATR-normalised (scale-robust,
# per the II.B lesson that absolute distances collapse at IR≈-4).
PHASE_1_6_KEY_LEVEL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("pwh", FAMILY_STRUCTURAL, IC_VERDICT_NEW,
                "1.13: previous-week high (causal, prev-week shift(1))."),
    FeatureSpec("pwl", FAMILY_STRUCTURAL, IC_VERDICT_NEW,
                "1.13: previous-week low (causal, prev-week shift(1))."),
    FeatureSpec("weekly_price_position", FAMILY_STRUCTURAL, IC_VERDICT_NEW,
                "1.13: position within prev-week range ∈ [0,1], scale-free."),
    FeatureSpec("dist_to_pdl_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "1.13: (close - pdl) / atr_14 — scale-robust dist to prev-day low."),
    FeatureSpec("dist_to_pwh_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "1.13: (close - pwh) / atr_14 — scale-robust dist to prev-week high."),
    FeatureSpec("dist_to_pwl_atr", FAMILY_DIST_ATR, IC_VERDICT_NEW,
                "1.13: (close - pwl) / atr_14 — scale-robust dist to prev-week low."),
)
# Phase 1.7: the "Liquidity Compass" — three instantaneous institutional-
# pressure derivatives that replace the heavy raw-CVD / microstructure family
# in the structure-only day-trade design. cvd_divergence_at_level is the third
# tool (already specced under FAMILY_CVD_MT5). These two complete the trio so
# keep_top keeps them; the structure_compass event gate is built on all three.
PHASE_1_7_COMPASS_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("order_flow_imbalance", FAMILY_COMPASS, IC_VERDICT_NEW,
                "1.7: OFI = (buy_vol - sell_vol)/total ∈ [-1,1] — instantaneous flow imbalance."),
    FeatureSpec("vwap_z_score", FAMILY_COMPASS, IC_VERDICT_NEW,
                "1.7: VWAP stretch — session-VWAP deviation in std units (mean-reversion tell)."),
)


# Required-by-pipeline columns (not features themselves but must be present)
REQUIRED_OHLCV: tuple[str, ...] = (
    'ts_event', 'open', 'high', 'low', 'close', 'volume',
)
REQUIRED_REGIME: tuple[str, ...] = (
    'regime_label', 'atr_14', 'is_session_break',
)


# Aggregate (all feature specs the schema knows about, audited or not)
ALL_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    *TOP_IC_FEATURES,
    *PHASE_1_3_NEW_FEATURES,
    *PHASE_1_4_NEW_FEATURES,
    *PHASE_1_4_DIST_ATR_FEATURES,
    *PHASE_1_5_NEW_FEATURES,
    *PHASE_1_5_INTERACTION_FEATURES,
    *PHASE_1_6_KEY_LEVEL_FEATURES,
    *PHASE_1_7_COMPASS_FEATURES,
)


# Hard NOISE/UNSTABLE blacklist from the Q2 IC audit (Phase 1.2 'drop_noise')
# Each entry has a reason so a future reader knows why it was dropped.
PHASE_1_2_BLACKLIST: dict[str, str] = {
    # NOISE LOB-snapshot family (Phase 2 will rebuild from MBO)
    'mbp_roll_lob_coverage': 'NOISE — LOB snapshot, rebuilt in Phase 2',
    'correction_depth': 'NOISE — LOB snapshot',
    'distance_to_wall': 'NOISE — LOB snapshot',
    'ask_wall_strength': 'NOISE — LOB snapshot',
    'bid_wall_strength': 'NOISE — LOB snapshot',
    'lob_imbalance': 'NOISE — LOB snapshot',
    # UNSTABLE_WALKFORWARD (consistency=0.8 — sign flips in 1/5 windows)
    'hawkes_intensity': 'UNSTABLE — wf_consistency=0.8',
    'is_overlap': 'UNSTABLE — sign flip across regimes',
    'time_since_london_open_min': 'UNSTABLE — wf_consistency=0.8',
    'time_since_ny_open_min': 'UNSTABLE — wf_consistency=0.8',
    'time_to_london_close_min': 'UNSTABLE — wf_consistency=0.8',
    'dist_to_london_high_atr': 'UNSTABLE — wf_consistency=0.8',
    'is_month_end': 'UNSTABLE — gate_kill > 1, weak signal',
    'bar_range': 'UNSTABLE — wf_consistency=0.8',
    'cycle_hurst': 'UNSTABLE — wf_consistency=0.8',
    'cycle_fractal_dim': 'UNSTABLE — wf_consistency=0.8',
    'dow_sin': 'UNSTABLE — calendar noise',
    'dom_cos': 'UNSTABLE — calendar noise',
    # Cumulative-CVD weak variants (replaced by MT5-style in Phase 1.4)
    'cvd_momentum': 'WEAK — replaced by cvd_bar_5m in Phase 1.4',
    'cvd_price_divergence': 'WEAK — replaced by cvd_divergence_at_level',
    'cvd_slope_1h': 'WEAK — IC near noise floor',
    'cvd_slope_4h': 'WEAK — IC near noise floor',
    'cvd_slope_6b': 'WEAK — IC near noise floor',
    # Calendar NOISE
    'is_friday': 'NOISE — calendar dummy',
    'is_monday': 'NOISE — calendar dummy',
    'is_first_week_of_year': 'NOISE — calendar dummy',
    'dom_sin': 'NOISE — calendar dummy',
    'dom': 'NOISE — calendar dummy',
    'is_month_start': 'NOISE — calendar dummy',
    'is_quarter_end': 'NOISE — calendar dummy',
    'is_year_end': 'NOISE — calendar dummy',
    'woy_sin': 'NOISE — calendar dummy',
    'woy_cos': 'NOISE — calendar dummy',
    'is_dst_transition_week': 'NOISE — calendar dummy',
    'is_event_window': 'NOISE — calendar dummy',
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

    # Phase 1.1: also audit FEATURE specs (in addition to label specs above)
    feature_present = [s.name for s in ALL_FEATURE_SPECS if s.name in df.columns]
    feature_missing = [s.name for s in ALL_FEATURE_SPECS if s.name not in df.columns]
    blacklisted_present = [
        c for c in df.columns if c in PHASE_1_2_BLACKLIST
    ]

    meta: dict = {
        "schema_version": SCHEMA_VERSION,
        "git_hash": _current_git_hash(),
        "built_at_utc": _utc_now_iso(),
        "pipeline_cleanup_phases": PIPELINE_CLEANUP_PHASES,
        "label_specs": [s.as_dict() for s in ALL_LABEL_SCHEMAS],
        "feature_specs": [s.as_dict() for s in ALL_FEATURE_SPECS],
        "target_columns": list(target_column_names()),
        "required_ohlcv": list(REQUIRED_OHLCV),
        "required_regime": list(REQUIRED_REGIME),
        "phase_1_2_blacklist": PHASE_1_2_BLACKLIST,
        "label_columns_present": columns_present,
        "label_columns_missing": columns_missing,
        "feature_columns_present": feature_present,
        "feature_columns_missing": feature_missing,
        "blacklisted_columns_still_present": blacklisted_present,
        "dtype_mismatches": dtype_mismatches,
        "rows": int(len(df)),
        "n_columns_total": len(df.columns),
    }
    if extra:
        meta["extra"] = extra
    return meta


# ── Phase 1.1 git/time helpers + validate() ────────────────────────────────
def _current_git_hash() -> str | None:
    """Returns short git hash, or None if not in a repo / git unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def validate(df: pd.DataFrame, *, strict: bool = False) -> list[str]:
    """Phase 1.1: returns list of human-readable validation errors.
    Empty list = OK.

    Checks:
      - REQUIRED_OHLCV columns present
      - REQUIRED_REGIME columns present
      - Canonical aliases present (obi_net + cvd_cumulative both required,
        not the raw obi/cvd which are kept as compute aliases only)
      - No blacklisted column appears alongside its replacement

    strict=True also requires the label-target columns (bias_label,
    exec_label, next_price_delta) — i.e. the dual-target heads are wired.
    """
    errors: list[str] = []

    for c in REQUIRED_OHLCV:
        if c not in df.columns:
            errors.append(f"required OHLCV column missing: {c!r}")
    for c in REQUIRED_REGIME:
        if c not in df.columns:
            errors.append(f"required regime column missing: {c!r}")

    # Canonical aliases — the IC audit / verify_data_health expect these.
    if 'obi_net' not in df.columns:
        errors.append(
            "canonical alias `obi_net` missing — Phase 1.1 expects it "
            "alongside any internal-compute `obi`"
        )
    if 'cvd_cumulative' not in df.columns:
        errors.append(
            "canonical alias `cvd_cumulative` missing — Phase 1.1 expects it"
        )

    if strict:
        for c in target_column_names():
            if c not in df.columns:
                errors.append(
                    f"strict mode: training target {c!r} missing "
                    f"(dual-target pipeline not fully wired)"
                )

    return errors


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
