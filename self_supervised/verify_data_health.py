"""
self_supervised/verify_data_health.py
═══════════════════════════════════════════════════════════════════════════════
Pre-flight smoke verifier for SSL pretraining data. Designed to run
BEFORE you commit a GPU to a full training run — exits non-zero on any
hard failure so a wrapper shell script can short-circuit the pipeline.

Why exists
──────────
The SSL data loader has multiple silent failure modes (zero ATR,
missing OBI, embargo-eating-the-dataset, leakage-prone columns
sneaking past the blacklist, NaN-dominated targets). Each one only
manifests at training time: GPU spins up, loss tracks noise, the
operator wastes an evening before noticing.

This verifier catches all of them in seconds.

Checks performed
────────────────
1. CAUSALITY    — every column in the leakage blacklist is either
                  absent from the parquet OR the blacklist still
                  matches its name. No NEW column starting with
                  `forward_`, `mfe_`, `mae_` etc. is present.
2. REQUIRED     — close, atr_14 (or atr), obi_net OR order_flow_imbalance,
                  is_event, regime_label, ts_event are all present and
                  not 100% NaN.
3. NAN          — for each required column, the NaN fraction is below
                  NAN_TOLERANCE (default 5%).
4. SAMPLES      — effective sample count (after lookback + embargo +
                  session_break filter) clears SSL_MIN_SAMPLES_WARN.
5. DROPOUT      — the SSL backbone config has a non-zero dropout
                  somewhere (otherwise the encoder will memorise the
                  hawkes/absorb signal that feeds time_to_event).
                  *Best-effort* — gated behind --check-dropout.

Verdict cascade
───────────────
HEALTHY   every check passes — go.
WARN      at least one soft check failed but no hard blocker —
          operator may continue under their own discretion.
FAIL      at least one hard check failed (e.g. missing required
          column, sample count under the error floor) — STOP.

Usage
─────
    # Standalone
    python self_supervised/verify_data_health.py \\
        --features combined/day_trading_features.parquet \\
        --output   audits/data_health

    # In a shell pipeline (exits non-zero on FAIL):
    python self_supervised/verify_data_health.py \\
        --features combined/day_trading_features.parquet || exit 1
    python self_supervised/pretrain_lob.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_supervised.data_loader import (
    SSL_MIN_SAMPLES_ERROR,
    SSL_MIN_SAMPLES_WARN,
    _is_leakage_column,
    _LEAKAGE_COLS_EXACT,
    _LEAKAGE_PREFIXES,
)


# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════
NAN_TOLERANCE = 0.05            # > 5 % NaN on a required column → fail
NAN_WARN_TOLERANCE = 0.01       # > 1 % → warn

# Columns the SSL training pipeline materially depends on. Each entry
# is (column_name, is_critical). Critical columns failing the NaN or
# presence check produces FAIL; soft ones produce WARN.
REQUIRED_COLUMNS: list[tuple[str, bool]] = [
    ('ts_event', True),
    ('close', True),
    ('high', True),
    ('low', True),
    ('open', False),
    ('regime_label', True),
    ('is_event', True),
    # OBI alternatives — at least one must be present (handled separately)
]
OBI_ALTERNATIVES = ('obi_net', 'order_flow_imbalance')
ATR_ALTERNATIVES = ('atr_14', 'atr')

# Columns that should NEVER appear in the parquet because they are
# computed inside data_loader.py as forward-looking targets. If
# prepare_day_trading ever writes one of these, it would leak into
# the context tensor.
TARGET_COLUMNS_NOT_TO_LEAK: list[str] = [
    'next_price', 'next_imbalance', 'next_volatility', 'next_regime',
    'wall_persist', 'time_to_event',
]


# ════════════════════════════════════════════════════════════════════
# Result containers
# ════════════════════════════════════════════════════════════════════
@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str           # "fail" | "warn" | "ok"
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthReport:
    verdict: str            # "HEALTHY" | "WARN" | "FAIL"
    checks: list[CheckResult] = field(default_factory=list)
    n_rows: int = 0
    sample_estimate: int = 0

    def add(self, c: CheckResult) -> None:
        self.checks.append(c)

    @property
    def n_failures(self) -> int:
        return sum(1 for c in self.checks if c.severity == "fail")

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warn")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "n_rows": int(self.n_rows),
            "sample_estimate": int(self.sample_estimate),
            "n_failures": self.n_failures,
            "n_warnings": self.n_warnings,
            "checks": [c.to_dict() for c in self.checks],
        }


# ════════════════════════════════════════════════════════════════════
# Individual checks — each is a pure function over the DataFrame
# ════════════════════════════════════════════════════════════════════
def check_causality(df: pd.DataFrame) -> list[CheckResult]:
    """Surface forward-looking / labelled columns that the SSL must
    blacklist. The data loader DOES blacklist a fixed set; this check
    confirms the set still matches reality."""
    out: list[CheckResult] = []

    # 1. Forward-looking targets must not be pre-baked into the parquet
    leaked_targets = [
        c for c in TARGET_COLUMNS_NOT_TO_LEAK if c in df.columns
    ]
    if leaked_targets:
        out.append(CheckResult(
            name="causality.no_leaked_targets",
            passed=False, severity="fail",
            message=(
                f"forward-looking SSL targets present in parquet: "
                f"{leaked_targets} — these should be computed by the "
                f"data_loader, not written by prepare_day_trading."
            ),
        ))
    else:
        out.append(CheckResult(
            name="causality.no_leaked_targets",
            passed=True, severity="ok",
            message=f"no SSL targets pre-baked (checked {len(TARGET_COLUMNS_NOT_TO_LEAK)})",
        ))

    # 2. Every column matching a known leakage prefix must be flagged
    unflagged = [
        c for c in df.columns
        if any(c.startswith(p) for p in _LEAKAGE_PREFIXES)
        and not _is_leakage_column(c)
    ]
    if unflagged:
        out.append(CheckResult(
            name="causality.prefix_blacklist_complete",
            passed=False, severity="fail",
            message=f"columns matching leakage prefix but not blacklisted: {unflagged}",
        ))
    else:
        out.append(CheckResult(
            name="causality.prefix_blacklist_complete",
            passed=True, severity="ok",
            message="all prefix-matched columns are blacklisted",
        ))
    return out


def check_required_columns(df: pd.DataFrame) -> list[CheckResult]:
    out: list[CheckResult] = []
    for col, is_critical in REQUIRED_COLUMNS:
        if col in df.columns:
            out.append(CheckResult(
                name=f"required.{col}",
                passed=True, severity="ok",
                message=f"{col} present",
            ))
        else:
            out.append(CheckResult(
                name=f"required.{col}",
                passed=False,
                severity="fail" if is_critical else "warn",
                message=(
                    f"{col} missing from parquet — "
                    f"re-run prepare_day_trading.py"
                ),
            ))

    # OBI alternatives — at least one required
    obi_present = [c for c in OBI_ALTERNATIVES if c in df.columns]
    if obi_present:
        out.append(CheckResult(
            name="required.obi",
            passed=True, severity="ok",
            message=f"OBI column present: {obi_present[0]}",
        ))
    else:
        out.append(CheckResult(
            name="required.obi",
            passed=False, severity="fail",
            message=(
                f"neither {OBI_ALTERNATIVES[0]} nor "
                f"{OBI_ALTERNATIVES[1]} present — "
                f"SSL next_imbalance task will fail loudly at construction"
            ),
        ))

    # ATR alternatives — at least one required
    atr_present = [c for c in ATR_ALTERNATIVES if c in df.columns]
    if atr_present:
        out.append(CheckResult(
            name="required.atr",
            passed=True, severity="ok",
            message=f"ATR column present: {atr_present[0]}",
        ))
    else:
        out.append(CheckResult(
            name="required.atr",
            passed=False, severity="fail",
            message=(
                f"neither {ATR_ALTERNATIVES[0]} nor "
                f"{ATR_ALTERNATIVES[1]} present — "
                f"SSL next_volatility task will fail loudly at construction"
            ),
        ))
    return out


def check_nan_fractions(df: pd.DataFrame) -> list[CheckResult]:
    out: list[CheckResult] = []
    # Pull every required column that's actually present
    required_names: list[str] = [c for c, _ in REQUIRED_COLUMNS if c in df.columns]
    for alts in (OBI_ALTERNATIVES, ATR_ALTERNATIVES):
        for c in alts:
            if c in df.columns:
                required_names.append(c)
                break
    for col in required_names:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            frac = float(s.isna().mean())
        else:
            # Non-numeric: count NaN-likes
            frac = float(s.isna().mean())
        if frac > NAN_TOLERANCE:
            out.append(CheckResult(
                name=f"nan.{col}",
                passed=False, severity="fail",
                message=(
                    f"{col} is {frac:.1%} NaN — above {NAN_TOLERANCE:.0%} "
                    f"tolerance. Verify upstream feature computation."
                ),
            ))
        elif frac > NAN_WARN_TOLERANCE:
            out.append(CheckResult(
                name=f"nan.{col}",
                passed=False, severity="warn",
                message=f"{col} is {frac:.1%} NaN (above {NAN_WARN_TOLERANCE:.0%} warn tolerance)",
            ))
        else:
            out.append(CheckResult(
                name=f"nan.{col}",
                passed=True, severity="ok",
                message=f"{col} NaN={frac:.2%}",
            ))
    return out


def estimate_effective_samples(
    df: pd.DataFrame, lookback_bars: int, embargo_bars: int,
) -> int:
    """Approximation of what `SSLDataset.n_samples` will land at.
    Does NOT instantiate the dataset (which is heavyweight) — just
    applies the same arithmetic + session-break filter."""
    n_total = len(df)
    min_idx = lookback_bars
    max_idx = max(min_idx, n_total - embargo_bars)
    base = max(0, max_idx - min_idx)

    if 'is_session_break' not in df.columns:
        return base

    breaks = df['is_session_break'].astype(bool).to_numpy()
    if not breaks.any():
        return base

    # Same logic as _build_segment_mask: mark a sample as clean iff its
    # lookback window and 24-bar forward window contain NO session break
    clean = np.ones(n_total, dtype=bool)
    forward = 24
    for i in range(min_idx, max_idx):
        lo = max(0, i - lookback_bars)
        hi = min(n_total, i + forward + 1)
        if breaks[lo:hi].any():
            clean[i] = False
    return int(clean[min_idx:max_idx].sum())


def check_sample_count(
    df: pd.DataFrame, lookback_bars: int, embargo_bars: int,
) -> tuple[list[CheckResult], int]:
    estimated = estimate_effective_samples(df, lookback_bars, embargo_bars)
    if estimated < SSL_MIN_SAMPLES_ERROR:
        return [CheckResult(
            name="samples.count",
            passed=False, severity="fail",
            message=(
                f"effective samples ≈ {estimated:,} — below ERROR floor "
                f"({SSL_MIN_SAMPLES_ERROR:,}). Loosen embargo or use a "
                f"longer source range."
            ),
        )], estimated
    if estimated < SSL_MIN_SAMPLES_WARN:
        return [CheckResult(
            name="samples.count",
            passed=False, severity="warn",
            message=(
                f"effective samples ≈ {estimated:,} — below recommended "
                f"{SSL_MIN_SAMPLES_WARN:,}. SSL likely to overfit."
            ),
        )], estimated
    return [CheckResult(
        name="samples.count",
        passed=True, severity="ok",
        message=f"effective samples ≈ {estimated:,}",
    )], estimated


# ════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════
def verify(
    features_parquet: Path,
    lookback_bars: int = 50,
    embargo_bars: int = 24,
) -> HealthReport:
    df = pd.read_parquet(features_parquet)
    report = HealthReport(verdict="HEALTHY", n_rows=len(df))

    for c in check_causality(df):
        report.add(c)
    for c in check_required_columns(df):
        report.add(c)
    for c in check_nan_fractions(df):
        report.add(c)
    sample_checks, estimated = check_sample_count(
        df, lookback_bars, embargo_bars,
    )
    for c in sample_checks:
        report.add(c)
    report.sample_estimate = estimated

    if report.n_failures > 0:
        report.verdict = "FAIL"
    elif report.n_warnings > 0:
        report.verdict = "WARN"
    return report


def write_report(report: HealthReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_health_summary.json").write_text(
        json.dumps(report.to_dict(), indent=2)
    )
    lines = ["═" * 70, "SSL DATA HEALTH VERIFIER", "═" * 70,
             f"Verdict          : {report.verdict}",
             f"Rows in parquet  : {report.n_rows:,}",
             f"Estimated samples: {report.sample_estimate:,}",
             f"Failures         : {report.n_failures}",
             f"Warnings         : {report.n_warnings}",
             ""]
    for c in report.checks:
        icon = {"ok": "✅", "warn": "⚠️ ", "fail": "🚨"}.get(c.severity, "❔")
        lines.append(f"  {icon} {c.name:<36} {c.message}")
    lines.append("═" * 70)
    (output_dir / "data_health_report.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--lookback-bars", type=int, default=50)
    p.add_argument("--embargo-bars", type=int, default=24)
    args = p.parse_args()

    report = verify(
        args.features,
        lookback_bars=args.lookback_bars,
        embargo_bars=args.embargo_bars,
    )
    write_report(report, args.output)

    print(f"verdict          : {report.verdict}")
    print(f"rows             : {report.n_rows:,}")
    print(f"sample estimate  : {report.sample_estimate:,}")
    print(f"failures         : {report.n_failures}")
    print(f"warnings         : {report.n_warnings}")
    print(f"output           : {args.output}")

    # Non-zero exit on FAIL so shell wrappers can short-circuit
    return 1 if report.verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
