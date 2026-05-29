"""
tools/run_production.py
═══════════════════════════════════════════════════════════════════════════════
CLI smoke-runner for the Production Runner. Mostly useful for:

  - confirming a checkpoint loads cleanly
  - dry-running predictions on synthetic features (so the operator
    sees the runner's SILENT-rate, drift scores, and refusal reasons
    before plugging it into a real tick stream)
  - exporting a default config to JSON for the operator to edit

NOT a live-trading entry point — that's the next layer. This is the
read-only audit / smoke harness.

Usage
─────
    # Dry-run a hundred random predictions against the runner
    python tools/run_production.py dry-run \\
        --checkpoint runs/fold_01/model.pt \\
        --daytrade-dim 247 --ssl-dim 32 \\
        --n-predictions 100 \\
        --allow-uncalibrated

    # Print a default config skeleton to stdout
    python tools/run_production.py print-default-config
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.production import (
    ConfidenceGateConfig,
    DriftGuardConfig,
    ProductionConfig,
    ProductionRunner,
    SizingConfig,
    THRESHOLD_SOURCE_OVERRIDDEN,
)


def _default_config_dict() -> dict:
    """Skeleton with every field a calibration team would need to fill."""
    return {
        "checkpoint_path": "runs/fold_01/model.pt",
        "device": "cpu",
        "confidence_gate": {
            "event_prob_threshold": 0.50,
            "confidence_threshold": 0.85,
            "confidence_threshold_source": "uncalibrated  # set to 'calibrated' after running tools/calibrate_*.py",
            "allow_uncalibrated": False,
            "calibration_artifact_path": None,
        },
        "drift_guard": {
            "enabled": True,
            "method": "mahalanobis",
            "threshold": 3.0,
            "artifact_path": "runs/fold_01/drift_artifact.npz",
        },
        "sizing": {
            "base_size": 1.0,
            "min_size": 0.0,
            "confidence_scaling": True,
        },
    }


def cmd_print_default_config(_args) -> int:
    print(json.dumps(_default_config_dict(), indent=2))
    return 0


def cmd_dry_run(args) -> int:
    cfg = ProductionConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_gate=ConfidenceGateConfig(
            event_prob_threshold=args.event_prob_threshold,
            confidence_threshold=args.confidence_threshold,
            confidence_threshold_source=(
                THRESHOLD_SOURCE_OVERRIDDEN if args.allow_uncalibrated
                else "uncalibrated"
            ),
            allow_uncalibrated=args.allow_uncalibrated,
        ),
        drift_guard=DriftGuardConfig(enabled=False),
        sizing=SizingConfig(
            base_size=args.base_size, min_size=0.0, confidence_scaling=True,
        ),
    )
    runner = ProductionRunner(
        cfg, warmup_predictions=args.warmup_predictions,
    )

    print(f"is_ready_to_trade : {cfg.is_ready_to_trade}")
    if not cfg.is_ready_to_trade:
        print(f"readiness_reason  : {cfg.readiness_reason()}")

    try:
        runner.start()
    except FileNotFoundError as exc:
        print(f"✗ checkpoint load failed: {exc}")
        return 1

    rng = np.random.RandomState(args.seed)
    counts: dict[str, int] = {}
    for _ in range(args.n_predictions):
        daytrade = rng.randn(args.daytrade_dim).astype(np.float32)
        ssl = rng.randn(args.ssl_dim).astype(np.float32)
        resp = runner.predict(daytrade, ssl)
        key = resp.action if resp.action != "SILENT" else (
            f"SILENT/{resp.refused_reason}"
        )
        counts[key] = counts.get(key, 0) + 1

    print(f"\nDry-run results ({args.n_predictions} predictions):")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        pct = counts[k] / args.n_predictions
        print(f"  {k:<40} {counts[k]:>5}  ({pct:.1%})")
    print(f"\nsilent_rate       : {runner.silent_rate:.1%}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("print-default-config")
    pp.set_defaults(func=cmd_print_default_config)

    pd = sub.add_parser("dry-run")
    pd.add_argument("--checkpoint", type=Path, default=None)
    pd.add_argument("--device", default="cpu")
    pd.add_argument("--daytrade-dim", type=int, default=247)
    pd.add_argument("--ssl-dim", type=int, default=32)
    pd.add_argument("--n-predictions", type=int, default=100)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--event-prob-threshold", type=float, default=0.50)
    pd.add_argument("--confidence-threshold", type=float, default=0.85)
    pd.add_argument("--allow-uncalibrated", action="store_true")
    pd.add_argument("--warmup-predictions", type=int, default=0)
    pd.add_argument("--base-size", type=float, default=1.0)
    pd.set_defaults(func=cmd_dry_run)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
