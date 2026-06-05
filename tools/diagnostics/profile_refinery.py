"""profile_refinery — time each top-level stage of the day_trading refinery.

Monkey-patches the heavy functions BEFORE calling run_day_trading_refinery, so:
  • the refinery's actual outputs (parquet, manifest, …) are identical to a
    normal run — patches only ADD timing,
  • a per-stage table is printed at the end (sorted by cumulative seconds).

Use:
    python tools/diagnostics/profile_refinery.py \\
        --mbo  /path/to/mbo.parquet  \\
        --output  pipeline_day_trading/probe_3m  \\
        --artifact-tag  probe_3m

Patching note: enrich_bars_with_intrabar / enrich_bars_with_intrabar_mbp are
imported into prepare_day_trading's namespace via `from … import …` at the top
of the file — so call sites inside the refinery resolve through P's namespace,
not the source module's. We patch P directly for those names (anything else
won't take effect).
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import wraps
from pathlib import Path

# Resolve sibling-import path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import prepare_day_trading as P
import modules.tick_intrabar_slices as TI
import modules.intrabar_mbp_microstructure as TIM
try:
    import modules.seasonal_map as SM
except ImportError:
    SM = None


_TIMES: dict[str, float] = {}
_CALLS: dict[str, int] = {}


def _wrap(holder, name: str) -> None:
    """Replace holder.name with a timing wrapper. Idempotent."""
    orig = getattr(holder, name, None)
    if orig is None or getattr(orig, "_profiled", False):
        return

    @wraps(orig)
    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            dt = time.perf_counter() - t
            _TIMES[name] = _TIMES.get(name, 0.0) + dt
            _CALLS[name] = _CALLS.get(name, 0) + 1
            print(
                f"  [TIME] {name:50s} +{dt:7.2f}s   total {_TIMES[name]:7.1f}s   calls={_CALLS[name]}",
                flush=True,
            )

    wrapper._profiled = True  # type: ignore[attr-defined]
    setattr(holder, name, wrapper)


def _instrument() -> None:
    # Functions defined IN prepare_day_trading.py (patched on P).
    for fn in (
        "aggregate_mbo_to_bars",
        "add_day_trading_features",
        "add_kalman_trend",
        "assign_regime_label",
        "label_by_outcome",
        "_apply_phase1_engineering_fixes",
        "_apply_phase1_feature_selection",
        "finalize_daytrade_parquet_export",
        "enrich_mbo_with_core_microstructure",
        "_attach_price_cycle_features",
        "compute_multitask_label_diagnostics",
        "load_mbo_ticks_enriched",
        "add_london_session_running_levels",
        "add_rolling_vwap_dist_features",
    ):
        _wrap(P, fn)

    # IMPORTANT: these are imported into P's namespace via top-level
    # `from modules.tick_intrabar_slices import enrich_bars_with_intrabar`.
    # Patching the source module alone would NOT intercept calls inside the
    # refinery — we must patch P's bound reference. Also patch the source for
    # any other consumers + clean module state.
    for fn in ("enrich_bars_with_intrabar",):
        _wrap(P, fn)
        _wrap(TI, fn)
    for fn in ("enrich_bars_with_intrabar_mbp",):
        _wrap(P, fn)
        _wrap(TIM, fn)

    # seasonal_map.add_seasonal_features is imported INSIDE the function body
    # (`from modules.seasonal_map import add_seasonal_features as _add_seasonal`)
    # — patching SM is enough (re-resolved at each call).
    if SM is not None:
        _wrap(SM, "add_seasonal_features")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Profile the refinery: real outputs + per-stage timing table."
    )
    ap.add_argument("--mbo", required=True,
                    help="MBO file or directory (passed to run_day_trading_refinery).")
    ap.add_argument("--output", required=True,
                    help="Output dir (persistent — refinery writes parquet/manifest here).")
    ap.add_argument("--artifact-tag", default=None)
    ap.add_argument("--mbp", default=None,
                    help="Optional MBP path (default: None = no MBP features).")
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--event-gate-mode", default="structure_compass",
                    choices=("microstructure", "structure_compass"))
    ap.add_argument("--build-lob", action="store_true",
                    help="Build LOB tensors (default off — matches decoder-probe workflow).")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Default 1 = sequential (byte-exact; no parallel artifacts).")
    args = ap.parse_args()

    _instrument()

    print(f"profile_refinery: mbo={args.mbo}")
    print(f"                  output={args.output}   tag={args.artifact_tag}")
    print(f"                  freq={args.freq}  horizon={args.horizon}  "
          f"gate={args.event_gate_mode}  lob={args.build_lob}  workers={args.n_workers}")
    print("-" * 84)

    T0 = time.perf_counter()
    try:
        P.run_day_trading_refinery(
            mbo_dir=args.mbo,
            mbp_path=args.mbp,
            output_dir=args.output,
            freq=args.freq,
            horizon_bars=args.horizon,
            build_lob_tensors=bool(args.build_lob),
            event_gate_mode=args.event_gate_mode,
            artifact_tag=args.artifact_tag,
            n_workers=args.n_workers,
        )
    finally:
        T_TOTAL = time.perf_counter() - T0
        print()
        print("=" * 84)
        print(f"TOTAL WALL TIME: {T_TOTAL:8.1f}s  ({T_TOTAL/60:.1f} min)")
        print("=" * 84)
        print(f"{'function':52s} {'cum_s':>10s} {'%total':>8s} {'calls':>6s}")
        print("-" * 84)
        sum_wrapped = 0.0
        for fn, dt in sorted(_TIMES.items(), key=lambda x: -x[1]):
            pct = 100.0 * dt / max(T_TOTAL, 1e-9)
            print(f"  {fn:50s} {dt:9.1f}s {pct:7.1f}% {_CALLS[fn]:>5d}")
            sum_wrapped += dt
        unaccounted = T_TOTAL - sum_wrapped
        print("-" * 84)
        print(f"  {'sum of wrapped':50s} {sum_wrapped:9.1f}s "
              f"{100.0*sum_wrapped/max(T_TOTAL,1e-9):7.1f}%")
        print(f"  {'UNACCOUNTED (I/O, parsing, …)':50s} {unaccounted:9.1f}s "
              f"{100.0*unaccounted/max(T_TOTAL,1e-9):7.1f}%")
        print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
