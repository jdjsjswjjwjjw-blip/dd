"""event_study — isolate a RARE pre-defined pattern from the 17k dead bars.

The motivation (mathematically sound): a mean IC blinds rare strong patterns.
100 bars at IC=0.30 diluted by 17,000 dead bars average to ~0.002. So the weak
IC decoder_probe found is CONSISTENT with a rare strong pattern the average
erases. An event study removes the dead bars from the measurement.

THE PATTERN (user-specified): a session-low SWEEP, defined ONLY by pre-move
conditions — never by the outcome.

    sweep at bar i  ⇔
        level[i] is defined (causal: london_sess_low / pdl, both shift-based)
        AND low[i] < level[i] − depth × atr[i]        (the wick pierces below)
    reclaim variant additionally requires close[i] > level[i] (closed back above)

We find EVERY such bar (winners and losers alike), then measure the forward
move. We do NOT condition on "the ones that bounced" — that is the cardinal
event-study sin (selecting survivors = guaranteed fake edge).

ANTI-LOOK-AHEAD (R1, the sharpest risk here):
  - The trigger uses level[i] (built from bars <= i), low[i], close[i] — all
    known at the close of bar i.
  - The outcome uses bars (i, i+h] ONLY.
  - A teeth test proves event detection at bar i is invariant to mutating any
    bar after i (truncate-invariance).

METRICS per (level × depth × reclaim × horizon):
  n_events, bounce_rate (fwd_return > 0), mean/median forward_return, MFE/MAE in
  ATR, triple-barrier win rate (TP +k·ATR before SL −k·ATR), and a
  directional-vs-volatility split (mean(ret) vs mean(|ret|)).

NULL TEST (R4): sample n_events RANDOM bars from the SAME session mix and
recompute the metrics, 500 reps. The pattern is real only if bounce_rate /
expectancy exceed the null 95th percentile.

A priori thresholds (R10): in EVENT_STUDY_THRESHOLDS, written before any run.

R6: reuses ic_audit primitives where applicable; otherwise self-contained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# A priori parameters (R10 — fixed before any result) -----------------------
DEFAULT_LEVELS: tuple[str, ...] = ("london_sess_low", "pdl")
DEFAULT_DEPTHS_ATR: tuple[float, ...] = (0.15, 0.30, 0.50)
DEFAULT_HORIZONS: tuple[int, ...] = (6, 12, 24, 48)
DEFAULT_TP_SL_ATR: float = 1.0           # triple-barrier ±1 ATR
DEFAULT_N_NULL: int = 500
MIN_EVENTS: int = 50                     # below this → UNDERPOWERED

# Verdict thresholds (a priori)
BOUNCE_RATE_EDGE = 0.55                  # >= AND above null p95
MEAN_RET_SIGMA = 1.5                     # mean fwd_return >= this × null std
WIN_RATE_EDGE = 0.55                    # TP-before-SL rate
DIRECTIONAL_VS_VOL = 0.5                # mean(ret) >= this × mean(|ret|)


def _session_label(hour: int) -> str:
    if 13 <= hour < 16:  return "overlap"
    if 7 <= hour < 13:   return "london"
    if 16 <= hour < 22:  return "ny"
    if 0 <= hour < 7:    return "asia"
    return "off"


def detect_sweep_events(
    df: pd.DataFrame, level_col: str, depth_atr: float, *, require_reclaim: bool,
) -> np.ndarray:
    """Boolean array: True at bar i iff a session-low sweep triggers there.
    PURE pre-move definition — uses only level[i], low[i], close[i], atr[i]."""
    n = len(df)
    if level_col not in df.columns:
        return np.zeros(n, dtype=bool)
    level = pd.to_numeric(df[level_col], errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
    atr = pd.to_numeric(df.get("atr_14", 0.0), errors="coerce").to_numpy(np.float64)

    valid = np.isfinite(level) & np.isfinite(low) & np.isfinite(atr) & (atr > 0)
    pierced = valid & (low < (level - depth_atr * atr))
    if require_reclaim:
        pierced = pierced & np.isfinite(close) & (close > level)
    return pierced


def _forward_metrics(
    df: pd.DataFrame, event_idx: np.ndarray, h: int, tp_sl_atr: float,
) -> dict[str, Any]:
    """Measure the forward move after each event over a fixed horizon h.
    Long bias (sweep-low → expect bounce up). Outcome window = (i, i+h]."""
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64)
    atr = pd.to_numeric(df.get("atr_14", 0.0), errors="coerce").to_numpy(np.float64)
    n = len(close)

    fwd_ret, mfe_atr, mae_atr, tp_first = [], [], [], []
    for i in event_idx:
        if i + h >= n or not np.isfinite(close[i]) or close[i] <= 0 or atr[i] <= 0:
            continue
        entry = close[i]
        win_close = close[i + h]
        fwd_ret.append((win_close - entry) / entry)
        hi = high[i + 1: i + h + 1]
        lo = low[i + 1: i + h + 1]
        if len(hi) == 0:
            continue
        mfe_atr.append(max(0.0, (np.nanmax(hi) - entry)) / atr[i])
        mae_atr.append(max(0.0, (entry - np.nanmin(lo))) / atr[i])
        # triple barrier (long): TP at +k·ATR, SL at -k·ATR, first touch
        up = entry + tp_sl_atr * atr[i]
        dn = entry - tp_sl_atr * atr[i]
        tp_hit = np.where(hi >= up)[0]
        sl_hit = np.where(lo <= dn)[0]
        t_tp = tp_hit[0] if len(tp_hit) else np.inf
        t_sl = sl_hit[0] if len(sl_hit) else np.inf
        if np.isfinite(t_tp) or np.isfinite(t_sl):
            tp_first.append(1 if t_tp < t_sl else 0)

    fr = np.asarray(fwd_ret, dtype=np.float64)
    if len(fr) == 0:
        return {"n": 0}
    return {
        "n": int(len(fr)),
        "bounce_rate": float((fr > 0).mean()),
        "mean_fwd_return": float(fr.mean()),
        "median_fwd_return": float(np.median(fr)),
        "mean_abs_fwd_return": float(np.abs(fr).mean()),
        "p25": float(np.percentile(fr, 25)),
        "p75": float(np.percentile(fr, 75)),
        "mean_mfe_atr": float(np.mean(mfe_atr)) if mfe_atr else float("nan"),
        "mean_mae_atr": float(np.mean(mae_atr)) if mae_atr else float("nan"),
        "win_rate_tp_before_sl": float(np.mean(tp_first)) if tp_first else float("nan"),
        "n_barrier_resolved": int(len(tp_first)),
    }


def _null_metrics(
    df: pd.DataFrame, event_idx: np.ndarray, h: int, tp_sl_atr: float,
    n_null: int, seed: int,
) -> dict[str, Any]:
    """Sample n_events random bars from the SAME session distribution as the
    events, recompute bounce_rate + mean_fwd_return. Returns the null p95/p05."""
    if len(event_idx) == 0:
        return {"bounce_p95": float("nan"), "mean_ret_std": float("nan")}
    ts = pd.to_datetime(df["ts_event"], utc=True)
    hours = ts.dt.hour.to_numpy()
    sess = np.array([_session_label(int(x)) for x in hours])
    ev_sess = sess[event_idx]
    # per-session candidate pools
    pools = {s: np.where(sess == s)[0] for s in np.unique(ev_sess)}
    rng = np.random.RandomState(seed)
    n_ev = len(event_idx)
    bounce_samples, mean_ret_samples = [], []
    for _ in range(n_null):
        picks = []
        for s in ev_sess:
            pool = pools[s]
            picks.append(pool[rng.randint(len(pool))])
        m = _forward_metrics(df, np.asarray(picks), h, tp_sl_atr)
        if m.get("n", 0) > 0:
            bounce_samples.append(m["bounce_rate"])
            mean_ret_samples.append(m["mean_fwd_return"])
    if not bounce_samples:
        return {"bounce_p95": float("nan"), "mean_ret_std": float("nan")}
    return {
        "bounce_p95": float(np.percentile(bounce_samples, 95)),
        "bounce_mean": float(np.mean(bounce_samples)),
        "mean_ret_std": float(np.std(mean_ret_samples)),
        "mean_ret_mean": float(np.mean(mean_ret_samples)),
        "n_null_used": int(len(bounce_samples)),
    }


def _classify(metrics: dict, null: dict) -> str:
    if metrics.get("n", 0) < MIN_EVENTS:
        return "UNDERPOWERED"
    bounce = metrics["bounce_rate"]
    mean_ret = metrics["mean_fwd_return"]
    mean_abs = metrics["mean_abs_fwd_return"]
    win = metrics.get("win_rate_tp_before_sl", float("nan"))
    bounce_p95 = null.get("bounce_p95", float("nan"))
    ret_std = null.get("mean_ret_std", float("nan"))
    ret_mean_null = null.get("mean_ret_mean", 0.0)

    above_bounce_null = np.isfinite(bounce_p95) and bounce >= bounce_p95
    above_ret_null = (np.isfinite(ret_std) and ret_std > 0
                      and (mean_ret - ret_mean_null) >= MEAN_RET_SIGMA * ret_std)
    directional = (np.isfinite(mean_abs) and mean_abs > 0
                   and abs(mean_ret) >= DIRECTIONAL_VS_VOL * mean_abs)

    if not directional and mean_abs > 0:
        # the move exists but is not directional → volatility, not edge
        if bounce >= BOUNCE_RATE_EDGE and above_bounce_null:
            return "EDGE_DIRECTIONAL"      # bounce-rate edge despite symmetric |ret|
        return "VOLATILITY_NOT_DIRECTION"
    if (bounce >= BOUNCE_RATE_EDGE and above_bounce_null
            and mean_ret > 0 and above_ret_null
            and (not np.isfinite(win) or win >= WIN_RATE_EDGE)):
        return "EDGE_DIRECTIONAL"
    if bounce >= BOUNCE_RATE_EDGE and above_bounce_null:
        return "EDGE_WEAK"
    return "NO_EDGE"


def run_event_study(
    features_parquet: Path, output_dir: Path,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
    depths_atr: tuple[float, ...] = DEFAULT_DEPTHS_ATR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    tp_sl_atr: float = DEFAULT_TP_SL_ATR,
    n_null: int = DEFAULT_N_NULL,
    seed: int = 42,
) -> dict[str, Any]:
    df = pd.read_parquet(features_parquet)
    for c in ("close", "high", "low", "ts_event"):
        if c not in df.columns:
            raise RuntimeError(f"event_study requires column {c!r}")

    cells: list[dict] = []
    for level in levels:
        for depth in depths_atr:
            for reclaim in (False, True):
                events = detect_sweep_events(df, level, depth, require_reclaim=reclaim)
                event_idx = np.where(events)[0]
                for h in horizons:
                    metrics = _forward_metrics(df, event_idx, h, tp_sl_atr)
                    null = (_null_metrics(df, event_idx, h, tp_sl_atr, n_null, seed)
                            if metrics.get("n", 0) >= MIN_EVENTS
                            else {"bounce_p95": float("nan"), "mean_ret_std": float("nan")})
                    cells.append({
                        "level": level, "depth_atr": depth, "reclaim": reclaim,
                        "horizon": h, "n_events_total": int(len(event_idx)),
                        "metrics": metrics, "null": null,
                        "verdict": _classify(metrics, null),
                    })

    any_edge = any(c["verdict"] in {"EDGE_DIRECTIONAL", "EDGE_WEAK"} for c in cells)
    summary = {
        "features_parquet": str(features_parquet),
        "n_rows": int(len(df)),
        "levels": list(levels), "depths_atr": list(depths_atr),
        "horizons": list(horizons), "tp_sl_atr": tp_sl_atr,
        "n_null": n_null, "min_events": MIN_EVENTS,
        "thresholds_a_priori": {
            "BOUNCE_RATE_EDGE": BOUNCE_RATE_EDGE, "MEAN_RET_SIGMA": MEAN_RET_SIGMA,
            "WIN_RATE_EDGE": WIN_RATE_EDGE, "DIRECTIONAL_VS_VOL": DIRECTIONAL_VS_VOL,
        },
        "cells": cells,
        "any_edge_found": bool(any_edge),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "event_study_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "event_study_report.txt", summary)
    return summary


def _write_report(path: Path, s: dict) -> None:
    L = []
    L.append("═" * 110)
    L.append("Event study — session-low SWEEP (pre-move defined; outcome measured; null-tested)")
    L.append("═" * 110)
    L.append(f"parquet : {s['features_parquet']}   n_rows: {s['n_rows']:,}")
    L.append(f"levels  : {s['levels']}   depths(ATR): {s['depths_atr']}   horizons: {s['horizons']}   TP/SL: ±{s['tp_sl_atr']} ATR")
    L.append("")
    L.append(f"  {'level':18s} {'depth':>6s} {'recl':>5s} {'h':>4s} {'n':>5s} {'bounce%':>8s} {'null95':>7s} "
             f"{'mean_ret':>9s} {'|ret|':>8s} {'win%':>6s} {'verdict':>22s}")
    L.append("  " + "-" * 116)
    for c in s["cells"]:
        m, nu = c["metrics"], c["null"]
        if m.get("n", 0) == 0:
            L.append(f"  {c['level']:18s} {c['depth_atr']:>6.2f} {str(c['reclaim'])[0]:>5s} {c['horizon']:>4d} "
                     f"{c['n_events_total']:>5d}  (no resolved events)")
            continue
        L.append(
            f"  {c['level']:18s} {c['depth_atr']:>6.2f} {str(c['reclaim'])[0]:>5s} {c['horizon']:>4d} "
            f"{m['n']:>5d} {m['bounce_rate']*100:>7.1f}% "
            f"{(nu.get('bounce_p95') or float('nan'))*100:>6.1f}% "
            f"{m['mean_fwd_return']*1e4:>+8.1f} {m['mean_abs_fwd_return']*1e4:>7.1f} "
            f"{(m.get('win_rate_tp_before_sl') or float('nan'))*100:>5.1f}% {c['verdict']:>22s}"
        )
    L.append("  " + "-" * 116)
    L.append("UNITS: mean_ret / |ret| in basis points (1e-4). bounce% vs null95 = bounce rate vs shuffle p95.")
    L.append("VERDICTS: EDGE_DIRECTIONAL (bounce+ret above null, directional) / EDGE_WEAK (bounce only) /")
    L.append("          VOLATILITY_NOT_DIRECTION / NO_EDGE / UNDERPOWERED (n<50).")
    L.append(f"ANY EDGE FOUND: {s['any_edge_found']}")
    L.append("R7 NOTE: an edge here is a historical association, NOT proof of tradeable profit after costs.")
    L.append("═" * 110)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Event study on session-low sweeps")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTHS_ATR))
    p.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    s = run_event_study(
        args.features, args.output,
        levels=tuple(args.levels), depths_atr=tuple(args.depths),
        horizons=tuple(args.horizons), n_null=args.n_null, seed=args.seed,
    )
    print(f"\n  cells: {len(s['cells'])}   any_edge_found: {s['any_edge_found']}")
    edges = [c for c in s["cells"] if c["verdict"] in {"EDGE_DIRECTIONAL", "EDGE_WEAK"}]
    for c in edges:
        m = c["metrics"]
        print(f"  EDGE: {c['level']} depth={c['depth_atr']} reclaim={c['reclaim']} h={c['horizon']} "
              f"n={m['n']} bounce={m['bounce_rate']:.0%} → {c['verdict']}")
    print(f"  report: {args.output}/event_study_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
