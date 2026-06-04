"""Breathing test (v2) — is there a STRUCTURED absorption tail in YOUR MBO?

WHY v2
══════
v1 fired the production threshold (absorb_z > 1.0) and reported a firing rate.
On 6B June 2025 that gave ~19% — which is just the BASE RATE of z>1 for a
roughly-normal variable (P(z>1)≈16%). The lesson: firing rate is not a signal,
it is a MECHANICAL function of the threshold (z>p95 fires 5% BY DEFINITION).
Choosing a threshold to hit a "healthy" firing rate would be circular (R10).

v2 asks the non-circular question instead: is the absorb_z TAIL genuinely
fatter than noise, and where it is, are those tail firings STRUCTURED
(concentrated in liquid sessions + clustered in time after removing the
overnight/session gaps that inflated v1's CV²)?

The objective anchor is EXCESS-OVER-NORMAL: P_observed(z>t) / P_normal(z>t).
absorb_z is a rolling z-score, so under pure noise it is ~N(0,1) and the ratio
is ≈1 at every t. A real fat tail makes the ratio ≫1 in the tail — independent
of any target firing rate. We anchor the "tail threshold" where excess first
reaches EXCESS_TAIL_ANCHOR, then judge STRUCTURE there.

LIMITS (R7): a STRUCTURED verdict means the tail is concentrated+clustered like
institutional absorption — NOT proof of trading edge (that needs the Phase-1
decoder probe). We do NOT touch the production z>1 threshold here — diagnose
first, calibrate later with a justified constant.

Uses the production AII path via AbsorptionDetector (zero re-implementation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.absorption import AbsorptionDetector, AbsorptionConfig


# ── MBO vocabulary (matches prepare_day_trading.py) ────────────────────────
TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "E", "0"}
BUY_SIDES = {"A", "ASK", "BUY", "BOT"}
SELL_SIDES = {"B", "BID", "S", "SELL"}

# ── v2 parameters ──────────────────────────────────────────────────────────
SWEEP_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
WARMUP_TICKS = 100

# Objective tail anchor: the tail is "genuinely fat" where the observed
# exceedance is at least this many times the normal exceedance.
EXCESS_TAIL_ANCHOR = 3.0
MIN_TAIL_FIRINGS = 200          # need enough tail firings to judge structure

# Structure judged AT the anchored tail threshold (NOT firing rate):
SESSION_STRUCT_RATIO = 2.0      # active/quiet firing-rate ratio → concentrated
SESSION_SUSPECT_RATIO = 1.3
CV2_CLUSTERED = 1.5             # gap-aware CV² (Poisson=1) → clustered

# A gap longer than this (seconds) is a session/overnight break, not an
# intra-session quiet period — excluded from the gap-aware CV².
GAP_SECONDS = 3600.0


def _exclusive_session(hour: int) -> str:
    """Priority overlap > ny > london > asia > off (matches M1)."""
    if 13 <= hour < 16:  return "overlap"
    if 7 <= hour < 13:   return "london"
    if 16 <= hour < 22:  return "ny"
    if 0 <= hour < 7:    return "asia"
    return "off"


def _load_trades_mbo(mbo_path: Path) -> pd.DataFrame:
    """Read MBO parquet (file or dir), keep only TRADE-action ticks with a
    recognized side. `.str.strip()` guards against whitespace in action codes."""
    p = Path(mbo_path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files in {p}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(p)
    for c in ("ts_event", "action", "side", "price", "size"):
        if c not in df.columns:
            raise ValueError(f"MBO column missing: {c!r}")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)
    act = df["action"].astype(str).str.upper().str.strip()
    sd = df["side"].astype(str).str.upper().str.strip()
    trades = df.loc[
        act.isin(TRADE_ACTIONS) & sd.isin(BUY_SIDES | SELL_SIDES),
        ["ts_event", "price", "side", "size"],
    ].reset_index(drop=True)
    if len(trades) == 0:
        raise RuntimeError(
            f"no recognized TRADE-action ticks with known side "
            f"(action values: {act.value_counts().head(5).to_dict()})"
        )
    return trades


def _gap_aware_cv2(fire_ts_ns: np.ndarray, gap_seconds: float) -> tuple[float, float, int]:
    """Return (cv2_raw, cv2_gap_aware, n_iet_used).

    cv2_raw includes ALL inter-event gaps (dominated by overnight/weekend
    breaks → inflated). cv2_gap_aware drops IETs longer than gap_seconds, so it
    measures intra-session clustering only."""
    if len(fire_ts_ns) < 30:
        return float("nan"), float("nan"), 0
    iet = np.diff(np.sort(fire_ts_ns)).astype(np.float64) / 1e9   # seconds
    iet = iet[iet > 0]
    if len(iet) < 20:
        return float("nan"), float("nan"), 0
    cv2_raw = float(np.var(iet) / max(np.mean(iet) ** 2, 1e-12))
    within = iet[iet <= gap_seconds]
    if len(within) < 20:
        return cv2_raw, float("nan"), int(len(within))
    cv2_gap = float(np.var(within) / max(np.mean(within) ** 2, 1e-12))
    return cv2_raw, cv2_gap, int(len(within))


def _threshold_stats(
    z: np.ndarray, ts_ns: np.ndarray, hours: np.ndarray,
    sess_labels: np.ndarray, bar_id: np.ndarray, t: float,
) -> dict:
    """All structure metrics for one z-threshold (firing rate is context only —
    the verdict reads STRUCTURE, not rate)."""
    fired = z > t
    n = len(z)
    p_obs = float(fired.mean()) if n else 0.0
    p_norm = float(norm.sf(t))
    excess = float(p_obs / p_norm) if p_norm > 1e-12 else float("inf")

    # session concentration (rate per session, active/quiet)
    sess_rate: dict[str, float] = {}
    for s in ("asia", "london", "overlap", "ny", "off"):
        m = sess_labels == s
        sess_rate[s] = float(fired[m].mean()) if m.any() else float("nan")
    active = max((sess_rate.get(k) or 0.0) for k in ("london", "overlap", "ny"))
    quiet = max(max((sess_rate.get(k) or 0.0) for k in ("asia", "off")), 1e-12)
    sess_ratio = float(active / quiet)

    # hour concentration — BOTH count-based and rate-based
    fire_hours = hours[fired]
    counts = np.array([int((fire_hours == h).sum()) for h in range(24)])
    rates = np.array([float(fired[hours == h].mean()) if (hours == h).any() else 0.0
                      for h in range(24)])
    nz_counts = counts[counts > 0]
    nz_rates = rates[rates > 0]
    hour_count_ratio = (float(counts.max() / max(np.median(nz_counts), 1e-9))
                        if len(nz_counts) >= 4 else float("nan"))
    hour_rate_ratio = (float(rates.max() / max(np.median(nz_rates), 1e-9))
                       if len(nz_rates) >= 4 else float("nan"))
    peak_hours = sorted(range(24), key=lambda h: -counts[h])[:5]

    # gap-aware clustering
    cv2_raw, cv2_gap, n_within = _gap_aware_cv2(ts_ns[fired], GAP_SECONDS)

    # per-bar median fire FRACTION (not any() — avoids saturation)
    n_fire = int(fired.sum())
    if n_fire > 0:
        s = pd.Series(fired.astype(np.float64))
        per_bar_frac = s.groupby(bar_id).mean()
        per_bar_median = float(per_bar_frac[per_bar_frac > 0].median()) if (per_bar_frac > 0).any() else 0.0
    else:
        per_bar_median = 0.0

    return {
        "threshold": float(t),
        "firing_rate": p_obs,
        "p_normal": p_norm,
        "excess_over_normal": excess,
        "n_firings": n_fire,
        "session_rates": sess_rate,
        "session_active_quiet": sess_ratio,
        "hour_count_ratio": hour_count_ratio,
        "hour_rate_ratio": hour_rate_ratio,
        "peak_hours_utc": peak_hours,
        "cv2_raw": cv2_raw,
        "cv2_gap_aware": cv2_gap,
        "n_intra_session_iet": n_within,
        "per_bar_median_fire_frac": per_bar_median,
    }


def _compute_stats(detected: pd.DataFrame, bar_freq: str = "5min") -> dict:
    settled = detected.iloc[WARMUP_TICKS:].reset_index(drop=True)
    z = pd.to_numeric(settled["absorb_z"], errors="coerce").to_numpy(np.float64)
    ts = pd.to_datetime(settled["ts_event"], utc=True)
    ts_ns = ts.astype("int64").to_numpy()
    hours = ts.dt.hour.to_numpy()
    sess_labels = np.array([_exclusive_session(int(h)) for h in hours])
    bar_id = ts.dt.floor(bar_freq).astype("int64").to_numpy()

    sweep = [_threshold_stats(z, ts_ns, hours, sess_labels, bar_id, t)
             for t in SWEEP_THRESHOLDS]
    pct = {f"p{p}": float(np.percentile(z, p)) for p in PERCENTILES}
    pct["max"] = float(z.max()) if len(z) else float("nan")
    return {"n_settled_ticks": int(len(z)), "absorb_z_percentiles": pct, "sweep": sweep}


def _overall_verdict(stats: dict) -> dict:
    """Anchor the tail threshold via excess-over-normal, then judge STRUCTURE.

    Verdicts:
      SIGNAL_LIKELY_ABSENT : no threshold reaches the excess anchor → the tail
                             is no fatter than noise.
      STRUCTURED           : at the tail anchor, firings concentrate in liquid
                             sessions AND cluster (gap-aware) → real structure,
                             the production z>1 was simply too loose.
      PARTIAL_STRUCTURE    : tail is fat + ONE of {concentrated, clustered}.
      DIFFUSE_TAIL         : tail is fat but flat across sessions + unclustered.
    """
    anchor = None
    for s in stats["sweep"]:
        if s["excess_over_normal"] >= EXCESS_TAIL_ANCHOR and s["n_firings"] >= MIN_TAIL_FIRINGS:
            anchor = s
            break
    if anchor is None:
        return {"verdict": "SIGNAL_LIKELY_ABSENT", "anchor_threshold": None,
                "reason": "absorb_z tail is no fatter than N(0,1) at any swept threshold"}

    sess_ok = np.isfinite(anchor["session_active_quiet"]) and anchor["session_active_quiet"] >= SESSION_STRUCT_RATIO
    clust_ok = np.isfinite(anchor["cv2_gap_aware"]) and anchor["cv2_gap_aware"] >= CV2_CLUSTERED
    if sess_ok and clust_ok:
        v = "STRUCTURED"
    elif sess_ok or clust_ok:
        v = "PARTIAL_STRUCTURE"
    else:
        v = "DIFFUSE_TAIL"
    return {
        "verdict": v,
        "anchor_threshold": anchor["threshold"],
        "anchor_excess": anchor["excess_over_normal"],
        "anchor_session_ratio": anchor["session_active_quiet"],
        "anchor_cv2_gap_aware": anchor["cv2_gap_aware"],
        "session_concentrated": bool(sess_ok),
        "time_clustered": bool(clust_ok),
    }


def run_breathing_test(
    mbo_path: Path, output_dir: Path, *, bar_freq: str = "5min",
    config: AbsorptionConfig | None = None,
) -> dict:
    trades = _load_trades_mbo(mbo_path)
    out = AbsorptionDetector(config).compute(trades)
    stats = _compute_stats(out, bar_freq=bar_freq)
    overall = _overall_verdict(stats)
    summary = {
        "mbo_path": str(mbo_path),
        "n_input_trades": int(len(trades)),
        "warmup_dropped": WARMUP_TICKS,
        "overall": overall,
        "stats": stats,
        "params": {
            "sweep_thresholds": list(SWEEP_THRESHOLDS),
            "excess_tail_anchor": EXCESS_TAIL_ANCHOR,
            "session_struct_ratio": SESSION_STRUCT_RATIO,
            "cv2_clustered": CV2_CLUSTERED,
            "gap_seconds": GAP_SECONDS,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breathing_test_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "breathing_test_report.txt", summary)
    return summary


def _write_report(path: Path, summary: dict) -> None:
    st = summary["stats"]
    ov = summary["overall"]
    L: list[str] = []
    L.append("═" * 92)
    L.append("Absorption breathing test v2 — is the absorb_z TAIL structured on YOUR data?")
    L.append("═" * 92)
    L.append(f"mbo            : {summary['mbo_path']}")
    L.append(f"input trades   : {summary['n_input_trades']:,}   settled: {st['n_settled_ticks']:,}")
    L.append("")
    pct = st["absorb_z_percentiles"]
    L.append("absorb_z distribution: " + "  ".join(f"{k}={v:.2f}" for k, v in pct.items()))
    L.append("")
    L.append(f"OVERALL: {ov['verdict']}")
    if ov.get("anchor_threshold") is not None:
        L.append(f"  tail anchored at z>{ov['anchor_threshold']:.1f} "
                 f"(excess={ov['anchor_excess']:.1f}× normal) → "
                 f"session_concentrated={ov['session_concentrated']} "
                 f"time_clustered={ov['time_clustered']}")
    else:
        L.append(f"  {ov.get('reason','')}")
    L.append("")
    L.append("THRESHOLD SWEEP (firing rate is context — verdict reads STRUCTURE):")
    L.append("  z>t  | firing% | excess× | sess a/q | hour(cnt) | hour(rate) | CV²raw  | CV²gap | bar_frac | n_fire")
    L.append("  " + "-" * 104)
    for s in st["sweep"]:
        def _f(x, w=7, p=2):
            return (f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else f"{'n/a':>{w}}")
        L.append(
            f"  {s['threshold']:.1f}  |"
            f"{s['firing_rate']*100:7.2f}% |"
            f"{_f(s['excess_over_normal'],7,1)} |"
            f"{_f(s['session_active_quiet'],8,2)}  |"
            f"{_f(s['hour_count_ratio'],8,2)} |"
            f"{_f(s['hour_rate_ratio'],9,2)}  |"
            f"{_f(s['cv2_raw'],7,1)} |"
            f"{_f(s['cv2_gap_aware'],6,1)} |"
            f"{_f(s['per_bar_median_fire_frac'],7,3)}  |"
            f"{s['n_firings']:>7,}"
        )
    L.append("  " + "-" * 104)
    L.append("KEY: excess× = P_observed(z>t) / P_normal(z>t); ≈1 = no fatter than noise, ≫1 = real tail.")
    L.append("     sess a/q = active(london/overlap/ny)/quiet(asia/off) firing-rate ratio (concentration).")
    L.append("     CV²gap = inter-event clustering with overnight/session gaps removed (Poisson=1).")
    L.append("R7 NOTE: STRUCTURED = institution-like tail structure, NOT proof of edge.")
    L.append("═" * 92)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Breathing v2: is the absorb_z tail structured (concentrated+clustered)?"
    )
    p.add_argument("--mbo", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bar-freq", default="5min")
    args = p.parse_args()
    summary = run_breathing_test(args.mbo, args.output, bar_freq=args.bar_freq)
    ov = summary["overall"]
    print(f"\n  OVERALL: {ov['verdict']}   (n_input_trades={summary['n_input_trades']:,})")
    if ov.get("anchor_threshold") is not None:
        print(f"  tail anchored at z>{ov['anchor_threshold']:.1f} (excess={ov['anchor_excess']:.1f}×): "
              f"concentrated={ov['session_concentrated']} clustered={ov['time_clustered']}")
    print(f"  report: {args.output}/breathing_test_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
