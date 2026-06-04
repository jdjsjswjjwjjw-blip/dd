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


def _bar_tick_counts(settled: pd.DataFrame, bar_freq: str) -> pd.Series:
    """Ticks per bar at bar_freq, indexed by bar timestamp."""
    bar_id = pd.to_datetime(settled["ts_event"], utc=True).dt.floor(bar_freq)
    return bar_id.value_counts().sort_index()


def _thin_tick_mask(settled: pd.DataFrame, min_ticks: int, bar_freq: str) -> np.ndarray:
    """Boolean mask: True iff the tick belongs to a bar with >= min_ticks ticks.
    min_ticks <= 0 returns an all-True mask (no filter — preserves v2 behaviour)."""
    n = len(settled)
    if min_ticks <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    bar_id = pd.to_datetime(settled["ts_event"], utc=True).dt.floor(bar_freq)
    counts = bar_id.value_counts()
    keep_bars = counts[counts >= min_ticks].index
    return bar_id.isin(keep_bars).to_numpy()


def _compute_stats_from_settled(settled: pd.DataFrame, bar_freq: str = "5min") -> dict:
    """Stat computation on already-settled (warmup-dropped) detector output.
    Used by _compute_stats (which drops warmup first) and by the filtered path
    in run_breathing_test (which filters AFTER warmup drop)."""
    n = len(settled)
    if n == 0:
        return {"n_settled_ticks": 0, "absorb_z_percentiles": {}, "sweep": []}
    z = pd.to_numeric(settled["absorb_z"], errors="coerce").to_numpy(np.float64)
    ts = pd.to_datetime(settled["ts_event"], utc=True)
    ts_ns = ts.astype("int64").to_numpy()
    hours = ts.dt.hour.to_numpy()
    sess_labels = np.array([_exclusive_session(int(h)) for h in hours])
    bar_id = ts.dt.floor(bar_freq).astype("int64").to_numpy()
    sweep = [_threshold_stats(z, ts_ns, hours, sess_labels, bar_id, t)
             for t in SWEEP_THRESHOLDS]
    pct = {f"p{p}": float(np.percentile(z, p)) for p in PERCENTILES}
    pct["max"] = float(z.max())
    return {"n_settled_ticks": int(n), "absorb_z_percentiles": pct, "sweep": sweep}


def _compute_stats(detected: pd.DataFrame, bar_freq: str = "5min") -> dict:
    """v2 entry point: drop warmup, then compute stats. Kept for back-compat
    with the existing test suite."""
    settled = detected.iloc[WARMUP_TICKS:].reset_index(drop=True)
    return _compute_stats_from_settled(settled, bar_freq=bar_freq)


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


def _run_pipeline(
    detected: pd.DataFrame, bar_freq: str, min_ticks_per_bar: int,
) -> dict:
    """Core pipeline (file-I/O free, so tests can call it directly).

    Always produces the unfiltered v2 result. If min_ticks_per_bar > 0, ALSO
    produces a filtered result (thin bars dropped) side-by-side — the report
    shows both so nothing is hidden (R10: visual diff, no silent filtering)."""
    settled = detected.iloc[WARMUP_TICKS:].reset_index(drop=True)
    # ── always: unfiltered (v2 baseline)
    stats_unfilt = _compute_stats_from_settled(settled, bar_freq=bar_freq)
    overall_unfilt = _overall_verdict(stats_unfilt)
    # ── tick-density distribution + p25 suggestion (R10: derived, not targeted)
    bar_counts = _bar_tick_counts(settled, bar_freq)
    tpb_dist: dict[str, float] = {}
    suggested_p25 = None
    if len(bar_counts) > 0:
        tpb_dist = {f"p{p}": float(np.percentile(bar_counts.to_numpy(), p))
                    for p in (10, 25, 50, 75, 90)}
        suggested_p25 = int(round(tpb_dist["p25"]))
    out: dict = {
        "warmup_dropped": WARMUP_TICKS,
        "bar_freq": bar_freq,
        "min_ticks_per_bar": int(min_ticks_per_bar),
        "ticks_per_bar_distribution": tpb_dist,
        "ticks_per_bar_suggested_n_p25": suggested_p25,
        "unfiltered": {"overall": overall_unfilt, "stats": stats_unfilt},
        "params": {
            "sweep_thresholds": list(SWEEP_THRESHOLDS),
            "excess_tail_anchor": EXCESS_TAIL_ANCHOR,
            "session_struct_ratio": SESSION_STRUCT_RATIO,
            "cv2_clustered": CV2_CLUSTERED,
            "gap_seconds": GAP_SECONDS,
        },
    }
    # ── optional: filtered (only when user requests, default 0 == v2 byte-equal)
    if min_ticks_per_bar > 0 and len(settled) > 0:
        mask = _thin_tick_mask(settled, min_ticks_per_bar, bar_freq)
        kept = settled.loc[mask].reset_index(drop=True)
        n_bars_total = int(len(bar_counts))
        n_bars_kept = int((bar_counts >= min_ticks_per_bar).sum())
        stats_filt = _compute_stats_from_settled(kept, bar_freq=bar_freq)
        overall_filt = _overall_verdict(stats_filt)
        out["filtered"] = {
            "n_bars_total": n_bars_total,
            "n_bars_kept": n_bars_kept,
            "fraction_bars_kept": float(n_bars_kept / max(n_bars_total, 1)),
            "n_ticks_kept": int(mask.sum()),
            "overall": overall_filt,
            "stats": stats_filt,
        }
    return out


def run_breathing_test(
    mbo_path: Path, output_dir: Path, *, bar_freq: str = "5min",
    min_ticks_per_bar: int = 0, config: AbsorptionConfig | None = None,
) -> dict:
    trades = _load_trades_mbo(mbo_path)
    detected = AbsorptionDetector(config).compute(trades)
    pipeline = _run_pipeline(detected, bar_freq=bar_freq,
                             min_ticks_per_bar=min_ticks_per_bar)
    summary = {
        "mbo_path": str(mbo_path),
        "n_input_trades": int(len(trades)),
        **pipeline,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breathing_test_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "breathing_test_report.txt", summary)
    return summary


def _sweep_table(stats: dict) -> list[str]:
    """Render the per-threshold sweep table rows + key."""
    def _f(x, w=7, p=2):
        return (f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else f"{'n/a':>{w}}")
    L = ["  z>t  | firing% | excess× | sess a/q | hour(cnt) | hour(rate) | CV²raw  | CV²gap | bar_frac | n_fire",
         "  " + "-" * 104]
    for s in stats["sweep"]:
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
    return L


def _section(title: str, stats: dict, overall: dict) -> list[str]:
    L = [title]
    pct = stats.get("absorb_z_percentiles", {})
    L.append("  absorb_z distribution: " + "  ".join(f"{k}={v:.2f}" for k, v in pct.items()))
    L.append(f"  OVERALL: {overall['verdict']}")
    if overall.get("anchor_threshold") is not None:
        L.append(f"    tail anchored at z>{overall['anchor_threshold']:.1f} "
                 f"(excess={overall['anchor_excess']:.1f}× normal) → "
                 f"session_concentrated={overall['session_concentrated']} "
                 f"time_clustered={overall['time_clustered']}")
    else:
        L.append(f"    {overall.get('reason','')}")
    L.extend(_sweep_table(stats))
    return L


def _write_report(path: Path, summary: dict) -> None:
    L: list[str] = []
    L.append("═" * 92)
    L.append("Absorption breathing test v2 — TAIL structure + thin-bar artifact diagnostic")
    L.append("═" * 92)
    L.append(f"mbo            : {summary['mbo_path']}")
    unfilt = summary["unfiltered"]
    L.append(f"input trades   : {summary['n_input_trades']:,}   "
             f"settled: {unfilt['stats']['n_settled_ticks']:,}")
    tpb = summary.get("ticks_per_bar_distribution") or {}
    if tpb:
        L.append("ticks-per-bar  : " + "  ".join(f"{k}={int(round(v))}" for k, v in tpb.items()))
        sug = summary.get("ticks_per_bar_suggested_n_p25")
        L.append(f"suggested min-ticks-per-bar (= p25, R10-derived): {sug}")
    L.append(f"filter applied : min_ticks_per_bar = {summary['min_ticks_per_bar']}")
    L.append("")
    L.extend(_section("ABSORB_Z TAIL (full data, no filter):",
                      unfilt["stats"], unfilt["overall"]))
    if "filtered" in summary:
        f = summary["filtered"]
        L.append("")
        L.append(f"FILTERED (min_ticks_per_bar = {summary['min_ticks_per_bar']}):  "
                 f"kept {f['n_bars_kept']:,}/{f['n_bars_total']:,} bars "
                 f"({100*f['fraction_bars_kept']:.1f}%)   "
                 f"ticks kept: {f['n_ticks_kept']:,}")
        L.extend(_section("ABSORB_Z TAIL (thin bars dropped):",
                          f["stats"], f["overall"]))
        L.append("")
        L.append("DIAGNOSTIC (R10: visual diff, no silent filtering):")
        unfilt_sess = unfilt["overall"].get("anchor_session_ratio")
        filt_sess = f["overall"].get("anchor_session_ratio")
        if isinstance(unfilt_sess, (int, float)) and isinstance(filt_sess, (int, float)):
            L.append(f"  anchor sess a/q : unfiltered={unfilt_sess:.2f}   filtered={filt_sess:.2f}")
            if filt_sess >= SESSION_STRUCT_RATIO and unfilt_sess < SESSION_STRUCT_RATIO:
                L.append("  → tail-structure REVEALED by thin-bar filter: institutional signal was "
                         "masked by thin-bar AII artifacts.")
            elif filt_sess < SESSION_STRUCT_RATIO and unfilt_sess < SESSION_STRUCT_RATIO:
                L.append("  → no jump after filter: tail is genuinely diffuse, NOT a thin-bar artifact.")
    L.append("")
    L.append("KEY: excess× = P_observed(z>t) / P_normal(z>t); ≈1 = no fatter than noise, ≫1 = real tail.")
    L.append("     sess a/q = active(london/overlap/ny)/quiet(asia/off) firing-rate ratio (concentration).")
    L.append("     CV²gap = inter-event clustering with overnight/session gaps removed (Poisson=1).")
    L.append("R7 NOTE: STRUCTURED = institution-like tail structure, NOT proof of edge.")
    L.append("═" * 92)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Breathing v2: tail structure + thin-bar artifact diagnostic."
    )
    p.add_argument("--mbo", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bar-freq", default="5min")
    p.add_argument("--min-ticks-per-bar", type=int, default=0,
                   help="If > 0, also compute a FILTERED result where bars with "
                        "fewer than this many ticks are dropped. The unfiltered "
                        "result is always reported. The report prints the "
                        "ticks-per-bar distribution and a p25 suggestion. "
                        "Default 0 = no filter (byte-equal to v2).")
    args = p.parse_args()
    summary = run_breathing_test(
        args.mbo, args.output, bar_freq=args.bar_freq,
        min_ticks_per_bar=args.min_ticks_per_bar,
    )

    def _print_ov(label: str, ov: dict) -> None:
        print(f"  {label}: {ov['verdict']}")
        if ov.get("anchor_threshold") is not None:
            print(f"    z>{ov['anchor_threshold']:.1f} (excess={ov['anchor_excess']:.1f}×) "
                  f"concentrated={ov['session_concentrated']} "
                  f"clustered={ov['time_clustered']}")

    print(f"\n  n_input_trades={summary['n_input_trades']:,}   "
          f"min_ticks_per_bar={summary['min_ticks_per_bar']}")
    sug = summary.get("ticks_per_bar_suggested_n_p25")
    if sug is not None:
        print(f"  suggested min_ticks_per_bar (p25 of ticks/bar): {sug}")
    _print_ov("OVERALL (unfiltered)", summary["unfiltered"]["overall"])
    if "filtered" in summary:
        _print_ov("OVERALL (filtered)  ", summary["filtered"]["overall"])
    print(f"  report: {args.output}/breathing_test_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
