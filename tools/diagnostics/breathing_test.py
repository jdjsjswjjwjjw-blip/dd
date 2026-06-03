"""Breathing test — does AbsorptionDetector fire on YOUR production MBO?

The simulator (tests/test_phase1_12_absorption_simulator.py) proves the
detector works when the signal is present. This tool answers the
complementary question — R7's necessary-but-not-sufficient bar:

    On real data, does the detector fire with patterns consistent with
    institutional absorption, or is the signal absent?

Six structural stats per run; each carries a {PASS, SUSPECT, DEAD,
BROKEN_THRESHOLD} verdict; the overall verdict aggregates them.

LIMITS (R7 honesty):
- "Breathing" does NOT prove trading edge. It proves the detector
  responds to real data with patterns matching market structure.
- The final answer to "is there edge?" requires a decoder probe over
  the raw depth tensor (Phase 1, SSL-trained).
- This tool covers absorption only. IcebergDetector requires a separate
  raw-MBO ↔ simulator-vocabulary translator (deferred).

Uses the production AII engine via AbsorptionDetector (modules/features_v2/
absorption.py) — zero re-implementation. Reports written utf-8.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modules.features_v2.absorption import AbsorptionDetector, AbsorptionConfig


# ── MBO vocabulary (matches prepare_day_trading.py) ────────────────────────
TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "E", "0"}
BUY_SIDES = {"A", "ASK", "BUY", "BOT"}
SELL_SIDES = {"B", "BID", "S", "SELL"}

# ── Verdict thresholds (literature-anchored; see report header) ────────────
# Firing rate
RATE_PASS = (0.005, 0.05)      # 0.5% — 5%
RATE_DEAD = 0.0005             # < 0.05% → signal absent
RATE_BROKEN = 0.25             # > 25%  → z-score broken
# Session contrast (active / quiet)
SESSION_PASS_RATIO = 2.0
SESSION_SUSPECT_RATIO = 1.2
# Hour-of-day concentration (max / median)
HOUR_PASS_RATIO = 2.0
HOUR_SUSPECT_RATIO = 1.2
# Inter-event CV² (Poisson = 1.0)
CV2_PASS = 1.5
CV2_SUSPECT = 1.2
# absorb_z percentile (production fires at z > 1.0)
P99_PASS = 2.0
P99_SUSPECT = 1.5
P99_BROKEN = 1.2
# Per-bar fraction
BAR_FIRE_PASS = (0.01, 0.10)
BAR_FIRE_DEAD = 0.005
BAR_FIRE_BROKEN = 0.50

WARMUP_TICKS = 100             # drop z-score warmup before stats


def _verdict_rate(r: float) -> str:
    if r > RATE_BROKEN:   return "BROKEN_THRESHOLD"
    if r < RATE_DEAD:     return "DEAD"
    if RATE_PASS[0] <= r <= RATE_PASS[1]: return "PASS"
    return "SUSPECT"


def _verdict_ratio(r: float, pass_t: float, susp_t: float) -> str:
    if not np.isfinite(r):       return "DEAD"
    if r >= pass_t:              return "PASS"
    if r >= susp_t:              return "SUSPECT"
    return "DEAD"


def _verdict_p99(p99: float) -> str:
    if p99 < P99_BROKEN:         return "BROKEN_THRESHOLD"
    if p99 >= P99_PASS:          return "PASS"
    if p99 >= P99_SUSPECT:       return "SUSPECT"
    return "DEAD"


def _verdict_bar_fire(f: float) -> str:
    if f > BAR_FIRE_BROKEN:      return "BROKEN_THRESHOLD"
    if f < BAR_FIRE_DEAD:        return "DEAD"
    if BAR_FIRE_PASS[0] <= f <= BAR_FIRE_PASS[1]: return "PASS"
    return "SUSPECT"


def _exclusive_session(hour: int) -> str:
    """Priority overlap > ny > london > asia > off (matches M1)."""
    if 13 <= hour < 16:  return "overlap"
    if 7 <= hour < 13:   return "london"
    if 16 <= hour < 22:  return "ny"
    if 0 <= hour < 7:    return "asia"
    return "off"


def _load_trades_mbo(mbo_path: Path) -> pd.DataFrame:
    """Read MBO parquet (file or dir), keep only TRADE-action ticks with
    a recognized side. Returns a frame ready for AbsorptionDetector."""
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
    is_trade = act.isin(TRADE_ACTIONS)
    is_known_side = sd.isin(BUY_SIDES | SELL_SIDES)
    trades = df.loc[is_trade & is_known_side, ["ts_event", "price", "side", "size"]].reset_index(drop=True)
    if len(trades) == 0:
        raise RuntimeError(
            "no recognized TRADE-action ticks with known side; check MBO "
            f"action vocabulary (got {act.value_counts().head(5).to_dict()})"
        )
    return trades


def _compute_stats(out: pd.DataFrame, bar_freq: str) -> dict:
    """Build the 6-stat dict from AbsorptionDetector.compute() output."""
    settled = out.iloc[WARMUP_TICKS:].reset_index(drop=True)
    n = len(settled)
    fired = settled["fired"].to_numpy(dtype=bool)
    z = settled["absorb_z"].to_numpy()
    ts = pd.to_datetime(settled["ts_event"], utc=True)
    hours = ts.dt.hour.to_numpy()

    # 1. firing rate
    fire_rate = float(fired.mean()) if n else 0.0

    # 2. session distribution
    sess_labels = np.array([_exclusive_session(int(h)) for h in hours])
    sess_rate: dict[str, float] = {}
    sess_n: dict[str, int] = {}
    for s in ("asia", "london", "overlap", "ny", "off"):
        m = sess_labels == s
        sess_n[s] = int(m.sum())
        sess_rate[s] = float(fired[m].mean()) if m.any() else float("nan")
    active = max(sess_rate.get("london", 0) or 0, sess_rate.get("ny", 0) or 0,
                 sess_rate.get("overlap", 0) or 0)
    quiet = max(sess_rate.get("asia", 0) or 0, sess_rate.get("off", 0) or 0, 1e-12)
    session_ratio = float(active / quiet) if quiet > 0 else float("nan")

    # 3. hour-of-day concentration
    hour_rate: dict[int, float] = {}
    for h in range(24):
        m = hours == h
        hour_rate[h] = float(fired[m].mean()) if m.any() else 0.0
    hr_vals = np.array([hour_rate[h] for h in range(24) if hour_rate[h] > 0 or any(hours == h)])
    if len(hr_vals) >= 4:
        median = float(np.median(hr_vals))
        hour_ratio = float(np.max(hr_vals) / max(median, 1e-12))
    else:
        hour_ratio = float("nan")

    # 4. inter-event clustering — CV² of inter-event time in seconds
    fire_ts = ts[fired]
    if len(fire_ts) >= 30:
        iet = fire_ts.diff().dt.total_seconds().dropna().to_numpy()
        iet = iet[iet > 0]
        if len(iet) >= 20:
            cv2 = float(np.var(iet) / max(np.mean(iet) ** 2, 1e-12))
        else:
            cv2 = float("nan")
    else:
        cv2 = float("nan")

    # 5. absorb_z percentile tail
    z_p = {p: float(np.percentile(z, p)) for p in (50, 90, 95, 99)}
    z_max = float(z.max()) if n else float("nan")

    # 6. per-bar firing
    df_bar = pd.DataFrame({"ts_event": ts, "fired": fired})
    df_bar["bar"] = df_bar["ts_event"].dt.floor(bar_freq)
    per_bar = df_bar.groupby("bar")["fired"].any()
    bar_fire_frac = float(per_bar.mean()) if len(per_bar) else 0.0
    n_bars = int(len(per_bar))

    # verdicts
    v_rate = _verdict_rate(fire_rate)
    v_sess = _verdict_ratio(session_ratio, SESSION_PASS_RATIO, SESSION_SUSPECT_RATIO)
    v_hour = _verdict_ratio(hour_ratio, HOUR_PASS_RATIO, HOUR_SUSPECT_RATIO)
    v_cv2 = _verdict_ratio(cv2, CV2_PASS, CV2_SUSPECT)
    v_p99 = _verdict_p99(z_p[99])
    v_bar = _verdict_bar_fire(bar_fire_frac)

    return {
        "n_settled_ticks": n,
        "firing_rate": {"value": fire_rate, "verdict": v_rate},
        "session_distribution": {
            "rates_per_session": sess_rate, "n_per_session": sess_n,
            "active_quiet_ratio": session_ratio, "verdict": v_sess,
        },
        "hour_of_day": {
            "rates": hour_rate, "max_over_median_ratio": hour_ratio,
            "verdict": v_hour,
        },
        "inter_event_clustering": {
            "cv2": cv2, "n_firings": int(len(fire_ts)), "verdict": v_cv2,
        },
        "absorb_z_distribution": {
            "p50": z_p[50], "p90": z_p[90], "p95": z_p[95], "p99": z_p[99],
            "max": z_max, "verdict": v_p99,
        },
        "per_bar_firing": {
            "bar_freq": bar_freq, "n_bars": n_bars,
            "bars_with_fire_fraction": bar_fire_frac, "verdict": v_bar,
        },
    }


def _overall_verdict(stats: dict) -> str:
    verdicts = [
        stats["firing_rate"]["verdict"],
        stats["session_distribution"]["verdict"],
        stats["hour_of_day"]["verdict"],
        stats["inter_event_clustering"]["verdict"],
        stats["absorb_z_distribution"]["verdict"],
        stats["per_bar_firing"]["verdict"],
    ]
    n_pass = verdicts.count("PASS")
    n_broken = verdicts.count("BROKEN_THRESHOLD")
    n_dead = verdicts.count("DEAD")

    # Detector wrongly calibrated (z-score never reaches threshold OR fires constantly)
    if (stats["firing_rate"]["verdict"] == "BROKEN_THRESHOLD"
            or stats["absorb_z_distribution"]["verdict"] == "BROKEN_THRESHOLD"
            or n_broken >= 2):
        return "DETECTOR_BROKEN"
    # Signal looks absent if multiple "absent" stats fire together
    if (stats["firing_rate"]["verdict"] == "DEAD"
            or (n_dead >= 3 and n_pass <= 1)):
        return "SIGNAL_LIKELY_ABSENT"
    if n_pass >= 4:
        return "HEALTHY"
    return "SUSPECT"


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
        "overall_verdict": overall,
        "stats": stats,
        "thresholds": {
            "firing_rate_pass": list(RATE_PASS), "firing_rate_dead": RATE_DEAD,
            "firing_rate_broken": RATE_BROKEN,
            "session_pass_ratio": SESSION_PASS_RATIO,
            "hour_pass_ratio": HOUR_PASS_RATIO,
            "cv2_pass": CV2_PASS, "p99_pass": P99_PASS,
            "p99_broken": P99_BROKEN,
            "bar_fire_pass": list(BAR_FIRE_PASS),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breathing_test_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "breathing_test_report.txt", summary)
    return summary


def _write_report(path: Path, summary: dict) -> None:
    s = summary["stats"]
    L: list[str] = []
    L.append("═" * 78)
    L.append("Absorption-detector breathing test — does it fire on YOUR data?")
    L.append("═" * 78)
    L.append(f"mbo            : {summary['mbo_path']}")
    L.append(f"input trades   : {summary['n_input_trades']:,}")
    L.append(f"settled ticks  : {s['n_settled_ticks']:,}  (warmup dropped: {summary['warmup_dropped']})")
    L.append("")
    L.append(f"OVERALL VERDICT: {summary['overall_verdict']}")
    L.append("─" * 78)
    fr = s["firing_rate"]
    L.append(f"1. firing rate                  : {fr['value']:.4%}    → {fr['verdict']}")
    sd = s["session_distribution"]
    L.append(f"2. session distribution         : active/quiet = {sd['active_quiet_ratio']:.2f}×    → {sd['verdict']}")
    for k in ("asia", "london", "overlap", "ny", "off"):
        r = sd["rates_per_session"].get(k, float("nan"))
        nq = sd["n_per_session"].get(k, 0)
        L.append(f"     {k:8s}: rate={r:.4%}   n={nq:,}")
    hd = s["hour_of_day"]
    L.append(f"3. hour-of-day concentration    : max/median = {hd['max_over_median_ratio']:.2f}×    → {hd['verdict']}")
    L.append("     peak hours (UTC) : " + ", ".join(
        f"{h:02d}h={hd['rates'][h]:.3%}"
        for h in sorted(hd["rates"], key=lambda k: -hd["rates"][k])[:5]
    ))
    iec = s["inter_event_clustering"]
    L.append(f"4. inter-event clustering CV²   : {iec['cv2']:.2f}   (Poisson=1.0)    → {iec['verdict']}")
    L.append(f"     n_firings used : {iec['n_firings']:,}")
    az = s["absorb_z_distribution"]
    L.append(f"5. absorb_z tail                : p99={az['p99']:.2f}  p95={az['p95']:.2f}  max={az['max']:.2f}    → {az['verdict']}")
    pb = s["per_bar_firing"]
    L.append(f"6. per-bar firing ({pb['bar_freq']})         : {pb['bars_with_fire_fraction']:.2%} of {pb['n_bars']:,} bars    → {pb['verdict']}")
    L.append("─" * 78)
    L.append("interpretation: HEALTHY = signal likely present; SIGNAL_LIKELY_ABSENT =")
    L.append("  rate low + flat sessions + Poisson-like; DETECTOR_BROKEN = z-score")
    L.append("  threshold miscalibrated for this instrument; SUSPECT = mixed.")
    L.append("R7 NOTE: this is a structural/breathing indicator, not proof of edge.")
    L.append("═" * 78)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run AbsorptionDetector on real MBO and assess whether "
                    "it 'breathes' (fires with patterns matching market structure)"
    )
    p.add_argument("--mbo", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bar-freq", default="5min")
    args = p.parse_args()
    summary = run_breathing_test(args.mbo, args.output, bar_freq=args.bar_freq)
    s = summary["stats"]
    print(f"\n  OVERALL: {summary['overall_verdict']}    (n_input_trades={summary['n_input_trades']:,})")
    for key, label in [
        ("firing_rate",          "1. firing rate                "),
        ("session_distribution", "2. session distribution       "),
        ("hour_of_day",          "3. hour-of-day concentration  "),
        ("inter_event_clustering","4. inter-event clustering CV²"),
        ("absorb_z_distribution", "5. absorb_z tail (p99)        "),
        ("per_bar_firing",       "6. per-bar firing fraction    "),
    ]:
        print(f"  {label}: {s[key]['verdict']}")
    print(f"\n  report: {args.output}/breathing_test_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
