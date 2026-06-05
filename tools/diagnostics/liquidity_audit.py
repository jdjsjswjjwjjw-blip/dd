"""liquidity_audit — is the extracted contract genuinely the LIQUID front month?

R8 data-integrity gate. The entire 4-tool conclusion ("no directional edge on
6B") is only valid if the extracted contract is the liquid front month. If the
continuous-contract extractor pulled a THIN/far-month contract, the order book
is a few ghost orders and every microstructure feature is meaningless — the
conclusion would be an artefact of bad data, not a market truth.

Checks (numbers, not vibes):
  1. SYMBOL identity — which contract(s) are present, is there one dominant
     front month, are there far-month rows mixed in?         (raw MBO)
  2. SPREAD — median / p10 / p90 bid-ask spread in ticks.    (raw MBO MBP-10)
  3. DEPTH — top-of-book size + how many of the 10 levels are filled. (raw MBO)
  4. VOLUME per bar — median / p10 of trade volume per 5-min bar. (features)
  5. LOW_LIQUIDITY regime fraction — high ⇒ thin?            (features)
  6. ROLL gaps — adjacent-bar price jumps that betray a bad continuous splice.
                                                              (features)

The raw-MBO checks (1-3) are the decisive ones; if the file lacks MBP-10 book
columns (pure MBO order stream), the tool reports that and falls back to the
checks it can run, never fabricating depth numbers.

A priori thresholds (R10) calibrated for 6B = GBP/USD futures (CME, tick 1e-4):
  liquid: median spread <= 2 ticks, top-of-book median size >= 5, >= 8/10
          levels filled, low_liquidity regime < 25%, no >0.5% adjacent-bar
          gaps outside session breaks.
Different instruments need re-calibration — thresholds are reported in the
output so the operator can judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TICK_SIZE = 0.0001          # 6B / GBPUSD futures
TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "E", "0"}

# A priori liquidity thresholds (R10) — calibrated for 6B; reported in output.
SPREAD_LIQUID_TICKS = 2.0
DEPTH_L0_LIQUID = 5.0
LEVELS_FILLED_LIQUID = 8
LOW_LIQ_REGIME_MAX = 0.25
ROLL_GAP_PCT = 0.005               # 0.5% adjacent-bar jump = suspicious splice


def _levels_present(cols: list[str], side: str, kind: str) -> list[str]:
    """e.g. side='bid' kind='px' → bid_px_00..09 that actually exist."""
    return [f"{side}_{kind}_{i:02d}" for i in range(10)
            if f"{side}_{kind}_{i:02d}" in cols]


# ── 1-3: raw MBO checks (symbol, spread, depth) ────────────────────────────
def audit_raw_mbo(mbo_path: Path, *, stride: int, tick_size: float) -> dict[str, Any]:
    import pyarrow.parquet as pq

    p = Path(mbo_path)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    schema_cols = pq.ParquetFile(files[0]).schema_arrow.names

    want = ["ts_event", "symbol", "action", "side", "price", "size"]
    bid_px = _levels_present(schema_cols, "bid", "px")
    ask_px = _levels_present(schema_cols, "ask", "px")
    bid_sz = _levels_present(schema_cols, "bid", "sz")
    ask_sz = _levels_present(schema_cols, "ask", "sz")
    read_cols = [c for c in want + bid_px + ask_px + bid_sz + ask_sz if c in schema_cols]

    frames = [pd.read_parquet(f, columns=read_cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    n = len(df)

    out: dict[str, Any] = {"n_ticks_audited": int(n), "stride": stride,
                           "has_mbp10_book": bool(bid_px and ask_px)}

    # 1. SYMBOL identity
    if "symbol" in df.columns:
        vc = df["symbol"].astype(str).value_counts()
        out["symbol"] = {
            "distinct": int(vc.shape[0]),
            "top": {str(k): int(v) for k, v in vc.head(5).items()},
            "dominant_share": float(vc.iloc[0] / max(vc.sum(), 1)),
        }
    else:
        out["symbol"] = {"missing": True}

    # 2. SPREAD (needs L0 bid/ask px)
    if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
        bid0 = pd.to_numeric(df["bid_px_00"], errors="coerce")
        ask0 = pd.to_numeric(df["ask_px_00"], errors="coerce")
        spread = (ask0 - bid0)
        valid = spread[np.isfinite(spread) & (spread > 0)]
        spr_ticks = (valid / tick_size)
        out["spread_ticks"] = {
            "median": float(spr_ticks.median()) if len(spr_ticks) else float("nan"),
            "p10": float(spr_ticks.quantile(0.10)) if len(spr_ticks) else float("nan"),
            "p90": float(spr_ticks.quantile(0.90)) if len(spr_ticks) else float("nan"),
            "n_valid": int(len(spr_ticks)),
        }
    else:
        out["spread_ticks"] = {"missing": "no bid_px_00/ask_px_00 (not MBP-10)"}

    # 3. DEPTH (L0 size + fill across 10 levels)
    if bid_sz and ask_sz:
        b0 = pd.to_numeric(df.get("bid_sz_00", np.nan), errors="coerce")
        a0 = pd.to_numeric(df.get("ask_sz_00", np.nan), errors="coerce")
        # fraction of the 10 levels with non-zero size (median across ticks)
        def _filled_frac(cols):
            if not cols:
                return float("nan")
            arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
            return float(np.nanmedian((arr > 0).sum(axis=1)))
        out["depth"] = {
            "median_bid_sz_L0": float(b0.median()) if b0.notna().any() else float("nan"),
            "median_ask_sz_L0": float(a0.median()) if a0.notna().any() else float("nan"),
            "median_bid_levels_filled": _filled_frac(bid_sz),
            "median_ask_levels_filled": _filled_frac(ask_sz),
            "n_bid_levels_in_schema": len(bid_sz),
        }
    else:
        out["depth"] = {"missing": "no bid_sz/ask_sz (not MBP-10)"}

    # trade tick density (works on any MBO with action/side)
    if "action" in df.columns:
        act = df["action"].astype(str).str.upper().str.strip()
        out["trade_tick_fraction"] = float(act.isin(TRADE_ACTIONS).mean())
    return out


# ── 4-6: features-parquet checks (volume, regime, roll gaps) ───────────────
def audit_features(features_path: Path) -> dict[str, Any]:
    df = pd.read_parquet(features_path)
    out: dict[str, Any] = {"n_bars": int(len(df))}

    # 4. volume per bar
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        out["volume_per_bar"] = {
            "median": float(v.median()), "p10": float(v.quantile(0.10)),
            "min": float(v.min()), "zero_frac": float((v == 0).mean()),
        }
    if "tick_count" in df.columns:
        t = pd.to_numeric(df["tick_count"], errors="coerce")
        out["tick_count_per_bar"] = {
            "median": float(t.median()), "p10": float(t.quantile(0.10)),
        }

    # 5. low_liquidity regime fraction
    if "regime_label" in df.columns:
        rl = df["regime_label"].astype(str)
        vc = rl.value_counts(normalize=True)
        out["regime_fraction"] = {str(k): float(v) for k, v in vc.items()}
        out["low_liquidity_fraction"] = float(vc.get("low_liquidity", 0.0))

    # coverage proxies (depth presence without raw book)
    for c in ("mbp_bar_coverage", "mbo_bar_coverage"):
        if c in df.columns:
            cov = pd.to_numeric(df[c], errors="coerce")
            out[c] = {"median": float(cov.median()), "zero_frac": float((cov == 0).mean())}

    # 6. roll gaps — adjacent-bar return outside session breaks
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy()
        ret = np.abs(np.diff(close) / np.clip(close[:-1], 1e-9, None))
        brk = (df["is_session_break"].astype(bool).to_numpy()[1:]
               if "is_session_break" in df.columns else np.zeros(len(ret), bool))
        non_break = ret[~brk]
        big = non_break[non_break > ROLL_GAP_PCT]
        out["roll_gaps"] = {
            "max_adjacent_return": float(non_break.max()) if len(non_break) else float("nan"),
            "n_gaps_over_threshold": int(len(big)),
            "threshold_pct": ROLL_GAP_PCT,
        }
    return out


def _verdict(raw: dict | None, feat: dict | None) -> dict[str, Any]:
    flags: list[str] = []
    liquid_signals, thin_signals = 0, 0

    if raw:
        spr = raw.get("spread_ticks", {})
        if isinstance(spr.get("median"), float) and np.isfinite(spr["median"]):
            if spr["median"] <= SPREAD_LIQUID_TICKS:
                liquid_signals += 1
            else:
                thin_signals += 1; flags.append(f"wide spread (median {spr['median']:.1f} ticks)")
        d = raw.get("depth", {})
        if isinstance(d.get("median_bid_sz_L0"), float) and np.isfinite(d["median_bid_sz_L0"]):
            if d["median_bid_sz_L0"] >= DEPTH_L0_LIQUID:
                liquid_signals += 1
            else:
                thin_signals += 1; flags.append(f"thin L0 depth (median {d['median_bid_sz_L0']:.1f})")
        if isinstance(d.get("median_bid_levels_filled"), float) and np.isfinite(d["median_bid_levels_filled"]):
            if d["median_bid_levels_filled"] >= LEVELS_FILLED_LIQUID:
                liquid_signals += 1
            else:
                thin_signals += 1; flags.append(f"few levels filled (median {d['median_bid_levels_filled']:.0f}/10)")
        sym = raw.get("symbol", {})
        if isinstance(sym.get("dominant_share"), float):
            if sym["dominant_share"] >= 0.80:
                liquid_signals += 1
            else:
                flags.append(f"multiple contracts mixed (dominant {sym['dominant_share']:.0%})")

    if feat:
        low_liq = feat.get("low_liquidity_fraction")
        if isinstance(low_liq, float):
            if low_liq < LOW_LIQ_REGIME_MAX:
                liquid_signals += 1
            else:
                thin_signals += 1; flags.append(f"high low_liquidity regime ({low_liq:.0%})")
        rg = feat.get("roll_gaps", {})
        if isinstance(rg.get("n_gaps_over_threshold"), int) and rg["n_gaps_over_threshold"] > 0:
            thr = rg.get("threshold_pct", ROLL_GAP_PCT)
            flags.append(f"{rg['n_gaps_over_threshold']} adjacent-bar gaps > {thr:.1%} (splice?)")
            thin_signals += 1

    if liquid_signals >= 3 and thin_signals == 0:
        verdict = "LIQUID_FRONT_MONTH (data integrity OK)"
    elif thin_signals >= 2:
        verdict = "THIN_CONTRACT_SUSPECTED (re-extract front month + redo analysis)"
    elif thin_signals == 1:
        verdict = "MIXED (one thin signal — investigate before trusting conclusions)"
    else:
        verdict = "INSUFFICIENT_DATA (book columns missing — checked what was available)"
    return {"verdict": verdict, "liquid_signals": liquid_signals,
            "thin_signals": thin_signals, "flags": flags}


def run_liquidity_audit(
    output_dir: Path, *, mbo_path: Path | None, features_path: Path | None,
    stride: int = 1, tick_size: float = DEFAULT_TICK_SIZE,
) -> dict[str, Any]:
    raw = audit_raw_mbo(mbo_path, stride=stride, tick_size=tick_size) if mbo_path else None
    feat = audit_features(features_path) if features_path else None
    verdict = _verdict(raw, feat)
    summary = {
        "mbo_path": str(mbo_path) if mbo_path else None,
        "features_path": str(features_path) if features_path else None,
        "tick_size": tick_size,
        "thresholds_a_priori": {
            "SPREAD_LIQUID_TICKS": SPREAD_LIQUID_TICKS, "DEPTH_L0_LIQUID": DEPTH_L0_LIQUID,
            "LEVELS_FILLED_LIQUID": LEVELS_FILLED_LIQUID,
            "LOW_LIQ_REGIME_MAX": LOW_LIQ_REGIME_MAX, "ROLL_GAP_PCT": ROLL_GAP_PCT,
        },
        "raw_mbo_audit": raw, "features_audit": feat, "overall": verdict,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "liquidity_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    _write_report(output_dir / "liquidity_audit_report.txt", summary)
    return summary


def _write_report(path: Path, s: dict) -> None:
    L = []
    L.append("═" * 88)
    L.append("Liquidity audit — is the extracted contract the LIQUID front month? (R8 data gate)")
    L.append("═" * 88)
    L.append(f"mbo      : {s['mbo_path']}")
    L.append(f"features : {s['features_path']}    tick_size: {s['tick_size']}")
    L.append("")
    r = s.get("raw_mbo_audit")
    if r:
        L.append(f"RAW MBO  (n_ticks audited: {r['n_ticks_audited']:,}, stride {r['stride']}, "
                 f"MBP-10 book: {r['has_mbp10_book']})")
        sym = r.get("symbol", {})
        if "top" in sym:
            L.append(f"  1. SYMBOL    : {sym['distinct']} distinct, dominant {sym['dominant_share']:.1%}  top={sym['top']}")
        spr = r.get("spread_ticks", {})
        if "median" in spr:
            L.append(f"  2. SPREAD    : median={spr['median']:.2f}  p10={spr['p10']:.2f}  p90={spr['p90']:.2f} ticks")
        else:
            L.append(f"  2. SPREAD    : {spr.get('missing')}")
        d = r.get("depth", {})
        if "median_bid_sz_L0" in d:
            L.append(f"  3. DEPTH     : L0 bid={d['median_bid_sz_L0']:.1f} ask={d['median_ask_sz_L0']:.1f}  "
                     f"levels filled bid={d['median_bid_levels_filled']:.0f}/10 ask={d['median_ask_levels_filled']:.0f}/10")
        else:
            L.append(f"  3. DEPTH     : {d.get('missing')}")
        if "trade_tick_fraction" in r:
            L.append(f"     trade tick fraction: {r['trade_tick_fraction']:.1%}")
    f = s.get("features_audit")
    if f:
        L.append("")
        L.append(f"FEATURES (n_bars: {f['n_bars']:,})")
        v = f.get("volume_per_bar", {})
        if v:
            L.append(f"  4. VOLUME/bar: median={v['median']:.0f}  p10={v['p10']:.0f}  min={v['min']:.0f}  zero={v['zero_frac']:.1%}")
        if "low_liquidity_fraction" in f:
            L.append(f"  5. LOW_LIQ   : {f['low_liquidity_fraction']:.1%}   regimes={ {k: round(v,3) for k,v in f.get('regime_fraction',{}).items()} }")
        for c in ("mbp_bar_coverage", "mbo_bar_coverage"):
            if c in f:
                L.append(f"     {c}: median={f[c]['median']:.3f}  zero={f[c]['zero_frac']:.1%}")
        rg = f.get("roll_gaps", {})
        if rg:
            L.append(f"  6. ROLL GAPS : max adj return={rg['max_adjacent_return']:.4%}  "
                     f"gaps>{rg['threshold_pct']:.1%}: {rg['n_gaps_over_threshold']}")
    L.append("")
    ov = s["overall"]
    L.append(f"VERDICT: {ov['verdict']}   (liquid signals: {ov['liquid_signals']}, thin: {ov['thin_signals']})")
    for fl in ov["flags"]:
        L.append(f"   ⚠️  {fl}")
    L.append("")
    L.append("R8 NOTE: if THIN, every microstructure conclusion on this data is suspect —")
    L.append("         re-extract the liquid front month and redo the analysis before trusting it.")
    L.append("═" * 88)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Liquidity / contract-integrity audit (R8)")
    p.add_argument("--mbo", type=Path, default=None, help="raw MBO/MBP-10 parquet (file or dir)")
    p.add_argument("--features", type=Path, default=None, help="refinery features parquet")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--stride", type=int, default=1, help="subsample raw MBO: take every Nth tick (speed)")
    p.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    args = p.parse_args()
    if not args.mbo and not args.features:
        p.error("provide at least one of --mbo / --features")
    s = run_liquidity_audit(args.output, mbo_path=args.mbo, features_path=args.features,
                            stride=args.stride, tick_size=args.tick_size)
    print(f"\n  VERDICT: {s['overall']['verdict']}")
    for fl in s["overall"]["flags"]:
        print(f"    ⚠️  {fl}")
    print(f"  report: {args.output}/liquidity_audit_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
