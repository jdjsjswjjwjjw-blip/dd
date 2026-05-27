"""
tools/diagnostics/regime_parity_test.py
═══════════════════════════════════════════════════════════════════════════════
Regime parity test — measures whether the strategy's performance is STABLE
across market regimes (trending / ranging / volatile / low_liquidity), or
whether it secretly depends on a single regime to be profitable.

This addresses Issue #1 from docs/PIPELINE_ISSUES_AUDIT.md (Regime Drift):
even a perfectly clean pipeline can collapse live if the market enters a
regime the model was undertrained on.

Methodology:
  1. Run the backtest once on the full holdout (or accept a pre-computed
     trades.csv via --trades-csv)
  2. Split trades by `regime` column (already populated by backtest_day_trade.py)
  3. Optionally also split by REALIZED-VOLATILITY QUARTILE — uses ATR
     scaled by close at entry time. This is finer-grained than the
     categorical regime label and catches cases where the classifier
     is too coarse.
  4. Compute per-regime: hit_rate, total P&L, mean pips/trade, annualized
     Sharpe, max drawdown, trade count.
  5. Coefficient of variation (CV = std / |mean|) across regimes — measures
     instability. Lower is better.
  6. Verdict + a JSON summary + a human-readable table.

The verdict:
  ROBUST              every regime profitable AND CV(hit_rate) < 0.30
  ACCEPTABLE          worst regime non-negative AND CV(hit_rate) < 0.50
  UNSTABLE            one or more regimes negative
  INSUFFICIENT_DATA   any regime has < min_trades_per_regime trades
  REGIME_BIAS         only one regime is profitable; others lose
                       (strategy is regime-conditional, not robust)

Usage:
  # Option A — point at an existing trades.csv from a prior backtest
  python tools/diagnostics/regime_parity_test.py \\
      --trades-csv stress_results/baseline/trades.csv \\
      --output regime_parity_results/

  # Option B — invoke the backtest then split (one step)
  python tools/diagnostics/regime_parity_test.py \\
      --features combined_6y/day_trading_features.parquet \\
      --output regime_parity_results/

Output:
  <output>/per_regime_metrics.csv       per-regime table
  <output>/regime_parity_summary.json   summary + verdict
  <output>/regime_parity_report.txt     human-readable summary
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKTEST_SCRIPT = REPO_ROOT / "self_supervised" / "backtest_day_trade.py"


# Minimum trades per regime to consider the per-regime metric statistically
# meaningful. With < 30 trades the CI is ±20% on hit rate alone.
DEFAULT_MIN_TRADES = 30
# 252 trading days × 24 hours × 4 (15-min bars) — but trades are sparse;
# for annualization we use ~252 × ~5 trades/day = 1260 trades/year as
# an order-of-magnitude factor when there's no time range available.
BARS_PER_YEAR = 252 * 24 * 4


def _safe_sharpe(pnl: np.ndarray, n_trades: int,
                 first_ts: pd.Timestamp | None,
                 last_ts: pd.Timestamp | None) -> float:
    """Annualized Sharpe estimate. Falls back to a rough estimate if the
    timestamps don't give us a span."""
    if len(pnl) < 2 or pnl.std() <= 1e-9:
        return 0.0
    per_trade_sharpe = pnl.mean() / pnl.std()
    # Trades per year — prefer empirical estimate if timestamps span
    # is sane, else default
    if first_ts is not None and last_ts is not None:
        days_span = max((last_ts - first_ts).days, 1)
        trades_per_yr = n_trades * 252 / days_span
    else:
        trades_per_yr = 1000.0
    return float(per_trade_sharpe * np.sqrt(max(trades_per_yr, 1)))


def _compute_max_drawdown(cum_pnl: np.ndarray) -> tuple[float, float]:
    """Returns (max_dd_dollars, max_dd_pct_of_starting_equity).
    Assumes starting_equity = first cum_pnl + 100000 if not provided —
    here we just report raw $ DD; pct is dollars / starting equity."""
    if len(cum_pnl) == 0:
        return 0.0, 0.0
    running_max = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - running_max
    return float(dd.min()), float(dd.min() / 100_000.0 * 100)


def per_regime_metrics(
    trades: pd.DataFrame, regime_col: str = "regime",
) -> pd.DataFrame:
    """Compute per-regime stats from a trades.csv DataFrame."""
    rows = []
    for regime, sub in trades.groupby(regime_col):
        if len(sub) == 0:
            continue
        pnl = sub["net_pnl"].to_numpy(dtype=np.float64)
        cum = np.cumsum(pnl)
        first_ts = pd.to_datetime(sub["ts_entry"]).min() if "ts_entry" in sub else None
        last_ts = pd.to_datetime(sub["ts_exit"]).max() if "ts_exit" in sub else None
        max_dd, max_dd_pct = _compute_max_drawdown(cum)
        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        rows.append({
            "regime": str(regime),
            "n_trades": int(len(sub)),
            "wins": int(wins),
            "losses": int(losses),
            "hit_rate": float(wins / max(wins + losses, 1)),
            "total_pnl": float(pnl.sum()),
            "mean_pips_per_trade": float(sub["pips"].mean())
                                    if "pips" in sub else 0.0,
            "annualized_sharpe": _safe_sharpe(pnl, len(sub), first_ts, last_ts),
            "max_drawdown_dollars": max_dd,
            "max_drawdown_pct": max_dd_pct,
        })
    return pd.DataFrame(rows).sort_values("n_trades", ascending=False).reset_index(drop=True)


def add_volatility_quartile(
    trades: pd.DataFrame, n_quartiles: int = 4,
) -> pd.DataFrame:
    """Add a `vol_quartile` column based on atr_at_entry / entry_price.

    Returns a copy of `trades` with the new column. If atr_at_entry is
    missing, this is a no-op (returns the input).
    """
    if "atr_at_entry" not in trades.columns or "entry_price" not in trades.columns:
        return trades
    out = trades.copy()
    vol_norm = out["atr_at_entry"].astype(np.float64) \
        / out["entry_price"].astype(np.float64).clip(lower=1e-9)
    # qcut may fail if too few unique values; fall back to numeric bins
    try:
        out["vol_quartile"] = pd.qcut(
            vol_norm, q=n_quartiles,
            labels=[f"Q{i+1}" for i in range(n_quartiles)],
            duplicates="drop",
        )
        out["vol_quartile"] = out["vol_quartile"].astype(str)
    except (ValueError, TypeError):
        # Fall back to absolute bins
        out["vol_quartile"] = "Q1"
    return out


def compute_cv(values: list[float]) -> float:
    """Coefficient of variation of a list of metrics.
    Lower = more stable across regimes. NaN-safe.
    """
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    mean = arr.mean()
    if abs(mean) < 1e-9:
        return float("inf")
    return float(arr.std() / abs(mean))


def compute_verdict(
    per_regime: pd.DataFrame, min_trades: int = DEFAULT_MIN_TRADES,
) -> tuple[str, dict]:
    """Classify the strategy as ROBUST / ACCEPTABLE / UNSTABLE / etc.
    Returns (verdict_string, diagnostic_dict).
    """
    if len(per_regime) == 0:
        return "INCOMPLETE — no per-regime data", {}

    # Filter to regimes with enough samples for stable stats
    significant = per_regime[per_regime["n_trades"] >= min_trades]
    if len(significant) == 0:
        return (
            f"INSUFFICIENT_DATA — no regime has >= {min_trades} trades. "
            f"Collect more holdout data or relax min_trades.",
            {"n_significant_regimes": 0},
        )

    hits = significant["hit_rate"].to_list()
    pnls = significant["total_pnl"].to_list()
    sharpes = significant["annualized_sharpe"].to_list()
    cv_hit = compute_cv(hits)
    cv_pnl = compute_cv(pnls)
    worst_pnl = min(pnls)
    n_positive = sum(1 for p in pnls if p > 0)
    n_total = len(pnls)

    diag = {
        "n_regimes_total": len(per_regime),
        "n_significant_regimes": len(significant),
        "cv_hit_rate": cv_hit,
        "cv_pnl": cv_pnl,
        "worst_regime_pnl": worst_pnl,
        "n_profitable_regimes": n_positive,
        "n_significant_regimes_total": n_total,
    }

    if n_positive == 1 and n_total > 1:
        return (
            f"REGIME_BIAS — only {n_positive}/{n_total} regimes profitable. "
            f"Strategy depends on a single regime; it will fail when the "
            f"market shifts.",
            diag,
        )
    if worst_pnl <= 0 and n_total > 1:
        return (
            f"UNSTABLE — at least one regime has non-positive P&L "
            f"(worst = ${worst_pnl:.2f}). Live edge is conditional on "
            f"regime classification staying accurate.",
            diag,
        )
    if all(s > 0 for s in sharpes) and cv_hit < 0.30:
        return (
            f"ROBUST — every regime profitable with hit-rate CV = "
            f"{cv_hit:.3f} (< 0.30). Strategy generalizes across regimes.",
            diag,
        )
    if cv_hit < 0.50:
        return (
            f"ACCEPTABLE — all regimes profitable, hit-rate CV = "
            f"{cv_hit:.3f}. Performance varies but remains positive.",
            diag,
        )
    return (
        f"MARGINAL — all regimes profitable but hit-rate CV = "
        f"{cv_hit:.3f} (>= 0.50). High regime sensitivity; monitor closely.",
        diag,
    )


def run_backtest_for_trades(
    features_path: Path, work_dir: Path,
) -> Path:
    """Invoke backtest_day_trade.py --strict and return the trades.csv path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        "--features", str(features_path),
        "--output", str(work_dir),
        "--strict",
        "--starting-equity", "100000",
        "--contracts-per-trade", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"backtest failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
        )
    trades = work_dir / "trades.csv"
    if not trades.exists():
        raise RuntimeError(f"backtest did not produce trades.csv at {trades}")
    return trades


def regime_parity_test(
    output_root: Path,
    trades_csv: Optional[Path] = None,
    features_path: Optional[Path] = None,
    min_trades: int = DEFAULT_MIN_TRADES,
    include_vol_quartiles: bool = True,
) -> dict:
    """Main entry point.

    If `trades_csv` is provided, use it directly. Else invoke backtest
    on `features_path` to produce one. Exactly one of the two must be set.
    """
    if trades_csv is None and features_path is None:
        raise ValueError("Must provide either --trades-csv or --features")
    output_root.mkdir(parents=True, exist_ok=True)

    if trades_csv is None:
        print(f"Running backtest on {features_path}...")
        trades_csv = run_backtest_for_trades(features_path, output_root / "_bt")
        print(f"  trades.csv at {trades_csv}")

    trades = pd.read_csv(trades_csv)
    if len(trades) == 0:
        raise RuntimeError(f"trades.csv at {trades_csv} is empty")
    if "regime" not in trades.columns:
        raise RuntimeError(
            f"trades.csv at {trades_csv} has no 'regime' column — "
            f"is this from backtest_day_trade.py?"
        )

    print(f"Loaded {len(trades):,} trades.")
    print()

    # ── By categorical regime ──
    print("═══ Per-regime metrics (categorical) ═══")
    by_regime = per_regime_metrics(trades, regime_col="regime")
    by_regime_path = output_root / "per_regime_metrics.csv"
    by_regime.to_csv(by_regime_path, index=False)
    print(by_regime.to_string(index=False))
    print()

    verdict_categorical, diag_categorical = compute_verdict(by_regime, min_trades)
    print(f"VERDICT (regime): {verdict_categorical}")
    print()

    # ── By volatility quartile (optional, finer-grained) ──
    quartile_metrics = None
    verdict_quartile = None
    diag_quartile = None
    if include_vol_quartiles:
        trades_with_q = add_volatility_quartile(trades, n_quartiles=4)
        if "vol_quartile" in trades_with_q.columns:
            print("═══ Per-volatility-quartile metrics ═══")
            quartile_metrics = per_regime_metrics(trades_with_q, regime_col="vol_quartile")
            q_path = output_root / "per_vol_quartile_metrics.csv"
            quartile_metrics.to_csv(q_path, index=False)
            print(quartile_metrics.to_string(index=False))
            print()
            verdict_quartile, diag_quartile = compute_verdict(quartile_metrics, min_trades)
            print(f"VERDICT (vol quartile): {verdict_quartile}")
            print()

    # ── Summary JSON ──
    summary = {
        "trades_csv": str(trades_csv),
        "n_trades": int(len(trades)),
        "min_trades_per_regime": min_trades,
        "regime_categorical": {
            "metrics": by_regime.to_dict(orient="records"),
            "verdict": verdict_categorical,
            "diagnostics": diag_categorical,
        },
    }
    if quartile_metrics is not None:
        summary["volatility_quartile"] = {
            "metrics": quartile_metrics.to_dict(orient="records"),
            "verdict": verdict_quartile,
            "diagnostics": diag_quartile,
        }
    summary_path = output_root / "regime_parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"💾 {summary_path}")

    # ── Human-readable report ──
    lines = []
    lines.append("═════════════════════════════════════════════════════════════════════════")
    lines.append("  REGIME PARITY TEST REPORT")
    lines.append("═════════════════════════════════════════════════════════════════════════")
    lines.append("")
    lines.append(f"  Trades source:   {trades_csv}")
    lines.append(f"  Total trades:    {len(trades):,}")
    lines.append(f"  Min/regime:      {min_trades}")
    lines.append("")
    lines.append("  CATEGORICAL REGIME (trending / ranging / volatile / ...):")
    lines.append("")
    lines.append("    regime          trades    hit       pnl       Sharpe    DD%")
    lines.append("    " + "─" * 65)
    for _, r in by_regime.iterrows():
        lines.append(
            f"    {r['regime']:<14}  {int(r['n_trades']):>6}   "
            f"{r['hit_rate']:.3f}   ${r['total_pnl']:>+9.2f}   "
            f"{r['annualized_sharpe']:+6.2f}   {r['max_drawdown_pct']:>5.2f}%"
        )
    lines.append("")
    lines.append(f"    Verdict: {verdict_categorical}")
    lines.append("")
    if quartile_metrics is not None:
        lines.append("  VOLATILITY QUARTILE (ATR/price, Q1=lowest, Q4=highest):")
        lines.append("")
        lines.append("    quartile        trades    hit       pnl       Sharpe    DD%")
        lines.append("    " + "─" * 65)
        for _, r in quartile_metrics.iterrows():
            lines.append(
                f"    {r['regime']:<14}  {int(r['n_trades']):>6}   "
                f"{r['hit_rate']:.3f}   ${r['total_pnl']:>+9.2f}   "
                f"{r['annualized_sharpe']:+6.2f}   {r['max_drawdown_pct']:>5.2f}%"
            )
        lines.append("")
        lines.append(f"    Verdict: {verdict_quartile}")
        lines.append("")
    report_path = output_root / "regime_parity_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"💾 {report_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trades-csv",
                     help="Path to trades.csv from a prior backtest_day_trade run")
    src.add_argument("--features",
                     help="Path to features.parquet — will run the backtest "
                          "to produce trades.csv, then split by regime")
    p.add_argument("--output", required=True)
    p.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES,
                   help="Minimum trades per regime for stable metrics (default 30)")
    p.add_argument("--no-vol-quartiles", action="store_true",
                   help="Skip the volatility-quartile breakdown")
    args = p.parse_args()
    regime_parity_test(
        output_root=Path(args.output),
        trades_csv=Path(args.trades_csv) if args.trades_csv else None,
        features_path=Path(args.features) if args.features else None,
        min_trades=args.min_trades,
        include_vol_quartiles=not args.no_vol_quartiles,
    )


if __name__ == "__main__":
    sys.exit(main())
