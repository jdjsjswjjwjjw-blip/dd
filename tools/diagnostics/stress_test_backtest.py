"""
tools/diagnostics/stress_test_backtest.py
═══════════════════════════════════════════════════════════════════════════════
Stress-test the day_trade backtest under deteriorating execution conditions
to measure strategy ROBUSTNESS vs FRAGILITY.

The honest question this answers:
  "If real-world latency and slippage are worse than my backtest assumes,
   does my P&L degrade gracefully — or does it cliff-fall?"

A profitable backtest under PERFECT execution can be useless live if a
small change in latency or slippage kills it. This tool quantifies that
risk before you put real money on the line.

Methodology:
  Run backtest_day_trade.py --strict on the same features parquet
  under N scenarios, each with progressively worse execution costs:

    baseline   1.0× slippage,  0 latency bars,  0 stop slip
    mild       1.5× slippage,  1 latency bar,   0.5 pip stop slip
    moderate   2.0× slippage,  2 latency bars,  1.0 pip stop slip
    severe     3.0× slippage,  3 latency bars,  2.0 pip stop slip
    extreme    5.0× slippage,  4 latency bars,  3.0 pip stop slip

  Then compare per-scenario:
    • Total P&L ($)
    • Hit rate
    • Annualized Sharpe
    • Max drawdown
    • Number of trades

  Verdict logic (configurable thresholds):
    ROBUST    P&L > 0 at severe stress AND Sharpe > 0 at moderate
    ACCEPTABLE  P&L > 0 at moderate stress, may degrade at severe
    FRAGILE   P&L turns negative at moderate stress
    BROKEN    P&L negative even at baseline (no edge to begin with)

Why latency is measured in BARS (not milliseconds):
  Our trading bars are 15-minute aggregates. Real-world millisecond
  latency is essentially zero at this granularity. The MEANINGFUL
  latency cost is "by the time the decision is acted on, how many
  more bars have elapsed?" — that's modeled by entry_at_next_open
  + extra latency_bars.

Usage:
  python tools/diagnostics/stress_test_backtest.py \\
      --features combined_6m/day_trading_features.parquet \\
      --output stress_test_results/ \\
      --scenarios baseline mild moderate severe extreme

Output:
  <output>/baseline/         backtest results at scenario 1
  <output>/mild/             ... scenario 2
  ...
  <output>/stress_summary.json   per-scenario metrics + verdict
  <output>/stress_report.txt     human-readable summary + degradation table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKTEST_SCRIPT = REPO_ROOT / "self_supervised" / "backtest_day_trade.py"


@dataclass
class StressScenario:
    """One stress configuration. All multipliers/penalties applied on top
    of the baseline backtest_day_trade.py defaults."""
    name: str
    slippage_multiplier: float          # multiplied into --slippage-pips-per-side
    extra_latency_bars: int             # added to --latency-bars on top of next-bar
    stop_slippage_pips: float           # passed to --stop-slippage-pips
    description: str


# Default scenario library — names match the CLI --scenarios filter
DEFAULT_SCENARIOS: dict[str, StressScenario] = {
    "baseline": StressScenario(
        name="baseline",
        slippage_multiplier=1.0,
        extra_latency_bars=0,
        stop_slippage_pips=0.0,
        description="Strict defaults — 0.5 pip/side, 1 pip spread, no stop slip",
    ),
    "mild": StressScenario(
        name="mild",
        slippage_multiplier=1.5,
        extra_latency_bars=1,
        stop_slippage_pips=0.5,
        description="1.5× slippage, +1 bar latency, 0.5 pip stop slip",
    ),
    "moderate": StressScenario(
        name="moderate",
        slippage_multiplier=2.0,
        extra_latency_bars=2,
        stop_slippage_pips=1.0,
        description="2× slippage, +2 bar latency, 1 pip stop slip",
    ),
    "severe": StressScenario(
        name="severe",
        slippage_multiplier=3.0,
        extra_latency_bars=3,
        stop_slippage_pips=2.0,
        description="3× slippage, +3 bar latency, 2 pip stop slip",
    ),
    "extreme": StressScenario(
        name="extreme",
        slippage_multiplier=5.0,
        extra_latency_bars=4,
        stop_slippage_pips=3.0,
        description="5× slippage, +4 bar latency, 3 pip stop slip",
    ),
}


def run_backtest_scenario(
    features_path: Path, output_dir: Path, scenario: StressScenario,
    base_slippage_pips: float = 0.5,
    base_spread_pips: float = 1.0,
    base_commission: float = 0.50,
) -> dict:
    """Invoke backtest_day_trade.py --strict with the scenario's parameters
    overlaid. Returns the parsed backtest_report.json."""
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        "--features", str(features_path),
        "--output", str(output_dir),
        "--strict",                # holdout-only + next-bar entry
        "--starting-equity", "100000",
        "--contracts-per-trade", "1",
        "--spread-pips", str(base_spread_pips),
        "--slippage-pips-per-side",
            str(base_slippage_pips * scenario.slippage_multiplier),
        "--commission-per-side", str(base_commission),
        "--stop-slippage-pips", str(scenario.stop_slippage_pips),
        "--latency-bars", str(1 + scenario.extra_latency_bars),  # strict = 1
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"backtest failed for scenario {scenario.name}:\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    report_path = output_dir / "backtest_report.json"
    if not report_path.exists():
        raise RuntimeError(f"no backtest_report.json at {report_path}")
    return json.loads(report_path.read_text())


def extract_key_metrics(report: dict) -> dict:
    """Pull the metrics that matter for stress comparison from a
    full backtest report. The backtest reports 'all_events' and
    optional 'by_slice'; we use the holdout slice if present, else
    all_events."""
    src = report.get("by_slice", {}).get("holdout") \
        or report.get("all_events", {})
    return {
        "n_trades": int(src.get("n_trades", 0)),
        "hit_rate": float(src.get("hit_rate", 0.0)),
        "total_pnl_dollars": float(src.get("total_pnl_dollars", 0.0)),
        "return_pct": float(src.get("return_pct", 0.0)),
        "annualized_sharpe": float(src.get("annualized_sharpe", 0.0)),
        "max_drawdown_dollars": float(src.get("max_drawdown_dollars", 0.0)),
        "max_drawdown_pct": float(src.get("max_drawdown_pct", 0.0)),
        "profit_factor": float(src.get("profit_factor", 0.0)),
        "avg_pips_per_trade": float(src.get("avg_pips_per_trade", 0.0)),
    }


def compute_verdict(per_scenario_metrics: dict[str, dict]) -> str:
    """Classify the strategy as ROBUST / ACCEPTABLE / FRAGILE / BROKEN
    based on P&L at each stress level."""
    baseline = per_scenario_metrics.get("baseline", {})
    moderate = per_scenario_metrics.get("moderate", {})
    severe = per_scenario_metrics.get("severe", {})

    if baseline.get("total_pnl_dollars", 0) <= 0:
        return "BROKEN — no edge even at baseline; check the strategy first"
    if not moderate:
        return "INCOMPLETE — 'moderate' scenario not run"

    moderate_pnl = moderate.get("total_pnl_dollars", 0)
    moderate_sharpe = moderate.get("annualized_sharpe", 0)
    severe_pnl = severe.get("total_pnl_dollars", 0) if severe else None

    if moderate_pnl <= 0:
        return ("FRAGILE — P&L turns negative under moderate stress "
                "(2× slip + 2 bar latency). Live execution will lose money.")
    if severe_pnl is not None and severe_pnl > 0 and moderate_sharpe > 0:
        return ("ROBUST — P&L stays positive through severe stress "
                "(3× slip + 3 bar latency). Strategy has true edge.")
    if moderate_pnl > 0 and moderate_sharpe > 0:
        return ("ACCEPTABLE — P&L positive at moderate stress but degrades "
                "at severe. Real-world performance will be reduced.")
    return ("MARGINAL — barely profitable at moderate stress; deploy with "
            "small size and monitor")


def stress_test(
    features_path: Path, output_root: Path,
    scenarios: list[StressScenario],
    base_slippage_pips: float = 0.5,
    base_spread_pips: float = 1.0,
    base_commission: float = 0.50,
) -> dict:
    """Run all stress scenarios and produce the summary."""
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"═══ Stress test: {len(scenarios)} scenarios ═══")
    print(f"  Features: {features_path}")
    print(f"  Output:   {output_root}")
    print()

    per_scenario: dict[str, dict] = {}
    for sc in scenarios:
        print(f"─── Scenario: {sc.name} ───")
        print(f"   {sc.description}")
        sc_dir = output_root / sc.name
        sc_dir.mkdir(parents=True, exist_ok=True)
        try:
            report = run_backtest_scenario(
                features_path, sc_dir, sc,
                base_slippage_pips=base_slippage_pips,
                base_spread_pips=base_spread_pips,
                base_commission=base_commission,
            )
            metrics = extract_key_metrics(report)
            per_scenario[sc.name] = metrics
            print(f"   ✓ trades={metrics['n_trades']:>4}  "
                  f"hit={metrics['hit_rate']:.3f}  "
                  f"PnL=${metrics['total_pnl_dollars']:>+10,.2f}  "
                  f"Sharpe={metrics['annualized_sharpe']:+.2f}")
        except RuntimeError as e:
            print(f"   ❌ {e}")
            per_scenario[sc.name] = {"error": str(e)}
        print()

    # ── Degradation analysis ──
    baseline = per_scenario.get("baseline", {})
    if baseline and "total_pnl_dollars" in baseline:
        base_pnl = baseline["total_pnl_dollars"]
        for name, m in per_scenario.items():
            if "total_pnl_dollars" in m and base_pnl != 0:
                m["pnl_degradation_pct"] = float(
                    100 * (base_pnl - m["total_pnl_dollars"]) / abs(base_pnl)
                )

    verdict = compute_verdict(per_scenario)
    summary = {
        "features_path": str(features_path),
        "n_scenarios": len(scenarios),
        "base_slippage_pips": base_slippage_pips,
        "base_spread_pips": base_spread_pips,
        "scenarios": [
            {
                "name": sc.name,
                "description": sc.description,
                "slippage_multiplier": sc.slippage_multiplier,
                "extra_latency_bars": sc.extra_latency_bars,
                "stop_slippage_pips": sc.stop_slippage_pips,
                "metrics": per_scenario.get(sc.name, {}),
            }
            for sc in scenarios
        ],
        "verdict": verdict,
    }

    # ── Write summary ──
    summary_path = output_root / "stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"💾 {summary_path}")

    # ── Write human-readable report ──
    lines = []
    lines.append("═════════════════════════════════════════════════════════════════════════")
    lines.append("  STRESS TEST REPORT")
    lines.append("═════════════════════════════════════════════════════════════════════════")
    lines.append("")
    lines.append(f"  Features:  {features_path}")
    lines.append(f"  Scenarios: {len(scenarios)}")
    lines.append("")
    lines.append("  Scenario     trades   hit_rate    total_pnl       Sharpe   max_DD%   Δ%PnL")
    lines.append("  " + "─" * 78)
    for sc in scenarios:
        m = per_scenario.get(sc.name, {})
        if "error" in m:
            lines.append(f"  {sc.name:<10}  ERROR: {m['error'][:60]}")
            continue
        lines.append(
            f"  {sc.name:<10}  {m.get('n_trades', 0):>5}   "
            f"{m.get('hit_rate', 0):.3f}   "
            f"${m.get('total_pnl_dollars', 0):>+11,.2f}   "
            f"{m.get('annualized_sharpe', 0):+6.2f}   "
            f"{m.get('max_drawdown_pct', 0):>6.2f}%   "
            f"{m.get('pnl_degradation_pct', 0):>+6.1f}%"
        )
    lines.append("")
    lines.append("─────────────────────────────────────────────────────────────────────────")
    lines.append(f"  VERDICT: {verdict}")
    lines.append("─────────────────────────────────────────────────────────────────────────")
    report_text = "\n".join(lines)
    report_path = output_root / "stress_report.txt"
    report_path.write_text(report_text)
    print(f"💾 {report_path}")
    print()
    print(report_text)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True,
                   help="Path to day_trading_features.parquet (with dataset_slice column)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--scenarios", nargs="+",
                   default=list(DEFAULT_SCENARIOS.keys()),
                   help=f"Subset of scenarios to run. Choices: "
                        f"{list(DEFAULT_SCENARIOS.keys())}")
    p.add_argument("--base-slippage-pips", type=float, default=0.5)
    p.add_argument("--base-spread-pips", type=float, default=1.0)
    p.add_argument("--base-commission", type=float, default=0.50)
    args = p.parse_args()

    unknown = set(args.scenarios) - set(DEFAULT_SCENARIOS.keys())
    if unknown:
        p.error(f"Unknown scenarios: {unknown}. "
                f"Available: {list(DEFAULT_SCENARIOS.keys())}")
    scenarios = [DEFAULT_SCENARIOS[n] for n in args.scenarios]
    stress_test(
        features_path=Path(args.features),
        output_root=Path(args.output),
        scenarios=scenarios,
        base_slippage_pips=args.base_slippage_pips,
        base_spread_pips=args.base_spread_pips,
        base_commission=args.base_commission,
    )


if __name__ == "__main__":
    sys.exit(main())
