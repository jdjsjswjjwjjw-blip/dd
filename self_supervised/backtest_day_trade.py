"""
self_supervised/backtest_day_trade.py
═══════════════════════════════════════════════════════════════════
Backtest نظيف ومركّز على day_trade events فقط.

بياخد day_trading_features.parquet (الـ events + path_outcome محسوبين بالفعل)
ويبني صورة كاملة للتداول:
  • per-trade P&L (pips + $ + %)
  • equity curve
  • drawdown analysis (max, duration)
  • win/loss distribution
  • breakdown by signal_quality tier
  • breakdown by regime
  • realistic costs (spread + slippage + commission)

تصميم محايد عن بيانات training/holdout — بياخد كل الـ events
وبيمشي عليها كأنه live trading (causal، بدون lookahead).

الـ Logic:
  لكل event_flag==1 في الـ bars:
    1. entry @ close[i] في direction = event_direction
    2. TP @ entry + dir * tp_mult * atr_14[i]
    3. SL @ entry - dir * sl_mult * atr_14[i]
    4. walk forward لـ label_horizon_steps bar
    5. أول touch (high≥TP لـ long، low≤SL لـ long، عكس لـ short) → exit
    6. لو وصلت timeout → exit @ close[i+horizon]
    7. P&L = (exit - entry) * dir - costs

التكاليف الواقعية لـ 6B (GBPUSD futures):
  • spread:      1.0 pip بـ default
  • slippage:    0.5 pip per side
  • commission:  $0.50 per side
  → round-trip ≈ 2 pip + $1 → ~$3 لكل صفقة 1 contract

استخدام:
  python self_supervised/backtest_day_trade.py \
      --features combined_6m/day_trading_features.parquet \
      --output   backtests/day_trade_6m \
      --starting-equity 100000 \
      --contracts-per-trade 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Constants for 6B (GBP/USD futures)
# ──────────────────────────────────────────────────────────────────
TICK_SIZE = 0.0001          # 1 tick = 0.0001 USD/GBP
TICK_VALUE_USD = 6.25       # 1 tick of 6B = $6.25 (CME spec)
# Note: 1 pip = 1 tick for 6B (decimal pip convention)


def simulate(
    df: pd.DataFrame,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    horizon_bars: int | None = None,
    spread_pips: float = 1.0,
    slippage_pips_per_side: float = 0.5,
    commission_per_side: float = 0.50,
    contracts: int = 1,
    starting_equity: float = 100_000.0,
    # ── strict-mode knobs (anti-leakage) ──
    entry_at_next_open: bool = False,
    stop_slippage_pips: float = 0.0,
    latency_bars: int = 0,
    cooldown_bars: int = 1,
    require_open_col: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Walk through bars; for each event, simulate a trade and track equity.

    Strict-mode flags (all OFF by default to preserve original behavior):
      • entry_at_next_open  — enter at open[i+1+latency_bars] not close[i].
                              avoids any "same-bar" leakage where event_flag
                              might have been computed using close[i] itself.
      • stop_slippage_pips  — penalty in pips when SL is hit (real stops
                              fill below the level in fast moves).
      • latency_bars        — extra bars between signal and entry (sim latency).
      • cooldown_bars       — bars to wait after exit before allowing a new
                              trade. Default 1.
      • require_open_col    — fail if 'open' column missing (next-bar mode).

    Returns:
        trades_df    — per-trade detail rows
        equity       — equity time series (indexed by ts_event)
        summary      — aggregate metrics dict
    """
    df = df.sort_values('ts_event').reset_index(drop=True)
    n = len(df)
    high = df['high'].to_numpy(np.float64)
    low = df['low'].to_numpy(np.float64)
    close = df['close'].to_numpy(np.float64)
    if 'open' in df.columns:
        open_p = df['open'].to_numpy(np.float64)
    elif require_open_col:
        raise ValueError("'open' column required for entry_at_next_open mode")
    else:
        open_p = close.copy()  # fall back to close as proxy
    atr = df['atr_14'].to_numpy(np.float64)
    ts = df['ts_event'].to_numpy()
    event_flag = df['event_flag'].fillna(0).astype(np.int8).to_numpy()
    event_dir = df['event_direction'].fillna(0).astype(np.int8).to_numpy()

    sig_q = (df['signal_quality'].fillna(0).astype(np.int8).to_numpy()
             if 'signal_quality' in df.columns else np.zeros(n, dtype=np.int8))
    regime = (df['regime_label'].astype(str).to_numpy()
              if 'regime_label' in df.columns else np.array(['?'] * n))

    if horizon_bars is None:
        h_col = df.get('label_horizon_steps')
        if h_col is not None and h_col.notna().any():
            horizon_bars = int(h_col.dropna().mode().iloc[0])
        else:
            horizon_bars = 6
    print(f"  horizon_bars used: {horizon_bars}")
    print(f"  entry mode: {'open[i+1+lat]' if entry_at_next_open else 'close[i] (legacy)'}")
    if entry_at_next_open:
        print(f"  latency_bars: {latency_bars}  cooldown_bars: {cooldown_bars}")
    if stop_slippage_pips > 0:
        print(f"  stop slippage: {stop_slippage_pips} pips (penalty on SL/sl_first)")

    # round-trip cost per contract in $
    rt_cost_pips = spread_pips + 2 * slippage_pips_per_side
    rt_cost_dollars = rt_cost_pips * TICK_VALUE_USD + 2 * commission_per_side
    print(f"  round-trip cost: {rt_cost_pips:.1f} pips + ${2*commission_per_side:.2f} comm = ${rt_cost_dollars:.2f}/contract")

    trades: list[dict] = []
    in_position_until = -1  # bar idx; no new trades before this

    for i in range(n - horizon_bars):
        if event_flag[i] != 1:
            continue
        if event_dir[i] == 0:
            continue
        # block overlapping trades (single-position-only)
        if i < in_position_until:
            continue
        if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            continue

        d = int(event_dir[i])  # +1 long, -1 short
        atr_i = float(atr[i])

        # ── Determine entry bar & price ──
        if entry_at_next_open:
            entry_bar = i + 1 + latency_bars
            if entry_bar >= n:
                continue
            if not np.isfinite(open_p[entry_bar]):
                continue
            entry = float(open_p[entry_bar])
        else:
            entry_bar = i
            entry = float(close[i])

        tp = entry + d * tp_mult * atr_i
        sl = entry - d * sl_mult * atr_i

        # Walk forward (start from bar AFTER entry_bar)
        exit_price = None
        exit_reason = 'timeout'
        exit_bar = entry_bar + horizon_bars
        for j in range(entry_bar + 1, min(entry_bar + 1 + horizon_bars, n)):
            hi, lo = high[j], low[j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue
            if d == 1:
                # LONG: TP if high >= tp, SL if low <= sl
                hit_tp = hi >= tp
                hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp
                hit_sl = hi >= sl
            if hit_tp and hit_sl:
                # Both touched same bar — conservative: assume SL hit first
                # with stop slippage applied (real stops fill worse)
                exit_price = sl - d * stop_slippage_pips * TICK_SIZE
                exit_reason = 'sl_first'
                exit_bar = j
                break
            if hit_tp:
                exit_price = tp  # limit-order fills exact at TP (best case)
                exit_reason = 'tp'
                exit_bar = j
                break
            if hit_sl:
                exit_price = sl - d * stop_slippage_pips * TICK_SIZE
                exit_reason = 'sl'
                exit_bar = j
                break
        if exit_price is None:
            # timeout — exit at close of last bar in horizon
            exit_idx = min(entry_bar + horizon_bars, n - 1)
            if not np.isfinite(close[exit_idx]):
                continue
            exit_price = float(close[exit_idx])
            exit_reason = 'timeout'
            exit_bar = exit_idx

        # P&L
        pips = (exit_price - entry) * d / TICK_SIZE
        gross_dollars = pips * TICK_VALUE_USD * contracts
        net_dollars = gross_dollars - rt_cost_dollars * contracts

        trades.append({
            'ts_signal': pd.Timestamp(ts[i]),
            'ts_entry': pd.Timestamp(ts[entry_bar]),
            'ts_exit': pd.Timestamp(ts[exit_bar]),
            'signal_bar': i,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'bars_held': exit_bar - entry_bar,
            'direction': 'LONG' if d == 1 else 'SHORT',
            'entry_price': entry,
            'tp_price': tp,
            'sl_price': sl,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'atr_at_entry': atr_i,
            'pips': pips,
            'gross_pnl': gross_dollars,
            'net_pnl': net_dollars,
            'signal_quality': int(sig_q[i]),
            'regime': str(regime[i]),
        })
        in_position_until = exit_bar + cooldown_bars

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return trades_df, pd.Series(dtype=float), {'n_trades': 0}

    # ── Equity curve ──
    trades_df = trades_df.sort_values('ts_exit').reset_index(drop=True)
    trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()
    trades_df['equity'] = starting_equity + trades_df['cum_pnl']
    equity = trades_df.set_index('ts_exit')['equity']

    # ── Drawdown ──
    running_max = equity.cummax()
    drawdown = equity - running_max
    drawdown_pct = drawdown / running_max * 100
    max_dd_dollars = float(drawdown.min())
    max_dd_pct = float(drawdown_pct.min())
    max_dd_at = drawdown.idxmin() if len(drawdown) else None

    # ── Summary stats ──
    n_trades = len(trades_df)
    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] < 0]
    flats = trades_df[trades_df['net_pnl'] == 0]

    avg_win = float(wins['net_pnl'].mean()) if len(wins) else 0.0
    avg_loss = float(losses['net_pnl'].mean()) if len(losses) else 0.0
    profit_factor = (
        wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
        if losses['net_pnl'].sum() < 0 else float('inf')
    )

    total_pnl = float(trades_df['net_pnl'].sum())
    return_pct = total_pnl / starting_equity * 100

    # Sharpe (per-trade, annualized assuming ~10 trades/month)
    if trades_df['net_pnl'].std() > 0:
        per_trade_sharpe = trades_df['net_pnl'].mean() / trades_df['net_pnl'].std()
        # Trades per year estimate
        days = (trades_df['ts_exit'].max() - trades_df['ts_entry'].min()).days
        trades_per_yr = n_trades * 252 / max(days, 1)
        annualized_sharpe = per_trade_sharpe * np.sqrt(trades_per_yr)
    else:
        annualized_sharpe = 0.0

    summary = {
        'n_trades': int(n_trades),
        'n_wins': int(len(wins)),
        'n_losses': int(len(losses)),
        'n_flats': int(len(flats)),
        'hit_rate': float(len(wins) / n_trades) if n_trades else 0.0,
        'total_pnl_dollars': total_pnl,
        'return_pct': return_pct,
        'avg_win_dollars': avg_win,
        'avg_loss_dollars': avg_loss,
        'profit_factor': float(profit_factor),
        'avg_pips_per_trade': float(trades_df['pips'].mean()),
        'sum_pips': float(trades_df['pips'].sum()),
        'max_drawdown_dollars': max_dd_dollars,
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_at': str(max_dd_at) if max_dd_at else None,
        'annualized_sharpe': float(annualized_sharpe),
        'exit_reason_breakdown': trades_df['exit_reason'].value_counts().to_dict(),
        'cost_per_trade_dollars': float(rt_cost_dollars * contracts),
        'starting_equity': starting_equity,
        'final_equity': float(equity.iloc[-1]) if len(equity) else starting_equity,
        'first_trade_ts': str(trades_df['ts_entry'].iloc[0]),
        'last_trade_ts': str(trades_df['ts_entry'].iloc[-1]),
    }
    return trades_df, equity, summary


def print_report(trades_df: pd.DataFrame, summary: dict, label: str = '') -> None:
    if summary['n_trades'] == 0:
        print(f"  [{label}] No trades.")
        return
    print(f"  ─── {label} ───")
    print(f"    Trades:           {summary['n_trades']:,}")
    print(f"    Wins / Losses:    {summary['n_wins']} / {summary['n_losses']} "
          f"(hit={summary['hit_rate']:.3f})")
    print(f"    Total P&L:        ${summary['total_pnl_dollars']:,.2f}  "
          f"({summary['return_pct']:+.2f}%)")
    print(f"    Avg pips/trade:   {summary['avg_pips_per_trade']:+.2f}")
    print(f"    Profit factor:    {summary['profit_factor']:.2f}")
    print(f"    Avg win / loss:   ${summary['avg_win_dollars']:+.2f} / ${summary['avg_loss_dollars']:+.2f}")
    print(f"    Max drawdown:     ${summary['max_drawdown_dollars']:,.2f} "
          f"({summary['max_drawdown_pct']:.2f}%) @ {summary['max_drawdown_at']}")
    print(f"    Annualized Sharpe: {summary['annualized_sharpe']:+.2f}")
    print(f"    Exit reasons:     {summary['exit_reason_breakdown']}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True,
                   help='day_trading_features.parquet (must have event_flag, event_direction, atr_14)')
    p.add_argument('--output', required=True, help='Output directory')
    p.add_argument('--start-ts', default='', help='Optional ISO start ts')
    p.add_argument('--end-ts', default='', help='Optional ISO end ts')
    p.add_argument('--starting-equity', type=float, default=100_000.0)
    p.add_argument('--contracts-per-trade', type=int, default=1)
    p.add_argument('--tp-mult', type=float, default=1.5,
                   help='TP = entry + dir * tp_mult * ATR')
    p.add_argument('--sl-mult', type=float, default=1.0)
    p.add_argument('--horizon-bars', type=int, default=0,
                   help='Override label_horizon_steps (default: from data)')
    p.add_argument('--spread-pips', type=float, default=1.0)
    p.add_argument('--slippage-pips-per-side', type=float, default=0.5)
    p.add_argument('--commission-per-side', type=float, default=0.50)
    p.add_argument('--min-signal-quality', type=int, default=0,
                   help='Only trade events with signal_quality >= this')
    # ── strict-mode flags (anti-leakage) ──
    p.add_argument('--strict', action='store_true',
                   help='Enable strict anti-leakage mode: '
                        'next-bar entry + stop slippage + holdout-only')
    p.add_argument('--entry-at-next-open', action='store_true',
                   help='Enter at open[i+1+latency_bars] instead of close[i]')
    p.add_argument('--latency-bars', type=int, default=0,
                   help='Bars between signal detection and entry (default 0)')
    p.add_argument('--stop-slippage-pips', type=float, default=0.0,
                   help='Penalty pips when SL is hit (real stops fill worse)')
    p.add_argument('--cooldown-bars', type=int, default=1,
                   help='Bars to wait after exit before allowing new trade')
    p.add_argument('--holdout-only', action='store_true',
                   help='Run backtest ONLY on dataset_slice==holdout rows')
    args = p.parse_args()

    # Strict mode = sane anti-leakage defaults (can be overridden individually)
    if args.strict:
        if not args.entry_at_next_open:
            args.entry_at_next_open = True
        if args.latency_bars == 0:
            args.latency_bars = 1
        if args.stop_slippage_pips == 0.0:
            args.stop_slippage_pips = 0.5
        if not args.holdout_only:
            args.holdout_only = True

    print("═" * 72)
    print("              DAY-TRADE BACKTEST" + (' (STRICT)' if args.strict else ''))
    print("═" * 72)
    print(f"Features:        {args.features}")
    print(f"Output:          {args.output}")
    print(f"Starting equity: ${args.starting_equity:,.2f}")
    print(f"Contracts/trade: {args.contracts_per_trade}")
    print(f"TP/SL:           {args.tp_mult}×ATR / {args.sl_mult}×ATR")
    print(f"Costs:           {args.spread_pips} spread + {args.slippage_pips_per_side}×2 slip + "
          f"${args.commission_per_side}×2 comm")
    print(f"Min signal_q:    {args.min_signal_quality}")
    if args.entry_at_next_open or args.strict:
        print(f"STRICT entry:    open[i+1+{args.latency_bars}]")
        print(f"STRICT stop slip: {args.stop_slippage_pips} pips")
        print(f"STRICT cooldown: {args.cooldown_bars} bars")
        print(f"STRICT holdout:  {args.holdout_only}")
    print()

    df = pd.read_parquet(args.features)
    df['ts_event'] = pd.to_datetime(df['ts_event'])

    # Optional time slice
    if args.start_ts:
        df = df[df['ts_event'] >= pd.Timestamp(args.start_ts)]
    if args.end_ts:
        df = df[df['ts_event'] <= pd.Timestamp(args.end_ts)]
    print(f"Loaded: {len(df):,} bars  ({df['ts_event'].iloc[0]} → {df['ts_event'].iloc[-1]})")

    # ── Holdout-only filter ──
    if args.holdout_only:
        if 'dataset_slice' not in df.columns:
            raise SystemExit("--holdout-only requires 'dataset_slice' column in features")
        before_h = len(df)
        df = df[df['dataset_slice'] == 'holdout'].copy()
        print(f"  holdout-only: {before_h:,} → {len(df):,} bars "
              f"({df['ts_event'].iloc[0]} → {df['ts_event'].iloc[-1]})")

    # Filter by signal quality
    if args.min_signal_quality > 0:
        before = (df['event_flag'] == 1).sum()
        df = df.copy()
        mask = ~(
            (df['event_flag'] == 1)
            & (df['signal_quality'].fillna(0) < args.min_signal_quality)
        )
        df.loc[~mask, 'event_flag'] = 0
        after = (df['event_flag'] == 1).sum()
        print(f"  signal_quality filter: {before} → {after} events")
    print()

    horizon = args.horizon_bars if args.horizon_bars > 0 else None

    # Shared simulate kwargs
    sim_kwargs = dict(
        tp_mult=args.tp_mult, sl_mult=args.sl_mult, horizon_bars=horizon,
        spread_pips=args.spread_pips,
        slippage_pips_per_side=args.slippage_pips_per_side,
        commission_per_side=args.commission_per_side,
        contracts=args.contracts_per_trade,
        starting_equity=args.starting_equity,
        entry_at_next_open=args.entry_at_next_open,
        stop_slippage_pips=args.stop_slippage_pips,
        latency_bars=args.latency_bars,
        cooldown_bars=args.cooldown_bars,
    )

    # ── Full backtest ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("═══ FULL DATA ═══")
    trades_all, equity_all, sum_all = simulate(df, **sim_kwargs)
    print_report(trades_all, sum_all, label='ALL EVENTS')

    # ── Train/holdout split if available AND we did NOT already filter to holdout ──
    splits = {}
    if 'dataset_slice' in df.columns and not args.holdout_only:
        for slice_name in ('train', 'holdout'):
            sub = df[df['dataset_slice'] == slice_name].copy()
            if len(sub) == 0:
                continue
            print(f"═══ {slice_name.upper()} SLICE ═══")
            t, e, s = simulate(sub, **sim_kwargs)
            print_report(t, s, label=slice_name.upper())
            splits[slice_name] = {'trades': t, 'equity': e, 'summary': s}

    # ── By signal_quality tier on full data ──
    if 'signal_quality' in df.columns:
        print("═══ BY signal_quality TIER ═══")
        for tier in sorted(df.loc[df['event_flag'] == 1, 'signal_quality'].dropna().unique()):
            sub = df.copy()
            sub.loc[(sub['event_flag'] == 1) & (sub['signal_quality'] != tier), 'event_flag'] = 0
            t, e, s = simulate(sub, **sim_kwargs)
            print_report(t, s, label=f'signal_quality={int(tier)}')

    # ── Save outputs ──
    trades_csv = out_dir / 'trades.csv'
    trades_all.to_csv(trades_csv, index=False)
    print(f"💾 {trades_csv}  ({len(trades_all)} rows)")

    equity_csv = out_dir / 'equity_curve.csv'
    equity_all.to_csv(equity_csv, header=['equity'])
    print(f"💾 {equity_csv}")

    report = {
        'config': vars(args),
        'all_events': sum_all,
        'by_slice': {k: v['summary'] for k, v in splits.items()},
    }
    report_path = out_dir / 'backtest_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"💾 {report_path}")


if __name__ == '__main__':
    main()
