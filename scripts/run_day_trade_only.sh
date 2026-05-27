#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/run_day_trade_only.sh
# ────────────────────────────────────────────────────────────────────────────
# Run subsystem A (day_trade rule pipeline) end-to-end without SSL or hybrid.
#
# Pipeline:
#   1. Extract continuous contract per monthly file (smart rollover)
#   2. prepare_day_trading per month
#   3. Combine all months into one parquet + LOB tensor
#   4. Run quality checks
#   5. Strict backtest on the holdout slice
#
# This is the "production-ready 78.3% hit rate" path — no ML, just rules.
#
# Usage:
#   ./scripts/run_day_trade_only.sh <raw_data_root> <output_root>
#
# Expected raw_data_root layout:
#   <raw_data_root>/
#     2019-01.mbo.parquet    2019-01.mbp10.parquet
#     2019-02.mbo.parquet    2019-02.mbp10.parquet
#     ...
#     2024-12.mbo.parquet    2024-12.mbp10.parquet
#
# Produces in <output_root>:
#   continuous/           — per-month continuous parquet (post-rollover)
#   features/             — per-month prepare_day_trading outputs
#   combined/             — single combined dataset
#   backtest/             — strict backtest results
#   pipeline.log          — full run log
# ════════════════════════════════════════════════════════════════════════════

set -eo pipefail

RAW_ROOT="${1:?Usage: $0 <raw_data_root> <output_root>}"
OUT_ROOT="${2:?Usage: $0 <raw_data_root> <output_root>}"
ROOT_SYMBOL="${3:-6B}"
START_YEAR="${4:-2019}"
END_YEAR="${5:-2024}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$OUT_ROOT/pipeline.log"

mkdir -p "$OUT_ROOT/continuous" "$OUT_ROOT/features" \
         "$OUT_ROOT/combined" "$OUT_ROOT/backtest"
exec > >(tee -a "$LOG") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  DAY_TRADE STANDALONE PIPELINE"
echo "════════════════════════════════════════════════════════════════"
echo "  raw:    $RAW_ROOT"
echo "  output: $OUT_ROOT"
echo "  root:   $ROOT_SYMBOL"
echo "  years:  $START_YEAR..$END_YEAR"
echo "  start:  $(date)"
echo ""

cd "$REPO_ROOT"

# ──────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract continuous contract per monthly file
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 1: Smart contract extraction (rollover) ═══"
n_extracted=0
for year in $(seq "$START_YEAR" "$END_YEAR"); do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        ym="${year}-${month}"
        for kind in mbo mbp10; do
            in_file="$RAW_ROOT/${ym}.${kind}.parquet"
            out_file="$OUT_ROOT/continuous/${ym}.${kind}.parquet"
            if [ ! -f "$in_file" ]; then
                # Try CSV fallback
                in_file="$RAW_ROOT/${ym}.${kind}.csv"
                [ ! -f "$in_file" ] && {
                    echo "  ⚠️  missing: ${ym}.${kind}"
                    continue
                }
            fi
            if [ -f "$out_file" ]; then
                echo "  ⏭️  skip: ${ym}.${kind} (already extracted)"
                continue
            fi
            python tools/extract_continuous_contract.py "$in_file" \
                --root "$ROOT_SYMBOL" --output "$out_file" \
                >/dev/null 2>&1 \
                || { echo "  ❌ extract failed: ${ym}.${kind}"; continue; }
            n_extracted=$((n_extracted + 1))
        done
    done
done
echo "  ✓ extracted $n_extracted files"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 2 — prepare_day_trading per month
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 2: prepare_day_trading per month ═══"
n_prepared=0
quarter_dirs=()
for year in $(seq "$START_YEAR" "$END_YEAR"); do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        ym="${year}-${month}"
        mbo_file="$OUT_ROOT/continuous/${ym}.mbo.parquet"
        mbp_file="$OUT_ROOT/continuous/${ym}.mbp10.parquet"
        out_dir="$OUT_ROOT/features/${ym}"

        if [ ! -f "$mbo_file" ] || [ ! -f "$mbp_file" ]; then
            echo "  ⏭️  skip ${ym}: missing inputs"
            continue
        fi
        if [ -f "$out_dir/day_trading_features.parquet" ]; then
            echo "  ⏭️  skip ${ym}: already prepared"
            quarter_dirs+=("$out_dir")
            continue
        fi

        mkdir -p "$out_dir"
        python prepare_day_trading.py \
            --mbo "$mbo_file" --mbp "$mbp_file" \
            --output "$out_dir" --freq 15min \
            --session_profile daytrade_default \
            >"$out_dir/prepare.log" 2>&1 \
            && { echo "  ✓ ${ym}"; n_prepared=$((n_prepared + 1));
                 quarter_dirs+=("$out_dir"); } \
            || { echo "  ❌ ${ym}: see $out_dir/prepare.log"; }
    done
done
echo "  ✓ prepared $n_prepared month-dirs"
echo ""

if [ "${#quarter_dirs[@]}" -lt 2 ]; then
    echo "❌ Need at least 2 month-dirs to continue."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────
# STEP 3 — Combine all months into one dataset
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 3: combine_quarter_artifacts ═══"
combined_dir="$OUT_ROOT/combined"
python self_supervised/combine_quarter_artifacts.py \
    --quarter-dirs "${quarter_dirs[@]}" \
    --output-dir "$combined_dir" \
    --overlap-mode auto \
    --carry-pips-per-quarter 40 \
    --lob-name lob_tensors.npy \
    --order-features-name NONE \
    --order-masks-name NONE \
    || { echo "❌ combine failed"; exit 1; }

echo "  ✓ combined to $combined_dir"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 4 — Quality checks on combined dataset
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 4: quality checks ═══"
python - <<PYEOF
import pandas as pd, numpy as np, json, sys
base = "$combined_dir"
df = pd.read_parquet(f"{base}/day_trading_features.parquet")
lob = np.load(f"{base}/lob_tensors.npy")
n = len(df)

errors = []

# (A) Alignment
if lob.shape[0] != n:
    errors.append(f"lob first dim {lob.shape[0]} != bars {n}")

# (B) Chronology
ts = pd.to_datetime(df["ts_event"])
gaps = ts.diff().dropna()
if not (gaps >= pd.Timedelta(0)).all():
    errors.append("timestamps not monotonic")

# (C) Session-break markers
n_breaks = int(df["is_session_break"].sum()) if "is_session_break" in df.columns else 0
n_rolls  = int(df["is_roll"].sum()) if "is_roll" in df.columns else 0

# (D) Duplicates
dup = int(df["ts_event"].duplicated().sum())
if dup > 0:
    errors.append(f"{dup} duplicate timestamps")

# (E) Event rate sane
ev_rate = float((df["event_flag"] == 1).mean()) if "event_flag" in df.columns else 0.0
if ev_rate < 0.02 or ev_rate > 0.20:
    errors.append(f"event_rate {ev_rate:.3f} out of [0.02, 0.20]")

# (F) Direction column on events
if "event_direction" in df.columns:
    ev_rows = df[df["event_flag"] == 1]
    zero_dir = int((ev_rows["event_direction"] == 0).sum())
    if len(ev_rows) > 0 and zero_dir / len(ev_rows) > 0.10:
        errors.append(f"{zero_dir}/{len(ev_rows)} events have direction=0")

print(f"  bars:              {n:,}")
print(f"  LOB shape:         {lob.shape}")
print(f"  is_session_break:  {n_breaks}")
print(f"  is_roll:           {n_rolls}")
print(f"  duplicates:        {dup}")
print(f"  event_rate:        {ev_rate*100:.2f}%")
print(f"  ts range:          {ts.iloc[0]} → {ts.iloc[-1]}")

if errors:
    print()
    print("❌ QUALITY ISSUES:")
    for e in errors:
        print(f"   • {e}")
    sys.exit(1)
else:
    print()
    print("✅ All quality checks passed")
PYEOF
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 5 — Strict backtest on holdout
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 5: strict backtest (holdout-only) ═══"
python self_supervised/backtest_day_trade.py \
    --features "$combined_dir/day_trading_features.parquet" \
    --output "$OUT_ROOT/backtest" \
    --strict \
    --starting-equity 100000 \
    --contracts-per-trade 1 \
    --tp-mult 1.5 --sl-mult 1.0 \
    --spread-pips 1.0 --slippage-pips-per-side 0.5 \
    --commission-per-side 0.50

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DAY_TRADE PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  end:    $(date)"
echo "  log:    $LOG"
echo ""
echo "  Outputs:"
echo "    $OUT_ROOT/continuous/        (per-month rollover-adjusted parquet)"
echo "    $OUT_ROOT/features/<ym>/     (per-month prepare_day_trading outputs)"
echo "    $OUT_ROOT/combined/          (single combined dataset)"
echo "    $OUT_ROOT/backtest/          (strict backtest results)"
echo ""
echo "  Next steps:"
echo "    cat $OUT_ROOT/backtest/backtest_report.json  | python -m json.tool | head -40"
echo "    head $OUT_ROOT/backtest/trades.csv"
