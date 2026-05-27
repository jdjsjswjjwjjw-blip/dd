#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/walk_forward.sh
# ────────────────────────────────────────────────────────────────────────────
# Walk-forward validation for the SSL + hybrid pipeline.
#
# Design:
#   • Pretrain SSL ONCE on the full date range (expensive)
#   • Extract embeddings ONCE for the full period
#   • N walk-forward folds with expanding train window + 6-month test windows
#     (folds generated automatically by tools/build_walk_forward_folds.py)
#   • For each fold: train hybrid on train slice, backtest on test slice
#
# Why frozen SSL: pretraining sees the full range (one-time, no fold-specific
# leakage because SSL has no labels). Walk-forward measures HYBRID
# generalization. Matches industry practice.
#
# Strict variant: --strict-ssl retrains SSL inside each fold (~5x compute).
#
# Usage:
#   ./scripts/walk_forward.sh <raw_data_root> <output_root> \
#       [start=2021-01] [end=2025-12] [root_symbol=6B] [--strict-ssl]
#
# Pre-requirement: scripts/run_day_trade_only.sh must have produced the
# day_trade features for the full date range. This script will invoke it
# automatically if missing.
# ════════════════════════════════════════════════════════════════════════════

set -eo pipefail

RAW_ROOT="${1:?Usage: $0 <raw_data_root> <output_root> [start=2021-01] [end=2025-12] [root=6B] [--strict-ssl]}"
OUT_ROOT="${2:?Usage: $0 <raw_data_root> <output_root> [start=2021-01] [end=2025-12] [root=6B] [--strict-ssl]}"
START_YM="${3:-2021-01}"
END_YM="${4:-2025-12}"
ROOT_SYMBOL="${5:-6B}"
STRICT_SSL=0
[ "${6:-}" = "--strict-ssl" ] && STRICT_SSL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$OUT_ROOT/walk_forward.log"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  WALK-FORWARD VALIDATION  ${START_YM} → ${END_YM}"
echo "════════════════════════════════════════════════════════════════"
echo "  raw:        $RAW_ROOT"
echo "  output:     $OUT_ROOT"
echo "  range:      $START_YM .. $END_YM"
echo "  root:       $ROOT_SYMBOL"
echo "  strict_ssl: $STRICT_SSL"
echo "  start:      $(date)"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 0: ensure day_trade pipeline has produced features for the range
# ──────────────────────────────────────────────────────────────────────────
START_YEAR="${START_YM%-*}"
END_YEAR="${END_YM%-*}"
DT_OUT="$OUT_ROOT/day_trade"
echo "═══ STEP 0: verify day_trade features ($START_YEAR..$END_YEAR) ═══"
if [ ! -d "$DT_OUT/combined" ]; then
    echo "  Running day_trade pipeline first..."
    ./scripts/run_day_trade_only.sh \
        "$RAW_ROOT" "$DT_OUT" "$ROOT_SYMBOL" "$START_YEAR" "$END_YEAR"
fi
echo "  ✓ day_trade features at $DT_OUT/combined/"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Pretrain SSL ONCE on full range (frozen-SSL design)
# ──────────────────────────────────────────────────────────────────────────
SSL_OUT="$OUT_ROOT/ssl_full"
SSL_COMBINED="$OUT_ROOT/ssl_combined"
if [ "$STRICT_SSL" = "0" ]; then
    echo "═══ STEP 1: pretrain SSL once on $START_YM..$END_YM ═══"
    if [ ! -f "$SSL_OUT/embeddings/embeddings.npy" ]; then
        echo "  Building per-month order tensors..."
        for m_dir in "$DT_OUT"/features/*/; do
            ym=$(basename "$m_dir")
            mbo_file="$DT_OUT/continuous/${ym}.mbo.parquet"
            [ ! -f "$mbo_file" ] && { echo "    ⏭️  ${ym}: no mbo"; continue; }
            [ -f "$m_dir/order_features.npy" ] && continue   # idempotent
            python self_supervised/build_order_batches.py \
                --mbo "$mbo_file" \
                --features "$m_dir/day_trading_features.parquet" \
                --output "$m_dir" \
                --freq 15min --lookback-bars 50 --n-orders 200 \
                >/dev/null 2>&1 || echo "    ❌ ${ym} order build failed"
        done

        # Combine all monthly artifacts into one SSL dataset
        quarter_dirs=("$DT_OUT"/features/*/)
        python self_supervised/combine_quarter_artifacts.py \
            --quarter-dirs "${quarter_dirs[@]}" \
            --output-dir "$SSL_COMBINED" \
            --overlap-mode auto

        # Pretrain SSL (LOB + cycle) end-to-end
        ./self_supervised/run_ssl_only.sh "$SSL_COMBINED" "$SSL_OUT" 0.85
    else
        echo "  ✓ existing checkpoint $SSL_OUT/embeddings/embeddings.npy"
    fi
fi
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Generate fold definitions
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 2: generate walk-forward folds ═══"
mapfile -t FOLDS < <(python tools/build_walk_forward_folds.py \
    --start "$START_YM" --end "$END_YM" --format bash)
n_folds="${#FOLDS[@]}"
echo "  → $n_folds folds"
python tools/build_walk_forward_folds.py --start "$START_YM" --end "$END_YM" \
    --format table
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Walk-forward — N folds
# ──────────────────────────────────────────────────────────────────────────
# Optional anti-collapse environment variables — picked up by ENHANCED runner.
# Defaults match the baseline behavior (all disabled).
#   WF_USE_SIMPLEX_ETF=1      → replace direction head with Simplex ETF
#   WF_ORTHO_WEIGHT=0.5       → orthogonality penalty against rule features
#   WF_USE_DBMTL=1            → DB-MTL gradient balancer
#   WF_SHARPE_WEIGHT=0.3      → differentiable Sharpe regularizer
# Set any to enable the corresponding research-driven counter-measure.
RUNNER="tools/run_walk_forward_fold.py"
EXTRA_FLAGS=()
if [ -n "${WF_USE_SIMPLEX_ETF:-}" ] || [ -n "${WF_ORTHO_WEIGHT:-}" ] \
   || [ -n "${WF_USE_DBMTL:-}" ] || [ -n "${WF_SHARPE_WEIGHT:-}" ]; then
    RUNNER="tools/run_walk_forward_fold_enhanced.py"
    [ "${WF_USE_SIMPLEX_ETF:-0}" = "1" ] && EXTRA_FLAGS+=("--use-simplex-etf")
    [ -n "${WF_ORTHO_WEIGHT:-}" ] && EXTRA_FLAGS+=("--ortho-weight" "$WF_ORTHO_WEIGHT")
    [ "${WF_USE_DBMTL:-0}" = "1" ] && EXTRA_FLAGS+=("--use-dbmtl")
    [ -n "${WF_SHARPE_WEIGHT:-}" ] && EXTRA_FLAGS+=("--sharpe-weight" "$WF_SHARPE_WEIGHT")
    echo "═══ Anti-collapse flags ON: ${EXTRA_FLAGS[*]}"
fi

echo "═══ STEP 3: run $n_folds folds (runner=$(basename $RUNNER)) ═══"
for fold_spec in "${FOLDS[@]}"; do
    read -r fold_id tr_s tr_e te_s te_e <<< "$fold_spec"
    fold_dir="$OUT_ROOT/fold_${fold_id}"
    mkdir -p "$fold_dir"

    echo "  ─── Fold $fold_id ─── train ${tr_s}..${tr_e} | test ${te_s}..${te_e} ───"

    python "$RUNNER" \
        --fold-id "$fold_id" \
        --combined-features "$DT_OUT/combined/day_trading_features.parquet" \
        --ssl-embeddings "$SSL_OUT/embeddings/embeddings.npy" \
        --embedding-valid "$SSL_OUT/embeddings/embedding_valid.npy" \
        --train-start "$tr_s" --train-end "$tr_e" \
        --test-start "$te_s" --test-end "$te_e" \
        --output-dir "$fold_dir" \
        "${EXTRA_FLAGS[@]}" \
        2>&1 | tail -10
done
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: Aggregate per-fold metrics
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 4: aggregate walk-forward results ═══"
python tools/aggregate_walk_forward.py \
    --folds-dir "$OUT_ROOT" \
    --output "$OUT_ROOT/walk_forward_summary.json"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  WALK-FORWARD COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  end:     $(date)"
echo "  log:     $LOG"
echo "  summary: $OUT_ROOT/walk_forward_summary.json"
