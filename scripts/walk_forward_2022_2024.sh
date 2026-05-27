#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# scripts/walk_forward_2022_2024.sh
# ────────────────────────────────────────────────────────────────────────────
# Walk-forward validation for the SSL + hybrid pipeline on 2022-2024 data.
#
# Design:
#   • Pretrain SSL ONCE on full 2022-2024 (expensive, ~4-7h on GPU)
#   • Extract embeddings ONCE for the full period
#   • 5 walk-forward folds with expanding train window + 6-month test:
#       Fold 1: train H1 2022      → test H2 2022
#       Fold 2: train 2022 full    → test H1 2023
#       Fold 3: train 2022+H1 2023 → test H2 2023
#       Fold 4: train 2022-2023    → test H1 2024
#       Fold 5: train ...H1 2024   → test H2 2024
#   • For each fold: train hybrid on train slice, backtest on test slice
#
# Why frozen SSL: pretraining sees all 3 years (one-time, no fold-specific
# leakage because SSL has no labels). Walk-forward measures HYBRID
# generalization, which is where the labels live. This matches industry
# practice (large pretrain, walk-forward fine-tune).
#
# Strict variant: pass --strict-ssl to retrain SSL inside each fold (5x more
# compute, ~20-30 hours total). Use only if you suspect SSL is overfitting
# the union of all years.
#
# Usage:
#   ./scripts/walk_forward_2022_2024.sh <raw_data_root> <output_root> [--strict-ssl]
#
# Pre-requirement: scripts/run_day_trade_only.sh must have already produced
# <day_trade_out>/combined/ for the full 2022-2024 period.
# ════════════════════════════════════════════════════════════════════════════

set -eo pipefail

RAW_ROOT="${1:?Usage: $0 <raw_data_root> <output_root> [--strict-ssl]}"
OUT_ROOT="${2:?Usage: $0 <raw_data_root> <output_root> [--strict-ssl]}"
STRICT_SSL=0
[ "${3:-}" = "--strict-ssl" ] && STRICT_SSL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$OUT_ROOT/walk_forward.log"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  WALK-FORWARD VALIDATION  2022 → 2024  (5 folds)"
echo "════════════════════════════════════════════════════════════════"
echo "  raw:       $RAW_ROOT"
echo "  output:    $OUT_ROOT"
echo "  strict_ssl: $STRICT_SSL"
echo "  start:     $(date)"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 0: ensure day_trade pipeline has produced features for all months
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 0: verify day_trade features exist ═══"
DT_OUT="$OUT_ROOT/day_trade"
if [ ! -d "$DT_OUT/combined" ]; then
    echo "  Running day_trade pipeline first..."
    ./scripts/run_day_trade_only.sh "$RAW_ROOT" "$DT_OUT" 6B 2022 2024
fi
echo "  ✓ day_trade features at $DT_OUT/combined/"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Pretrain SSL ONCE on full 2022-2024 (frozen-SSL design)
# ──────────────────────────────────────────────────────────────────────────
SSL_OUT="$OUT_ROOT/ssl_full"
if [ "$STRICT_SSL" = "0" ]; then
    echo "═══ STEP 1: pretrain SSL once on full 2022-2024 ═══"
    if [ ! -f "$SSL_OUT/embeddings/embeddings.npy" ]; then
        # Need order tensors too — build per month + combine
        echo "  Building per-month order tensors..."
        for m_dir in "$DT_OUT"/features/*/; do
            ym=$(basename "$m_dir")
            mbo_file="$DT_OUT/continuous/${ym}.mbo.parquet"
            if [ ! -f "$mbo_file" ]; then
                echo "    ⏭️  ${ym}: no mbo file"
                continue
            fi
            if [ -f "$m_dir/order_features.npy" ]; then
                continue   # already done
            fi
            python self_supervised/build_order_batches.py \
                --mbo "$mbo_file" \
                --features "$m_dir/day_trading_features.parquet" \
                --output "$m_dir" \
                --freq 15min --lookback-bars 50 --n-orders 200 \
                >/dev/null 2>&1 || echo "    ❌ ${ym} order build failed"
        done

        # Combine all months with order tensors
        quarter_dirs=("$DT_OUT"/features/*/)
        SSL_COMBINED="$OUT_ROOT/ssl_combined"
        python self_supervised/combine_quarter_artifacts.py \
            --quarter-dirs "${quarter_dirs[@]}" \
            --output-dir "$SSL_COMBINED" \
            --overlap-mode auto

        # Pretrain SSL
        ./self_supervised/run_ssl_only.sh "$SSL_COMBINED" "$SSL_OUT" 0.85
    else
        echo "  ✓ existing checkpoint $SSL_OUT/embeddings/embeddings.npy"
    fi
fi
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Walk-forward — 5 folds
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 2: 5-fold walk-forward (hybrid training + backtest) ═══"

# Fold definitions:  fold_id, train_start_ym, train_end_ym, test_start_ym, test_end_ym
declare -a FOLDS=(
    "1 2022-01 2022-06 2022-07 2022-12"
    "2 2022-01 2022-12 2023-01 2023-06"
    "3 2022-01 2023-06 2023-07 2023-12"
    "4 2022-01 2023-12 2024-01 2024-06"
    "5 2022-01 2024-06 2024-07 2024-12"
)

for fold_spec in "${FOLDS[@]}"; do
    read -r fold_id tr_s tr_e te_s te_e <<< "$fold_spec"
    fold_dir="$OUT_ROOT/fold_${fold_id}"
    mkdir -p "$fold_dir"

    echo "  ─── Fold $fold_id ─── train ${tr_s}..${tr_e} | test ${te_s}..${te_e} ───"

    python tools/run_walk_forward_fold.py \
        --fold-id "$fold_id" \
        --combined-features "$DT_OUT/combined/day_trading_features.parquet" \
        --ssl-embeddings "$SSL_OUT/embeddings/embeddings.npy" \
        --embedding-valid "$SSL_OUT/embeddings/embedding_valid.npy" \
        --train-start "$tr_s" --train-end "$tr_e" \
        --test-start "$te_s" --test-end "$te_e" \
        --output-dir "$fold_dir" \
        2>&1 | tail -10
done
echo ""

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Aggregate per-fold metrics
# ──────────────────────────────────────────────────────────────────────────
echo "═══ STEP 3: aggregate walk-forward results ═══"
python tools/aggregate_walk_forward.py \
    --folds-dir "$OUT_ROOT" \
    --output "$OUT_ROOT/walk_forward_summary.json"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  WALK-FORWARD COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  end:    $(date)"
echo "  log:    $LOG"
echo "  summary: $OUT_ROOT/walk_forward_summary.json"
