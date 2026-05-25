#!/bin/bash
# self_supervised/run_ssl_only.sh
# ═══════════════════════════════════════════════════════════════
# نفس run_full_pipeline.sh بس بدون Phase A0 — لأن الـ order_batches
# اتبنت يدوياً قبل (per-quarter + combine_quarter_artifacts).
#
# الاستخدام:
#   ./self_supervised/run_ssl_only.sh <combined_dir> [ssl_output] [train_split]
#
# الـ combined_dir لازم يحتوي:
#   day_trading_features.parquet
#   lob_tensors.npy
#   order_features.npy   (الناتج من build_order_batches/combine)
#   order_masks.npy

set -eo pipefail

COMBINED_DIR="${1:?Usage: $0 <combined_dir> [ssl_output] [train_split]}"
SSL_OUTPUT_DIR="${2:-checkpoints/ssl_run}"
TRAIN_SPLIT="${3:-0.75}"

FEATURES="${COMBINED_DIR}/day_trading_features.parquet"
LOB_TENSORS="${COMBINED_DIR}/lob_tensors.npy"
LOB_TIMESTAMPS="${COMBINED_DIR}/lob_tensor_timestamps.npy"
ORDER_BATCHES_DIR="${COMBINED_DIR}"  # order_features.npy + order_masks.npy live here

# ── Sanity ──
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL PIPELINE (no Phase A0 — orders pre-built)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Combined dir: $COMBINED_DIR"
echo "  Output dir:   $SSL_OUTPUT_DIR"
echo "  Train split:  $TRAIN_SPLIT"
echo ""

for f in "$FEATURES" "$LOB_TENSORS"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing required file: $f"
        exit 1
    fi
done

if [ ! -f "${ORDER_BATCHES_DIR}/order_features.npy" ]; then
    echo "❌ Missing order_features.npy in $ORDER_BATCHES_DIR"
    echo "   Run build_order_batches.py per quarter + combine_quarter_artifacts.py first"
    exit 1
fi

mkdir -p "$SSL_OUTPUT_DIR"
ORDER_BATCHES_FLAG="--order-batches-dir $ORDER_BATCHES_DIR"

# ── Phase A: Pretrain LOB Transformer ──
echo "═══ PHASE A: Pretrain LOB Transformer ═══"
python self_supervised/pretrain_lob.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
    $ORDER_BATCHES_FLAG \
    --output "$SSL_OUTPUT_DIR/lob" \
    --epochs 30 \
    --batch-size 8 \
    --num-workers 0 \
    --lr 1e-5 \
    --train-split "$TRAIN_SPLIT" \
    --embargo-bars 24 \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_a_lob.log"

LOB_CKPT="$SSL_OUTPUT_DIR/lob/best_ssl_lob.pt"
[ -f "$LOB_CKPT" ] || { echo "❌ Phase A failed"; exit 1; }
echo "✅ Phase A complete"
echo ""

# ── Phase B: Pretrain Price Cycle ──
echo "═══ PHASE B: Pretrain Price Cycle Model ═══"
python self_supervised/pretrain_cycle.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
    $ORDER_BATCHES_FLAG \
    --output "$SSL_OUTPUT_DIR/cycle" \
    --epochs 25 \
    --batch-size 32 \
    --num-workers 0 \
    --lr 2e-5 \
    --train-split "$TRAIN_SPLIT" \
    --embargo-bars 24 \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_b_cycle.log"

CYCLE_CKPT="$SSL_OUTPUT_DIR/cycle/best_ssl_cycle.pt"
[ -f "$CYCLE_CKPT" ] || { echo "❌ Phase B failed"; exit 1; }
echo "✅ Phase B complete"
echo ""

# ── Phase C: Extract Embeddings ──
echo "═══ PHASE C: Extract Embeddings ═══"
python self_supervised/extract_embeddings.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
    $ORDER_BATCHES_FLAG \
    --lob-checkpoint "$LOB_CKPT" \
    --cycle-checkpoint "$CYCLE_CKPT" \
    --output-dir "$SSL_OUTPUT_DIR/embeddings" \
    --train-split "$TRAIN_SPLIT" \
    --embargo-bars 24 \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_c_embeddings.log"

EMBEDDINGS="$SSL_OUTPUT_DIR/embeddings/embeddings.npy"
[ -f "$EMBEDDINGS" ] || { echo "❌ Phase C failed"; exit 1; }
echo "✅ Phase C complete"
echo ""

# ── Phase D: Fine-Tune Direction Head ──
echo "═══ PHASE D: Fine-Tune Direction Head ═══"
python self_supervised/fine_tune_direction.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --output "$SSL_OUTPUT_DIR/direction" \
    --epochs 100 \
    --batch-size 32 \
    --hidden-dim 128 \
    --train-split "$TRAIN_SPLIT" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_d_direction.log"

DIRECTION_CKPT="$SSL_OUTPUT_DIR/direction/best_direction_head.pt"
[ -f "$DIRECTION_CKPT" ] || { echo "❌ Phase D failed"; exit 1; }
echo "✅ Phase D complete"
echo ""

# ── Phase E: Validation ──
echo "═══ PHASE E: Validation ═══"
python self_supervised/validate_ssl.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --direction-head "$DIRECTION_CKPT" \
    --train-split "$TRAIN_SPLIT" \
    --output "$SSL_OUTPUT_DIR/validation_report.json" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_e_validation.log"

# ── Phase F: Pattern Discovery ──
echo ""
echo "═══ PHASE F: Pattern Discovery ═══"
python self_supervised/explain_patterns.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --direction-head "$DIRECTION_CKPT" \
    --output "$SSL_OUTPUT_DIR/interpretability" \
    --n-clusters 8 \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_f_patterns.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL PIPELINE COMPLETE"
echo "  📊 Validation: $SSL_OUTPUT_DIR/validation_report.json"
echo "  🔍 Patterns:   $SSL_OUTPUT_DIR/interpretability/interpretability_report.md"
echo "═══════════════════════════════════════════════════════════════"
