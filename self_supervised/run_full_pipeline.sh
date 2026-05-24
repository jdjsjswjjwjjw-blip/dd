#!/bin/bash
# self_supervised/run_full_pipeline.sh
# ═══════════════════════════════════════════════════════════════
# Full SSL training + validation pipeline (MAX SIGNAL, NO DATA LOSS).
#
# الاستخدام:
#   ./self_supervised/run_full_pipeline.sh <pipeline_dir> <mbo_path> [ssl_output] [train_split]
#
#   Example:
#   ./self_supervised/run_full_pipeline.sh pipeline_3months mbo_3months.parquet
#
# الـ phases:
#   A0. Build OrderBatches من raw MBO (يحافظ على iceberg signals)
#   A.  Pretrain LOB Transformer (6 SSL heads، ~ساعة GPU)
#   B.  Pretrain Price Cycle Model (4 SSL heads، ~30 دقيقة GPU)
#   C.  Extract embeddings (دقائق)
#   D.  Fine-tune direction head (دقائق)
#   E.  Validation report (~5 ثواني)

set -e

# ── Configuration ──
PIPELINE_DIR="${1:-pipeline_clean}"
MBO_PATH="${2:-}"
SSL_OUTPUT_DIR="${3:-checkpoints/ssl_run}"
TRAIN_SPLIT="${4:-0.75}"

FEATURES="${PIPELINE_DIR}/day_trading_features.parquet"
LOB_TENSORS="${PIPELINE_DIR}/lob_tensors.npy"
LOB_TIMESTAMPS="${PIPELINE_DIR}/lob_tensor_timestamps.npy"
ORDER_BATCHES_DIR="${SSL_OUTPUT_DIR}/order_batches"

# ── Sanity checks ──
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL FULL PIPELINE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline dir: $PIPELINE_DIR"
echo "  MBO path:     ${MBO_PATH:-<not provided — will use pseudo-orders>}"
echo "  Output dir:   $SSL_OUTPUT_DIR"
echo "  Train split:  $TRAIN_SPLIT"
echo ""

for f in "$FEATURES" "$LOB_TENSORS"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing required file: $f"
        echo "   Run prepare_day_trading.py first to generate it"
        exit 1
    fi
done

mkdir -p "$SSL_OUTPUT_DIR"

# ── Phase A0: Build OrderBatches from raw MBO (الأهم لـ iceberg detection) ──
ORDER_BATCHES_FLAG=""
if [ -n "$MBO_PATH" ] && [ -f "$MBO_PATH" ]; then
    echo "═══ PHASE A0: Build OrderBatches من raw MBO ═══"
    echo "  هذا يحافظ على iceberg signals + order_id + action لكل tick"
    python self_supervised/build_order_batches.py \
        --mbo "$MBO_PATH" \
        --features "$FEATURES" \
        --output "$ORDER_BATCHES_DIR" \
        --lookback-bars 50 --n-orders 200 \
        2>&1 | tee "$SSL_OUTPUT_DIR/phase_a0_order_batches.log"

    if [ -f "$ORDER_BATCHES_DIR/order_features.npy" ]; then
        ORDER_BATCHES_FLAG="--order-batches-dir $ORDER_BATCHES_DIR"
        echo "✅ Phase A0 complete: real orders جاهزة"
    else
        echo "⚠️  Phase A0 failed — falling back to pseudo-orders"
    fi
    echo ""
else
    echo "⚠️  No MBO file provided — using pseudo-orders from LOB tensors"
    echo "    (iceberg detection سيكون ضعيفاً)"
    echo ""
fi

# ── Phase A: Pretrain LOB Transformer ──
echo "═══ PHASE A: Pretrain LOB Transformer ═══"
python self_supervised/pretrain_lob.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
    $ORDER_BATCHES_FLAG \
    --output "$SSL_OUTPUT_DIR/lob" \
    --epochs 50 \
    --batch-size 32 \
    --train-split "$TRAIN_SPLIT" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_a_lob.log"

LOB_CKPT="$SSL_OUTPUT_DIR/lob/best_ssl_lob.pt"
if [ ! -f "$LOB_CKPT" ]; then
    echo "❌ Phase A failed — no checkpoint produced"
    exit 1
fi
echo "✅ Phase A complete: $LOB_CKPT"
echo ""

# ── Phase B: Pretrain Price Cycle ──
echo "═══ PHASE B: Pretrain Price Cycle Model ═══"
python self_supervised/pretrain_cycle.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
    $ORDER_BATCHES_FLAG \
    --output "$SSL_OUTPUT_DIR/cycle" \
    --epochs 40 \
    --batch-size 32 \
    --train-split "$TRAIN_SPLIT" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_b_cycle.log"

CYCLE_CKPT="$SSL_OUTPUT_DIR/cycle/best_ssl_cycle.pt"
if [ ! -f "$CYCLE_CKPT" ]; then
    echo "❌ Phase B failed — no checkpoint produced"
    exit 1
fi
echo "✅ Phase B complete: $CYCLE_CKPT"
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
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_c_embeddings.log"

EMBEDDINGS="$SSL_OUTPUT_DIR/embeddings/embeddings.npy"
if [ ! -f "$EMBEDDINGS" ]; then
    echo "❌ Phase C failed"
    exit 1
fi
echo "✅ Phase C complete: $EMBEDDINGS"
echo ""

# ── Phase D: Fine-Tune Direction Head ──
echo "═══ PHASE D: Fine-Tune Direction Head ═══"
python self_supervised/fine_tune_direction.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --output "$SSL_OUTPUT_DIR/direction" \
    --epochs 100 \
    --batch-size 32 \
    --train-split "$TRAIN_SPLIT" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_d_direction.log"

DIRECTION_CKPT="$SSL_OUTPUT_DIR/direction/best_direction_head.pt"
if [ ! -f "$DIRECTION_CKPT" ]; then
    echo "❌ Phase D failed"
    exit 1
fi
echo "✅ Phase D complete: $DIRECTION_CKPT"
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

# ── Phase F: Pattern Discovery & Interpretability ──
echo ""
echo "═══ PHASE F: Pattern Discovery — ما تعلّمه النموذج ═══"
python self_supervised/explain_patterns.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --direction-head "$DIRECTION_CKPT" \
    --output "$SSL_OUTPUT_DIR/interpretability" \
    --n-clusters 8 \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_f_patterns.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL FULL PIPELINE COMPLETE"
echo "  Output: $SSL_OUTPUT_DIR"
echo ""
echo "  📊 Validation: $SSL_OUTPUT_DIR/validation_report.json"
echo "  🔍 Patterns:   $SSL_OUTPUT_DIR/interpretability/interpretability_report.md"
echo "  📈 Clusters:   $SSL_OUTPUT_DIR/interpretability/cluster_statistics.csv"
echo "═══════════════════════════════════════════════════════════════"
