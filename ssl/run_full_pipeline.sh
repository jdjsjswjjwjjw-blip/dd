#!/bin/bash
# ssl/run_full_pipeline.sh
# ═══════════════════════════════════════════════════════════════
# Full SSL training + validation pipeline.
#
# الاستخدام:
#   1. شغّل prepare_day_trading.py أولاً لتحضير features + LOB tensors
#   2. شغّل هذا الـ script (سيكتشف الـ output ويكمل من هناك)
#
# الـ phases:
#   A. Pretrain LOB Transformer (6 SSL heads، ~ساعة GPU)
#   B. Pretrain Price Cycle Model (4 SSL heads، ~30 دقيقة GPU)
#   C. Extract embeddings (دقائق)
#   D. Fine-tune direction head (دقائق)
#   E. Validation report (~5 ثواني)

set -e

# ── Configuration ──
PIPELINE_DIR="${1:-pipeline_clean}"
SSL_OUTPUT_DIR="${2:-checkpoints/ssl_run}"
TRAIN_SPLIT="${3:-0.75}"

FEATURES="${PIPELINE_DIR}/day_trading_features.parquet"
LOB_TENSORS="${PIPELINE_DIR}/lob_tensors.npy"
LOB_TIMESTAMPS="${PIPELINE_DIR}/lob_tensor_timestamps.npy"

# ── Sanity checks ──
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL FULL PIPELINE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline dir: $PIPELINE_DIR"
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

# ── Phase A: Pretrain LOB Transformer ──
echo "═══ PHASE A: Pretrain LOB Transformer ═══"
python ssl/pretrain_lob.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
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
python ssl/pretrain_cycle.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
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
python ssl/extract_embeddings.py \
    --features "$FEATURES" \
    --lob-tensors "$LOB_TENSORS" \
    --lob-timestamps "$LOB_TIMESTAMPS" \
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
python ssl/fine_tune_direction.py \
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
python ssl/validate_ssl.py \
    --features "$FEATURES" \
    --embeddings "$EMBEDDINGS" \
    --direction-head "$DIRECTION_CKPT" \
    --train-split "$TRAIN_SPLIT" \
    --output "$SSL_OUTPUT_DIR/validation_report.json" \
    2>&1 | tee "$SSL_OUTPUT_DIR/phase_e_validation.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SSL FULL PIPELINE COMPLETE"
echo "  Output: $SSL_OUTPUT_DIR"
echo "  Final verdict: see $SSL_OUTPUT_DIR/validation_report.json"
echo "═══════════════════════════════════════════════════════════════"
