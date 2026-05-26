#!/bin/bash
# ════════════════════════════════════════════════════════════════
# QuantSystem V19.2 — تشغيل كامل
# ════════════════════════════════════════════════════════════════

set -e

# ── إعدادات ──
SYMBOL="${SYMBOL:-6B}"
MBO="${MBO:-data/mbo.parquet}"
MBP="${MBP:-data/mbp.parquet}"
OUT="${OUT:-out}"
FREQ="${FREQ:-5min}"
HORIZON="${HORIZON:-6}"
CONFIDENCE="${CONFIDENCE:-medium}"

mkdir -p $OUT

echo "════════════════════════════════════════════════════"
echo " QuantSystem V19.2 — $SYMBOL"
echo " MBO: $MBO"
echo " MBP: $MBP"
echo " Freq: $FREQ, Horizon: $HORIZON, Confidence: $CONFIDENCE"
echo "════════════════════════════════════════════════════"

# ── 0. الاختبارات أولاً ──
echo ""
echo "🧪 تشغيل اختبارات الوحدة..."
python tests/test_simulators.py || { echo "❌ الاختبارات فشلت"; exit 1; }

# ── 1. prepare_day_trading (يحوي microstructure engines) ──
echo ""
echo "① prepare_day_trading (microstructure من MBO خام)..."
python prepare_day_trading.py \
  --mbo "$MBO" --mbp "$MBP" \
  --output $OUT \
  --freq $FREQ --horizon $HORIZON

# ── 2. المحاكيات الخمسة ──
echo ""
echo "② feature_simulators (18 مخرج)..."
python feature_simulators.py \
  --input $OUT/day_trading_features.parquet \
  --output $OUT/sim_${SYMBOL}.parquet --window 200

# ── 3. wall depth ──
echo ""
echo "③ wall_depth_simulator (8 مخرجات من MBP-10)..."
python wall_depth_simulator.py \
  --bars $OUT/sim_${SYMBOL}.parquet --mbp "$MBP" \
  --output $OUT/walls_${SYMBOL}.parquet --freq $FREQ --window 200

# ── 4. iceberg ──
echo ""
echo "④ iceberg_simulator (5 مخرجات من MBO)..."
python iceberg_simulator.py \
  --bars $OUT/walls_${SYMBOL}.parquet --mbo "$MBO" \
  --output $OUT/icebergs_${SYMBOL}.parquet --freq $FREQ --window 200

# ── 5. session mapper ──
echo ""
echo "⑤ session_mapper (16 zones × 12 levels × 5 events)..."
python session_mapper.py \
  --input $OUT/icebergs_${SYMBOL}.parquet \
  --output $OUT/mapped_${SYMBOL}.parquet

# ── 6. edge scanner ──
echo ""
echo "⑥ edge_scanner (FDR + 3-way + permutation)..."
python edge_scanner.py \
  --input $OUT/mapped_${SYMBOL}.parquet \
  --output $OUT/edge_candidates_${SYMBOL}.json \
  --symbol $SYMBOL --confidence $CONFIDENCE \
  --horizons 3,6,12

# ── 7. cluster ──
echo ""
echo "⑦ cluster_engine..."
python cluster_engine.py \
  --candidates $OUT/edge_candidates_${SYMBOL}.json \
  --data $OUT/mapped_${SYMBOL}.parquet \
  --output $OUT/alphas_${SYMBOL}.json

echo ""
echo "════════════════════════════════════════════════════"
echo " ✅ اكتمل!"
echo "   النتيجة: $OUT/alphas_${SYMBOL}.json"
echo "════════════════════════════════════════════════════"
