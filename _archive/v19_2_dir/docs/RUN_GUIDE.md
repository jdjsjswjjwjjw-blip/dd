# دليل تشغيل QuantSystem V19 الكامل

## ⚠️ توضيح مهم — order_id والتجديد المؤسسي

السوق *لا* يجدّد بنفس `order_id`. كل order جديد له معرّف جديد:

```
المؤسسة:
  Order_id=123 (5 lot) → Filled
  Order_id=456 (6 lot) جديد في نفس السعر
  Order_id=789 (4 lot) جديد آخر...
```

**نظامنا لا يعتمد على order_id إطلاقاً في كشف iceberg**.

`iceberg_simulator.py` يكشف التجديد بنمط (سعر + زمن + حجم متشابه ±30%)
لا بمطابقة order_id.

`prepare_day_trading.py` يستخدم order_id فقط لكشف cancellation
(هل نفس الـ order أُضيف ثم أُلغي = spoofing) — لا علاقة بـ iceberg.

---

## متطلبات الداتا

### MBO خام (databento format)
```
الأعمدة المطلوبة:
  ts_event   — timestamp بدقة نانوسكند
  action     — A (Add) / C (Cancel) / M (Modify) / T (Trade) / F (Fill) / R (Reset)
  side       — B (Bid side) / A (Ask side) / N (None)
  price      — السعر
  size       — الحجم
  order_id   — معرّف الـ order (للـ cancellation tracking فقط)
  sequence   — رقم تسلسل (اختياري)
  symbol     — رمز العقد (مثل 6BM5)
```

### MBP خام (10 مستويات)
```
الأعمدة المطلوبة:
  ts_event
  bid_px_00 .. bid_px_09   — أسعار 10 مستويات شراء
  bid_sz_00 .. bid_sz_09   — أحجام 10 مستويات شراء
  ask_px_00 .. ask_px_09   — أسعار 10 مستويات بيع
  ask_sz_00 .. ask_sz_09   — أحجام 10 مستويات بيع
```

---

## التشغيل خطوة بخطوة

### الإعداد الأولي

```bash
# المسار
mkdir -p ~/quant_v19 && cd ~/quant_v19
mkdir -p data out logs

# انسخ الملفات من ZIP
unzip QuantSystem_V19_Final.zip
mv QuantSystem_V19_Final/*.py .

# المتطلبات
pip install pandas numpy scipy pyarrow
```

---

### الخطوة ① — تنظيف الداتا الخام

```bash
python data_pipeline.py \
  --mbo  data/6BM5_mbo.csv \
  --mbp  data/6BM5_mbp.csv \
  --output out/ \
  --freq 5min \
  --horizon 6 \
  --max-move-pct 0.015 \
  --norm-window 200 \
  --tag 6BM5

# الناتج:
#   out/clean_features_6BM5.parquet
#   out/pipeline_manifest_6BM5.json
```

**ماذا يفعل**:
- ينظّف التيكات (تكرار، أسعار خاطئة، BBO متقاطع)
- يجمّع MBO إلى شمعات 5min (high/low من التيكات الفعلية)
- يدمج MBP depth (book_imbalance, depth_ratio, spread_mean)
- يصلح الـ features المكسورة (hawkes delta, kyle mean, clips)
- normalization تكيفي (rank + zscore)

---

### الخطوة ② — labels + LOB tensors

```bash
python prepare_day_trading.py \
  --preflight-parquet out/clean_features_6BM5.parquet \
  --output out/ \
  --sanitize-max-bar-return 0.015 \
  --tag 6BM5

# الناتج:
#   out/day_trading_features_6BM5.parquet  (مع labels + soft_labels)
#   out/lob_tensors_6BM5.npz               (للـ DeepLOB CNN)
```

**ماذا يفعل**:
- يضيف bias_label (LONG/SHORT/NEUTRAL)
- soft labels لكل horizon
- LOB tensors (50×20×3) للـ deep learning
- regime classification
- Kalman trend filter

⚠️ هذا الملف يحفظ كل MBO/MBP — لا يفقد شيئاً. تأكدنا.

---

### الخطوة ③ — المحاكيات الخمسة

```bash
python feature_simulators.py \
  --input out/day_trading_features_6BM5.parquet \
  --output out/sim_6BM5.parquet \
  --window 200

# الناتج: 18 عمود sim_*
#   sim_absorb_intensity, sim_absorb_direction, ...
#   sim_flow_strength, sim_flow_direction, ...
#   sim_informed_prob, sim_informed_urgency, ...
#   sim_wall_real, sim_depth_imbalance, sim_depth_pressure
#   sim_liquidity_state, sim_volatility_regime, sim_data_quality
```

---

### الخطوة ④ — محاكي الجدران العميق

```bash
python wall_depth_simulator.py \
  --bars out/sim_6BM5.parquet \
  --mbp  data/6BM5_mbp.csv \
  --output out/walls_6BM5.parquet \
  --freq 5min \
  --window 200

# الناتج: + 8 أعمدة جدران
#   sim_wall_bid_level, sim_wall_ask_level
#   sim_wall_bid_size, sim_wall_ask_size
#   sim_wall_growth, sim_wall_consumed
#   sim_wall_persist, sim_wall_shift
```

**يقرأ الدفتر الكامل** (bid_sz_00..09) ويكتشف:
- موقع الحائط بالضبط (أي مستوى عمق)
- نموه/تقلّصه
- استهلاكه
- صموده
- انتقاله

---

### الخطوة ⑤ — محاكي iceberg المؤسسي

```bash
python iceberg_simulator.py \
  --bars out/walls_6BM5.parquet \
  --mbo  data/6BM5_mbo.csv \
  --output out/icebergs_6BM5.parquet \
  --freq 5min \
  --window 200

# الناتج: + 5 أعمدة iceberg
#   sim_iceberg_prob, sim_iceberg_side
#   sim_iceberg_strength, sim_iceberg_replenish
#   sim_iceberg_stealth
```

**يكشف iceberg المؤسسي** بـ 4 طبقات:
- Volume imbalance > 3×
- Replenishment بحجم متشابه ±30% خلال 500ms
- Cross-level coordination
- Wall persist × consumed

🔑 **لا يعتمد على order_id** — يكشف النمط (السعر + الزمن + الحجم).

---

### الخطوة ⑥ — Session Mapper

```bash
python session_mapper.py \
  --input out/icebergs_6BM5.parquet \
  --output out/mapped_6BM5.parquet \
  --near-atr 0.5 \
  --touch-atr 0.15

# الناتج: + zones + levels + events
#   zone, quarter, zone_full
#   12 مستوى: PDH/PDL/PD_50/PWH/PWL/PW_50/asia_H/L/london_H/L/ny_H/L
#   12 × dist_to_<level>
#   12 × <level>_event (touch/break/reject/failed_break/near)
```

**16 zone**: آسيا/لندن/NY × Q1/Q2/Q3 + التداخلات (asia_london, london_ny)

---

### الخطوة ⑦ — Edge Scanner

```bash
python edge_scanner.py \
  --input out/mapped_6BM5.parquet \
  --output out/edge_candidates_6BM5.json \
  --horizons 3,6,12

# الناتج: edge_candidates_6BM5.json
#   كل edge اجتاز معايير 6B الصارمة:
#     t-stat ≥ 2.5, n ≥ 25, 12+ يوم
#     WR ≥ 53% أو avg ≥ 3 pip
```

**33 تركيبة بحث**:
- 13 عمق سوق (depth_pressure, informed, flow, ...)
- 10 جدران عميقة (wall_growing, ask_wall_consumed, ...)
- 10 iceberg مؤسسي (iceberg_buy_replenish, ...)

× 3 آفاق (@3/@6/@12 شمعة)
× 1,200 خلية محتملة (16 zone × 12 level × 5 event)

---

### الخطوة ⑧ — Cluster Engine

```bash
python cluster_engine.py \
  --candidates out/edge_candidates_6BM5.json \
  --data out/mapped_6BM5.parquet \
  --output out/alphas_6BM5.json \
  --corr-threshold 0.70

# الناتج: alphas_6BM5.json
#   alphas مميّزة (لا تكرار، لا ترابط > 70%)
#   مرتّبة بالقوة (t-stat × WR × ثبات)
```

---

## أمر واحد لتشغيل كل شيء

أنشئ `run_all.sh`:

```bash
#!/bin/bash
set -e

SYMBOL="6BM5"
MBO="data/${SYMBOL}_mbo.csv"
MBP="data/${SYMBOL}_mbp.csv"
OUT="out"
FREQ="5min"
HORIZON=6

mkdir -p $OUT

echo "① تنظيف..."
python data_pipeline.py --mbo $MBO --mbp $MBP --output $OUT \
  --freq $FREQ --horizon $HORIZON --tag $SYMBOL

echo "② labels + LOB..."
python prepare_day_trading.py \
  --preflight-parquet $OUT/clean_features_${SYMBOL}.parquet \
  --output $OUT --tag $SYMBOL

echo "③ المحاكيات الخمسة..."
python feature_simulators.py \
  --input $OUT/day_trading_features_${SYMBOL}.parquet \
  --output $OUT/sim_${SYMBOL}.parquet

echo "④ الجدران العميقة..."
python wall_depth_simulator.py \
  --bars $OUT/sim_${SYMBOL}.parquet --mbp $MBP \
  --output $OUT/walls_${SYMBOL}.parquet --freq $FREQ

echo "⑤ iceberg المؤسسي..."
python iceberg_simulator.py \
  --bars $OUT/walls_${SYMBOL}.parquet --mbo $MBO \
  --output $OUT/icebergs_${SYMBOL}.parquet --freq $FREQ

echo "⑥ session mapper..."
python session_mapper.py \
  --input $OUT/icebergs_${SYMBOL}.parquet \
  --output $OUT/mapped_${SYMBOL}.parquet

echo "⑦ edge scanner..."
python edge_scanner.py \
  --input $OUT/mapped_${SYMBOL}.parquet \
  --output $OUT/edge_candidates_${SYMBOL}.json

echo "⑧ cluster engine..."
python cluster_engine.py \
  --candidates $OUT/edge_candidates_${SYMBOL}.json \
  --data $OUT/mapped_${SYMBOL}.parquet \
  --output $OUT/alphas_${SYMBOL}.json

echo ""
echo "✅ اكتمل! alphas في: $OUT/alphas_${SYMBOL}.json"
```

التشغيل:
```bash
chmod +x run_all.sh
./run_all.sh
```

---

## التحقق من سلامة الـ MBO

```bash
# هل الـ MBO يحوي كل actions؟
python -c "
import pandas as pd
df = pd.read_csv('data/6BM5_mbo.csv', nrows=100000)
print('Actions:', df['action'].value_counts().to_dict())
print('Sides:  ', df['side'].value_counts().to_dict())
print('Trades:', (df['action']=='T').sum())
print('Adds:  ', (df['action']=='A').sum())
print('Cancels:',(df['action']=='C').sum())
"

# يجب أن ترى:
#   T (Trades): مئات/آلاف
#   A (Adds):   آلاف/عشرات الآلاف
#   C (Cancels): آلاف
#   إن كانت كلها A فقط = الـ feed لا يحوي trades → iceberg لن يعمل
```

---

## استكشاف الأخطاء

### "iceberg_simulator يخرج كل القيم = 0"
```
السبب: لا توجد Trades (action='T') في MBO
الحل: تحقق من الـ feed — يجب أن يحوي T/F
```

### "wall_depth_simulator يخرج قيم flat"
```
السبب: MBP لا يحوي bid_sz_00..09 / ask_sz_00..09
الحل: تأكد من أنه MBP-10 ليس MBP-1
```

### "edge_scanner: 0 candidates"
```
السبب: الداتا قصيرة جداً (شهر فقط) لمعايير 6B الصارمة
الحل: ① اجمع 6 أشهر
      ② مؤقتاً خفّف المعايير في edge_scanner.CRITERIA_6B
```

---

## ملاحظات تشغيلية

- شموع 5min افتراضية. للـ scalping استخدم 1min:
  `--freq 1min --horizon 3`

- horizon = 6 شمعات (30 دقيقة على 5min). للأطول:
  `--horizon 12` (ساعة)

- نافذة normalization 200 شمعة (~16 ساعة). للسوق المتقلب:
  `--window 100`

- معايير 6B في `edge_scanner.py` قابلة للتعديل في `CRITERIA_6B`
