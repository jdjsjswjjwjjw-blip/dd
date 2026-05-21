# QuantSystem V19.2 — Edge Discovery System (GBP/USD Futures)

نظام كامل لاكتشاف edges إحصائياً موثوقة من بيانات MBO/MBP لـ 6B.

```
VERSION:  V19.2
STATUS:   Production-ready (مع 6 أشهر داتا)
TESTS:    8/8 PASS
```

---

## الميزات الرئيسية

### الصرامة الإحصائية
- **FDR (Benjamini-Hochberg)** بدل Bonferroni المتطرف
- **Permutation Test** يكمّل t-test
- **3-Way Time Split**: train + validation + holdout (لمسة واحدة)
- **Transaction Costs** مدمجة في المعايير

### المحاكيات (31 إشارة عمق سوق)
- **5 محاكيات أساسية** (`feature_simulators.py`): 18 مخرج
  - Absorption, Order Flow, Informed, Wall Dynamics, Liquidity
- **محاكي الجدران العميق** (`wall_depth_simulator.py`): 8 مخرجات
  - من MBP-10 الكامل (bid_sz_00..09)
- **محاكي iceberg المؤسسي** (`iceberg_simulator.py`): 5 مخرجات
  - بدون الاعتماد على order_id (يكشف بالنمط)

### البحث المنظّم
- **16 zone زمني**: آسيا/لندن/NY × Q1/Q2/Q3 + 2 تداخل
- **12 مستوى مرجعي**: PDH/PDL/PD_50/PWH/PWL/PW_50/جلسات H&L
- **5 أحداث**: touch / break / reject / failed_break / near
- **33 تركيبة محاكيات** للاختبار

---

## بدء سريع

```bash
# 1. الإعداد
cd QuantSystem_V19_Final
pip install pandas numpy scipy pyarrow

# 2. تشغيل الاختبارات (يجب 8/8)
python tests/test_simulators.py

# 3. التشغيل الكامل
SYMBOL=6B \
MBO=path/to/mbo.parquet \
MBP=path/to/mbp.parquet \
./run_all.sh

# النتيجة: out/alphas_6B.json
```

---

## المعمارية

```
MBO/MBP خام
  ↓
① data_pipeline.py        → تنظيف
② prepare_day_trading.py  → labels + LOB tensors
③ feature_simulators.py   → 18 محاكي عمق
④ wall_depth_simulator.py → 8 محاكي جدران
⑤ iceberg_simulator.py    → 5 محاكي iceberg
⑥ session_mapper.py       → zones + levels + events
⑦ edge_scanner.py         → اكتشاف الـ edges (FDR + 3-way + perm)
⑧ cluster_engine.py       → تجميع وتمييز
  ↓
alphas.json (مُتحقَّق منها على holdout)
```

---

## ملفات المشروع

| فئة | الملفات |
|------|---------|
| **النظري** | `market_specs.py`, `statistics_module.py` |
| **Pipeline** | `data_pipeline.py`, `prepare_day_trading.py`, `feature_simulators.py`, `wall_depth_simulator.py`, `iceberg_simulator.py`, `session_mapper.py`, `edge_scanner.py`, `cluster_engine.py` |
| **الاختبارات** | `tests/test_simulators.py` |
| **الإدارة** | `README.md`, `CHANGELOG.md`, `MANIFEST.md`, `AUDIT_REPORT.md`, `RUN_GUIDE.md`, `run_all.sh` |

التفاصيل في [MANIFEST.md](MANIFEST.md).

---

## الفرق عن V19.1

اقرأ [CHANGELOG.md](CHANGELOG.md) للتفاصيل الكاملة:

- ❌ V19.1: Bonferroni على 89,000 (t≥5.66) → 0 edges حتى الحقيقية
- ✅ V19.2: FDR (BH) + permutation → يلتقط الـ edges، يرفض الضوضاء

- ❌ V19.1: train/test (test يُستخدم في الضبط بدون قصد)
- ✅ V19.2: train/validation/holdout (holdout = لمسة واحدة)

- ❌ V19.1: CRITERIA hardcoded → تناقض بين الوثائق
- ✅ V19.2: market_specs.py + derive_edge_criteria()

---

## المعايير الحالية (مُشتقة من السوق)

لـ 6B بـ confidence="medium":

```
transaction_cost: 1.80 pips/trade  (commission + slippage)

min_t_stat:      2.0   (train IS)
min_t_oos:       1.0   (validation + holdout)
min_wr:          0.53
min_avg_pips:    2.5   (= max(2.5, cost × 1.2))
fdr_alpha:       0.10  (Benjamini-Hochberg)

3-way split:
  train:      60%
  validation: 20%
  holdout:    20%
  purge:      12 bars × 2
```

---

## التوقعات

```
شهر واحد:
  ~ 20 مرشّح في stage 1
  ~ 0 يجتاز FDR (حجم الداتا قليل)
  
3 أشهر:
  ~ 100 مرشّح
  ~ 2-5 يجتازون كل المراحل
  
6 أشهر:
  ~ 300 مرشّح
  ~ 5-15 alpha مُتحقَّق منها
  ← الإعداد الإنتاجي الموصى به
```

---

## الترخيص

داخلي — للاستخدام الشخصي/البحثي.

---

## التواصل

للأسئلة والملاحظات: راجع CHANGELOG.md و AUDIT_REPORT.md أولاً.
