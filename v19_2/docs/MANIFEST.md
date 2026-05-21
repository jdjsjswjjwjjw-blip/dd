# MANIFEST — QuantSystem V19.2

دليل شامل لكل ملفات المشروع.

## الإصدار: V19.2 (2025-05-20)

```
VERSION:   V19.2
COMMIT:    initial
PYTHON:    ≥ 3.10
DEPENDS:   pandas ≥ 2.0, numpy ≥ 1.24, scipy ≥ 1.10, pyarrow
```

---

## هيكل المشروع

```
QuantSystem_V19_Final/
├── README.md
├── CHANGELOG.md                  ← تتبّع التغييرات بين الإصدارات
├── MANIFEST.md                   ← هذا الملف
├── AUDIT_REPORT.md               ← تقرير الفحص العلمي
├── RUN_GUIDE.md                  ← دليل التشغيل التفصيلي
├── run_all.sh                    ← script تشغيل تلقائي
│
├── market_specs.py               ← 🆕 مواصفات السوق (single source of truth)
├── statistics_module.py          ← 🆕 FDR + permutation + 3-way split
│
├── data_pipeline.py              ← المرحلة ①: تنظيف وتجهيز
├── prepare_day_trading.py        ← المرحلة ②: labels + LOB tensors
├── feature_simulators.py         ← المرحلة ③: 18 محاكي عمق
├── wall_depth_simulator.py       ← المرحلة ④: 8 محاكي جدران
├── iceberg_simulator.py          ← المرحلة ⑤: 5 محاكي iceberg
├── session_mapper.py             ← المرحلة ⑥: zones + levels + events
├── edge_scanner.py               ← المرحلة ⑦: اكتشاف الـ edges
├── cluster_engine.py             ← المرحلة ⑧: تجميع وتمييز
│
└── tests/
    └── test_simulators.py        ← 🆕 8 اختبارات وحدة شاملة
```

---

## دور كل ملف بالتفصيل

### الملفات الإدارية

| الملف | الوصف | متى تقرأه |
|------|-------|-----------|
| `README.md` | نظرة عامة | أول مرة |
| `CHANGELOG.md` | تتبّع التغييرات | عند upgrade |
| `MANIFEST.md` | (هذا) دليل الملفات | للمراجعة |
| `AUDIT_REPORT.md` | الفحص العلمي | قبل الإنتاج |
| `RUN_GUIDE.md` | التشغيل المفصّل | عند التشغيل |
| `run_all.sh` | script تلقائي | للتشغيل السريع |

### الملفات النظرية

| الملف | الدور | الاعتمادية |
|------|------|-----------|
| `market_specs.py` | مواصفات الأسواق + transaction costs | (لا اعتماديات) |
| `statistics_module.py` | FDR + permutation + 3-way split | scipy, numpy |

### Pipeline (8 مراحل)

| # | الملف | المدخل | المخرج | يعتمد على |
|---|------|--------|---------|----------|
| ① | `data_pipeline.py` | MBO + MBP خام | `clean_features.parquet` | (لا) |
| ② | `prepare_day_trading.py` | clean_features | `day_trading_features.parquet` | ① |
| ③ | `feature_simulators.py` | day_trading_features | `sim.parquet` (+18) | ② |
| ④ | `wall_depth_simulator.py` | sim + MBP | `walls.parquet` (+8) | ③ |
| ⑤ | `iceberg_simulator.py` | walls + MBO | `icebergs.parquet` (+5) | ④ |
| ⑥ | `session_mapper.py` | icebergs | `mapped.parquet` (+zones+levels) | ⑤ |
| ⑦ | `edge_scanner.py` | mapped | `edge_candidates.json` | ⑥, market_specs, statistics |
| ⑧ | `cluster_engine.py` | edge_candidates + mapped | `alphas.json` | ⑦ |

### الاختبارات

| الملف | الدور |
|------|------|
| `tests/test_simulators.py` | 8 unit tests على المحاكيات والإحصاء |

---

## معايير الجودة

كل ملف يجب أن يجتاز:

### ① Causality
- لا lookahead (`shift(-N)`, `center=True`, `bfill`)
- تغيير صف i+1 لا يؤثر على صف i
- اختُبر في `tests/test_simulators.py::test_causality_full_pipeline`

### ② Range Validation
- كل مخرج في نطاقه المعلن (مثلاً [-1, 1] أو [0, 1])
- اختُبر في `tests/test_simulators.py::test_output_ranges`

### ③ No Bias
- على عشوائية: المحاكيات الـ direction-like متوسطها ≈ 0
- continuous probability متوسطها ≈ 0.5
- اختُبر في `tests/test_simulators.py::test_no_bias_random_data`

### ④ NaN Handling
- لا NaN في المخرج النهائي (حتى مع NaN في الإدخال)
- اختُبر في `tests/test_simulators.py::test_no_nan_in_output`

### ⑤ Statistical Rigor
- FDR control (Benjamini-Hochberg)
- Permutation test
- 3-way split (no overlap, purge respected)
- اختُبرت في `tests/test_simulators.py::test_*`

---

## كيف تتفرّع وتطوّر

```bash
# إنشاء فرع جديد لميزة
git checkout -b feature/<name>

# قبل أي push:
python tests/test_simulators.py
# يجب 8/8

# سجّل التغييرات في CHANGELOG.md قبل merge
```

---

## كيف تُعدّل سوقاً جديداً (مثلاً ES)

```python
# market_specs.py
SPEC_ES = MarketSpec(
    symbol="ES",
    description="E-mini S&P 500",
    tick_size=0.25,
    tick_value=12.50,
    contract_size=50,
    # ... بقية الحقول من CME spec
)

# في get_market_spec
specs = {"6B": SPEC_6B, "ES": SPEC_ES}
```

ثم:
```bash
python edge_scanner.py --input ... --symbol ES
```

كل شيء بقية في الـ pipeline يبقى كما هو.

---

## الملفات التي تخضع لـ Git

```
# kept under version control
*.py
*.md
*.sh
tests/

# ignored
out/
*.parquet
*.json (نتائج)
__pycache__/
*.pyc
```
