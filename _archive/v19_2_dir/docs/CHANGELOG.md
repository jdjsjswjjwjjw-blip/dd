# CHANGELOG — QuantSystem

## V19.2 (Current) — 2025-05-20

### 🔬 إصلاحات علمية صارمة

#### Multiple Testing — إعادة بناء كاملة
- ❌ **حُذف**: `_bonferroni_threshold` المتطرف (sqrt(2 ln(N/α)))
- ✅ **أُضيف**: `benjamini_hochberg` FDR control
  - أقوى من Bonferroni للتركيبات المترابطة
  - يضبط expected false discovery rate لا FWER
- ✅ **أُضيف**: `permutation_test_edge`
  - يبني توزيع null من الداتا الفعلية
  - لا يفترض توزيعاً طبيعياً
  - أصرم في وجود heavy tails

#### Train/Test Split — 3-way بدل 2-way
- ❌ **القديم**: train (70%) + test (30%) — test يُستخدم في الضبط
- ✅ **الجديد**: train (60%) + validation (20%) + holdout (20%)
  - `train`: اكتشاف
  - `validation`: ضبط ومعايرة (يمكن النظر فيه)
  - `holdout`: لمسة واحدة فقط للحكم النهائي

#### Single Source of Truth
- ❌ **القديم**: `CRITERIA_6B` hardcoded في edge_scanner
- ✅ **الجديد**: `market_specs.py` يحوي كل أرقام السوق
  - `MarketSpec` dataclass لكل سوق
  - `derive_edge_criteria()` يشتق المعايير من السوق + حجم الداتا
  - transaction costs مدمجة (`round_trip_cost_pips`)

#### Unit Tests الصارمة
- ✅ **جديد**: `tests/test_simulators.py`
  - `test_causality_full_pipeline` — تغيير المستقبل ≠ تأثير الماضي
  - `test_no_bias_random_data` — على عشوائية، الإشارات بلا انحياز
  - `test_output_ranges` — كل مخرج في نطاقه المعلن
  - `test_no_nan_in_output` — لا NaN في النتائج
  - `test_benjamini_hochberg` — FDR على p-values معلومة
  - `test_permutation_with_real_edge` — يميّز edge من ضوضاء
  - `test_3way_split_no_overlap` — التقسيم لا يتداخل
  - **8/8 اختبارات تنجح**

### 🧱 ملفات جديدة
- `market_specs.py` (139 سطر) — مواصفات الأسواق
- `statistics_module.py` (251 سطر) — FDR + permutation + 3-way split
- `tests/test_simulators.py` (294 سطر) — وحدة اختبارات شاملة
- `CHANGELOG.md` (هذا الملف) — تتبّع التغييرات

### 🔄 ملفات مُعدَّلة
- `edge_scanner.py` — إعادة بناء كاملة (FDR + 3-way + permutation)
- `feature_simulators.py` — bug fix في `_rank` (min_periods)
- `wall_depth_simulator.py` — نفس bug fix
- `iceberg_simulator.py` — نفس bug fix + توضيح "لا order_id"
- `data_pipeline.py` — نفس bug fix

### 📋 ملفات الإدارة
- `MANIFEST.md` — قائمة شاملة بالملفات وغرض كل منها
- `RUN_GUIDE.md` — محدّث للأوامر الجديدة
- `AUDIT_REPORT.md` — محدّث ليعكس V19.2

---

## V19.1 — 2025-05-19 (Deprecated)

### مشاكل اكتُشِفت لاحقاً
- ❌ Bonferroni على 89,000 اختبار = صرامة مبالغ فيها (t≥5.66)
- ❌ Train/Test ثنائي = test يُستخدم في الضبط (data leakage subtle)
- ❌ Bug في `_rank`: `min_periods > window` على نوافذ صغيرة
- ❌ Hardcoded CRITERIA_6B في edge_scanner
- ❌ لا unit tests
- ❌ تناقض بين الـ docstrings والـ CRITERIA الفعلية

### إصلاحات V19.1 (احتُفظ بها)
- ✅ FDR/Bonferroni concept (تطوّر إلى FDR كامل في V19.2)
- ✅ سببية المحاكيات (اختُبرت رسمياً في V19.2)
- ✅ Wall Depth Simulator (8 outputs)
- ✅ Iceberg Simulator (5 outputs, no order_id dependency)
- ✅ Session Mapper (16 zones, 12 levels, 5 events)

---

## V19.0 — 2025-05-18

### المعمارية الأولى
- 5 محاكيات عمق (`feature_simulators.py`)
- `session_mapper.py` (zones × levels × events)
- `edge_scanner.py` v1 (بدون OOS)
- `cluster_engine.py`

---

## الخارطة المستقبلية (V20+)

### الأولوية ①
- [ ] alpha_validation.py — اختبار walk-forward كامل
- [ ] backtest_engine.py — محاكاة كاملة مع slippage/commission
- [ ] live_signal_monitor.py — تنفيذ الـ alphas مباشرةً

### الأولوية ②
- [ ] دمج موديل ML (gradient boosting) لكل alpha مُتحقَّق منها
- [ ] Risk management module (position sizing, drawdown limits)
- [ ] Multi-symbol support (ES, NQ, 6E)

### الأولوية ③
- [ ] Documentation موسّع (Jupyter notebooks)
- [ ] CI/CD مع pytest على كل commit
- [ ] Performance optimization (Numba/Cython للـ tight loops)
