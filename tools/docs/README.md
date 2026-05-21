# tools/docs/ — Documentation Generators

Scripts لتوليد الـ documentation artifacts (PDFs، reports، etc).

## `build_integration_report.py`

يبني تقرير `QuantSystem_Integration_Report_v2_Quantum.pdf`
(31 صفحة، عربي/إنجليزي مختلط، 161 KB).

التقرير = source-of-truth التشخيصي للنظام، يحوي:
- 10 مشاكل مُشخَّصة في الـ main project
- 15 ملف V19.2 + وظائفها
- خطة الدمج بـ 7 مراحل
- تأثير الدمج على DL
- المزايا الكمية + الجدول الشامل
- الطبقة الكمية (Phases A, B, C, D)
- الخلاصة والتوصيات

### المتطلبات

```bash
# فونتات DejaVu (Linux)
sudo apt-get install fonts-dejavu

# فونتات DejaVu (macOS)
brew install --cask font-dejavu

# Python dependencies
pip install -r requirements-docs.txt
```

### التشغيل

```bash
# الافتراضي: يُكتب في docs/QuantSystem_Integration_Report_v2_Quantum.pdf
python tools/docs/build_integration_report.py

# أو مع مسار مخصص
python tools/docs/build_integration_report.py -o /tmp/report.pdf
```

### Output
- ملف PDF بحجم ~160 KB
- 31 صفحة A4
- نص عربي مع reshape + bidi
- code blocks ملوّنة (light + dark themes)
- جداول مقارنة + boxes للمشاكل + section headers

### الـ Layout
- Title page → ToC → 6 أجزاء رئيسية + خلاصة
- Section headers ملوّنة (primary blue #1a365d)
- Problem boxes بـ severity colors (HIGH/MED/LOW)
- Code blocks بـ syntax-like highlighting
- Comparison tables (before/after مع التحسّن)

### تخصيص المسار للـ fonts
```bash
DEJAVU_FONTS_DIR=/custom/path python tools/docs/build_integration_report.py
```

### ملاحظات
- ليس جزء من نظام التداول (QuantSystem core)
- مولّد للتوثيق فقط
- مفيد للـ reproducibility (لو حد يريد تعديل التقرير)
