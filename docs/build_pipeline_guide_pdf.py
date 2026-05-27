"""
docs/build_pipeline_guide_pdf.py
═══════════════════════════════════════════════════════════════════════════════
Build PIPELINE_GUIDE.pdf — bilingual (English + Arabic) operational
walkthrough covering extraction → merge → quality checks → SSL → hybrid.

Uses:
  • reportlab for PDF layout
  • arabic_reshaper + python-bidi for proper RTL Arabic rendering
  • DejaVuSans.ttf for Unicode coverage (Latin + Arabic)

Run:
  python docs/build_pipeline_guide_pdf.py
  → produces docs/PIPELINE_GUIDE.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Preformatted,
    Spacer, Table, TableStyle, PageBreak,
)


# ── Font registration (DejaVu Sans for Unicode + Arabic glyph coverage) ──
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONT_REGULAR))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONT_BOLD))
pdfmetrics.registerFont(TTFont("DejaVu-Mono", FONT_MONO))


def ar(text: str) -> str:
    """Reshape + bidi an Arabic string for correct PDF rendering."""
    return get_display(arabic_reshaper.reshape(text))


# ── Styles ──
STYLES = getSampleStyleSheet()

H1_EN = ParagraphStyle(
    "H1_EN", parent=STYLES["Heading1"], fontName="DejaVu-Bold", fontSize=18,
    textColor=HexColor("#0E2C5C"), spaceAfter=8, alignment=TA_LEFT,
)
H1_AR = ParagraphStyle(
    "H1_AR", parent=H1_EN, alignment=TA_RIGHT,
)
H2_EN = ParagraphStyle(
    "H2_EN", parent=STYLES["Heading2"], fontName="DejaVu-Bold", fontSize=14,
    textColor=HexColor("#1A4080"), spaceAfter=6, alignment=TA_LEFT,
)
H2_AR = ParagraphStyle(
    "H2_AR", parent=H2_EN, alignment=TA_RIGHT,
)
BODY_EN = ParagraphStyle(
    "BODY_EN", parent=STYLES["BodyText"], fontName="DejaVu", fontSize=10,
    leading=14, alignment=TA_LEFT, spaceAfter=4,
)
BODY_AR = ParagraphStyle(
    "BODY_AR", parent=BODY_EN, alignment=TA_RIGHT, spaceAfter=4,
)
BULLET_EN = ParagraphStyle(
    "BULLET_EN", parent=BODY_EN, leftIndent=20, bulletIndent=10, spaceAfter=2,
)
BULLET_AR = ParagraphStyle(
    "BULLET_AR", parent=BODY_AR, rightIndent=20, spaceAfter=2,
)
CODE = ParagraphStyle(
    "CODE", parent=STYLES["Code"], fontName="DejaVu-Mono", fontSize=8,
    leading=11, leftIndent=10, rightIndent=10,
    backColor=HexColor("#F4F4F4"), borderColor=HexColor("#CCCCCC"),
    borderWidth=0.5, borderPadding=6, spaceAfter=8, spaceBefore=4,
)
CAPTION = ParagraphStyle(
    "CAPTION", parent=BODY_EN, fontSize=9, textColor=HexColor("#555555"),
    spaceAfter=8,
)
COVER_TITLE = ParagraphStyle(
    "COVER_TITLE", fontName="DejaVu-Bold", fontSize=28,
    textColor=HexColor("#0E2C5C"), alignment=TA_CENTER, spaceAfter=20,
)
COVER_SUBTITLE = ParagraphStyle(
    "COVER_SUBTITLE", fontName="DejaVu", fontSize=14,
    textColor=HexColor("#444444"), alignment=TA_CENTER, spaceAfter=10,
)
COVER_AR = ParagraphStyle(
    "COVER_AR", fontName="DejaVu-Bold", fontSize=22,
    textColor=HexColor("#0E2C5C"), alignment=TA_CENTER, spaceAfter=30,
)


def code_block(text: str):
    """Code blocks are English-only (commands are English), monospace."""
    # Strip leading/trailing newlines and use Preformatted to preserve whitespace
    return Preformatted(text.strip("\n"), CODE)


def bullet_en(items: list[str]):
    return [Paragraph("• " + t, BULLET_EN) for t in items]


def bullet_ar(items: list[str]):
    return [Paragraph(ar("• " + t), BULLET_AR) for t in items]


def divider():
    """A thin horizontal rule."""
    from reportlab.platypus import HRFlowable
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC"),
                       spaceBefore=8, spaceAfter=12)


def en_ar_pair(en_h: str, ar_h: str, en_body: str, ar_body: str):
    """Bilingual paragraph pair: English on left, Arabic on right (stacked)."""
    flow = []
    flow.append(Paragraph(en_h, H2_EN))
    flow.append(Paragraph(ar(ar_h), H2_AR))
    flow.append(Paragraph(en_body, BODY_EN))
    flow.append(Paragraph(ar(ar_body), BODY_AR))
    return flow


# ────────────────────────────────────────────────────────────────────
# DOCUMENT CONTENT
# ────────────────────────────────────────────────────────────────────

def build_story() -> list:
    s = []

    # ── COVER PAGE ──
    s.append(Spacer(1, 6 * cm))
    s.append(Paragraph("Pipeline Guide", COVER_TITLE))
    s.append(Paragraph(ar("دليل خط الأنابيب الكامل"), COVER_AR))
    s.append(Paragraph("From Raw Monthly Contracts to Hybrid Trading Model",
                       COVER_SUBTITLE))
    s.append(Paragraph(ar("من ملفات العقود الشهرية الخام إلى النموذج الهجين"),
                       ParagraphStyle("cs", parent=COVER_SUBTITLE, fontSize=12)))
    s.append(Spacer(1, 4 * cm))
    s.append(Paragraph("6B (GBP/USD Futures) — End-to-End Operational Guide",
                       CAPTION))
    s.append(Paragraph(ar("عقود الجنيه الإسترليني — دليل تشغيل من البداية للنهاية"),
                       ParagraphStyle("cs2", parent=CAPTION, alignment=TA_CENTER)))
    s.append(PageBreak())

    # ── TABLE OF CONTENTS ──
    s.append(Paragraph("Table of Contents", H1_EN))
    s.append(Paragraph(ar("فهرس المحتويات"), H1_AR))
    s.append(divider())
    toc_items = [
        ("1.  Overview & Prerequisites", "نظرة عامة + المتطلبات"),
        ("2.  Step 1 — Smart Contract Extraction (Rollover)",
         "الخطوة 1 — استخراج العقد الذكي (Rollover)"),
        ("3.  Step 2 — Per-Month prepare_day_trading",
         "الخطوة 2 — تشغيل prepare_day_trading لكل شهر"),
        ("4.  Step 3 — Build Order Batches",
         "الخطوة 3 — بناء Order Batches"),
        ("5.  Step 4 — Combine Quarterly / Yearly Artifacts",
         "الخطوة 4 — دمج الأرباع / السنوات"),
        ("6.  Step 5 — Quality Checks after Merge (CRITICAL)",
         "الخطوة 5 — فحوصات الجودة بعد الدمج (حرجة)"),
        ("7.  Step 6 — SSL Pretraining",
         "الخطوة 6 — التدريب السببي SSL"),
        ("8.  Step 7 — Train Hybrid Model",
         "الخطوة 7 — تدريب النموذج الهجين"),
        ("9.  Step 8 — Strict Backtest & Decision Policy",
         "الخطوة 8 — Backtest صارم + سياسة القرار"),
        ("10. Appendix — Troubleshooting & Common Issues",
         "ملحق — استكشاف الأخطاء"),
    ]
    for en, ar_t in toc_items:
        s.append(Paragraph(en, BODY_EN))
        s.append(Paragraph(ar(ar_t), BODY_AR))
        s.append(Spacer(1, 4))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 1 — Overview
    # ────────────────────────────────────────────────
    s.append(Paragraph("1. Overview & Prerequisites", H1_EN))
    s.append(Paragraph(ar("١. نظرة عامة + المتطلبات"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "What this pipeline does",
        "ما يفعله هذا الـ pipeline",
        "Takes 6 years of raw monthly Databento files (each containing both "
        "near-month and far-month contracts) and produces: (a) a continuous "
        "back-adjusted dataset, (b) 161-dim SSL embeddings per bar, and (c) "
        "a trained HybridModel that fuses rule labels + SSL into per-bar "
        "trading decisions with adaptive TP and regime-aware sizing.",
        "يأخد بيانات ٦ سنوات من ملفات Databento الشهرية (كل ملف فيه العقد "
        "القريب + العقد البعيد) ويُنتج: (أ) داتا continuous مُعدّلة بـ Panama، "
        "(ب) embeddings 161-dim لكل bar من SSL، (ج) نموذج هجين مدرَّب يدمج "
        "labels القواعد + SSL إلى قرارات تداول لكل bar مع TP تكيُّفي وتحديد "
        "حجم مبني على الـ regime.",
    ))

    s.extend(en_ar_pair(
        "Prerequisites",
        "المتطلبات قبل البدء",
        "(1) Repository on branch claude/task-d-RcDhu, latest commit. "
        "(2) Python 3.11 with the requirements installed. "
        "(3) Disk space: ~200GB for raw 6-year data + ~50GB for derived "
        "artifacts. (4) GPU recommended for SSL pretraining (CPU works "
        "but 5-10x slower).",
        "(١) المستودع على فرع claude/task-d-RcDhu، آخر commit. "
        "(٢) Python 3.11 مع المتطلبات مثبتة. "
        "(٣) مساحة قرص: ~٢٠٠ جيجا للداتا الخام لـ ٦ سنوات + ~٥٠ جيجا "
        "للمخرجات. (٤) GPU مفضّل لـ SSL pretraining (CPU شغّال لكن أبطأ ٥-١٠ مرات).",
    ))

    s.append(Paragraph("Sanity check before starting:", BODY_EN))
    s.append(Paragraph(ar("فحص أولي قبل البدء:"), BODY_AR))
    s.append(code_block(
"""# 1. Verify branch + tag
git status              # → claude/task-d-RcDhu, clean
git tag -l "v*"         # → v1-baseline-pre-cleanup must exist

# 2. Verify subsystem boundaries
python tools/check_subsystem_boundaries.py
# Expected: ✅ Subsystem boundaries clean

# 3. Verify test suite passes
python -m pytest tests/ -q
# Expected: 337 passed, 2 skipped, 0 failed
"""))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 2 — Step 1: Extraction
    # ────────────────────────────────────────────────
    s.append(Paragraph("2. Step 1 — Smart Contract Extraction (Rollover)", H1_EN))
    s.append(Paragraph(ar("٢. الخطوة ١ — استخراج العقد الذكي"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Each monthly Databento file contains BOTH the near-month and the "
        "far-month contract. The extractor produces ONE continuous file per "
        "month: it uses the near contract while it's the most liquid, then "
        "switches to the far contract at the volume-crossover day, and "
        "back-adjusts the older contract's prices so there's no jump.",
        "كل ملف شهري من Databento فيه العقد القريب + العقد البعيد. الـ "
        "extractor بيُنتج ملف continuous واحد للشهر: يستخدم العقد القريب طول "
        "ما هو الأنشط، ثم يحوّل للعقد البعيد عند يوم تقاطع الفوليوم، ويُعدّل "
        "أسعار العقد القديم عشان مفيش jump في السعر.",
    ))

    s.append(Paragraph("Command:", BODY_EN))
    s.append(Paragraph(ar("الأمر:"), BODY_AR))
    s.append(code_block(
"""# Run per monthly file (loop for 6 years × 12 months = 72 files per data type)
for year in 2019 2020 2021 2022 2023 2024; do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    # MBO file (order-level)
    python tools/extract_continuous_contract.py \\
        raw_data/${year}-${month}.mbo.parquet \\
        --root 6B \\
        --output continuous/${year}-${month}.mbo.parquet

    # MBP-10 file (depth)
    python tools/extract_continuous_contract.py \\
        raw_data/${year}-${month}.mbp10.parquet \\
        --root 6B \\
        --output continuous/${year}-${month}.mbp10.parquet
  done
done
"""))

    s.append(Paragraph("Expected output per file:", BODY_EN))
    s.append(Paragraph(ar("المخرج المتوقع لكل ملف:"), BODY_AR))
    s.append(code_block(
"""Reading: raw_data/2025-03.mbo.parquet
  rows: 2,001  cols: 4

Result:
  mode:            continuous_with_rollover
  near:            6BH5
  far:             6BM5
  rollover_day:    2025-03-11 00:00:00+00:00
  raw_offset:      0.020005   ← carry detected automatically
  applied_offset:  0.020005
  output_rows:     1,500

💾 continuous/2025-03.mbo.parquet (...)
💾 continuous/2025-03.mbo.parquet.log.json
"""))

    s.append(Paragraph("Quality checks after Step 1:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة بعد الخطوة ١:"), H2_AR))
    s.extend(bullet_en([
        "Every monthly file produced a `*.log.json` with mode != 'no_rollover_picked_dominant' (rare in normal data)",
        "Output row count > 70% of input row count (otherwise the rollover detection is too aggressive)",
        "applied_offset is finite and within plausible range (for 6B: ±0.005 typical)",
        "No file failed with an error — check the loop exit code",
    ]))
    s.extend(bullet_ar([
        "كل ملف شهري أنتج log.json بقيمة mode مختلفة عن no_rollover_picked_dominant",
        "عدد rows الخارج > ٧٠٪ من المدخل (لو أقل، الـ rollover صارم زيادة)",
        "applied_offset قيمته منطقية (لـ 6B: حول ±٠.٠٠٥)",
        "مفيش ملف فشل — تأكد من exit code للـ loop",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 3 — Step 2: prepare_day_trading
    # ────────────────────────────────────────────────
    s.append(Paragraph("3. Step 2 — Per-Month prepare_day_trading", H1_EN))
    s.append(Paragraph(ar("٣. الخطوة ٢ — prepare_day_trading لكل شهر"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Convert continuous MBO + MBP-10 monthly files into 15-min bars + "
        "135 hand-crafted features + rule labels (event_flag, "
        "event_direction, signal_quality, path_outcome) + 9-channel LOB tensor.",
        "تحويل ملفات MBO + MBP-10 الشهرية إلى bars 15 دقيقة + ١٣٥ "
        "feature يدوي + labels قواعد (event_flag, event_direction, "
        "signal_quality, path_outcome) + LOB tensor 9 قنوات.",
    ))

    s.append(code_block(
"""# Run per monthly continuous file
for year in 2019 2020 2021 2022 2023 2024; do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    out_dir="features/${year}-${month}"
    mkdir -p "$out_dir"

    python prepare_day_trading.py \\
        --mbo continuous/${year}-${month}.mbo.parquet \\
        --mbp continuous/${year}-${month}.mbp10.parquet \\
        --output "$out_dir" \\
        --freq 15min \\
        --session_profile daytrade_default \\
        2>&1 | tee "$out_dir/prepare.log"
  done
done
"""))

    s.append(Paragraph("Expected outputs in each features/YYYY-MM/ dir:", BODY_EN))
    s.append(Paragraph(ar("المخرجات المتوقعة في كل مجلد:"), BODY_AR))
    s.append(code_block(
"""features/2019-01/
├── day_trading_features.parquet     # ~2000 bars × 135 cols + labels
├── lob_tensors.npy                   # (N_bars, T=50, P=20, C=9)
├── lob_tensor_timestamps.npy         # (N_bars,) datetime64
├── day_trading_manifest.json         # metadata
└── artifact_manifest.json
"""))

    s.append(Paragraph("Quality checks after Step 2:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة بعد الخطوة ٢:"), H2_AR))
    s.extend(bullet_en([
        "Every month produced day_trading_features.parquet with N rows ≈ 2000-2200 (15-min bars per month)",
        "lob_tensors.npy first dim equals features parquet row count (alignment must be exact)",
        "Event rate (event_flag.sum() / len(df)) is 3-8% — too low → labeler broken, too high → false events",
        "regime_label distribution: trending + ranging dominate (~95%), volatile rare",
        "atr_14 has no NaN after warmup (first 14 bars may be NaN — acceptable)",
    ]))
    s.extend(bullet_ar([
        "كل شهر أنتج day_trading_features.parquet عدد صفوف ≈ ٢٠٠٠-٢٢٠٠ (شموع ١٥ دقيقة شهرياً)",
        "lob_tensors.npy أول بُعد = عدد صفوف الـ parquet (المحاذاة يجب أن تكون دقيقة)",
        "نسبة الحدث (event_flag) ٣-٨٪ — الأقل = labeler مكسور، الأعلى = events زائفة",
        "توزيع regime_label: trending + ranging هما الأغلب (~٩٥٪)، volatile نادر",
        "atr_14 خالي من NaN بعد warmup (أول ١٤ شمعة ممكن NaN — مقبول)",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 4 — Step 3: Order Batches
    # ────────────────────────────────────────────────
    s.append(Paragraph("4. Step 3 — Build Order Batches", H1_EN))
    s.append(Paragraph(ar("٤. الخطوة ٣ — بناء Order Batches"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Build per-bar order tensors from raw MBO ticks: (N_bars, T=50, "
        "N_orders=200, F=7). These feed the SSL transformer alongside the "
        "LOB tensor.",
        "بناء order tensors لكل bar من ticks الـ MBO الخام: (N_bars, T=50, "
        "N_orders=200, F=7). هتدخل على SSL transformer مع LOB tensor.",
    ))

    s.append(code_block(
"""for year in 2019 2020 2021 2022 2023 2024; do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    out_dir="features/${year}-${month}"

    python self_supervised/build_order_batches.py \\
        --mbo continuous/${year}-${month}.mbo.parquet \\
        --features "$out_dir/day_trading_features.parquet" \\
        --output "$out_dir" \\
        --freq 15min \\
        --lookback-bars 50 \\
        --n-orders 200 \\
        --tick-size 0.0001
  done
done
"""))

    s.append(Paragraph("Quality checks:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة:"), H2_AR))
    s.extend(bullet_en([
        "order_features.npy shape: (N_bars, 50, 200, 7) — first dim matches features parquet",
        "order_masks.npy shape: (N_bars, 50, 200) — bool",
        "Sparsity (order_masks.mean()) should be 80-95% — too low means too many empty bars",
        "order_batches_meta.json present and parseable",
    ]))
    s.extend(bullet_ar([
        "order_features.npy شكله: (N_bars, 50, 200, 7) — أول بُعد = عدد bars",
        "order_masks.npy شكله: (N_bars, 50, 200) — bool",
        "الكثافة (mask.mean()) يجب أن تكون ٨٠-٩٥٪ — الأقل = bars فاضية كتيرة",
        "order_batches_meta.json موجود وقابل للقراءة",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 5 — Step 4: Combine
    # ────────────────────────────────────────────────
    s.append(Paragraph("5. Step 4 — Combine Quarterly / Yearly Artifacts", H1_EN))
    s.append(Paragraph(ar("٥. الخطوة ٤ — دمج الأرباع / السنوات"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Stitch all monthly features + LOB tensors + order tensors into ONE "
        "combined dataset (parquet + npy). Handles contract rollovers via "
        "the overlap-mode=auto detector. Marks inter-month boundaries with "
        "is_session_break=1 so the SSL filter respects them.",
        "تجميع كل الـ features الشهرية + LOB tensors + order tensors في "
        "dataset واحد (parquet + npy). يتعامل مع rollovers العقود عبر "
        "overlap-mode=auto. يُعلِّم الحدود بين الشهور بـ is_session_break=1 "
        "عشان SSL filter يحترمها.",
    ))

    s.append(code_block(
"""# Combine all 72 monthly outputs into one 6-year dataset
python self_supervised/combine_quarter_artifacts.py \\
    --quarter-dirs features/2019-01 features/2019-02 features/2019-03 \\
                   features/2019-04 features/2019-05 features/2019-06 \\
                   features/2019-07 features/2019-08 features/2019-09 \\
                   features/2019-10 features/2019-11 features/2019-12 \\
                   features/2020-01 ... features/2024-12 \\
    --output-dir combined_6y \\
    --overlap-mode auto \\
    --carry-pips-per-quarter 40
"""))

    s.append(Paragraph("Expected outputs:", BODY_EN))
    s.append(Paragraph(ar("المخرجات المتوقعة:"), BODY_AR))
    s.append(code_block(
"""combined_6y/
├── day_trading_features.parquet     # ~145,000 bars × 137 cols
├── lob_tensors.npy                   # (145000, 50, 20, 9) ≈ 5 GB
├── order_features.npy                # (145000, 50, 200, 7) ≈ 40 GB
├── order_masks.npy                   # (145000, 50, 200) ≈ 1.4 GB
└── combine_meta.json                 # rollover diagnostics
"""))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 6 — Step 5: Quality Checks (CRITICAL)
    # ────────────────────────────────────────────────
    s.append(Paragraph("6. Step 5 — Quality Checks after Merge (CRITICAL)", H1_EN))
    s.append(Paragraph(ar("٦. الخطوة ٥ — فحوصات الجودة بعد الدمج (حرجة)"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Why this step exists",
        "ليه الخطوة دي مهمة",
        "Skipping these checks is the single biggest cause of model failures. "
        "Run ALL of them after every combine_quarter_artifacts call. Failing "
        "any one means a data bug — fix it before SSL pretraining.",
        "تجاهل الفحوصات دي هو السبب الأول لفشل النموذج. شغّل كلها بعد كل "
        "combine_quarter_artifacts. أي فحص يفشل = bug في الداتا، صلِّحه قبل "
        "ما تشغّل SSL pretraining.",
    ))

    s.append(Paragraph("(A) Alignment check — bars vs npy first-dim", H2_EN))
    s.append(Paragraph(ar("(أ) فحص المحاذاة — bars مع أول بُعد npy"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
lob = np.load('combined_6y/lob_tensors.npy')
of  = np.load('combined_6y/order_features.npy')
om  = np.load('combined_6y/order_masks.npy')
n = len(df)
assert lob.shape[0] == n, f'lob first dim {lob.shape[0]} != bars {n}'
assert of.shape[0]  == n, f'of  first dim {of.shape[0]}  != bars {n}'
assert om.shape[0]  == n, f'om  first dim {om.shape[0]}  != bars {n}'
print(f'✅ Alignment OK ({n:,} bars on all artifacts)')
"
"""))

    s.append(Paragraph("(B) Chronological order — ts must be monotonic", H2_EN))
    s.append(Paragraph(ar("(ب) ترتيب زمني — ts يجب أن يكون متصاعد"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
ts = pd.to_datetime(df['ts_event'])
gaps = ts.diff().dropna()
print(f'monotonic: {(gaps >= pd.Timedelta(0)).all()}')
print(f'min gap:   {gaps.min()}')
print(f'max gap:   {gaps.max()}  (weekend/roll gaps OK)')
print(f'median gap: {gaps.median()}  (should be 15min)')
"
"""))

    s.append(Paragraph("(C) Session-break markers at boundaries", H2_EN))
    s.append(Paragraph(ar("(ج) علامات session_break عند الحدود"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
n_breaks = int(df['is_session_break'].sum())
n_rolls  = int(df['is_roll'].sum())
print(f'is_session_break: {n_breaks}  (weekends + rolls; expect ~250-400 for 6yr)')
print(f'is_roll: {n_rolls}  (contract rolls; expect ~24-30 for 6yr)')
# Show a few break ts to verify they look like Sunday opens
print('First 5 breaks:')
print(df.loc[df['is_session_break']==1, 'ts_event'].head().tolist())
"
"""))

    s.append(Paragraph("(D) Price continuity at rollover boundaries", H2_EN))
    s.append(Paragraph(ar("(د) استمرارية السعر عند حدود الـ rollover"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('combined_6y/day_trading_features.parquet').reset_index(drop=True)
rolls = df.index[df['is_roll']==1].tolist()
print(f'Rolls at indices: {rolls[:10]}')
for ri in rolls[:5]:
    if ri == 0: continue
    c_before = float(df['close'].iloc[ri-1])
    c_after  = float(df['close'].iloc[ri])
    pips = (c_after - c_before) / 0.0001
    mark = '✅' if abs(pips) < 200 else '⚠️'
    print(f'  {mark} roll @ idx={ri}: {c_before:.5f} -> {c_after:.5f} ({pips:+.1f} pips)')
"
"""))

    s.append(Paragraph("(E) Label sanity — events have non-zero direction", H2_EN))
    s.append(Paragraph(ar("(هـ) فحص الـ labels — events لها direction غير صفر"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
n = len(df)
events = (df['event_flag'] == 1).sum()
event_dir_dist = df.loc[df['event_flag']==1, 'event_direction'].value_counts().to_dict()
print(f'Total bars: {n:,}')
print(f'Events: {events:,} ({100*events/n:.2f}%)')
print(f'Event direction distribution: {event_dir_dist}')
# Sanity: most events should have non-zero direction
zero_dir = event_dir_dist.get(0, 0)
assert zero_dir / max(events, 1) < 0.05, f'{zero_dir} events have direction=0 (>5%)'
print('✅ Most events have non-zero direction')
"
"""))

    s.append(Paragraph("(F) Train / holdout slice has both classes", H2_EN))
    s.append(Paragraph(ar("(و) train + holdout كلاهما يحوي events"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
if 'dataset_slice' in df.columns:
    for sl in ('train', 'holdout'):
        sub = df[df['dataset_slice'] == sl]
        ev = (sub['event_flag'] == 1).sum()
        print(f'{sl}: {len(sub):,} bars | {ev:,} events')
else:
    print('dataset_slice column missing — pipeline will use default 75/25 split')
"
"""))

    s.append(Paragraph("(G) NO duplicate timestamps", H2_EN))
    s.append(Paragraph(ar("(ز) لا توجد ts مكررة"), H2_AR))
    s.append(code_block(
"""python -c "
import pandas as pd
df = pd.read_parquet('combined_6y/day_trading_features.parquet')
dup = df['ts_event'].duplicated().sum()
print(f'Duplicate timestamps: {dup}')
assert dup == 0, f'{dup} duplicate timestamps — combine_quarter_artifacts has a bug'
print('✅ No duplicates')
"
"""))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 7 — Step 6: SSL Pretraining
    # ────────────────────────────────────────────────
    s.append(Paragraph("7. Step 6 — SSL Pretraining", H1_EN))
    s.append(Paragraph(ar("٧. الخطوة ٦ — التدريب السببي SSL"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Pretrain HierarchicalLOBTransformer + PriceCycleModel on the "
        "combined 6-year dataset, then extract 161-dim embeddings per bar.",
        "تدريب HierarchicalLOBTransformer + PriceCycleModel على dataset "
        "الـ ٦ سنوات، ثم استخراج embeddings 161-dim لكل bar.",
    ))

    s.append(code_block(
"""./self_supervised/run_ssl_only.sh combined_6y checkpoints/ssl_6y 0.75

# Outputs (~5-8 hours on GPU):
#   checkpoints/ssl_6y/lob/best_ssl_lob.pt        ← LOB transformer
#   checkpoints/ssl_6y/cycle/best_ssl_cycle.pt    ← Price cycle
#   checkpoints/ssl_6y/embeddings/embeddings.npy  ← (145000, 161)
#   checkpoints/ssl_6y/embeddings/embedding_valid.npy  ← bool mask
"""))

    s.append(Paragraph("Quality checks:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة:"), H2_AR))
    s.extend(bullet_en([
        "Phase A (LOB) val loss decreased monotonically and reached ~0.5-0.8 final",
        "Phase B (Cycle) val loss reached ~2.0-2.5 final",
        "embedding_valid.npy.sum() / len(df) > 70% (with 6yr data, expect 80-90%)",
        "embeddings.npy shape is (N_bars, 161) and dtype float32",
        "No NaN in embeddings (np.isnan(emb).any() == False for valid rows)",
    ]))
    s.extend(bullet_ar([
        "Phase A (LOB) val loss نزل بانتظام ووصل ~٠.٥-٠.٨",
        "Phase B (Cycle) val loss وصل ~٢.٠-٢.٥",
        "embedding_valid > ٧٠٪ (مع ٦ سنوات بيانات: ٨٠-٩٠٪)",
        "embeddings.npy شكله (N_bars, 161) و dtype float32",
        "مفيش NaN في embeddings للصفوف الصحيحة",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 8 — Step 7: Hybrid
    # ────────────────────────────────────────────────
    s.append(Paragraph("8. Step 7 — Train Hybrid Model", H1_EN))
    s.append(Paragraph(ar("٨. الخطوة ٧ — تدريب النموذج الهجين"), H1_AR))
    s.append(divider())

    s.extend(en_ar_pair(
        "Goal",
        "الهدف",
        "Train the HybridModel that fuses day_trade features + SSL "
        "embeddings into event/direction/confidence predictions.",
        "تدريب HybridModel الذي يدمج features الـ day_trade + embeddings "
        "الـ SSL إلى تنبؤات event/direction/confidence.",
    ))

    s.append(code_block(
"""python -m modules.trading_intel.training.train_hybrid \\
    --features        combined_6y/day_trading_features.parquet \\
    --ssl-embeddings  checkpoints/ssl_6y/embeddings/embeddings.npy \\
    --embedding-valid checkpoints/ssl_6y/embeddings/embedding_valid.npy \\
    --output          checkpoints/hybrid_6y \\
    --use-dataset-slice \\
    --epochs 80 \\
    --hidden-dim 256 \\
    --n-layers 3 \\
    --dropout 0.3 \\
    --lr 3e-4
"""))

    s.append(Paragraph("Quality checks:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة:"), H2_AR))
    s.extend(bullet_en([
        "Train loss decreases monotonically; val loss plateaus then early-stops",
        "Val event_acc > 0.65 (random ≈ 0.05)",
        "Val dir_acc > 0.60 (random ≈ 0.33)",
        "predictions.parquet contains all expected columns (event_prob, p_long, p_short, p_neutral, confidence)",
        "best_hybrid.pt contains norm_daytrade, norm_ssl, norm_cnn (if CNN used)",
    ]))
    s.extend(bullet_ar([
        "Train loss ينزل بانتظام؛ val loss يثبت ثم early stop",
        "Val event_acc > ٠.٦٥ (عشوائي ≈ ٠.٠٥)",
        "Val dir_acc > ٠.٦٠ (عشوائي ≈ ٠.٣٣)",
        "predictions.parquet يحوي كل الأعمدة المتوقعة",
        "best_hybrid.pt يحوي norm_daytrade, norm_ssl, norm_cnn",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 9 — Step 8: Backtest
    # ────────────────────────────────────────────────
    s.append(Paragraph("9. Step 8 — Strict Backtest & Decision Policy", H1_EN))
    s.append(Paragraph(ar("٩. الخطوة ٨ — Backtest صارم + سياسة القرار"), H1_AR))
    s.append(divider())

    s.append(Paragraph("Baseline strict backtest (rule pipeline only):", H2_EN))
    s.append(Paragraph(ar("الـ Baseline الصارم (قواعد فقط):"), H2_AR))
    s.append(code_block(
"""python self_supervised/backtest_day_trade.py \\
    --features combined_6y/day_trading_features.parquet \\
    --output backtests/baseline_6y_strict \\
    --strict \\
    --starting-equity 100000 \\
    --contracts-per-trade 1

# Expected: ~78% hit, ~3-6% annualized over the 6-year holdout.
"""))

    s.append(Paragraph("Hybrid-enhanced backtest (rule + SSL + hybrid):", H2_EN))
    s.append(Paragraph(ar("Backtest الهجين (قواعد + SSL + hybrid):"), H2_AR))
    s.append(Paragraph(
        "(After hybrid predictions are integrated as a filter / re-rank — "
        "see modules.trading_intel.hybrid.policy.decide() for the policy logic.)",
        CAPTION,
    ))

    s.append(Paragraph("Quality checks:", H2_EN))
    s.append(Paragraph(ar("فحوصات الجودة:"), H2_AR))
    s.extend(bullet_en([
        "Baseline holdout hit rate within ±3% of expected (~75-81%)",
        "Hybrid hit rate >= baseline (otherwise the hybrid is hurting, not helping)",
        "Max drawdown < 1.5% (with 1 contract on $100k)",
        "Sharpe (annualized, non-overlapping trades) is statistically significant given the trade count",
        "Trade count is plausible (~150-300 trades / year for 6B)",
    ]))
    s.extend(bullet_ar([
        "Baseline hit rate في الـ holdout ضمن ±٣٪ من المتوقع (~٧٥-٨١٪)",
        "Hybrid hit rate >= baseline (وإلا الـ hybrid بيضر)",
        "Max drawdown < ١.٥٪ (مع contract واحد على $١٠٠ك)",
        "Sharpe (annualized) ذو معنى إحصائي مع عدد الصفقات",
        "عدد الصفقات معقول (~١٥٠-٣٠٠ سنوياً لـ 6B)",
    ]))
    s.append(PageBreak())

    # ────────────────────────────────────────────────
    # SECTION 10 — Troubleshooting
    # ────────────────────────────────────────────────
    s.append(Paragraph("10. Appendix — Troubleshooting", H1_EN))
    s.append(Paragraph(ar("١٠. ملحق — استكشاف الأخطاء"), H1_AR))
    s.append(divider())

    problems = [
        ("Extractor: 'no_rollover_picked_dominant' on most months",
         "الـ extractor: 'no_rollover_picked_dominant' في معظم الشهور",
         "Likely cause: the file contains only ONE contract per month. "
         "This is normal for non-rollover months — the dominant contract is used as-is.",
         "السبب الأرجح: الملف يحوي عقد واحد فقط في الشهر. ده طبيعي لشهور "
         "غير الـ rollover — العقد المسيطر يُستخدم كما هو."),

        ("Extractor: applied_offset is NaN",
         "الـ extractor: applied_offset = NaN",
         "Pre-rollover window had no overlap with both contracts. Check "
         "raw data — both contracts must have at least 3 days of overlap "
         "before the rollover_day.",
         "نافذة ما قبل الـ rollover لم تتداخل مع كلا العقدين. تحقق من البيانات "
         "الخام — كلا العقدين يجب أن يكون لديهما ٣ أيام تداخل على الأقل قبل rollover."),

        ("combine_quarter_artifacts: shape mismatch",
         "combine_quarter_artifacts: shape mismatch",
         "Means LOB tensors / order tensors don't have the same first dim "
         "as the features parquet in one of the quarter dirs. Re-run "
         "prepare_day_trading + build_order_batches for that quarter with "
         "matching --freq / --lookback-bars / --n-orders settings.",
         "يعني LOB tensors / order tensors عندها أول بُعد مختلف عن الـ features parquet "
         "في أحد المجلدات. أعد تشغيل prepare_day_trading + build_order_batches للربع "
         "بنفس --freq / --lookback-bars / --n-orders."),

        ("SSL embedding_valid is too low (< 50%)",
         "SSL embedding_valid منخفض (< ٥٠٪)",
         "Likely cause: too many session_break markers, possibly because "
         "the bar duration < daily maintenance gap. With 15min bars, expect "
         "≥80% valid. If lower, check is_session_break distribution.",
         "السبب الأرجح: علامات session_break كثيرة جداً، ربما بسبب bar duration "
         "< maintenance gap اليومي. مع شموع ١٥ دقيقة، توقع >=٨٠٪ صالحة. "
         "لو أقل، تحقق من توزيع is_session_break."),

        ("Hybrid model val accuracy plateaus near random",
         "Hybrid val accuracy يثبت قرب العشوائي",
         "Likely cause: SSL embeddings have collapsed during pretraining "
         "OR feature columns include leakage. Check that LEAKAGE_PATTERNS "
         "in train_hybrid.py covers all label-derived columns.",
         "السبب الأرجح: SSL embeddings انهارت في pretraining، أو features تحوي "
         "leakage. تحقق أن LEAKAGE_PATTERNS في train_hybrid.py يغطي كل الأعمدة "
         "المشتقة من الـ labels."),

        ("Backtest hit rate << 78%",
         "Backtest hit rate أقل بكتير من ٧٨٪",
         "Likely cause: not running in --strict mode, OR holdout split is "
         "different from training. Always backtest with --strict and verify "
         "split boundaries.",
         "السبب الأرجح: لم تشغّل --strict mode، أو الـ holdout مختلف عن التدريب. "
         "دائماً شغّل --strict وتحقق من حدود الـ split."),
    ]

    for en_t, ar_t, en_b, ar_b in problems:
        s.append(Paragraph(f"<b>Q:</b> {en_t}", BODY_EN))
        s.append(Paragraph(ar(f"<b>س:</b> {ar_t}"), BODY_AR))
        s.append(Paragraph(f"<b>A:</b> {en_b}", BODY_EN))
        s.append(Paragraph(ar(f"<b>ج:</b> {ar_b}"), BODY_AR))
        s.append(Spacer(1, 8))

    s.append(divider())
    s.append(Paragraph(
        "For deeper architecture details, see SUBSYSTEMS.md and "
        "modules/trading_intel/README.md.",
        CAPTION,
    ))
    s.append(Paragraph(ar(
        "للمزيد من تفاصيل المعمارية، راجع SUBSYSTEMS.md و "
        "modules/trading_intel/README.md."
    ), ParagraphStyle("c2", parent=CAPTION, alignment=TA_RIGHT)))

    return s


# ── Page template with footer ──
def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(HexColor("#666666"))
    canvas.drawString(2 * cm, 1 * cm,
                       "QuantSystem Pipeline Guide  •  6B (GBP/USD)")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    out_path = Path(__file__).resolve().parent / "PIPELINE_GUIDE.pdf"
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="QuantSystem Pipeline Guide",
        author="QuantSystem",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="normal",
    )
    template = PageTemplate(id="main", frames=frame, onPage=_on_page)
    doc.addPageTemplates([template])

    story = build_story()
    doc.build(story)

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ Built: {out_path}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
