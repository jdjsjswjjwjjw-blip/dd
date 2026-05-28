"""
docs/build_verification_pipeline_pdf.py
═══════════════════════════════════════════════════════════════════════════════
Builds VERIFICATION_PIPELINE.pdf — focused bilingual walkthrough of every
verification + audit layer added on top of the day-trade and SSL stacks.

Sections
────────
  1. Overview                  the why
  2. Stage A — DAY_TRADE       prepare_day_trading.py + event-gate auto-hook
  3. Stage B — Hardening       FrozenScaler / redundancy / stress / regime
  4. Stage C — SSL training    walk-forward folds + anti-collapse modules
  5. Stage D — Execution       replay engine + slippage calibration
  6. Verdict cascades          all classifiers in one place
  7. End-to-end checklist      ordered runbook

Run
───
    python docs/build_verification_pipeline_pdf.py
    → docs/VERIFICATION_PIPELINE.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)


# ── Fonts (DejaVu covers Latin + Arabic) ─────────────────────────────────
pdfmetrics.registerFont(TTFont(
    "DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont(
    "DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont(
    "DejaVu-Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))


def ar(text: str) -> str:
    """Reshape + bidi an Arabic string for PDF rendering."""
    return get_display(arabic_reshaper.reshape(text))


# ── Styles ───────────────────────────────────────────────────────────────
STYLES = getSampleStyleSheet()

COVER_TITLE = ParagraphStyle(
    "COVER_TITLE", fontName="DejaVu-Bold", fontSize=28,
    textColor=HexColor("#0E2C5C"), alignment=TA_CENTER, spaceAfter=18,
)
COVER_SUB = ParagraphStyle(
    "COVER_SUB", fontName="DejaVu", fontSize=14,
    textColor=HexColor("#555555"), alignment=TA_CENTER, spaceAfter=16,
)
COVER_AR = ParagraphStyle(
    "COVER_AR", fontName="DejaVu-Bold", fontSize=22,
    textColor=HexColor("#0E2C5C"), alignment=TA_CENTER, spaceAfter=24,
)
H1_EN = ParagraphStyle(
    "H1_EN", fontName="DejaVu-Bold", fontSize=18,
    textColor=HexColor("#0E2C5C"), alignment=TA_LEFT,
    spaceBefore=4, spaceAfter=6,
)
H1_AR = ParagraphStyle("H1_AR", parent=H1_EN, alignment=TA_RIGHT)
H2_EN = ParagraphStyle(
    "H2_EN", fontName="DejaVu-Bold", fontSize=13,
    textColor=HexColor("#1A4080"), alignment=TA_LEFT,
    spaceBefore=8, spaceAfter=4,
)
H2_AR = ParagraphStyle("H2_AR", parent=H2_EN, alignment=TA_RIGHT)
H3_EN = ParagraphStyle(
    "H3_EN", fontName="DejaVu-Bold", fontSize=11,
    textColor=HexColor("#2C5AA0"), alignment=TA_LEFT,
    spaceBefore=6, spaceAfter=3,
)
BODY_EN = ParagraphStyle(
    "BODY_EN", fontName="DejaVu", fontSize=10, leading=14,
    alignment=TA_LEFT, spaceAfter=4,
)
BODY_AR = ParagraphStyle("BODY_AR", parent=BODY_EN, alignment=TA_RIGHT)
BULLET_EN = ParagraphStyle(
    "BULLET_EN", parent=BODY_EN, leftIndent=18, bulletIndent=8, spaceAfter=2,
)
BULLET_AR = ParagraphStyle(
    "BULLET_AR", parent=BODY_AR, rightIndent=18, spaceAfter=2,
)
CODE = ParagraphStyle(
    "CODE", fontName="DejaVu-Mono", fontSize=8, leading=11,
    leftIndent=10, rightIndent=10,
    backColor=HexColor("#F4F4F4"), borderColor=HexColor("#CCCCCC"),
    borderWidth=0.5, borderPadding=6, spaceAfter=6, spaceBefore=4,
)
CAPTION = ParagraphStyle(
    "CAPTION", parent=BODY_EN, fontSize=9, textColor=HexColor("#555555"),
)


# ── Helpers ──────────────────────────────────────────────────────────────
def code(text: str):
    return Preformatted(text.strip("\n"), CODE)


def bullets_en(items):
    return [Paragraph("• " + t, BULLET_EN) for t in items]


def bullets_ar(items):
    return [Paragraph(ar("• " + t), BULLET_AR) for t in items]


def hr():
    return HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#CCCCCC"),
        spaceBefore=6, spaceAfter=10,
    )


def section_pair(en_h: str, ar_h: str, en_body: str = "", ar_body: str = ""):
    out = [Paragraph(en_h, H2_EN), Paragraph(ar(ar_h), H2_AR)]
    if en_body:
        out.append(Paragraph(en_body, BODY_EN))
    if ar_body:
        out.append(Paragraph(ar(ar_body), BODY_AR))
    return out


def status_table(headers, rows, col_widths=None):
    """Two-column-aware table. Headers white-on-blue, alternating row tint."""
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1A4080")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), HexColor("#F4F8FF")))
    t.setStyle(TableStyle(style))
    return t


# ════════════════════════════════════════════════════════════════════════
# CONTENT
# ════════════════════════════════════════════════════════════════════════
def build_story() -> list:
    s = []

    # ──────────────────────────────────────────────────────────────────
    # COVER
    # ──────────────────────────────────────────────────────────────────
    s.append(Spacer(1, 5 * cm))
    s.append(Paragraph("Verification Pipeline", COVER_TITLE))
    s.append(Paragraph(ar("خط أنابيب التحقق والتدقيق"), COVER_AR))
    s.append(Paragraph(
        "DAY_TRADE → Hardening → SSL → Execution Realism", COVER_SUB))
    s.append(Paragraph(ar(
        "من قواعد التداول اليومي إلى محرّك التنفيذ الواقعي"), COVER_SUB))
    s.append(Spacer(1, 3 * cm))
    s.append(Paragraph(
        "Every audit, verdict, and feedback loop in one document.", CAPTION))
    s.append(Paragraph(ar(
        "كل عملية تدقيق وكل verdict وكل feedback loop في وثيقة واحدة."),
        ParagraphStyle("c", parent=CAPTION, alignment=TA_CENTER)))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 1. OVERVIEW
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("1. Overview", H1_EN))
    s.append(Paragraph(ar("١. نظرة عامة"), H1_AR))
    s += section_pair(
        "Why a verification pipeline",
        "لماذا خط أنابيب للتحقق",
        "A trading system has many points where a silent bug can survive "
        "the test suite and only show up in production: a scaler that "
        "re-fits on the full dataset, a feature that is 95 % correlated "
        "with another, a strategy that only works in trending markets, a "
        "labeling gate that mis-classifies 80 % of bars as NEUTRAL, a "
        "slippage assumption that underestimates real execution by 3×. "
        "This pipeline catches each of those classes of bug with a "
        "dedicated, testable, automated tool.",
        "نظام التداول مليء بنقاط ممكن فيها bug صامت ينجو من الـ test suite "
        "ولا يظهر إلا في الـ production: scaler يعيد الـ fit على كامل البيانات، "
        "ميزة مرتبطة 95% بميزة أخرى، استراتيجية تنجح فقط في trending markets، "
        "labeling gate يصنّف 80% من الصفوف NEUTRAL خطأ، slippage assumption "
        "يقلّل الواقع بـ 3 أضعاف. خط الأنابيب ده بيلاقي كل نوع منهم بأداة "
        "مخصّصة قابلة للاختبار ومؤتمتة.",
    )
    s += section_pair(
        "How to read this document",
        "كيف تقرأ هذه الوثيقة",
        "Each stage has: (1) what it produces, (2) the verification "
        "command, (3) the verdict it outputs, (4) what to do if the "
        "verdict is bad. Code blocks are copy-paste-runnable.",
        "كل مرحلة فيها: (١) المخرجات، (٢) أمر التحقّق، (٣) الـ verdict اللي "
        "بيطلع، (٤) إيه تعمل لو الـ verdict سيء. الـ code blocks جاهزة "
        "للنسخ والتشغيل مباشرةً.",
    )

    s.append(Paragraph("Pipeline at a glance", H2_EN))
    s.append(Paragraph(ar("نظرة سريعة على المسار"), H2_AR))
    s.append(code("""
   ┌──────────────────┐
   │  Raw MBO/MBP-10  │
   └────────┬─────────┘
            ▼
   ┌──────────────────────────────────────┐
   │ A. prepare_day_trading.py            │
   │    └─ event-gate auto-hook (audit)   │
   └────────┬─────────────────────────────┘
            ▼
   ┌──────────────────────────────────────┐
   │ B. Hardening (4 layers)              │
   │    ├─ FrozenScaler (norm drift)      │
   │    ├─ audit_feature_redundancy       │
   │    ├─ stress_test_backtest           │
   │    └─ regime_parity_test             │
   └────────┬─────────────────────────────┘
            ▼
   ┌──────────────────────────────────────┐
   │ C. SSL + Hybrid (walk-forward folds) │
   │    └─ anti-collapse modules          │
   └────────┬─────────────────────────────┘
            ▼
   ┌──────────────────────────────────────┐
   │ D. Execution realism (Replay Engine) │
   │    └─ calibrate_slippage             │
   └──────────────────────────────────────┘
"""))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 2. STAGE A — DAY_TRADE
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("2. Stage A — DAY_TRADE pipeline", H1_EN))
    s.append(Paragraph(ar("٢. المرحلة أ — خط أنابيب DAY_TRADE"), H1_AR))
    s += section_pair(
        "What it does",
        "ما الذي يفعله",
        "Ingests raw Databento MBO/MBP-10 ticks, aggregates them into "
        "session-aware bars, enriches with regime + Kalman trend + "
        "Hawkes/absorption/Kyle microstructure signals, decides which "
        "bars are EVENTS via a six-component gate, and produces "
        "directional labels for the ones that pass.",
        "بيقرأ ticks خام من Databento (MBO أو MBP-10)، يجمّعهم في bars "
        "حسب الجلسة، يضيف regime + Kalman trend + Hawkes/absorption/Kyle "
        "microstructure signals، يقرّر أي bars EVENTS عبر gate بـ ٦ مكوّنات، "
        "ويولّد directional labels للـ bars اللي عدّت.",
    )
    s += section_pair(
        "Run it",
        "كيف تشغّله",
        "", "",
    )
    s.append(code("""python prepare_day_trading.py \\
    --raw-mbo-dir raw/mbo \\
    --output-dir  combined/ \\
    --symbol 6B"""))

    s += section_pair(
        "Auto-hook: event-gate audit",
        "Auto-hook: تدقيق Event Gate تلقائي",
        "After writing the features parquet, the pipeline runs the event-"
        "gate audit automatically and prints a verdict in line with the "
        "existing coverage diagnostics. The labels default to NEUTRAL on "
        "every bar where is_event == 0, so a starved gate is the single "
        "biggest cause of high-NEUTRAL training data. The hook makes that "
        "cause visible at write time instead of hiding it.",
        "بعد ما الـ pipeline يكتب الـ features parquet، بيشغّل تلقائياً "
        "audit_event_gate ويطبع الـ verdict مع الـ coverage diagnostics. "
        "الـ labels بتكون NEUTRAL على كل bar فيه is_event=0، فلو الـ gate "
        "starved ده أكبر سبب لـ NEUTRAL عالي في training data. الـ hook "
        "بيكشف السبب لحظة الكتابة بدل ما يبقى مخفي.",
    )
    s.append(Paragraph("Example output:", H3_EN))
    s.append(code("""💾 Dataset saved: combined/day_trading_features.parquet
   Rows: 18,000 | Columns: 247
   📊 MBP coverage: bar_mean=82.3%, low(<30%)=4.2%
   📊 Training pool:
      train_event_flag=True : 3,820 rows (21.2%)
      LONG: 1,950 | SHORT: 1,870

   🚨 Event gate: COMPONENT_DOMINATED (rate=12.4%) → combined/_audit_event_gate/
      ← high NEUTRAL traces to gate, not market behavior
      ← dominated_components: ['cvd_align_above_06']"""))

    s.append(Paragraph(
        "Per-verdict response (event gate)", H3_EN))
    s.append(status_table(
        ["Verdict", "Meaning", "Action"],
        [
            ["HEALTHY", "20-35% events, all regions clean",
             "proceed to Stage B"],
            ["LOW_RATE", "10-20% events", "loosen REGIME_EVENT_THRESHOLD"],
            ["STARVED", "<10% events",
             "switch event_score to continuous_score"],
            ["WARMUP_HEAVY", "≥5% of bars killed by _zscore min_periods",
             "drop warm-up bars from training set"],
            ["COMPONENT_DOMINATED", "one gate component fails >90% of bars",
             "fix the source signal (eg. cvd_direction_pct fillna)"],
            ["REGIME_STARVED", "non-volatile regime <15% event rate",
             "regime-specific threshold"],
        ],
        col_widths=[3.6 * cm, 6.4 * cm, 6.2 * cm],
    ))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 3. STAGE B — HARDENING
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("3. Stage B — Operational hardening", H1_EN))
    s.append(Paragraph(ar("٣. المرحلة ب — التصلّب التشغيلي"), H1_AR))
    s += section_pair(
        "Four layers, four classes of bug",
        "أربع طبقات، أربع فئات من الـ bugs",
        "Each layer addresses one production-realistic failure mode "
        "documented in docs/PIPELINE_ISSUES_AUDIT.md.",
        "كل طبقة بتعالج نوع واحد من الـ failures المُوَثّقة في "
        "docs/PIPELINE_ISSUES_AUDIT.md.",
    )
    s.append(status_table(
        ["Layer", "Tool", "What it solves"],
        [
            ["Normalization drift",
             "modules/trading_intel/hybrid/inference.py (FrozenScaler)",
             "Cannot accidentally fit_transform at inference"],
            ["Input redundancy",
             "tools/audit_feature_redundancy.py",
             "Drops 14 CVD / 12 ATR / 7 imbalance duplicate variants"],
            ["Execution stress",
             "tools/diagnostics/stress_test_backtest.py",
             "5 scenarios: baseline → mild → moderate → severe → extreme"],
            ["Regime sensitivity",
             "tools/diagnostics/regime_parity_test.py",
             "Catches one-regime strategies that look profitable in agg"],
        ],
        col_widths=[3.6 * cm, 5.6 * cm, 7.0 * cm],
    ))

    s.append(Paragraph("B.1 FrozenScaler — normalization drift", H2_EN))
    s.append(Paragraph(ar("ب.١ FrozenScaler — انحراف التطبيع"), H2_AR))
    s.append(Paragraph(
        "Trains with mu/sigma fit only on the train slice. Inference "
        "loads a checkpoint and the scaler is frozen — every transform "
        "uses the stored stats; any fit() raises RuntimeError. There is "
        "no public path to accidentally re-fit at inference.", BODY_EN))
    s.append(Paragraph(ar(
        "التدريب بيـfit الـ mu/sigma على train slice فقط. الـ inference "
        "بيحمّل checkpoint والـ scaler مجمّد — كل transform بيستخدم "
        "stats محفوظة؛ أي fit() بيرمي RuntimeError. مفيش طريق عام لإعادة "
        "الـ fit بالغلط أثناء الـ inference."), BODY_AR))
    s.append(code("""from modules.trading_intel.hybrid.inference import (
    load_hybrid_for_inference, predict_one,
)
model, scaler, cfg = load_hybrid_for_inference("runs/fold_01")
pred = predict_one(model, scaler, latest_features_dict)
# scaler.fit(...)  → RuntimeError"""))

    s.append(Paragraph("B.2 Feature-redundancy audit", H2_EN))
    s.append(Paragraph(ar("ب.٢ تدقيق التكرار في الميزات"), H2_AR))
    s.append(Paragraph(
        "Builds a Pearson correlation graph at threshold τ (default "
        "0.9), takes connected components, marks all but the most-"
        "predictive feature in each cluster as redundant. Writes "
        "redundancy_summary.json with a drop_list consumed by both fold "
        "runners via --drop-features-from-audit.", BODY_EN))
    s.append(Paragraph(ar(
        "بيبني correlation graph بين الميزات عند threshold τ (default 0.9)، "
        "ياخد connected components، يحتفظ بأقوى ميزة في كل cluster ويعتبر "
        "الباقي تكرار. بيكتب redundancy_summary.json فيه drop_list يستهلكه "
        "الـ fold runners عبر --drop-features-from-audit."), BODY_AR))
    s.append(code("""python tools/audit_feature_redundancy.py \\
    --features combined/day_trading_features.parquet \\
    --output   audits/redundancy --threshold 0.9"""))

    s.append(Paragraph("B.3 Stress test", H2_EN))
    s.append(Paragraph(ar("ب.٣ اختبار الإجهاد التنفيذي"), H2_AR))
    s.append(Paragraph(
        "Re-runs the strict backtest under five execution-quality "
        "scenarios from baseline to extreme (latency bars + slippage "
        "pips both monotonically worsening). Verdict from final-fold "
        "Sharpe stability across scenarios.", BODY_EN))
    s.append(Paragraph(ar(
        "بيعيد الـ backtest تحت ٥ سيناريوهات تنفيذ متدرّجة من baseline لـ "
        "extreme (latency bars + slippage pips بيزيدوا monotonically). الـ "
        "verdict بناءً على ثبات Sharpe عبر السيناريوهات."), BODY_AR))
    s.append(status_table(
        ["Verdict", "Sharpe stability"],
        [
            ["ROBUST",     "All 5 scenarios profitable"],
            ["ACCEPTABLE", "Worst case still ≥ 0.5 Sharpe"],
            ["FRAGILE",    "Severe scenario breaks the edge"],
            ["BROKEN",     "Mild scenario already breaks the edge"],
        ],
        col_widths=[3.5 * cm, 12.7 * cm],
    ))

    s.append(Paragraph("B.4 Regime parity", H2_EN))
    s.append(Paragraph(ar("ب.٤ تكافؤ الـ regime"), H2_AR))
    s.append(Paragraph(
        "Splits trades by categorical regime AND by realised-vol "
        "quartile, computes per-subset Sharpe/PF/DD and a coefficient "
        "of variation across subsets. Catches strategies that secretly "
        "depend on one regime to be profitable.", BODY_EN))
    s.append(Paragraph(ar(
        "بيقسّم الصفقات حسب الـ regime (تصنيفي) وحسب volatility quartile، "
        "ويحسب Sharpe/PF/DD لكل subset مع coefficient of variation عبر "
        "الـ subsets. بيكشف الاستراتيجيات اللي خفية بتعتمد على regime واحد "
        "علشان تكون رابحة."), BODY_AR))
    s.append(code("""python tools/diagnostics/regime_parity_test.py \\
    --trades-csv backtests/baseline_strict/trades.csv \\
    --output     diagnostics/regime --min-trades 30"""))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 4. STAGE C — SSL TRAINING
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("4. Stage C — SSL training", H1_EN))
    s.append(Paragraph(ar("٤. المرحلة ج — تدريب الـ SSL"), H1_AR))
    s += section_pair(
        "Walk-forward + anti-collapse",
        "Walk-forward + مضادّ الانهيار",
        "The SSL encoder + hybrid head are trained per fold on an "
        "expanding train window with a forward holdout. Four modules "
        "(individually toggleable) protect against the documented "
        "failure modes of multi-task SSL.",
        "الـ SSL encoder + hybrid head بيتدرّبوا per fold على expanding "
        "train window مع forward holdout. أربع modules (كل واحدة قابلة "
        "للتفعيل المستقل) بتحمي من failure modes الموثّقة في multi-task SSL.",
    )
    s.append(status_table(
        ["Module", "Failure it prevents", "Activation flag"],
        [
            ["Fixed Simplex ETF",
             "direction-head collapse to single class",
             "--use-simplex-etf"],
            ["Orthogonal Representation",
             "feature-collapse (all heads share one subspace)",
             "--ortho-weight W"],
            ["DB-MTL balancer",
             "small tasks dominated by large ones",
             "--use-dbmtl"],
            ["Differentiable Sharpe",
             "loss is proxy for accuracy, not return",
             "--sharpe-weight W"],
        ],
        col_widths=[4.0 * cm, 7.2 * cm, 5.0 * cm],
    ))
    s.append(Paragraph("Run a fold (all four enabled):", H3_EN))
    s.append(code("""python tools/run_walk_forward_fold_enhanced.py \\
    --fold-id 1 \\
    --combined-features combined/day_trading_features.parquet \\
    --ssl-embeddings    embeddings/ssl_macro.parquet \\
    --embedding-valid   embeddings/ssl_valid_mask.parquet \\
    --train-start 2021-01-01 --train-end 2023-12-31 \\
    --test-start  2024-01-01 --test-end  2024-06-30 \\
    --output-dir  runs/fold_01 \\
    --drop-features-from-audit audits/redundancy/redundancy_summary.json \\
    --use-simplex-etf \\
    --ortho-weight 0.05 \\
    --use-dbmtl \\
    --sharpe-weight 0.1"""))
    s.append(Paragraph("Aggregate across folds:", H3_EN))
    s.append(code("""python tools/aggregate_walk_forward.py \\
    --runs-dir runs/ \\
    --output   aggregated_metrics.json"""))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 5. STAGE D — EXECUTION REALISM
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("5. Stage D — Execution realism", H1_EN))
    s.append(Paragraph(ar("٥. المرحلة د — واقعية التنفيذ"), H1_AR))
    s += section_pair(
        "Replay engine + slippage calibration",
        "Replay engine + معايرة الـ slippage",
        "Bar-level backtest assumes constant 0.5-pip slippage and "
        "integer-bar latency — fine for ranking strategies, blind to "
        "what a 50-lot order does to the level-2 book. Stage D fixes "
        "that with two stand-alone tools that form a closed feedback "
        "loop.",
        "الـ backtest على مستوى الـ bars بيفترض slippage ثابت 0.5 pip "
        "و latency بـ bars صحيحة — كويس لمقارنة استراتيجيات، لكن أعمى "
        "عن تأثير أمر 50 lot على L2 book. المرحلة د بتحلّ ده بأداتين "
        "مستقلّتين بتشكّلوا closed feedback loop.",
    )

    s.append(Paragraph("D.1 Replay engine", H2_EN))
    s.append(Paragraph(ar("د.١ محرّك الإعادة (Replay Engine)"), H2_AR))
    s.append(Paragraph(
        "modules/replay_engine/ — event-by-event MBP-10 simulator. "
        "Walks the book level-by-level on each market order, samples "
        "lognormal latency between decision and arrival, logs every "
        "fill with the LOB snapshot at fill time. Pure separation: "
        "snapshots are frozen dataclasses; fill_market_order is a pure "
        "function; only the orchestrator owns time and RNG.", BODY_EN))
    s.append(Paragraph(ar(
        "modules/replay_engine/ — محاكي MBP-10 حدث-بـحدث. بيمشي على الـ "
        "book مستوى مستوى لكل market order، بيسحب latency لـlognormal بين "
        "الـ decision والـ arrival، بيسجّل كل fill مع LOB snapshot لحظة "
        "التنفيذ. فصل صافي: الـ snapshots frozen dataclasses؛ "
        "fill_market_order pure function؛ الـ orchestrator هو الوحيد اللي "
        "بيملك الوقت والـ RNG."), BODY_AR))
    s.append(code("""python tools/run_replay_backtest.py \\
    --signals  backtests/baseline/signals.csv \\
    --mbp      raw/mbp_10.parquet \\
    --output   replay_results/baseline \\
    --tick-size 0.0001 \\
    --latency-mean-us 5000 --latency-jitter-us 2000 \\
    --slip-base-ticks 0.5 --slip-mid-ticks 1.5"""))
    s.append(Paragraph("Outputs", H3_EN))
    s += bullets_en([
        "execution_log.jsonl — one JSON object per fill (snapshot included)",
        "replay_summary.json — aggregate latency/slippage/fill-quality stats",
    ])
    s += bullets_ar([
        "execution_log.jsonl — كل سطر JSON واحد بيمثّل fill (snapshot مع الـ fill)",
        "replay_summary.json — إحصاءات تجميعية للـ latency والـ slippage وجودة الـ fill",
    ])

    s.append(Paragraph("D.2 AdaptiveSlippage — three regions", H2_EN))
    s.append(Paragraph(ar("د.٢ AdaptiveSlippage — ثلاث مناطق"), H2_AR))
    s.append(status_table(
        ["Region", "Condition", "Predicted ticks"],
        [
            ["A — sub-L1",  "size / L1_depth ≤ 0.10",     "base_ticks (flat)"],
            ["B — linear",  "0.10 < ratio ≤ 1.0",
             "base + t · (mid − base), t = (ratio − 0.10) / 0.90"],
            ["C — walking", "ratio > 1.0",
             "mid + extra_per_level · (levels_consumed − 1)"],
        ],
        col_widths=[3.0 * cm, 4.6 * cm, 8.4 * cm],
    ))

    s.append(Paragraph("D.3 Calibration loop", H2_EN))
    s.append(Paragraph(ar("د.٣ حلقة المعايرة"), H2_AR))
    s.append(Paragraph(
        "Reads execution_log.jsonl, splits events by region, refits "
        "each constant via OLS-through-origin on its own subset, then "
        "applies safety clamps (non-negativity + base ≤ mid). Writes "
        "slippage_config_suggested.json as a drop-in for the next "
        "replay run.", BODY_EN))
    s.append(Paragraph(ar(
        "بيقرأ execution_log.jsonl، يقسّم الـ events حسب الـ region، يعيد "
        "fit كل constant عبر OLS-through-origin على subset بتاعها، ثم "
        "يطبّق safety clamps (عدم سالبية + base ≤ mid). يكتب "
        "slippage_config_suggested.json كـdrop-in للـ replay اللي بعده."),
        BODY_AR))
    s.append(code("""python tools/calibrate_slippage.py \\
    --log    replay_results/baseline/execution_log.jsonl \\
    --output replay_results/baseline/calibration \\
    --tick-size 0.0001 \\
    --current-base 0.5 --current-mid 1.5 --current-extra 1.0"""))

    s.append(Paragraph("Closed feedback loop", H3_EN))
    s.append(code("""  run_replay_backtest → execution_log.jsonl
          ↓
  calibrate_slippage  → slippage_config_suggested.json
          ↓
  run_replay_backtest --slip-base X --slip-mid Y --slip-extra Z
          ↓
  (repeat until verdict = WELL_CALIBRATED)"""))
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 6. VERDICT CASCADES — ALL IN ONE PLACE
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("6. Verdict cascades — all in one place", H1_EN))
    s.append(Paragraph(ar("٦. شجرات الـ verdicts — في مكان واحد"), H1_AR))
    s.append(Paragraph(
        "Every audit emits a categorical verdict. Use this table to "
        "decide whether the run is shippable.", BODY_EN))
    s.append(Paragraph(ar(
        "كل audit بيخرج verdict تصنيفي. الجدول ده بيقولّك إذا الـ run "
        "جاهز للـ production أم لأ."), BODY_AR))
    s.append(status_table(
        ["Audit", "OK", "Marginal", "Block"],
        [
            ["Event gate",
             "HEALTHY",
             "LOW_RATE / WARMUP_HEAVY / REGIME_STARVED",
             "STARVED / COMPONENT_DOMINATED"],
            ["Redundancy",
             "drop_list applied",
             "drop_list large but applied",
             "drop_list not applied"],
            ["Stress test",
             "ROBUST",
             "ACCEPTABLE",
             "FRAGILE / BROKEN"],
            ["Regime parity",
             "ROBUST",
             "ACCEPTABLE",
             "UNSTABLE / REGIME_BIAS / INSUFFICIENT_DATA"],
            ["Slippage calibration",
             "WELL_CALIBRATED",
             "OVER_CALIBRATED (safe)",
             "UNDER_CALIBRATED (dangerous)"],
        ],
        col_widths=[3.4 * cm, 3.5 * cm, 4.6 * cm, 4.5 * cm],
    ))

    s.append(Paragraph("Why some 'Block' verdicts are tolerable", H2_EN))
    s.append(Paragraph(ar("لماذا بعض verdicts الـ Block محتملة"), H2_AR))
    s += bullets_en([
        "OVER_CALIBRATED (slippage) — model is conservative; the system "
        "loses some marginal trades that would have been profitable but "
        "will never put on a trade it shouldn't.",
        "WARMUP_HEAVY (event gate) — fixable by dropping the warm-up "
        "bars from the training set; the live system never sees them.",
        "FRAGILE (stress) — acceptable as a documented risk only if the "
        "broker is known to provide consistent execution.",
    ])
    s += bullets_ar([
        "OVER_CALIBRATED (slippage) — الـ model conservative؛ النظام بيخسر "
        "بعض الصفقات الـ marginal لكن لن يدخل في صفقة كان مفروض ميدخلش فيها.",
        "WARMUP_HEAVY (event gate) — قابل للإصلاح بطرد الـ warm-up bars "
        "من training set؛ الـ live system مش بيشوفهم أصلاً.",
        "FRAGILE (stress) — مقبول كمخاطرة موثّقة فقط لو الـ broker معروف "
        "بتنفيذ ثابت.",
    ])
    s.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────
    # 7. END-TO-END RUNBOOK
    # ──────────────────────────────────────────────────────────────────
    s.append(Paragraph("7. End-to-end runbook", H1_EN))
    s.append(Paragraph(ar("٧. خطوات التشغيل من البداية للنهاية"), H1_AR))
    s.append(Paragraph(
        "Run these in order. Each command's verdict gates the next.",
        BODY_EN))
    s.append(Paragraph(ar(
        "شغّلهم بالترتيب. verdict كل أمر بيحدّد إذا اللي بعده يشتغل."),
        BODY_AR))

    s.append(Paragraph("Step 1 — Build features + auto-audit", H3_EN))
    s.append(code("""python prepare_day_trading.py \\
    --raw-mbo-dir raw/mbo --output-dir combined/ --symbol 6B
# expect: 🚨/⚠️/✅ Event gate verdict on stdout"""))

    s.append(Paragraph("Step 2 — Feature redundancy audit", H3_EN))
    s.append(code("""python tools/audit_feature_redundancy.py \\
    --features combined/day_trading_features.parquet \\
    --output   audits/redundancy --threshold 0.9"""))

    s.append(Paragraph("Step 3 — Walk-forward folds (loop over fold_id)", H3_EN))
    s.append(code("""for FOLD in 1 2 3 4 5 6 7 8; do
    python tools/run_walk_forward_fold_enhanced.py \\
        --fold-id $FOLD \\
        --combined-features combined/day_trading_features.parquet \\
        --ssl-embeddings    embeddings/ssl_macro.parquet \\
        --embedding-valid   embeddings/ssl_valid_mask.parquet \\
        --output-dir        runs/fold_$FOLD \\
        --drop-features-from-audit audits/redundancy/redundancy_summary.json \\
        --use-simplex-etf --ortho-weight 0.05 \\
        --use-dbmtl --sharpe-weight 0.1
done
python tools/aggregate_walk_forward.py --runs-dir runs/ --output runs/agg.json"""))

    s.append(Paragraph("Step 4 — Stress test", H3_EN))
    s.append(code("""python tools/diagnostics/stress_test_backtest.py \\
    --features combined/day_trading_features.parquet \\
    --output   diagnostics/stress
# expect: verdict ROBUST or ACCEPTABLE"""))

    s.append(Paragraph("Step 5 — Regime parity", H3_EN))
    s.append(code("""python tools/diagnostics/regime_parity_test.py \\
    --trades-csv backtests/baseline_strict/trades.csv \\
    --output     diagnostics/regime
# expect: verdict ROBUST or ACCEPTABLE"""))

    s.append(Paragraph("Step 6 — Replay backtest (execution realism)", H3_EN))
    s.append(code("""python tools/run_replay_backtest.py \\
    --signals  backtests/baseline_strict/signals.csv \\
    --mbp      raw/mbp_10.parquet \\
    --output   replay_results/baseline \\
    --tick-size 0.0001"""))

    s.append(Paragraph("Step 7 — Calibrate slippage", H3_EN))
    s.append(code("""python tools/calibrate_slippage.py \\
    --log    replay_results/baseline/execution_log.jsonl \\
    --output replay_results/baseline/calibration \\
    --tick-size 0.0001
# expect: verdict WELL_CALIBRATED after 1-2 iterations"""))

    s.append(Paragraph("Step 8 — Verify in inference mode", H3_EN))
    s.append(code("""python -c "
from modules.trading_intel.hybrid.inference import (
    load_hybrid_for_inference, predict_one,
)
m, sc, cfg = load_hybrid_for_inference('runs/fold_01')
print('FrozenScaler intact:', sc.is_frozen())
" """))

    s.append(hr())
    s.append(Paragraph(
        "If every step ends with a green verdict, the build is "
        "shippable. If any step blocks, fix that step before moving on "
        "— don't override.", BODY_EN))
    s.append(Paragraph(ar(
        "لو كل خطوة طلعت verdict أخضر، الـ build جاهز للـ production. "
        "لو أي خطوة بلوك، أصلح المرحلة دي قبل ما تمشي للي بعدها — متعملش "
        "override."), BODY_AR))

    return s


# ════════════════════════════════════════════════════════════════════════
# DOCUMENT BUILD
# ════════════════════════════════════════════════════════════════════════
def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 1 * cm,
                             f"Verification Pipeline — page {doc.page}")
    canvas.restoreState()


def main():
    out = Path(__file__).parent / "VERIFICATION_PIPELINE.pdf"
    doc = BaseDocTemplate(
        str(out), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="main",
    )
    doc.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=_on_page))
    doc.build(build_story())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
