"""
regime_config.py — المرجع المركزي لإعدادات كل Regime
═══════════════════════════════════════════════════════
QuantSystem V19 | Section 9 من المرجع التقني

يُستورد من:
    prepare_day_trading.py  → REGIME_TP_SL, REGIME_MAX_BARS, REGIME_EVENT_THRESHOLD
    initial_training_6_years.py → REGIME_MODEL_CONFIGS, REGIME_EXTRA_FEATURES
    modules/online_learning.py  → REGIME_DRIFT_CONFIG, REGIME_PRED_THRESHOLD
    modules/live_predictor.py   → REGIME_EXTRA_FEATURES, REGIME_PRED_THRESHOLD

قاعدة: لا تُعرِّف هذه الثوابت في أي ملف آخر — استورد دائماً من هنا.
"""

from __future__ import annotations

# ── تعريف الـ Regimes المدعومة ────────────────────────────────────────────────
# low_liquidity أُضيف لمطابقة assign_regime_label في prepare_day_trading.py
# الذي يصنّف الـ bars بنسبة mbp_bar_coverage < 0.30 كـ low_liquidity منذ V19.
REGIMES: tuple[str, ...] = ('trending', 'ranging', 'volatile', 'low_liquidity')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Event Detection — عتبة كشف الأحداث الحقيقية
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: عتبة أعلى — فقط إشارات قوية جداً تمر (السوق فوضوي والإشارات الضعيفة مضللة)
# low_liquidity: أعلى عتبة — إشارات ضعيفة في عمق ضعيف = ضوضاء غالباً
REGIME_EVENT_THRESHOLD: dict[str, float] = {
    'trending'      : 0.60,
    'ranging'       : 0.60,
    'volatile'      : 0.75,
    'low_liquidity' : 0.80,
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Labels: TP/SL Multipliers
# ──────────────────────────────────────────────────────────────────────────────
# (tp_mult, sl_mult) — مضروبان في ATR
# Trending:  حركة طويلة → TP = 2× ATR كافٍ، SL = 1× ATR عادي
# Ranging:   رفع TP قليلًا لتقليل break-even المطلوب مع الحفاظ على SL مضبوط
# Volatile:  TP أوسع وSL أضيق نسبيًا حتى لا يصبح BE غير واقعي
# low_liquidity: أهداف صغيرة + SL ضيق — السبريد أوسع، الحركة محدودة
REGIME_TP_SL: dict[str, tuple[float, float]] = {
    'trending'      : (2.0, 1.0),
    'ranging'       : (1.1, 0.6),
    'volatile'      : (1.8, 1.0),
    'low_liquidity' : (1.0, 0.5),
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Labels: Max Horizon (بالـ bars)
# ──────────────────────────────────────────────────────────────────────────────
# كل bar = 5 دقائق (افتراضياً)
# Trending:  الحركة تستمر طويلاً → 12 bar = 60 دقيقة
# Ranging:   الحركة تعكس بسرعة → 6 bars = 30 دقيقة
# Volatile:  الحركة تنتهي بسرعة → 3 bars = 15 دقيقة
# low_liquidity: نافذة قصيرة — العمق الضعيف لا يدعم حركة مستدامة
REGIME_MAX_BARS: dict[str, int] = {
    'trending'      : 12,
    'ranging'       :  6,
    'volatile'      :  3,
    'low_liquidity' :  4,
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Training: CatBoost Model Config
# ──────────────────────────────────────────────────────────────────────────────
# Trending:  عمق أوسط + iterations كثيرة → يتعلم تسلسل اتجاهي طويل
# Ranging:   عمق أكبر → يتعلم أنماط انعكاس دقيقة
# Volatile:  عمق ضحل + learning_rate أسرع → يتعلم بسرعة ولا يُفرط في التخصيص
# low_liquidity: نموذج معتدل — البيانات أقل، تجنّب overfitting
REGIME_MODEL_CONFIGS: dict[str, dict] = {
    'trending'      : {'iterations': 800,  'learning_rate': 0.02,  'depth': 7},
    'ranging'       : {'iterations': 1000, 'learning_rate': 0.015, 'depth': 8},
    'volatile'      : {'iterations': 500,  'learning_rate': 0.05,  'depth': 5},
    'low_liquidity' : {'iterations': 600,  'learning_rate': 0.03,  'depth': 6},
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Training: Features الإضافية لكل Regime
# ──────────────────────────────────────────────────────────────────────────────
# تُضاف فوق BASE_FEATURES المشتركة
# Trending:  يحتاج لقياس التسارع والزخم الاتجاهي
# Ranging:   يحتاج لقياس الاضطراب والانعكاسات وعمق السيولة
# Volatile:  يحتاج للحظات الذروة والضغط الشديد
# low_liquidity: يحتاج لقياس عمق/تغطية الـ LOB والإلغاءات
REGIME_EXTRA_FEATURES: dict[str, list[str]] = {
    'trending': [
        'cvd_velocity',           # تسارع الضغط الاتجاهي
        'hawkes_intensity',       # كثافة وصول الأوردرات
        'kyle_lambda',            # استجابة السعر للـ flow
        'cvd_direction_pct',      # نسبة الـ slices في اتجاه واحد
    ],
    'ranging': [
        'absorption_std',         # تقلب صانع السوق
        'imb_reversals',          # عدد انعكاسات الـ OBI
        'cancel_std',             # تقلب الإلغاءات
        'mbp_bid_slope_intrabar', # ميل عمق الـ bid
        'mbp_ask_slope_intrabar', # ميل عمق الـ ask
    ],
    'volatile': [
        'kyle_lambda',            # تأثير flow (max يُحسب في التجميع)
        'cancel_volume_ratio',    # نسبة السبوفينغ الكلي
        'micro_atr_max',          # أعلى تقلب intrabar
        'absorption_intensity',   # شدة الامتصاص
    ],
    'low_liquidity': [
        'mbp_bar_coverage',       # تغطية MBP داخل الشمعة
        'mbo_bar_coverage',       # كثافة تيكات MBO
        'cancel_ratio',           # نسبة الإلغاءات
        'spoofing_ratio',         # نسبة السبوفينغ
        'inter_event_time',       # فجوات زمنية بين الأحداث
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Online Learning: Drift Detection
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: delta أكبر + lambda أصغر = يكتشف drift بسرعة أكبر
#           لأن السوق الـ volatile يتغير أسرع من الـ trending/ranging
# low_liquidity: متوسط — التعلّم بسرعة معتدلة لتفادي ضجيج البيانات الناقصة
REGIME_DRIFT_CONFIG: dict[str, dict] = {
    'trending'      : {'delta': 0.005, 'lambda_': 50},
    'ranging'       : {'delta': 0.005, 'lambda_': 50},
    'volatile'      : {'delta': 0.008, 'lambda_': 30},
    'low_liquidity' : {'delta': 0.006, 'lambda_': 40},
}

# ──────────────────────────────────────────────────────────────────────────────
# 7. Live Prediction: Confidence Threshold
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: ثقة أعلى مطلوبة لأن الإشارات الخاطئة في volatile أكثر تكلفةً
# low_liquidity: أعلى ثقة مطلوبة — كلفة الخطأ في سيولة ضعيفة كبيرة جداً (slippage)
REGIME_PRED_THRESHOLD: dict[str, float] = {
    'trending'      : 0.60,
    'ranging'       : 0.60,
    'volatile'      : 0.65,
    'low_liquidity' : 0.70,
}

# ──────────────────────────────────────────────────────────────────────────────
# 8. Event Score Weights — أوزان مكونات event_score
# ──────────────────────────────────────────────────────────────────────────────
# المجموع = 1.0 — تغطية LOB (MBP roll) + كثافة شريط MBO تربط الحدث بعمق السوق وجودة التيكات
EVENT_SCORE_WEIGHTS: dict[str, float] = {
    'hawkes_z_above_1'        : 0.26,
    'absorb_z_above_1'        : 0.26,
    'kyle_z_above_05'         : 0.165,
    'cvd_align_above_06'      : 0.165,
    'mbp_roll_cov_above_cut'  : 0.075,
    'mbo_tick_cov_above_cut'  : 0.075,
}

# عتبة ثنائية على mbp_roll_lob_coverage ∈ [0,1] (نفس روح _coverage_stats low_threshold≈0.50)
EVENT_LOB_COVERAGE_CUT: float = 0.50

# عتبة ثنائية على mbo_bar_coverage ∈ [0,1] (كثافة التيكات مقابل وسط محلي)
EVENT_MBO_COVERAGE_CUT: float = 0.50

# Z-score window للـ rolling baseline
EVENT_ZSCORE_WINDOW: int = 100
EVENT_ZSCORE_MIN_PERIODS: int = 20

# ──────────────────────────────────────────────────────────────────────────────
# 9. DayTrade labels — TP/SL/horizon حسب شريحة event_score (صفوف is_event فقط)
# ──────────────────────────────────────────────────────────────────────────────
# عند التفعيل: يُستبدل REGIME_TP_SL و REGIME_MAX_BARS لكل صف حدث بثلاث شرائح من event_score
DAYTRADE_EVENT_SCORE_TIER_LABELS: bool = True

EVENT_LABEL_SCORE_STRONG_MIN: float = 0.70   # >= → شريحة قوية
EVENT_LABEL_SCORE_MID_MIN: float = 0.50      # >= وبحد أدنى أقل من strong → وسط؛ وإلا ضعيف

EVENT_LABEL_TIER_STRONG: tuple[float, float, int] = (2.0, 1.0, 24)   # tp_mult, sl_mult, max_bars
EVENT_LABEL_TIER_MID: tuple[float, float, int] = (1.5, 1.0, 12)
EVENT_LABEL_TIER_WEAK: tuple[float, float, int] = (1.0, 1.0, 6)

# ──────────────────────────────────────────────────────────────────────────────
# جدول ملخص سريع للمراجعة
# ──────────────────────────────────────────────────────────────────────────────
def print_regime_summary() -> None:
    """يطبع جدول مقارنة الإعدادات لكل Regime — للتشخيص فقط."""
    print("\n" + "═" * 86)
    print("  Regime Configuration Summary — QuantSystem V19")
    print("═" * 86)
    header = f"  {'Setting':<24} {'Trending':>12} {'Ranging':>12} {'Volatile':>12} {'LowLiq':>14}"
    print(header)
    print("─" * 86)

    def _row(label: str, fmt) -> tuple:
        return (label,) + tuple(fmt(r) for r in REGIMES)

    rows = [
        _row("Event threshold", lambda r: str(REGIME_EVENT_THRESHOLD[r])),
        _row("TP multiplier",   lambda r: f"{REGIME_TP_SL[r][0]}× ATR"),
        _row("SL multiplier",   lambda r: f"{REGIME_TP_SL[r][1]}× ATR"),
        _row("Max horizon (bars)", lambda r: str(REGIME_MAX_BARS[r])),
        _row("CatBoost depth",  lambda r: str(REGIME_MODEL_CONFIGS[r]['depth'])),
        _row("CatBoost iters",  lambda r: str(REGIME_MODEL_CONFIGS[r]['iterations'])),
        _row("Learning rate",   lambda r: str(REGIME_MODEL_CONFIGS[r]['learning_rate'])),
        _row("Drift lambda_",   lambda r: str(REGIME_DRIFT_CONFIG[r]['lambda_'])),
        _row("Pred threshold",  lambda r: str(REGIME_PRED_THRESHOLD[r])),
    ]
    for row in rows:
        label, t, r, v, llq = row
        print(f"  {label:<24} {t:>12} {r:>12} {v:>12} {llq:>14}")
    print("═" * 86 + "\n")


if __name__ == '__main__':
    print_regime_summary()
