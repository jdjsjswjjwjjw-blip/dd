"""
market_specs.py — مواصفات السوق (Single Source of Truth)
══════════════════════════════════════════════════════════════════════

يحوي كل الافتراضات المتعلقة بسوق محدد في مكان واحد.
لا يتم hardcode أي رقم سوق في باقي الملفات — كلها تُقرأ من هنا.

عند تغيير السوق (من 6B إلى ES مثلاً)، عدّل هذا الملف فقط.

كل قيمة موثّقة بمصدرها:
  - بعضها قياس من الداتا الفعلية (measured)
  - بعضها معيار صناعة (industry standard)
  - بعضها يحتاج معايرة على 6 أشهر (calibrated)

⚠️ تحذير علمي:
   الأرقام أدناه افتراضات يجب تأكيدها على الداتا الفعلية.
   لا تستخدمها في الإنتاج بدون OOS validation على 6+ أشهر.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MarketSpec:
    """مواصفات سوق محدد — قيم مُقاسة أو من مصادر موثّقة."""
    
    symbol: str
    description: str
    
    # ── معايير العقد (من CME spec) ──
    tick_size: float          # أصغر حركة سعرية
    tick_value: float         # قيمة tick بالدولار
    contract_size: int        # حجم العقد
    
    # ── سيولة (measured: لازم تأكيد من داتا) ──
    typical_daily_volume: int      # متوسط الحجم اليومي (contracts)
    typical_spread_ticks: float    # السبريد المعتاد بـ ticks
    typical_atr_daily_pips: float  # ATR يومي (للسياق)
    
    # ── transaction costs (مهم للـ edge بعد التكاليف) ──
    commission_per_contract: float  # يحدّده الوسيط
    typical_slippage_ticks: float   # slippage متوسط (محافظ)
    
    # ── جلسات نشطة (UTC) — منطقية لـ FX/index futures ──
    asia_session:   tuple[str, str] = ("00:00", "07:00")
    london_session: tuple[str, str] = ("07:00", "13:00")
    ny_session:     tuple[str, str] = ("16:00", "20:00")
    overlap_asia_london: tuple[str, str] = ("07:00", "08:00")
    overlap_london_ny:   tuple[str, str] = ("13:00", "16:00")
    
    def round_trip_cost_pips(self) -> float:
        """تكلفة الذهاب والإياب بالـ pips (commission + 2× slippage)."""
        # كل tick = 1 pip في 6B
        slippage_cost = self.typical_slippage_ticks * 2  # دخول + خروج
        commission_pips = self.commission_per_contract / self.tick_value * 2
        return slippage_cost + commission_pips
    
    def min_profitable_pips(self) -> float:
        """الحد الأدنى لـ edge ليكون ربحاً صافياً بعد التكاليف."""
        return self.round_trip_cost_pips() * 1.5  # 1.5× التكلفة كحد أدنى


# ══════════════════════════════════════════════════════════════════
# مواصفات 6B (GBP/USD Futures)
# ══════════════════════════════════════════════════════════════════
# المصادر:
#   - CME spec: https://www.cmegroup.com/markets/fx/g10/british-pound.contractSpecs.html
#   - typical_daily_volume: قياس تقريبي من CME daily reports
#   - typical_spread/slippage: قياس من الداتا الخام (يحتاج تأكيد)

SPEC_6B = MarketSpec(
    symbol="6B",
    description="GBP/USD Futures (CME)",
    
    # CME spec (موثّق):
    tick_size=0.0001,           # 1 pip
    tick_value=6.25,            # $6.25 per tick
    contract_size=62_500,       # £62,500 per contract
    
    # measured (يحتاج تأكيد):
    typical_daily_volume=70_000,    # ~60-80K يومياً
    typical_spread_ticks=1.0,       # 1 tick في الساعات النشطة
    typical_atr_daily_pips=80.0,    # 60-100 pips
    
    # محافظ (يحدّده الوسيط):
    commission_per_contract=2.50,   # round-trip
    typical_slippage_ticks=0.5,     # نصف tick لكل صفقة
)

# ══════════════════════════════════════════════════════════════════
# الافتراضي
# ══════════════════════════════════════════════════════════════════

DEFAULT_MARKET = SPEC_6B


def get_market_spec(symbol: str = "6B") -> MarketSpec:
    """يرجع مواصفات السوق المطلوب."""
    specs = {"6B": SPEC_6B}
    if symbol not in specs:
        raise ValueError(
            f"غير معروف: {symbol}. المتاح: {list(specs.keys())}. "
            f"أضف مواصفات السوق في market_specs.py"
        )
    return specs[symbol]


# ══════════════════════════════════════════════════════════════════
# معايير الـ Edge (مُشتقة من السوق، ليست hardcoded)
# ══════════════════════════════════════════════════════════════════

def derive_edge_criteria(
    spec: MarketSpec,
    n_train_days: int,
    confidence: Literal["loose", "medium", "strict"] = "medium",
) -> dict:
    """
    يشتق معايير القبول من مواصفات السوق وحجم الداتا.
    
    المنطق:
      - min_avg_pips يجب أن يتجاوز round_trip_cost (وإلا لا ربح)
      - min_n مرتبط بحجم الداتا (لا منطقي 25 على شهر)
      - min_days_recur ~ربع أيام train
      - min_t_stat من toleranace الـ false positive
    """
    cost = spec.round_trip_cost_pips()
    
    # ① عتبات الـ t-stat حسب الثقة المطلوبة
    t_thresholds = {
        "loose":  {"is": 1.8, "oos": 0.8},   # استكشاف
        "medium": {"is": 2.0, "oos": 1.0},   # production تقليدي
        "strict": {"is": 2.5, "oos": 1.5},   # مؤسسي
    }
    t = t_thresholds[confidence]
    
    # ② حجم العينة (يتسق مع حجم البيانات)
    min_n_total = max(15, n_train_days * 2)   # 2 إشارات/يوم على الأقل
    min_days = max(5, int(n_train_days * 0.25))  # ربع أيام train
    
    # ③ العتبات النهائية
    return {
        # إحصائية
        "min_t_stat":      t["is"],
        "min_t_oos":       t["oos"],
        "min_n":           min_n_total,
        "min_days_recur":  min_days,
        "min_n_test":      max(5, min_n_total // 4),
        
        # أداء (مُشتق من تكاليف التنفيذ)
        "min_wr":          0.53,                          # > 50% بهامش
        "min_avg_pips":    max(2.5, cost * 1.2),          # 1.2× التكلفة
        "min_profit_after_cost": cost * 0.5,              # ربح صافٍ بعد التكلفة
        
        # OOS validation (الحارس الأهم)
        "max_degradation": 0.60,
        "stability_check": True,
        
        # FDR (يستبدل Bonferroni)
        "fdr_alpha":       0.10,    # 10% expected false discoveries
        "use_fdr":         True,
        "use_permutation": True,    # إضافة permutation test
        
        # holdout صارم
        "holdout_ratio":   0.20,    # آخر 20% لا تُمس
        "train_ratio":     0.60,    # الـ 60% الأولى للاكتشاف
        "validation_ratio":0.20,    # 20% بين train و holdout للضبط
        
        # transaction costs
        "round_trip_cost": cost,
        "min_profitable":  spec.min_profitable_pips(),
    }
