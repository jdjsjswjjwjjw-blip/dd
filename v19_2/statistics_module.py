"""
statistics.py — الإحصاء الصارم (Multi-testing + Permutation)
══════════════════════════════════════════════════════════════════════

يحوي الأدوات الإحصائية الصارمة التي تحلّ:
  ① Multiple testing بطريقة Benjamini-Hochberg (FDR)
     - أقوى من Bonferroni لأنه لا يعاقب التركيبات المترابطة كثيراً
     - يضبط expected proportion of false discoveries
     
  ② Permutation test
     - يبني توزيع null من إعادة خلط البيانات الفعلية
     - لا يعتمد على افتراض التوزيع الطبيعي
     - أكثر صرامة من t-test الكلاسيكي
     
  ③ Strict 3-way time split
     - train (60%) → اكتشاف
     - validation (20%) → ضبط ومعايرة
     - holdout (20%) → اختبار وحيد لا يتكرر

كل دالة هنا قابلة للاختبار unit-test.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════
# Benjamini-Hochberg FDR Control
# ══════════════════════════════════════════════════════════════════

def benjamini_hochberg(
    p_values: np.ndarray,
    fdr_alpha: float = 0.10,
) -> tuple[np.ndarray, float]:
    """
    Benjamini-Hochberg procedure للتحكم بـ False Discovery Rate.
    
    على عكس Bonferroni الذي يضبط FWER (false positive واحد كافٍ للرفض)،
    BH يضبط expected proportion من الـ discoveries التي قد تكون false.
    
    لـ N=100 اختبار و alpha=0.10:
      - Bonferroni: p < 0.001 (شديد، يضيع كثيراً)
      - BH: يقبل تقريباً أعلى p حتى p_(k) ≤ k/N × alpha
    
    Returns:
      rejected: array boolean — أي اختبار يُرفض H0 له
      threshold: أعلى p_value قُبل
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), 0.0
    
    # رتّب من الأصغر للأكبر
    order = np.argsort(p)
    p_sorted = p[order]
    
    # ابحث عن أكبر k حيث p_(k) ≤ k/n × alpha
    ranks = np.arange(1, n + 1)
    threshold_line = ranks / n * fdr_alpha
    below = p_sorted <= threshold_line
    
    if not below.any():
        rejected = np.zeros(n, dtype=bool)
        return rejected, 0.0
    
    # أكبر k يحقق الشرط
    k_max = np.max(np.where(below)[0])
    threshold_p = p_sorted[k_max]
    
    rejected = np.zeros(n, dtype=bool)
    rejected[order[:k_max + 1]] = True
    return rejected, float(threshold_p)


# ══════════════════════════════════════════════════════════════════
# Permutation Test
# ══════════════════════════════════════════════════════════════════

def permutation_test_edge(
    returns: np.ndarray,
    direction: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permutation test للـ edge — أصرم من t-test الكلاسيكي.
    
    الفكرة:
      الـ edge يدّعي أن returns في اتجاه معين معنوية.
      نبني توزيع null عبر random sign flipping (يحفظ الـ magnitude
      لكن يكسر الاتجاه).
    
      observed_stat = mean(returns × direction)
      null_distribution = [mean(returns × random_sign) for _ in range(N)]
      p_value = P(null >= observed)
    
    لا يفترض توزيع طبيعي، أكثر صرامة من t-test في وجود heavy tails.
    
    Returns:
      observed: القيمة المُلاحَظة
      p_value: احتمال أن يكون عشوائياً
      null_mean / null_std: للتشخيص
    """
    rng = np.random.default_rng(seed)
    
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 5:
        return {"observed": 0.0, "p_value": 1.0, "n": n, "valid": False}
    
    # الإحصاءة الملاحَظة
    signed = r * direction
    observed = float(signed.mean())
    
    # null distribution: نقلب علامات عشوائياً
    # هذا يحفظ توزيع |returns| لكن يكسر الاتجاه
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        random_signs = rng.choice([-1, 1], size=n)
        null_stats[i] = (r * random_signs).mean()
    
    # one-tailed p-value: P(null >= observed) لو observed > 0
    # أو P(null <= observed) لو observed < 0
    if observed > 0:
        p_value = float((null_stats >= observed).mean())
    else:
        p_value = float((null_stats <= observed).mean())
    
    return {
        "observed":   round(observed, 6),
        "p_value":    round(p_value, 4),
        "null_mean":  round(float(null_stats.mean()), 6),
        "null_std":   round(float(null_stats.std()), 6),
        "n":          n,
        "n_perm":     n_permutations,
        "valid":      True,
    }


# ══════════════════════════════════════════════════════════════════
# 3-Way Time Split
# ══════════════════════════════════════════════════════════════════

@dataclass
class TimeSplit:
    """تقسيم زمني صارم: train / validation / holdout."""
    
    train_idx:      np.ndarray
    validation_idx: np.ndarray
    holdout_idx:    np.ndarray
    purge_bars:     int
    
    n_train:        int
    n_validation:   int
    n_holdout:      int
    
    @property
    def total(self) -> int:
        return self.n_train + self.n_validation + self.n_holdout


def make_3way_split(
    n: int,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
    purge_bars: int = 12,
) -> TimeSplit:
    """
    تقسيم زمني صارم:
      [train 60%] [purge] [validation 20%] [purge] [holdout 20%]
    
    الـ purge بين كل قسمين يمنع تسريب fwd_ret.
    
    holdout يُمس مرة واحدة فقط في النهاية للتقييم النهائي.
    إذا اكتشفت نمطاً على train، تختبره على validation للضبط،
    ثم تختبره على holdout مرة واحدة فقط للحكم النهائي.
    """
    holdout_ratio = 1.0 - train_ratio - validation_ratio
    if holdout_ratio < 0.10:
        raise ValueError(f"holdout صغير جداً: {holdout_ratio:.2f}")
    
    train_end       = int(n * train_ratio)
    val_start       = train_end + purge_bars
    val_end         = val_start + int(n * validation_ratio)
    holdout_start   = val_end + purge_bars
    
    if holdout_start >= n:
        raise ValueError(
            f"البيانات قليلة جداً للـ 3-way split مع purge={purge_bars}. "
            f"تحتاج على الأقل {(purge_bars * 2 + 100)} صف، عندك {n}"
        )
    
    return TimeSplit(
        train_idx=     np.arange(0, train_end),
        validation_idx=np.arange(val_start, val_end),
        holdout_idx=   np.arange(holdout_start, n),
        purge_bars=    purge_bars,
        n_train=       train_end,
        n_validation=  val_end - val_start,
        n_holdout=     n - holdout_start,
    )


# ══════════════════════════════════════════════════════════════════
# Edge Statistics — كل المقاييس في مكان واحد
# ══════════════════════════════════════════════════════════════════

def edge_statistics(
    returns: np.ndarray,
    days:    np.ndarray,
    direction: int,
    pips_multiplier: float = 10000.0,
) -> dict:
    """
    يحسب كل الإحصاءات المتعلقة بـ edge على subset معطى.
    
    Args:
      returns: fwd_ret لكل إشارة (pct returns)
      days: array من dates لتجميع يومي
      direction: +1 (LONG) or -1 (SHORT)
      pips_multiplier: 10000 لـ FX
    
    Returns dict مع كل المقاييس.
    """
    r = np.asarray(returns, dtype=float)
    d = np.asarray(days)
    valid = ~np.isnan(r)
    r = r[valid]
    d = d[valid]
    
    n = len(r)
    if n < 5:
        return {"valid": False, "n": n}
    
    # تحويل لـ pips في اتجاه الـ edge
    dir_ret_pips = r * direction * pips_multiplier
    
    # الإحصاءات الأساسية
    wr = float((dir_ret_pips > 0).mean())
    avg_pips = float(dir_ret_pips.mean())
    median_pips = float(np.median(dir_ret_pips))
    std_pips = float(dir_ret_pips.std(ddof=1)) if n > 1 else 0.0
    
    # متوسطات يومية (لـ t-stat صحيح)
    df_daily = pd.DataFrame({"d": d, "r": dir_ret_pips}).groupby("d")["r"].agg(["mean", "count"])
    df_daily = df_daily[df_daily["count"] >= 2]  # على الأقل صفقتين لليوم
    daily_means = df_daily["mean"].values
    n_days = len(daily_means)
    
    if n_days < 3:
        return {"valid": False, "n": n, "n_days": n_days}
    
    # t-stat مبني على daily means (يحترم الـ autocorrelation داخل اليوم)
    mean_d = float(daily_means.mean())
    std_d = float(daily_means.std(ddof=1)) if n_days > 1 else 0.0
    t_stat = mean_d / (std_d / np.sqrt(n_days) + 1e-9) if std_d > 0 else 0.0
    
    # p-value من t-distribution
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1)))
    
    # Sharpe-like (annualized assuming 252 days)
    sharpe = (mean_d / (std_d + 1e-9)) * np.sqrt(252) if std_d > 0 else 0.0
    
    # Stability — نصفي الفترة
    mid = n_days // 2
    if mid >= 3:
        first = daily_means[:mid]
        second = daily_means[mid:]
        t_first = (first.mean() / (first.std(ddof=1) / np.sqrt(len(first)) + 1e-9)
                   if first.std(ddof=1) > 0 else 0.0)
        t_second = (second.mean() / (second.std(ddof=1) / np.sqrt(len(second)) + 1e-9)
                    if second.std(ddof=1) > 0 else 0.0)
        same_sign = np.sign(t_first) == np.sign(t_second)
        stable = bool(same_sign and abs(t_first) >= 1.0 and abs(t_second) >= 1.0)
    else:
        t_first = t_second = 0.0
        same_sign = stable = False
    
    return {
        "valid":     True,
        "n":         int(n),
        "n_days":    int(n_days),
        "wr":        round(wr, 4),
        "avg_pips":  round(avg_pips, 3),
        "med_pips":  round(median_pips, 3),
        "std_pips":  round(std_pips, 3),
        "t_stat":    round(float(t_stat), 3),
        "p_value":   round(float(p_value), 5),
        "sharpe":    round(float(sharpe), 2),
        "t_first":   round(float(t_first), 3),
        "t_second":  round(float(t_second), 3),
        "same_sign": bool(same_sign),
        "stable":    bool(stable),
    }


# ══════════════════════════════════════════════════════════════════
# Edge Validation — Full Pipeline
# ══════════════════════════════════════════════════════════════════

def validate_edge_pipeline(
    stats_train: dict,
    stats_validation: dict | None,
    stats_holdout: dict | None,
    criteria: dict,
    p_value_train: float | None = None,
) -> dict:
    """
    التحقّق الكامل من edge عبر 3 مراحل.
    
    المنطق:
      ① train: اكتشاف — يجب أن يجتاز معايير IS
      ② validation: ضبط/فلترة — نفس الإشارة + لا يتدهور كثيراً
      ③ holdout: حكم نهائي — نفس الإشارة + معنوي
    
    Returns dict مع:
      passed: bool — هل اجتاز كل المراحل
      stage_failed: أي مرحلة فشل فيها (None لو نجح)
      reasons: قائمة بأسباب الفشل
    """
    reasons = []
    
    # ── ① Train ──
    if not stats_train.get("valid", False):
        return {"passed": False, "stage_failed": "train", "reasons": ["train invalid"]}
    
    if abs(stats_train["t_stat"]) < criteria["min_t_stat"]:
        reasons.append(f"train t={stats_train['t_stat']:.2f} < {criteria['min_t_stat']}")
    if stats_train["n"] < criteria["min_n"]:
        reasons.append(f"train n={stats_train['n']} < {criteria['min_n']}")
    if stats_train["n_days"] < criteria["min_days_recur"]:
        reasons.append(f"train days={stats_train['n_days']} < {criteria['min_days_recur']}")
    if stats_train["wr"] < criteria["min_wr"] and stats_train["avg_pips"] < criteria["min_avg_pips"]:
        reasons.append(f"train WR={stats_train['wr']:.0%} & avg={stats_train['avg_pips']:.1f}p غير كافيين")
    if criteria.get("stability_check", True) and not stats_train.get("stable", False):
        reasons.append("train غير مستقر (نصفي الفترة)")
    
    if reasons:
        return {"passed": False, "stage_failed": "train", "reasons": reasons}
    
    # ── ② Validation ──
    if stats_validation is None or not stats_validation.get("valid", False):
        return {"passed": False, "stage_failed": "validation", 
                "reasons": ["لا توجد validation"]}
    
    if stats_validation["n"] < criteria.get("min_n_test", 5):
        reasons.append(f"validation n={stats_validation['n']} قليل")
    if np.sign(stats_validation["avg_pips"]) != np.sign(stats_train["avg_pips"]):
        reasons.append("validation عكس الإشارة")
    if abs(stats_validation["t_stat"]) < criteria["min_t_oos"]:
        reasons.append(f"validation t={stats_validation['t_stat']:.2f} < {criteria['min_t_oos']}")
    
    if reasons:
        return {"passed": False, "stage_failed": "validation", "reasons": reasons}
    
    # ── ③ Holdout (الحكم النهائي) ──
    if stats_holdout is None or not stats_holdout.get("valid", False):
        return {"passed": False, "stage_failed": "holdout",
                "reasons": ["لا توجد holdout"]}
    
    if stats_holdout["n"] < criteria.get("min_n_test", 5):
        reasons.append(f"holdout n={stats_holdout['n']} قليل")
    if np.sign(stats_holdout["avg_pips"]) != np.sign(stats_train["avg_pips"]):
        reasons.append("holdout عكس الإشارة")
    if abs(stats_holdout["t_stat"]) < criteria["min_t_oos"] * 0.8:  # holdout أخف قليلاً
        reasons.append(f"holdout t={stats_holdout['t_stat']:.2f} ضعيف")
    
    # degradation: من train إلى holdout
    avg_tr = abs(stats_train["avg_pips"])
    avg_ho = abs(stats_holdout["avg_pips"])
    degradation = (avg_tr - avg_ho) / (avg_tr + 1e-9)
    if degradation > criteria.get("max_degradation", 0.60):
        reasons.append(f"holdout degradation={degradation:.0%} > {criteria['max_degradation']:.0%}")
    
    if reasons:
        return {"passed": False, "stage_failed": "holdout", "reasons": reasons,
                "degradation": round(degradation, 3)}
    
    return {
        "passed": True,
        "stage_failed": None,
        "reasons": [],
        "degradation": round(degradation, 3),
    }
