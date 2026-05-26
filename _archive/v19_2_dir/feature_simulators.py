"""
feature_simulators.py — محاكيات عمق السوق (Market Depth Simulators)
══════════════════════════════════════════════════════════════════════

النطاق: عمق السوق فقط (MBO + MBP).
  لا مستويات سعرية، لا PDH/PDL — تلك تُمرَّر كـ context مباشر للنموذج.

الفكرة:
  المحاكي يفهم *إشارة عمق غامضة* ويحوّلها لرقم نظيف مستقر.
  الإشارة الواضحة (قمة لندن = رقم) لا تحتاج محاكياً — تُمرَّر مباشرة.

المحاكيات الخمسة (كلها عمق سوق):
  ① Absorption  — عمق يُمتص بدون حركة سعر
  ② Order Flow  — تدفق الأوامر التراكمي الحقيقي
  ③ Informed    — طرف مؤسسي يتحرك في الدفتر
  ④ Wall        — حوائط حقيقية (iceberg) vs وهمية (spoofing)
  ⑤ Liquidity   — حالة سيولة الدفتر وجودة الداتا

مترابطة:
  ④ Wall يستخدم مخرج ① و ②
  ⑤ Liquidity مستقل لكن يُكمّل البقية

كل محاكي adaptive (rank/zscore) + causal (لا lookahead).

الاستخدام:
  from feature_simulators import run_all_simulators, get_model_input_columns
  df = run_all_simulators(df)

CLI:
  python feature_simulators.py --input clean.parquet --output sim.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# أدوات Normalization التكيفي
# ══════════════════════════════════════════════════════════════════

def _rank(s: pd.Series, window: int = 200) -> pd.Series:
    """Percentile rank متحرك ∈ [0,1] — مستقل عن الـ scale."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s.rolling(window, min_periods=max(2, min(window // 10, window))).rank(pct=True).fillna(0.5)


def _zscore(s: pd.Series, window: int = 200) -> pd.Series:
    """Rolling z-score ∈ [-4,4]."""
    s  = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mu = s.rolling(window, min_periods=max(2, min(window // 10, window))).mean()
    sd = s.rolling(window, min_periods=max(2, min(window // 10, window))).std().clip(lower=1e-9)
    return ((s - mu) / sd).clip(-4, 4).fillna(0.0)


def _safe(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """قراءة عمود بأمان."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=np.float64)


def _sign3(s: pd.Series, dead_zone: float = 0.05) -> pd.Series:
    """تحويل لـ -1/0/+1 مع منطقة محايدة."""
    return pd.Series(
        np.where(s > dead_zone, 1.0, np.where(s < -dead_zone, -1.0, 0.0)),
        index=s.index, dtype=np.float64,
    )


def _persistence(condition: pd.Series, window: int = 6) -> pd.Series:
    """عدد مرات تحقق الشرط في آخر window شمعة."""
    return condition.astype(float).rolling(window, min_periods=1).sum()


def _bar_cvd_delta(df: pd.DataFrame) -> pd.Series:
    """فرق CVD داخل الشمعة — يستخدم bar_cvd_delta أو cvd.diff كبديل."""
    d = _safe(df, "bar_cvd_delta")
    if d.std() < 1e-9:
        d = _safe(df, "cvd").diff().fillna(0)
    return d


# ══════════════════════════════════════════════════════════════════
# المحاكي ① — Absorption Engine (عمق يُمتص)
# ══════════════════════════════════════════════════════════════════

def simulate_absorption(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    عمق السوق يُمتص بدون أن يتحرك السعر.

    المبدأ الفيزيائي:
      امتصاص = حجم كبير ÷ حركة سعر صغيرة
      طرف كبير يبتلع العرض/الطلب في مستوى ثابت

    المدخل (عمق فقط):
      absorption_intensity, volume_burst, volume,
      bar_cvd_delta, bid_wall_strength, ask_wall_strength,
      book_imbalance, high/low (للحركة فقط)

    المخرج (4D):
      sim_absorb_intensity  ← قوة الامتصاص [0,1]
      sim_absorb_direction  ← +1 امتصاص شرائي / -1 بيعي
      sim_absorb_persist    ← استمرارية الامتصاص [0,1]
      sim_absorb_at_wall    ← امتصاص يحدث عند حائط؟ [0,1]
    """
    out = df.copy()

    high = _safe(out, "high"); low = _safe(out, "low")
    bar_range = (high - low).abs()
    avg_range = bar_range.rolling(window, min_periods=10).median().clip(lower=1e-9)
    range_norm = (bar_range / avg_range).clip(0, 5)

    vol_burst = _safe(out, "volume_burst")
    if vol_burst.std() < 1e-9:
        vol_burst = _safe(out, "volume")
    vol_norm = _rank(vol_burst, window)

    absorb_raw = _safe(out, "absorption_intensity")

    # النسبة الفيزيائية: حجم ÷ حركة
    phys_absorb = vol_norm / (range_norm + 0.20)
    if absorb_raw.std() > 1e-6:
        absorb_combined = 0.6 * _rank(phys_absorb, window) + 0.4 * _rank(absorb_raw, window)
    else:
        absorb_combined = _rank(phys_absorb, window)
    out["sim_absorb_intensity"] = absorb_combined.clip(0, 1).astype(np.float32)

    # اتجاه الامتصاص — من CVD delta + book imbalance
    cvd_delta = _bar_cvd_delta(out)
    book_imb  = _safe(out, "book_imbalance")
    if book_imb.std() < 1e-9:
        bid_w = _safe(out, "bid_wall_strength")
        ask_w = _safe(out, "ask_wall_strength")
        book_imb = _zscore(ask_w - bid_w, window) / 3.0
    dir_score = _sign3(_zscore(cvd_delta, window)) * 0.6 + _sign3(book_imb) * 0.4
    out["sim_absorb_direction"] = _sign3(dir_score, dead_zone=0.10).astype(np.float32)

    # الاستمرارية
    strong = out["sim_absorb_intensity"] > 0.70
    out["sim_absorb_persist"] = (_persistence(strong, 6) / 6.0).clip(0, 1).astype(np.float32)

    # امتصاص عند حائط؟
    bid_w = _safe(out, "bid_wall_strength")
    ask_w = _safe(out, "ask_wall_strength")
    wall_max = pd.concat([bid_w, ask_w], axis=1).max(axis=1)
    wall_strong = _rank(wall_max, window) > 0.65
    out["sim_absorb_at_wall"] = (
        ((out["sim_absorb_intensity"] > 0.60) & wall_strong).astype(np.float32)
    )

    return out


# ══════════════════════════════════════════════════════════════════
# المحاكي ② — Order Flow Engine (تدفق الأوامر)
# ══════════════════════════════════════════════════════════════════

def simulate_order_flow(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    تدفق الأوامر التراكمي الحقيقي — هل دفع متراكم في اتجاه واحد؟

    المبدأ:
      CVD يتراكم + OBI يتفق + VNET موجب = ضغط حقيقي
      التوافق = إشارة | التضارب = ضجيج

    المدخل:
      cvd, session_cvd, bar_cvd_delta, obi, vnet,
      vnet_intrabar_last, cvd_momentum, cvd_price_divergence

    المخرج (4D):
      sim_flow_strength      ← قوة الضغط [0,1]
      sim_flow_direction     ← +1/-1/0
      sim_flow_consistency   ← توافق المصادر [0,1]
      sim_flow_divergence    ← CVD يخالف السعر؟ [0,1]
    """
    out = df.copy()

    cvd_delta = _bar_cvd_delta(out)
    obi       = _safe(out, "obi")
    vnet      = _safe(out, "vnet")
    cvd_mom   = _safe(out, "cvd_momentum")
    cvd_div   = _safe(out, "cvd_price_divergence")

    cvd_z  = (_zscore(cvd_delta, window) / 3.0).clip(-1, 1)
    obi_n  = obi.clip(-1, 1)
    vnet_z = (_zscore(vnet, window) / 3.0).clip(-1, 1)
    mom_n  = _sign3(cvd_mom) * _rank(cvd_mom.abs(), window)

    flow_score = 0.35 * cvd_z + 0.30 * obi_n + 0.20 * vnet_z + 0.15 * mom_n
    out["sim_flow_direction"] = _sign3(flow_score, dead_zone=0.10).astype(np.float32)
    out["sim_flow_strength"]  = _rank(flow_score.abs(), window).clip(0, 1).astype(np.float32)

    # consistency
    signs = pd.concat([_sign3(cvd_z), _sign3(obi_n), _sign3(vnet_z)], axis=1)
    majority = _sign3(flow_score)
    agree = signs.apply(lambda r: (r == majority.loc[r.name]).sum(), axis=1) / 3.0
    out["sim_flow_consistency"] = agree.clip(0, 1).fillna(0.33).astype(np.float32)

    # divergence
    if cvd_div.std() > 1e-6:
        out["sim_flow_divergence"] = _rank(cvd_div.abs(), window).clip(0, 1).astype(np.float32)
    else:
        close = _safe(out, "close")
        price_dir = _sign3(close.diff(3))
        cvd_dir   = _sign3(cvd_delta.rolling(3, min_periods=1).sum())
        diverge   = (price_dir != cvd_dir) & (price_dir != 0) & (cvd_dir != 0)
        out["sim_flow_divergence"] = (_persistence(diverge, 4) / 4.0).clip(0, 1).astype(np.float32)

    return out


# ══════════════════════════════════════════════════════════════════
# المحاكي ③ — Informed Flow Engine (طرف مؤسسي)
# ══════════════════════════════════════════════════════════════════

def simulate_informed_flow(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    يكشف وجود طرف مؤسسي في الدفتر = تجمع أوامر غير عادي.

    المبدأ:
      hawkes عالٍ = أوامر تتسارع (عدوى)
      kyle عالٍ   = السعر يستجيب بقوة للحجم
      inter_event منخفض = أوامر سريعة متلاحقة
      anomaly عالٍ = شذوذ في الدفتر

    المدخل:
      hawkes_intrabar_sum / hawkes_intensity,
      kyle_lambda_intrabar_mean / kyle_lambda,
      inter_event_time, cancel_ratio, fisher_signal,
      anomaly, liquidity_sweep

    المخرج (4D):
      sim_informed_prob      ← احتمال طرف مؤسسي [0,1]
      sim_informed_urgency   ← استعجال الأوامر [0,1]
      sim_informed_direction ← اتجاهه (+1/-1/0)
      sim_sweep_signal       ← هل حدث ضرب سيولة؟ [0,1]
    """
    out = df.copy()

    hawkes = _safe(out, "hawkes_intrabar_sum")
    if hawkes.std() < 1e-9:
        hawkes = _safe(out, "hawkes_intensity")

    kyle = _safe(out, "kyle_lambda_intrabar_mean")
    if kyle.std() < 1e-9:
        kyle = _safe(out, "kyle_lambda")

    iet     = _safe(out, "inter_event_time")
    cancel  = _safe(out, "cancel_ratio")
    fisher  = _safe(out, "fisher_signal")
    anomaly = _safe(out, "anomaly")
    sweep   = _safe(out, "liquidity_sweep")

    hawkes_r  = _rank(hawkes, window)
    kyle_r    = _rank(kyle, window)
    iet_r     = 1.0 - _rank(iet, window)
    anomaly_r = _rank(anomaly, window)

    informed = 0.35 * hawkes_r + 0.30 * kyle_r + 0.20 * iet_r + 0.15 * anomaly_r
    out["sim_informed_prob"] = informed.clip(0, 1).astype(np.float32)

    cancel_inv = 1.0 - _rank(cancel, window)
    out["sim_informed_urgency"] = (0.6 * iet_r + 0.4 * cancel_inv).clip(0, 1).astype(np.float32)

    cvd_delta = _bar_cvd_delta(out)
    dir_score = _sign3(fisher) * 0.5 + _sign3(_zscore(cvd_delta, window)) * 0.5
    out["sim_informed_direction"] = _sign3(dir_score, dead_zone=0.10).astype(np.float32)

    if sweep.std() > 1e-6:
        out["sim_sweep_signal"] = _rank(sweep.abs(), window).clip(0, 1).astype(np.float32)
    else:
        out["sim_sweep_signal"] = np.float32(0.0)

    return out


# ══════════════════════════════════════════════════════════════════
# المحاكي ④ — Wall Dynamics Engine (مترابط مع ①②)
# ══════════════════════════════════════════════════════════════════

def simulate_wall_dynamics(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    يفهم الحوائط = حقيقية (iceberg) أم وهمية (spoofing)؟
    مترابط: يستخدم مخرج Absorption + OrderFlow.

    المبدأ المُكتشَف من الداتا (6BM5):
      الجدران تُحلَّل الآن في wall_depth_simulator.py من MBP الخام.
      هذا المحاكي يبقى مسؤولاً عن *ضغط العمق* فقط (إشارة قوية t=3.99).

    المدخل:
      book_imbalance, depth_ratio, distance_to_wall,
      bid/ask_wall_strength (للـ imbalance فقط)
      + sim_absorb_intensity, sim_flow_consistency (مترابط)

    المخرج (3D):
      sim_wall_real          ← ثقة إشارة العمق [0,1]
      sim_depth_imbalance    ← اختلال العمق المُصفّى [-1,1]
      sim_depth_pressure     ← ضغط العمق الكلي [-1,1]  ← الأقوى

    ملاحظة: sim_wall_direction / sim_wall_strength أُزيلا.
      بدائلهما الأقوى في wall_depth_simulator.py:
      sim_wall_growth / consumed / persist / shift / bid_level / ...
    """
    out = df.copy()

    bid_w   = _safe(out, "bid_wall_strength")
    ask_w   = _safe(out, "ask_wall_strength")
    spoof   = _safe(out, "spoofing_ratio")
    spoof_d = _safe(out, "spoofing_duration")
    trap    = _safe(out, "liquidity_trap")
    book_imb= _safe(out, "book_imbalance")
    depth_r = _safe(out, "depth_ratio", 1.0)
    dist_w  = _safe(out, "distance_to_wall")

    absorb_int = _safe(out, "sim_absorb_intensity")
    flow_cons  = _safe(out, "sim_flow_consistency", 0.33)

    wall_imb = ask_w - bid_w

    # ثقة إشارة العمق (يبقى — يُستخدم لتصفية depth_imbalance)
    fake = (
        0.45 * _rank(spoof, window) +
        0.30 * _rank(spoof_d, window) +
        0.25 * _rank(trap, window)
    )
    real = 0.5 * absorb_int + 0.5 * flow_cons
    out["sim_wall_real"] = (real * (1.0 - fake)).clip(0, 1).astype(np.float32)

    # depth imbalance
    if book_imb.std() > 1e-6:
        depth_imb = book_imb.clip(-1, 1)
    else:
        depth_imb = (_zscore(wall_imb, window) / 3.0).clip(-1, 1)
    out["sim_depth_imbalance"] = (
        depth_imb * out["sim_wall_real"]
    ).clip(-1, 1).astype(np.float32)

    # depth pressure — الإشارة الأقوى (t=3.99)
    depth_r_n  = (_zscore(np.log(depth_r.clip(0.1, 10)), window) / 3.0).clip(-1, 1)
    dist_close = 1.0 - _rank(dist_w, window)
    wall_dir_proxy = _sign3(_zscore(wall_imb, window), dead_zone=0.15)
    pressure = (
        0.5 * depth_imb +
        0.3 * depth_r_n +
        0.2 * (dist_close * wall_dir_proxy)
    )
    out["sim_depth_pressure"] = pressure.clip(-1, 1).astype(np.float32)

    return out


# ══════════════════════════════════════════════════════════════════
# المحاكي ⑤ — Liquidity Engine (حالة سيولة الدفتر)
# ══════════════════════════════════════════════════════════════════

def simulate_liquidity(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    يقيس حالة سيولة الدفتر وجودة الداتا.

    المبدأ:
      دفتر سائل = عمق كثيف + spread ضيق + تيكات كثيرة
      دفتر جاف  = عمق رقيق + spread واسع + تيكات قليلة
      جودة الداتا = هل التغطية كافية للثقة بالإشارة؟

    المدخل:
      micro_atr, liquidity_density, gap_size, tick_count,
      volume_burst, spread_mean, mbo_bar_coverage, mbp_bar_coverage

    المخرج (3D):
      sim_liquidity_state    ← سائل [1] / جاف [0]
      sim_volatility_regime  ← هادئ [0] / متفجر [1]
      sim_data_quality       ← موثوقية الإشارة [0,1]
    """
    out = df.copy()

    micro_atr = _safe(out, "micro_atr")
    liq_dens  = _safe(out, "liquidity_density")
    gap       = _safe(out, "gap_size")
    tick_ct   = _safe(out, "tick_count")
    vol_burst = _safe(out, "volume_burst")
    spread    = _safe(out, "spread_mean")
    mbo_cov   = _safe(out, "mbo_bar_coverage")
    mbp_cov   = _safe(out, "mbp_bar_coverage")

    dens_r   = _rank(liq_dens, window)
    tick_r   = _rank(tick_ct, window)
    spread_inv = (1.0 - _rank(spread, window)) if spread.std() > 1e-6 \
                 else pd.Series(0.5, index=out.index)
    liquidity = 0.4 * dens_r + 0.35 * tick_r + 0.25 * spread_inv
    out["sim_liquidity_state"] = liquidity.clip(0, 1).astype(np.float32)

    atr_r   = _rank(micro_atr, window)
    burst_r = _rank(vol_burst, window)
    gap_r   = _rank(gap, window)
    volat   = 0.5 * atr_r + 0.3 * burst_r + 0.2 * gap_r
    out["sim_volatility_regime"] = volat.clip(0, 1).astype(np.float32)

    if mbp_cov.std() > 1e-6:
        quality = 0.5 * mbo_cov.clip(0, 1) + 0.5 * mbp_cov.clip(0, 1)
    else:
        quality = mbo_cov.clip(0, 1)
    out["sim_data_quality"] = quality.clip(0, 1).astype(np.float32)

    return out


# ══════════════════════════════════════════════════════════════════
# المُشغّل
# ══════════════════════════════════════════════════════════════════

SIMULATOR_OUTPUTS = [
    # Absorption (4)
    "sim_absorb_intensity", "sim_absorb_direction",
    "sim_absorb_persist", "sim_absorb_at_wall",
    # Order Flow (4)
    "sim_flow_strength", "sim_flow_direction",
    "sim_flow_consistency", "sim_flow_divergence",
    # Informed (4)
    "sim_informed_prob", "sim_informed_urgency",
    "sim_informed_direction", "sim_sweep_signal",
    # Depth Engine (3) — اتجاه/قوة الجدران الآن في wall_depth_simulator.py
    "sim_wall_real", "sim_depth_imbalance", "sim_depth_pressure",
    # Liquidity (3)
    "sim_liquidity_state", "sim_volatility_regime", "sim_data_quality",
]

# مخرجات محاكي الجدران العميق — تُضاف بعد wall_depth_simulator.simulate_wall_depth
WALL_DEPTH_OUTPUTS = [
    "sim_wall_bid_level", "sim_wall_ask_level",
    "sim_wall_bid_size", "sim_wall_ask_size",
    "sim_wall_growth", "sim_wall_consumed",
    "sim_wall_persist", "sim_wall_shift",
]

# السياق المباشر — مستويات ووقت (بلا محاكي)
CONTEXT_PASSTHROUGH = [
    "dist_to_pdh", "price_position",
    "london_sess_high", "london_sess_low",
    "dist_to_london_high_atr", "dist_to_london_low_atr",
    "is_london", "is_overlap", "is_ny",
    "regime_label",
]


def run_all_simulators(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    يشغّل محاكيات عمق السوق الخمسة بالترتيب الصحيح:

      ① Absorption  ← مستقل
      ② OrderFlow   ← مستقل
      ③ Informed    ← مستقل
      ④ Wall        ← يحتاج مخرج ① و ②
      ⑤ Liquidity   ← مستقل

    يُرجع df + 20 عمود sim_*.
    الـ52 الخام تبقى للتشخيص؛ النموذج يقرأ sim_* + context فقط.
    """
    df = df.sort_values("ts_event").reset_index(drop=True) if "ts_event" in df.columns else df.copy()

    print("🔮 تشغيل محاكيات عمق السوق...")

    df = simulate_absorption(df, window=window)
    print("  ✅ ① Absorption Engine")

    df = simulate_order_flow(df, window=window)
    print("  ✅ ② Order Flow Engine")

    df = simulate_informed_flow(df, window=window)
    print("  ✅ ③ Informed Flow Engine")

    df = simulate_wall_dynamics(df, window=window)
    print("  ✅ ④ Wall Dynamics Engine (مترابط ①②)")

    df = simulate_liquidity(df, window=window)
    print("  ✅ ⑤ Liquidity Engine")

    missing = [c for c in SIMULATOR_OUTPUTS if c not in df.columns]
    if missing:
        print(f"  ⚠️ مخرجات ناقصة: {missing}")
        for c in missing:
            df[c] = np.float32(0.0)

    print(f"  ✅ {len(SIMULATOR_OUTPUTS)} مخرج عمق سوق جاهز")
    return df


def get_model_input_columns(df: pd.DataFrame) -> list[str]:
    """
    الأعمدة التي يقرأها النموذج:
      19 محاكي عمق + 8 محاكي جدران (إن وُجدت) + السياق المتاح.
    """
    cols = list(SIMULATOR_OUTPUTS)
    # أضف مخرجات الجدران العميقة لو حُسبت
    cols += [c for c in WALL_DEPTH_OUTPUTS if c in df.columns]
    cols += [c for c in CONTEXT_PASSTHROUGH if c in df.columns]
    return cols


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Market Depth Simulators — Layer 1")
    ap.add_argument("--input",  required=True, help="parquet المدخل")
    ap.add_argument("--output", default=None,  help="parquet المخرج")
    ap.add_argument("--window", type=int, default=200, help="نافذة normalization")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"📥 {len(df):,} rows | {len(df.columns)} cols")

    df = run_all_simulators(df, window=args.window)

    out_path = args.output or str(
        Path(args.input).with_name(Path(args.input).stem + "_sim.parquet")
    )
    df.to_parquet(out_path, index=False)
    print(f"\n💾 محفوظ: {out_path}")

    mc = get_model_input_columns(df)
    print(f"\n📋 النموذج يقرأ {len(mc)} عمود = 20 محاكي + {len(mc)-20} سياق")


if __name__ == "__main__":
    main()
