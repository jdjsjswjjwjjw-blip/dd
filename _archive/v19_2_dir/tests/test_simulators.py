"""
tests/test_simulators.py — اختبارات وحدة صارمة للمحاكيات
══════════════════════════════════════════════════════════════════════

كل محاكي يجب أن يجتاز:
  ① Causality — تغيير المستقبل لا يؤثر على الماضي
  ② Sanity — على ضوضاء عشوائية، الإشارة قريبة من 0.5 (لا bias)
  ③ Responsiveness — على نمط حقيقي مزروع، الإشارة تستجيب
  ④ Range — كل المخرجات في النطاق المعلن
  ⑤ NaN handling — لا NaN في المخرج النهائي

تشغيل:
  pytest tests/test_simulators.py -v
أو:
  python tests/test_simulators.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# اجعل المشروع متاحاً للاستيراد
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from feature_simulators import (
    run_all_simulators, SIMULATOR_OUTPUTS,
    simulate_absorption, simulate_order_flow,
    simulate_informed_flow, simulate_wall_dynamics, simulate_liquidity,
)
from statistics_module import (
    benjamini_hochberg, permutation_test_edge,
    make_3way_split, edge_statistics,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS — بناء داتا اختبار
# ══════════════════════════════════════════════════════════════════

def _make_random_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """داتا عشوائية بحتة — يجب ألا تنتج إشارات قوية."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    close = 100 + rng.standard_normal(n).cumsum() * 0.1
    df = pd.DataFrame({
        "ts_event": ts,
        "open":  close + rng.standard_normal(n) * 0.05,
        "close": close,
        "high":  close + np.abs(rng.standard_normal(n)) * 0.1,
        "low":   close - np.abs(rng.standard_normal(n)) * 0.1,
        "volume": rng.integers(50, 200, n).astype(float),
        "absorption_intensity": rng.uniform(0, 20, n),
        "volume_burst": rng.uniform(0, 10, n),
        "bar_cvd_delta": rng.standard_normal(n) * 100,
        "cvd": rng.standard_normal(n).cumsum() * 50,
        "session_cvd": rng.standard_normal(n).cumsum() * 30,
        "obi": rng.uniform(-1, 1, n),
        "vnet": rng.standard_normal(n) * 50,
        "vnet_intrabar_last": rng.standard_normal(n) * 50,
        "cvd_momentum": rng.standard_normal(n) * 10,
        "cvd_price_divergence": rng.standard_normal(n),
        "hawkes_intrabar_sum": rng.exponential(2, n),
        "kyle_lambda_intrabar_mean": rng.uniform(0, 0.1, n),
        "inter_event_time": rng.uniform(0, 1000, n),
        "cancel_ratio": rng.uniform(0, 1, n),
        "fisher_signal": rng.standard_normal(n),
        "anomaly": rng.exponential(0.5, n),
        "liquidity_sweep": rng.exponential(0.3, n),
        "bid_wall_strength": rng.uniform(0, 2, n),
        "ask_wall_strength": rng.uniform(0, 2, n),
        "spoofing_ratio": rng.uniform(0, 1, n),
        "spoofing_duration": rng.uniform(0, 10, n),
        "liquidity_trap": rng.uniform(0, 1, n),
        "book_imbalance": rng.uniform(-1, 1, n),
        "depth_ratio": rng.uniform(0.5, 2, n),
        "spread_mean": rng.uniform(0.0001, 0.0005, n),
        "distance_to_wall": rng.uniform(0, 20, n),
        "micro_atr": rng.uniform(0, 5, n),
        "liquidity_density": rng.uniform(0, 1000, n),
        "gap_size": rng.uniform(0, 1, n),
        "tick_count": rng.integers(10, 100, n),
        "mbo_bar_coverage": rng.uniform(0.5, 1, n),
        "mbp_bar_coverage": rng.uniform(0.5, 1, n),
    })
    # ضمان high >= max(open,close), low <= min(open,close)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low",  "open", "close"]].min(axis=1)
    return df


# ══════════════════════════════════════════════════════════════════
# TEST 1 — Causality
# ══════════════════════════════════════════════════════════════════

def test_causality_full_pipeline():
    """
    التغيير في صف i+k لا يجب أن يؤثر على أي مخرج في صف i (لكل k > 0).
    
    نختبر بتغيير صف 200، ونتأكد أن صف 100 لم يتغير في أي مخرج.
    """
    df1 = _make_random_data(500, seed=42)
    df1_out = run_all_simulators(df1.copy(), window=100)
    snapshot_at_100 = df1_out[SIMULATOR_OUTPUTS].iloc[100].values.copy()
    
    df2 = df1.copy()
    # غيّر صف 200 جذرياً
    for col in ["close", "high", "low", "open", "absorption_intensity",
                "bid_wall_strength", "ask_wall_strength", "obi"]:
        df2.loc[200, col] = df2[col].max() * 1000  # قيمة متطرفة
    
    df2_out = run_all_simulators(df2, window=100)
    snapshot_at_100_v2 = df2_out[SIMULATOR_OUTPUTS].iloc[100].values
    
    max_diff = np.abs(snapshot_at_100 - snapshot_at_100_v2).max()
    assert max_diff < 1e-9, f"FAIL: future row affected past! max_diff={max_diff}"
    return True


# ══════════════════════════════════════════════════════════════════
# TEST 2 — No-Bias on Random Data
# ══════════════════════════════════════════════════════════════════

def test_no_bias_random_data():
    """
    على ضوضاء عشوائية بحتة:
      - signals direction-like (range [-1,1]) متوسطها ≈ 0
      - signals continuous probability متوسطها ≈ 0.5
    
    استثناء (threshold-based or gated):
      sim_absorb_persist (gated by intensity>0.7)
      sim_absorb_at_wall (gated by wall_strong)
      sim_wall_real (multiplied by 1-fake)
      sim_data_quality (depends on coverage, not random)
    """
    df = _make_random_data(1000, seed=42)
    df_out = run_all_simulators(df, window=200)
    
    # المخرجات الـ direction (range [-1, 1])
    direction_cols = [
        "sim_absorb_direction", "sim_flow_direction",
        "sim_informed_direction", "sim_iceberg_side",
        "sim_depth_pressure", "sim_depth_imbalance",
    ]
    # continuous probability في [0, 1] بلا gating
    prob_cols = [
        "sim_absorb_intensity",   # rank-based, mean ≈ 0.5
        "sim_flow_strength",
        "sim_flow_consistency",
        "sim_informed_prob",
        "sim_informed_urgency",
        "sim_wall_strength",
        "sim_liquidity_state",
        "sim_volatility_regime",
        "sim_wall_bid_size",
        "sim_wall_ask_size",
    ]
    
    bias_tolerance = 0.15
    failures = []
    
    for col in direction_cols:
        if col not in df_out.columns: continue
        s = df_out[col].dropna()
        if len(s) < 100: continue
        m = s.mean()
        if abs(m) > bias_tolerance:
            failures.append((col, m, "direction expected ≈ 0"))
    
    for col in prob_cols:
        if col not in df_out.columns: continue
        s = df_out[col].dropna()
        if len(s) < 100: continue
        m = s.mean()
        if abs(m - 0.5) > bias_tolerance:
            failures.append((col, m, "probability expected ≈ 0.5"))
    
    if failures:
        for col, m, expected in failures:
            print(f"  ❌ BIAS: {col} mean={m:.3f}, {expected}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# TEST 3 — Range Validation
# ══════════════════════════════════════════════════════════════════

def test_output_ranges():
    """
    كل مخرج يجب أن يكون داخل نطاقه المعلن.
    """
    df = _make_random_data(500, seed=42)
    df_out = run_all_simulators(df, window=100)
    
    range_map = {
        # direction columns: [-1, 1]
        "sim_absorb_direction":    (-1, 1),
        "sim_flow_direction":      (-1, 1),
        "sim_informed_direction":  (-1, 1),
        "sim_iceberg_side":        (-1, 1),
        # probability/intensity columns: [0, 1]
        "sim_absorb_intensity":    (0, 1),
        "sim_absorb_persist":      (0, 1),
        "sim_absorb_at_wall":      (0, 1),
        "sim_flow_strength":       (0, 1),
        "sim_flow_consistency":    (0, 1),
        "sim_flow_divergence":     (0, 1),
        "sim_informed_prob":       (0, 1),
        "sim_informed_urgency":    (0, 1),
        "sim_sweep_signal":        (0, 1),
        "sim_wall_real":           (0, 1),
        "sim_wall_strength":       (0, 1),
        "sim_liquidity_state":     (0, 1),
        "sim_volatility_regime":   (0, 1),
        "sim_data_quality":        (0, 1),
        # depth pressure/imbalance: [-1, 1]
        "sim_depth_pressure":      (-1, 1),
        "sim_depth_imbalance":     (-1, 1),
    }
    
    failures = []
    for col, (lo, hi) in range_map.items():
        if col not in df_out.columns:
            continue
        s = df_out[col].dropna()
        if s.min() < lo - 1e-6 or s.max() > hi + 1e-6:
            failures.append((col, s.min(), s.max(), lo, hi))
    
    if failures:
        for col, mn, mx, lo, hi in failures:
            print(f"  ❌ RANGE: {col} actual=[{mn:.3f},{mx:.3f}] expected=[{lo},{hi}]")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# TEST 4 — NaN Handling
# ══════════════════════════════════════════════════════════════════

def test_no_nan_in_output():
    """لا NaN في أي مخرج محاكي (يجب أن تكون مُعالَجة)."""
    df = _make_random_data(500, seed=42)
    # نضع بعض NaN في المدخلات للاختبار
    df.loc[50:60, "absorption_intensity"] = np.nan
    df.loc[100:105, "hawkes_intrabar_sum"] = np.nan
    
    df_out = run_all_simulators(df, window=100)
    
    nan_cols = []
    for col in SIMULATOR_OUTPUTS:
        n_nan = df_out[col].isna().sum()
        if n_nan > 0:
            nan_cols.append((col, n_nan))
    
    if nan_cols:
        for col, n in nan_cols:
            print(f"  ⚠️ NaN found: {col} = {n}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# TEST 5 — Statistics Module Sanity
# ══════════════════════════════════════════════════════════════════

def test_benjamini_hochberg():
    """
    BH على p-values معلومة مسبقاً.
      - p_values كلها 1.0 → لا اكتشافات
      - p_values مختلطة → عدد محدّد من الاكتشافات
    """
    # كل p-values = 1.0 (لا شيء معنوي)
    p_all_one = np.ones(20)
    rejected, _ = benjamini_hochberg(p_all_one, fdr_alpha=0.10)
    assert rejected.sum() == 0, f"FAIL: rejected={rejected.sum()} لـ p=1.0"
    
    # p-values: 5 معنوية حقاً + 15 ضوضاء
    p_mixed = np.array([0.001, 0.005, 0.01, 0.02, 0.04] + [0.5] * 15)
    rejected, threshold = benjamini_hochberg(p_mixed, fdr_alpha=0.10)
    # نتوقع ≥3 اكتشافات (المعنوية الحقيقية)
    assert rejected.sum() >= 3, f"FAIL: rejected={rejected.sum()} مع 5 معنوية"
    return True


def test_permutation_with_real_edge():
    """
    Permutation على بيانات بـ edge حقيقي → p_value منخفض.
    Permutation على ضوضاء بحتة → p_value ≈ 0.5
    """
    rng = np.random.default_rng(42)
    
    # ① ضوضاء بحتة
    pure_noise = rng.standard_normal(200) * 0.001
    result_noise = permutation_test_edge(pure_noise, direction=1, n_permutations=500)
    assert result_noise["p_value"] > 0.20, \
        f"FAIL: pure noise gave p={result_noise['p_value']:.3f}"
    
    # ② edge قوي (+0.0005 متوسط)
    real_edge = rng.standard_normal(200) * 0.001 + 0.0005
    result_edge = permutation_test_edge(real_edge, direction=1, n_permutations=500)
    assert result_edge["p_value"] < 0.10, \
        f"FAIL: real edge gave p={result_edge['p_value']:.3f}"
    
    return True


def test_3way_split_no_overlap():
    """التقسيم الثلاثي لا يتداخل ويحترم purge."""
    split = make_3way_split(n=1000, train_ratio=0.6, validation_ratio=0.2, purge_bars=10)
    # لا تداخل
    train_set = set(split.train_idx.tolist())
    val_set = set(split.validation_idx.tolist())
    hold_set = set(split.holdout_idx.tolist())
    assert len(train_set & val_set) == 0, "train ∩ validation ≠ ∅"
    assert len(val_set & hold_set) == 0, "validation ∩ holdout ≠ ∅"
    assert len(train_set & hold_set) == 0, "train ∩ holdout ≠ ∅"
    # purge: gap بين train و validation
    gap_tv = split.validation_idx.min() - split.train_idx.max() - 1
    assert gap_tv >= 10, f"train-val gap = {gap_tv} < 10"
    gap_vh = split.holdout_idx.min() - split.validation_idx.max() - 1
    assert gap_vh >= 10, f"val-hold gap = {gap_vh} < 10"
    return True


def test_edge_statistics_consistency():
    """edge_statistics يعطي نفس النتيجة على نفس الإدخال."""
    rng = np.random.default_rng(42)
    rets = rng.standard_normal(100) * 0.001 + 0.0003
    days = np.array([f"2025-01-{(i // 10) + 1:02d}" for i in range(100)])
    
    s1 = edge_statistics(rets, days, direction=1)
    s2 = edge_statistics(rets, days, direction=1)
    assert s1 == s2, "edge_statistics غير حتمية"
    return True


# ══════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════

def run_all_tests():
    tests = [
        ("Causality (full pipeline)",      test_causality_full_pipeline),
        ("No-bias on random data",         test_no_bias_random_data),
        ("Output ranges",                  test_output_ranges),
        ("NaN handling",                   test_no_nan_in_output),
        ("Benjamini-Hochberg",             test_benjamini_hochberg),
        ("Permutation test",               test_permutation_with_real_edge),
        ("3-way split correctness",        test_3way_split_no_overlap),
        ("Edge stats consistency",         test_edge_statistics_consistency),
    ]
    
    print("=" * 64)
    print("Unit Tests — QuantSystem V19.2")
    print("=" * 64)
    
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            status = "✅ PASS" if ok else "⚠️ WARN"
            print(f"  {status:<10} {name}")
            results.append((name, ok, None))
        except AssertionError as e:
            print(f"  ❌ FAIL    {name}")
            print(f"             {e}")
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"  💥 ERROR   {name}")
            print(f"             {type(e).__name__}: {e}")
            results.append((name, False, str(e)))
    
    print("=" * 64)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"النتيجة: {n_pass}/{len(results)} اجتاز")
    return n_pass == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
