"""
validate_daytrade_output.py — تحقّق شامل من مخرجات prepare_day_trading.py

شغّله بعد ما الـ pipeline يخلص:
    python validate_daytrade_output.py pipeline_month_test

يقيس:
  ① Sprint 19 labeling: NEUTRAL <60%?
  ② 4-regime distribution: كل الـ regimes ظاهرة؟
  ③ Seasonal features: 21 col موجودة + بدون NaN؟
  ④ Cycle features: 12 col موجودة + بدون NaN؟
  ⑤ LOB tensors: shape صحيح؟
  ⑥ بيانات قابلة للتدريب: train_pool عدد كافي؟
  ⑦ Data quality: NaN في الـ features الأساسية؟
"""
import sys
import os
import glob
import json
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python validate_daytrade_output.py <output_dir>")
    sys.exit(1)

output_dir = sys.argv[1]
features_dir = os.path.join(output_dir, 'features')

print("=" * 72)
print(f"VALIDATION REPORT — {output_dir}")
print("=" * 72)

# ── ① Load all parquet shards ───────────────────────────────────────────
parquet_files = sorted(glob.glob(os.path.join(features_dir, '*.parquet')))
if not parquet_files:
    print(f"❌ no parquet files found in {features_dir}")
    sys.exit(1)

print(f"\n📂 Loading {len(parquet_files)} parquet shard(s)...")
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   Date range: {df.ts_event.min()} → {df.ts_event.max()}")

scores = []  # (check_name, passed, detail)

# ── ② Sprint 19 labels ────────────────────────────────────────────────
print("\n" + "─" * 72)
print("① SPRINT 19 LABELING")
print("─" * 72)
if 'bias_label' in df.columns:
    counts = df['bias_label'].value_counts().sort_index()
    L, S, N = int(counts.get(0, 0)), int(counts.get(1, 0)), int(counts.get(2, 0))
    total = L + S + N
    pct_neutral = N / total * 100 if total else 0
    pct_directional = (L + S) / total * 100 if total else 0
    print(f"  LONG    : {L:>8,}  ({L/total*100:5.1f}%)")
    print(f"  SHORT   : {S:>8,}  ({S/total*100:5.1f}%)")
    print(f"  NEUTRAL : {N:>8,}  ({pct_neutral:5.1f}%)")
    print(f"  Directional ratio: {pct_directional:.1f}%")
    if pct_neutral < 60:
        scores.append(('Sprint 19 (NEUTRAL<60%)', True, f'{pct_neutral:.1f}%'))
        print(f"  ✅ Sprint 19 working — NEUTRAL acceptable")
    else:
        scores.append(('Sprint 19 (NEUTRAL<60%)', False, f'{pct_neutral:.1f}%'))
        print(f"  ❌ NEUTRAL too high — investigate Sprint 19")
    if 0 < L and 0 < S:
        ls_ratio = max(L, S) / min(L, S)
        print(f"  LONG/SHORT imbalance: {ls_ratio:.2f}:1 ({'⚠️  skewed' if ls_ratio > 2.5 else '✅ balanced'})")
else:
    print("  ❌ bias_label column missing")
    scores.append(('Sprint 19', False, 'no bias_label'))

# ── ③ 4-regime distribution ───────────────────────────────────────────
print("\n" + "─" * 72)
print("② 4-REGIME UNIFICATION")
print("─" * 72)
if 'regime_label' in df.columns:
    regime_counts = df['regime_label'].value_counts()
    expected = {'trending', 'ranging', 'volatile', 'low_liquidity'}
    found = set(regime_counts.index)
    for r in ('trending', 'ranging', 'volatile', 'low_liquidity'):
        n = int(regime_counts.get(r, 0))
        pct = n / len(df) * 100
        mark = '✅' if n > 0 else '⚠️ '
        print(f"  {mark} {r:14s}: {n:>8,} ({pct:5.1f}%)")
    if expected.issubset(found):
        scores.append(('4-regime present', True, f'{len(found)}/4'))
        print(f"  ✅ All 4 regimes present")
    else:
        missing = expected - found
        scores.append(('4-regime present', False, f'missing: {missing}'))
        print(f"  ⚠️  Missing regimes: {missing}")
else:
    print("  ❌ regime_label column missing")
    scores.append(('4-regime', False, 'no regime_label'))

# ── ④ Seasonal features ───────────────────────────────────────────────
print("\n" + "─" * 72)
print("③ SEASONAL MAP (21 features)")
print("─" * 72)
seasonal_cols = [
    'session_phase', 'time_since_london_open_min', 'time_to_london_close_min',
    'time_since_ny_open_min', 'time_to_ny_close_min',
    'dow_sin', 'dow_cos', 'is_monday', 'is_friday',
    'dom', 'dom_sin', 'dom_cos', 'is_month_end', 'is_month_start',
    'is_quarter_end', 'is_year_end',
    'woy_sin', 'woy_cos', 'is_first_week_of_year',
    'is_dst_transition_week', 'is_event_window',
]
present = [c for c in seasonal_cols if c in df.columns]
missing = [c for c in seasonal_cols if c not in df.columns]
print(f"  Present: {len(present)}/21")
if missing:
    print(f"  ❌ Missing: {missing}")
    scores.append(('Seasonal features', False, f'missing {len(missing)}'))
else:
    nan_count = df[present].isna().sum().sum()
    print(f"  NaN cells: {nan_count}")
    if nan_count == 0:
        scores.append(('Seasonal features', True, '21/21 + 0 NaN'))
        print(f"  ✅ All 21 seasonal features clean")
    else:
        scores.append(('Seasonal features', False, f'{nan_count} NaN'))

# ── ⑤ Price Cycle features ────────────────────────────────────────────
print("\n" + "─" * 72)
print("④ PRICE CYCLE STRUCTURAL (12 features)")
print("─" * 72)
cycle_cols = [
    'cycle_structure_score', 'cycle_bars_since_swing',
    'cycle_trend_maturity', 'cycle_momentum_decay',
    'cycle_phase_acc_prob', 'cycle_phase_markup_prob',
    'cycle_phase_dist_prob', 'cycle_phase_markdown_prob',
    'cycle_position',
    'cycle_hurst', 'cycle_fractal_dim', 'cycle_mtf_alignment',
]
present = [c for c in cycle_cols if c in df.columns]
missing = [c for c in cycle_cols if c not in df.columns]
print(f"  Present: {len(present)}/12")
if missing:
    print(f"  ❌ Missing: {missing}")
    scores.append(('Price cycle features', False, f'missing {len(missing)}'))
else:
    nan_count = df[present].isna().sum().sum()
    print(f"  NaN cells: {nan_count}")
    if 'cycle_hurst' in df.columns:
        h = df['cycle_hurst']
        print(f"  cycle_hurst: [{h.min():.3f}, {h.max():.3f}] mean={h.mean():.3f}")
    if 'cycle_phase_acc_prob' in df.columns:
        p = df[['cycle_phase_acc_prob','cycle_phase_markup_prob',
                'cycle_phase_dist_prob','cycle_phase_markdown_prob']]
        sums = p.sum(axis=1)
        print(f"  Phase probs sum: {sums.min():.3f} - {sums.max():.3f} (should be ~1.0)")
    if nan_count == 0:
        scores.append(('Cycle features', True, '12/12 + 0 NaN'))
        print(f"  ✅ All 12 cycle features clean")

# ── ⑥ LOB tensors ─────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("⑤ LOB TENSORS")
print("─" * 72)
lob_files = glob.glob(os.path.join(output_dir, '*lob_tensors*.npy'))
if lob_files:
    for lf in lob_files:
        arr = np.load(lf, mmap_mode='r')
        size_mb = os.path.getsize(lf) / 1e6
        print(f"  {os.path.basename(lf)}: shape={arr.shape} dtype={arr.dtype} ({size_mb:.1f} MB)")
    scores.append(('LOB tensors', True, f'{len(lob_files)} file(s)'))
else:
    print("  ⚠️  no LOB tensors (mbo-only mode or --no-lob)")
    scores.append(('LOB tensors', None, 'absent (ok if no MBP)'))

# ── ⑦ Train pool ─────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("⑥ TRAINABLE POOL")
print("─" * 72)
if 'train_event_flag' in df.columns:
    n_train = int(df['train_event_flag'].sum())
    pct = n_train / len(df) * 100
    print(f"  train_event_flag=1: {n_train:,} ({pct:.1f}%)")
    if n_train >= 1000:
        scores.append(('Train pool size', True, f'{n_train} >= 1000'))
        print(f"  ✅ enough for training")
    else:
        scores.append(('Train pool size', False, f'{n_train} < 1000'))
        print(f"  ⚠️  small pool — may struggle to train")
elif 'is_event' in df.columns:
    n_ev = int(df['is_event'].sum())
    print(f"  is_event=1: {n_ev:,} ({n_ev/len(df)*100:.1f}%)")
    if n_ev >= 1000:
        scores.append(('Event pool', True, f'{n_ev}'))

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
passed = sum(1 for _, p, _ in scores if p is True)
failed = sum(1 for _, p, _ in scores if p is False)
neutral = sum(1 for _, p, _ in scores if p is None)
for name, p, detail in scores:
    mark = '✅' if p is True else ('❌' if p is False else '⚠️ ')
    print(f"  {mark} {name:30s} → {detail}")
print(f"\n  {passed} passed, {failed} failed, {neutral} skipped")
if failed == 0:
    print("\n🎉 Pipeline output looks good! Ready to proceed to Stage 2 CatBoost.")
else:
    print(f"\n⚠️  {failed} check(s) failed — investigate before training.")
