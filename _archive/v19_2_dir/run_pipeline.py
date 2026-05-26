"""
run_pipeline.py — تشغيل كل المحاكيات + edge scanner كنظام واحد
══════════════════════════════════════════════════════════════════════

يستدعي كل المراحل من داخل Python (لا subprocess) =
  - أسرع (لا إعادة تحميل numpy/pandas في كل خطوة)
  - أنظف (error handling موحّد)
  - يحفظ الحالة بين الخطوات (DataFrame في الذاكرة)

الاستخدام:
  python run_pipeline.py \
    --input out/day_trading_features.parquet \
    --mbo glbx-...mbo.parquet \
    --mbp glbx-...mbp-10.parquet \
    --output out/ \
    --symbol 6B --confidence medium

أو من Jupyter:
  %run run_pipeline.py --input ... --mbo ... --mbp ...
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# استيراد كل المراحل
from feature_simulators import run_all_simulators, SIMULATOR_OUTPUTS
from wall_depth_simulator import simulate_wall_depth, WALL_DEPTH_OUTPUTS
from iceberg_simulator import simulate_iceberg, ICEBERG_OUTPUTS
from session_mapper import map_sessions
from edge_scanner import scan_edges
from cluster_engine import select_distinct_alphas


def _step(name: str):
    """Decorator يطبع توقيت كل مرحلة."""
    def wrapper(fn):
        def inner(*args, **kwargs):
            print(f"\n{'═'*64}")
            print(f"🚀 {name}")
            print('═'*64)
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            print(f"   ⏱️  {elapsed:.1f}s")
            return result
        return inner
    return wrapper


def run_full_pipeline(
    input_parquet: str,
    mbo_path: str,
    mbp_path: str,
    output_dir: str = "out",
    symbol: str = "6B",
    confidence: str = "medium",
    horizons: list[int] = [3, 6, 12],
    freq: str = "5min",
    window: int = 200,
    n_permutations: int = 1000,
    corr_threshold: float = 0.70,
    save_intermediate: bool = True,
    verbose: bool = True,
) -> dict:
    """
    يشغّل الـ pipeline الكامل في الذاكرة.
    
    Returns dict:
      df_final: pd.DataFrame النهائي مع كل المحاكيات + session mapping
      edge_result: نتيجة edge_scanner
      alphas: نتيجة cluster_engine
      timings: وقت كل مرحلة
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    
    # ════════════════════════════════════════════════════════
    # تحميل البيانات الأولية
    # ════════════════════════════════════════════════════════
    print(f"\n📥 تحميل {input_parquet}...")
    t0 = time.time()
    df = pd.read_parquet(input_parquet)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event').reset_index(drop=True)
    print(f"   {len(df):,} شمعة | {len(df.columns)} عمود | "
          f"{pd.to_datetime(df.ts_event).dt.date.nunique()} يوم")
    timings['load'] = time.time() - t0
    
    # تنظيف outliers احتياطي
    ret = pd.to_numeric(df['close'], errors='coerce').pct_change().abs()
    bad = ret > 0.015
    if bad.any():
        for col in ('open', 'high', 'low', 'close'):
            if col in df.columns:
                df.loc[bad, col] = np.nan
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
        print(f"   نُظِّفت {bad.sum()} شمعة outlier")
    
    # ATR احتياطي
    if 'atr_14' not in df.columns:
        df['atr_14'] = (df['high'] - df['low']).rolling(14, min_periods=1).mean()
    
    # ════════════════════════════════════════════════════════
    # المرحلة ① — 18 محاكي عمق
    # ════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"\n{'═'*64}\n🚀 المرحلة ①: feature_simulators (18 محاكي عمق)\n{'═'*64}")
    df = run_all_simulators(df, window=window)
    timings['feature_simulators'] = time.time() - t0
    print(f"   ⏱️  {timings['feature_simulators']:.1f}s")
    if save_intermediate:
        df.to_parquet(out_dir / f"sim_{symbol}.parquet", index=False)
        print(f"   💾 {out_dir / f'sim_{symbol}.parquet'}")
    
    # ════════════════════════════════════════════════════════
    # المرحلة ② — 8 محاكي جدران (من MBP-10)
    # ════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"\n{'═'*64}\n🚀 المرحلة ②: wall_depth_simulator (من MBP)\n{'═'*64}")
    mbp_df = pd.read_parquet(mbp_path)
    print(f"   MBP: {len(mbp_df):,} snapshot")
    df = simulate_wall_depth(df, mbp_df, freq=freq, window=window)
    del mbp_df  # حرّر الذاكرة
    timings['wall_depth'] = time.time() - t0
    print(f"   ⏱️  {timings['wall_depth']:.1f}s")
    if save_intermediate:
        df.to_parquet(out_dir / f"walls_{symbol}.parquet", index=False)
    
    # ════════════════════════════════════════════════════════
    # المرحلة ③ — 5 محاكي iceberg (من MBO)
    # ════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"\n{'═'*64}\n🚀 المرحلة ③: iceberg_simulator (من MBO)\n{'═'*64}")
    mbo_df = pd.read_parquet(mbo_path)
    print(f"   MBO: {len(mbo_df):,} event")
    df = simulate_iceberg(df, mbo_df, freq=freq, window=window)
    del mbo_df  # حرّر الذاكرة
    timings['iceberg'] = time.time() - t0
    print(f"   ⏱️  {timings['iceberg']:.1f}s")
    if save_intermediate:
        df.to_parquet(out_dir / f"icebergs_{symbol}.parquet", index=False)
    
    # ════════════════════════════════════════════════════════
    # المرحلة ④ — session mapper (16 zone × 12 level × 5 events)
    # ════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"\n{'═'*64}\n🚀 المرحلة ④: session_mapper\n{'═'*64}")
    df = map_sessions(df)
    timings['session_mapper'] = time.time() - t0
    print(f"   ⏱️  {timings['session_mapper']:.1f}s")
    if save_intermediate:
        df.to_parquet(out_dir / f"mapped_{symbol}.parquet", index=False)
        print(f"   💾 {out_dir / f'mapped_{symbol}.parquet'}")
    
    # ملخص الإشارات
    n_alive = sum(1 for c in df.columns
                  if c.startswith('sim_') and df[c].std() > 0)
    print(f"   ✅ المحاكيات الحية: {n_alive}/{len(SIMULATOR_OUTPUTS) + len(WALL_DEPTH_OUTPUTS) + len(ICEBERG_OUTPUTS)}")
    
    # ════════════════════════════════════════════════════════
    # المرحلة ⑤ — edge scanner (FDR + 3-way + permutation)
    # ════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"\n{'═'*64}\n🚀 المرحلة ⑤: edge_scanner (FDR + 3-way + permutation)\n{'═'*64}")
    edge_result = scan_edges(
        df,
        horizons=horizons,
        symbol=symbol,
        confidence=confidence,
        run_permutation=True,
        n_permutations=n_permutations,
        verbose=verbose,
    )
    timings['edge_scanner'] = time.time() - t0
    print(f"   ⏱️  {timings['edge_scanner']:.1f}s")
    
    # حفظ
    def _clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(x) for x in obj]
        return obj
    
    edges_path = out_dir / f"edge_candidates_{symbol}.json"
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump(_clean(edge_result), f, indent=2, ensure_ascii=False, default=str)
    print(f"   💾 {edges_path}")
    
    # ════════════════════════════════════════════════════════
    # المرحلة ⑥ — cluster engine (تجميع وتمييز)
    # ════════════════════════════════════════════════════════
    alphas_result = None
    if edge_result['candidates']:
        t0 = time.time()
        print(f"\n{'═'*64}\n🚀 المرحلة ⑥: cluster_engine\n{'═'*64}")
        alphas, report = select_distinct_alphas(
            edge_result['candidates'], df, corr_threshold=corr_threshold
        )
        timings['cluster_engine'] = time.time() - t0
        print(f"   ⏱️  {timings['cluster_engine']:.1f}s")
        
        alphas_result = {
            'alphas': alphas,
            'report': report,
        }
        alphas_path = out_dir / f"alphas_{symbol}.json"
        with open(alphas_path, 'w', encoding='utf-8') as f:
            json.dump(_clean(alphas_result), f, indent=2, ensure_ascii=False, default=str)
        print(f"   💾 {alphas_path}")
    else:
        print(f"\n⚠️ لا توجد edges لتجميعها — تخطّى cluster_engine")
    
    # ════════════════════════════════════════════════════════
    # تقرير نهائي
    # ════════════════════════════════════════════════════════
    print(f"\n{'═'*64}")
    print(f"📊 ملخص النتائج")
    print('═'*64)
    diag = edge_result.get('diagnostics', {})
    print(f"  Stage 1 (مرشّحات أولية):      {diag.get('stage1_candidates', 0)}")
    print(f"  Stage 2 (FDR):                 {diag.get('stage2_passed_fdr', 0)}")
    print(f"  Stage 3 (Permutation):         {diag.get('stage3_passed_permutation', 0)}")
    print(f"  Stage 4 (Validation):          {diag.get('stage4_passed_validation', 0)}")
    print(f"  Stage 5 (Holdout — نهائي):     {diag.get('stage5_passed_holdout', 0)}")
    if alphas_result:
        print(f"  ⭐ alphas مميّزة:              {len(alphas_result['alphas'])}")
    
    print(f"\n⏱️  أوقات التنفيذ:")
    total = 0
    for k, v in timings.items():
        print(f"   {k:<22}: {v:6.1f}s")
        total += v
    print(f"   {'─'*30}")
    print(f"   {'TOTAL':<22}: {total:6.1f}s")
    
    # أقوى الـ alphas
    candidates = edge_result.get('candidates', [])
    if candidates:
        print(f"\n🏆 أقوى 5 edges:")
        for i, c in enumerate(candidates[:5], 1):
            st = c['stats_train']
            sh = c.get('stats_holdout', {})
            print(f"\n  {i}. {c.get('edge_id', 'unknown')}")
            print(f"     train:   t={st['t_stat']:+.2f}  wr={st['wr']:.0%}  avg={st['avg_pips']:+.1f}p")
            print(f"     holdout: t={sh.get('t_stat',0):+.2f}  wr={sh.get('wr',0):.0%}  avg={sh.get('avg_pips',0):+.1f}p")
    
    return {
        'df': df,
        'edge_result': edge_result,
        'alphas': alphas_result,
        'timings': timings,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run full pipeline (simulators + session + edge scanner)"
    )
    ap.add_argument("--input",   required=True, help="day_trading_features.parquet")
    ap.add_argument("--mbo",     required=True, help="MBO الخام (parquet)")
    ap.add_argument("--mbp",     required=True, help="MBP-10 الخام (parquet)")
    ap.add_argument("--output",  default="out", help="مجلد المخرجات")
    ap.add_argument("--symbol",  default="6B")
    ap.add_argument("--confidence", default="medium",
                    choices=["loose", "medium", "strict"])
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--corr-threshold", type=float, default=0.70)
    ap.add_argument("--no-intermediate", action="store_true",
                    help="لا تحفظ ملفات وسيطة (أسرع، RAM أقل لكن لا قابلية رجوع)")
    args = ap.parse_args()
    
    horizons = [int(x) for x in args.horizons.split(',')]
    
    result = run_full_pipeline(
        input_parquet=args.input,
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        symbol=args.symbol,
        confidence=args.confidence,
        horizons=horizons,
        freq=args.freq,
        window=args.window,
        n_permutations=args.n_permutations,
        corr_threshold=args.corr_threshold,
        save_intermediate=not args.no_intermediate,
    )
    
    print(f"\n✅ اكتمل! النتيجة الرئيسية في:")
    print(f"   {args.output}/alphas_{args.symbol}.json")
    print(f"   {args.output}/edge_candidates_{args.symbol}.json")


if __name__ == "__main__":
    main()
