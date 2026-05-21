"""
combine_months.py — دمج عدة شهور من day_trading_features
══════════════════════════════════════════════════════════════════════

يأخذ ملفات شهرية ويدمجها في ملف واحد جاهز للـ pipeline.

استخدامات:
  ① دمج محدّد:
     python combine_months.py \
         --inputs out/2024-11/day_trading_features.parquet \
                  out/2024-12/day_trading_features.parquet \
                  out/2025-01/day_trading_features.parquet \
         --output out/combined.parquet
  
  ② دمج كل ما في مجلد (يأخذ كل *.parquet بـ glob pattern):
     python combine_months.py \
         --pattern "out/202*/day_trading_features.parquet" \
         --output out/combined.parquet

  ③ تخطي شهور معيّنة:
     python combine_months.py \
         --pattern "out/*/day_trading_features.parquet" \
         --skip 2024-12 \
         --output out/combined.parquet
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def combine_months(
    paths: list[str],
    output_path: str,
    apply_fix_ohlc: bool = True,
    symbol: str = "6B",
) -> dict:
    """
    يدمج عدة ملفات شهرية في ملف واحد.
    
    خطوات:
      ① اقرأ كل ملف
      ② تحقق من الـ schema (نفس الأعمدة)
      ③ ادمج بترتيب زمني
      ④ احذف التكرار (لو في overlap)
      ⑤ أصلح OHLC outliers (اختياري لكن موصى به)
      ⑥ أعد حساب PDH/PDL/ATR على الـ DataFrame الكامل
    """
    if not paths:
        raise ValueError("لا ملفات للدمج")
    
    print(f"📥 دمج {len(paths)} ملف:")
    
    dfs = []
    info_per_file = []
    schema_ref = None
    
    for i, path in enumerate(paths, 1):
        if not Path(path).exists():
            print(f"  ❌ {path} — غير موجود، تخطّى")
            continue
        
        df = pd.read_parquet(path)
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        
        # تحقق من schema
        cols = set(df.columns)
        if schema_ref is None:
            schema_ref = cols
        else:
            missing = schema_ref - cols
            extra = cols - schema_ref
            if missing or extra:
                print(f"  ⚠️ {path}: schema مختلف")
                if missing: print(f"     مفقود: {list(missing)[:5]}")
                if extra:   print(f"     زائد: {list(extra)[:5]}")
                # احتفظ بالـ intersection فقط
                schema_ref = schema_ref & cols
        
        n_days = pd.to_datetime(df['ts_event']).dt.date.nunique()
        date_range = f"{df.ts_event.min().date()} → {df.ts_event.max().date()}"
        print(f"  {i}. {Path(path).name}: {len(df):,} صف, {n_days} يوم ({date_range})")
        
        info_per_file.append({
            'path': path,
            'rows': len(df),
            'days': n_days,
            'date_range': date_range,
        })
        dfs.append(df)
    
    if not dfs:
        raise ValueError("لا ملفات صالحة")
    
    # احتفظ بالأعمدة المشتركة فقط
    common_cols = list(schema_ref)
    dfs = [df[common_cols] for df in dfs]
    
    # ──  دمج ──
    combined = pd.concat(dfs, ignore_index=True)
    
    # ⚠️ ترتيب زمني صارم — لا يهم ترتيب inputs
    combined = combined.sort_values('ts_event').reset_index(drop=True)
    
    print(f"\n📊 بعد الدمج: {len(combined):,} صف × {len(combined.columns)} عمود")
    print(f"   ⏰ الفترة (مرتّبة زمنياً تلقائياً):")
    print(f"      أقدم صف: {combined.ts_event.min()}")
    print(f"      أحدث صف: {combined.ts_event.max()}")
    
    # تأكيد أن الـ DataFrame مرتّب
    is_sorted = (combined['ts_event'].diff().dt.total_seconds().dropna() >= 0).all()
    if is_sorted:
        print(f"   ✅ مرتّب تصاعدياً (شيخوخة → أحدث)")
    else:
        print(f"   ⚠️ مشكلة في الترتيب!")
    
    # ── احذف التكرار (overlap) ──
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=['ts_event'], keep='first')
    n_dupes = n_before - len(combined)
    if n_dupes > 0:
        print(f"   ⚠️ حُذفت {n_dupes} شمعة مكررة (overlap)")
    
    # ── إصلاح OHLC على الكل ──
    if apply_fix_ohlc:
        print(f"\n🔧 إصلاح OHLC outliers...")
        try:
            from fix_ohlc import fix_ohlc
            combined, fix_stats = fix_ohlc(combined, symbol=symbol)
            print(f"   إصلاحات: {fix_stats['fixes']}")
        except ImportError:
            print(f"   ⚠️ fix_ohlc غير متاح — تخطّى")
    
    # ── إعادة حساب PDH/PDL على الـ DataFrame الكامل ──
    # (لأن PDH/PDL الأصلي محسوب لكل شهر منفصلاً، وأول يوم في كل شهر
    #  يفقد PDH/PDL لو ما كان موجود في الشهر السابق)
    print(f"\n📐 إعادة حساب PDH/PDL على كل الفترة...")
    combined['__day'] = pd.to_datetime(combined['ts_event']).dt.date
    daily = combined.groupby('__day').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min'),
    ).reset_index()
    daily['pdh_new'] = daily['day_high'].shift(1)
    daily['pdl_new'] = daily['day_low'].shift(1)
    daily['pd_50_new'] = (daily['pdh_new'] + daily['pdl_new']) / 2
    
    combined = combined.merge(
        daily[['__day', 'pdh_new', 'pdl_new', 'pd_50_new']],
        on='__day', how='left'
    )
    
    for src, dst_options in (
        ('pdh_new', ['pdh', 'PDH']),
        ('pdl_new', ['pdl', 'PDL']),
        ('pd_50_new', ['pd_50', 'PD_50']),
    ):
        for dst in dst_options:
            if dst in combined.columns:
                combined[dst] = combined[src]
    
    combined = combined.drop(columns=['__day', 'pdh_new', 'pdl_new', 'pd_50_new'])
    
    # ── حفظ ──
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    
    # ── ملخص ──
    total_days = pd.to_datetime(combined['ts_event']).dt.date.nunique()
    stats = {
        'n_files': len(dfs),
        'total_rows': len(combined),
        'total_days': total_days,
        'duplicates_removed': n_dupes,
        'common_columns': len(common_cols),
        'date_range': f"{combined.ts_event.min()} → {combined.ts_event.max()}",
        'output': output_path,
    }
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="دمج عدة شهور من day_trading_features")
    ap.add_argument("--inputs", nargs="+", help="قائمة ملفات صراحةً")
    ap.add_argument("--pattern", help="glob pattern (مثل 'out/*/day_trading_features.parquet')")
    ap.add_argument("--output", required=True, help="الملف الناتج")
    ap.add_argument("--skip", nargs="+", default=[], help="أنماط لتخطّيها")
    ap.add_argument("--no-fix-ohlc", action="store_true", help="لا تطبّق fix_ohlc")
    ap.add_argument("--symbol", default="6B")
    args = ap.parse_args()
    
    # جمع المسارات
    paths = []
    if args.inputs:
        paths.extend(args.inputs)
    if args.pattern:
        matches = sorted(glob.glob(args.pattern))
        paths.extend(matches)
    
    if not paths:
        raise ValueError("لا --inputs ولا --pattern. حدد ملفاتك")
    
    # حذف التكرار في القائمة + تخطّي
    seen = set()
    paths_filtered = []
    for p in paths:
        if p in seen: continue
        seen.add(p)
        if any(s in p for s in args.skip):
            print(f"⏭️  تخطّى: {p}")
            continue
        paths_filtered.append(p)
    
    if not paths_filtered:
        raise ValueError("لا ملفات بعد الفلترة")
    
    stats = combine_months(
        paths=paths_filtered,
        output_path=args.output,
        apply_fix_ohlc=not args.no_fix_ohlc,
        symbol=args.symbol,
    )
    
    print(f"\n✅ اكتمل!")
    for k, v in stats.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
