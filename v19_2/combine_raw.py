"""
combine_raw.py — دمج ملفات MBO/MBP الخام من شهور متعددة
══════════════════════════════════════════════════════════════════════

الاستخدام:
  python combine_raw.py \
      --inputs apr2025.mbo.parquet may2025.mbo.parquet \
      --output combined_mbo.parquet
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
import pandas as pd


def combine_raw_files(paths: list[str], output_path: str) -> dict:
    """دمج ملفات parquet خامة (MBO أو MBP) بترتيب زمني."""
    if not paths:
        raise ValueError("لا ملفات")
    
    print(f"📥 دمج {len(paths)} ملف:")
    dfs = []
    for i, path in enumerate(paths, 1):
        if not Path(path).exists():
            print(f"  ❌ {path} — غير موجود، تخطّى")
            continue
        df = pd.read_parquet(path)
        ts_col = 'ts_event' if 'ts_event' in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col])
        size_mb = Path(path).stat().st_size / (1024*1024)
        print(f"  {i}. {Path(path).name}: {len(df):,} صف ({size_mb:.0f} MB)")
        dfs.append(df)
    
    if not dfs:
        raise ValueError("لا ملفات صالحة")
    
    combined = pd.concat(dfs, ignore_index=True)
    ts_col = 'ts_event' if 'ts_event' in combined.columns else combined.columns[0]
    
    # ⚠️ ترتيب زمني صارم
    combined = combined.sort_values(ts_col).reset_index(drop=True)
    
    print(f"\n   ⏰ الفترة (مرتّبة زمنياً تلقائياً):")
    print(f"      أقدم: {combined[ts_col].min()}")
    print(f"      أحدث: {combined[ts_col].max()}")
    
    # حذف التكرار
    n_before = len(combined)
    # استخدم كل الأعمدة كـ subset (مع limit للسرعة)
    subset = [c for c in combined.columns if c in 
              ('ts_event', 'price', 'size', 'action', 'side', 'order_id', 'sequence')]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep='first')
    
    n_dup = n_before - len(combined)
    if n_dup > 0:
        print(f"  ⚠️ حُذفت {n_dup:,} صف مكرر")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    
    out_size_mb = Path(output_path).stat().st_size / (1024*1024)
    return {
        'n_files': len(dfs),
        'total_rows': len(combined),
        'output_size_mb': round(out_size_mb, 1),
        'output': output_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="ملفات صراحةً")
    ap.add_argument("--pattern", help="glob pattern")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    
    paths = []
    if args.inputs:
        paths.extend(args.inputs)
    if args.pattern:
        paths.extend(sorted(glob.glob(args.pattern)))
    
    if not paths:
        raise ValueError("لا --inputs ولا --pattern")
    
    stats = combine_raw_files(paths, args.output)
    print(f"\n✅ اكتمل!")
    for k, v in stats.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
