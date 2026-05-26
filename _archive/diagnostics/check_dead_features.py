import pandas as pd
import numpy as np

df = pd.read_parquet("pipeline_mbo2_refinery_latest/daytrade/day_trading_features.parquet")
num = df.apply(pd.to_numeric, errors="coerce")
std = num.std(numeric_only=True)

dead = [c for c in std.index if pd.notna(std[c]) and std[c] == 0]
weak = [c for c in std.index if pd.notna(std[c]) and 0 < std[c] < 0.001]
null = [c for c in df.columns if df[c].isnull().mean() > 0.5]

print("Total columns:", len(df.columns))
print("Dead (std=0):", len(dead))
print("Weak (std<0.001):", len(weak))
print("Null (>50% null):", len(null))
print("")
print("Dead list:")
print(dead)
print("")
print("Weak list:")
print(weak)
print("")
print("Null list:")
print(null)
