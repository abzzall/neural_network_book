"""Chapter 4: cleaning a small table step by step with pandas."""

import numpy as np
import pandas as pd

df = pd.DataFrame({
    "district": ["centre", " Centre", "north", "NORTH", "south", "north", "north", "east"],
    "area_m2": [52.0, np.nan, 61.0, 61.0, 900.0, 45.0, 61.0, 70.0],
    "rooms": [2, 3, 2, 2, 3, 1, 2, 3],
})
print(f"start: {len(df)} rows, {df['district'].nunique()} district spellings")

# 1. One spelling per district: strip spaces, lower case.
df["district"] = df["district"].str.strip().str.lower()
print(f"after normalising text: {df['district'].nunique()} districts")

# 2. Exact duplicates are the same listing entered twice.
df = df.drop_duplicates().reset_index(drop=True)
print(f"after removing duplicates: {len(df)} rows")

# 3. An impossible value is treated as missing, not silently kept.
df.loc[df["area_m2"] > 400, "area_m2"] = np.nan

# 4. Record which values were missing, then fill them with the median.
df["area_missing"] = df["area_m2"].isna().astype(int)
median = df["area_m2"].median()
df["area_m2"] = df["area_m2"].fillna(median)
print(f"median area used for {df['area_missing'].sum()} missing values: {median}")

# 5. One column per district.
df = pd.get_dummies(df, columns=["district"], dtype=int)
print("columns:", list(df.columns))
