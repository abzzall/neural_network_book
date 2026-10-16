"""Chapter 4: a simulated table of apartment listings with typical data defects.

The data are simulated, so every defect is known exactly:
- district names written inconsistently (" Centre", "NORTH", ...);
- missing values in area and year_built;
- some areas entered ten times too large (a units mistake);
- duplicated rows;
- valuation_after_sale_k, recorded only after a sale: target leakage.
"""

import numpy as np
import pandas as pd

DISTRICT_EFFECT = {"centre": 60.0, "north": 10.0, "south": 0.0, "east": -10.0}


def make_listings(n=500, seed=42):
    rng = np.random.default_rng(seed)
    district = rng.choice(list(DISTRICT_EFFECT), n, p=[0.25, 0.3, 0.25, 0.2])
    area = rng.uniform(25.0, 150.0, n)
    rooms = np.clip(np.round(area / 30 + rng.normal(0, 0.5, n)), 1, 6)
    year = rng.integers(1960, 2025, n).astype(float)
    effect = np.array([DISTRICT_EFFECT[d] for d in district])
    price = 2.0 * area + 15.0 * rooms - 0.3 * (2025 - year) + effect + rng.normal(0, 15, n)

    df = pd.DataFrame({"district": district, "area_m2": area.round(1), "rooms": rooms,
                       "year_built": year, "price_k": price.round(1)})
    # Recorded by the listing site after the sale, so never known for a new listing.
    df["valuation_after_sale_k"] = (df["price_k"] + rng.normal(0, 2, n)).round(1)

    messy = rng.random(n) < 0.15                                      # inconsistent text
    df.loc[messy, "district"] = [rng.choice([d.upper(), " " + d.capitalize(), d + " "])
                                 for d in df.loc[messy, "district"]]
    df.loc[rng.random(n) < 0.05, "area_m2"] = np.nan                  # missing values
    df.loc[rng.random(n) < 0.08, "year_built"] = np.nan
    wrong = (rng.random(n) < 0.03) & df["area_m2"].notna()            # units mistake
    df.loc[wrong, "area_m2"] = df.loc[wrong, "area_m2"] * 10
    duplicates = df.sample(15, random_state=seed)                      # duplicated rows
    return pd.concat([df, duplicates], ignore_index=True)


if __name__ == "__main__":
    df = make_listings()
    print(f"rows = {len(df)}, duplicated rows = {df.duplicated().sum()}")
    print("missing:", df.isna().sum()[["area_m2", "year_built"]].to_dict())
    print(f"district spellings = {df['district'].nunique()}")
    print(f"areas above 400 m2 = {(df['area_m2'] > 400).sum()}")
