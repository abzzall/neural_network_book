"""Chapter 4: the cleaning and preparation of the listings, as one scikit-learn pipeline."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from messy_data import make_listings

df = make_listings()
df = df.assign(district=df["district"].str.strip().str.lower()).drop_duplicates()
df.loc[df["area_m2"] > 400, "area_m2"] = np.nan          # impossible values -> missing
X = df[["district", "area_m2", "rooms", "year_built"]]   # the valuation is NOT used
y = df["price_k"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
prepare = ColumnTransformer([
    ("num", numeric, ["area_m2", "rooms", "year_built"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["district"]),
])
model = Pipeline([("prepare", prepare), ("regress", LinearRegression())])
model.fit(X_train, y_train)                               # every statistic from training rows
rmse = np.sqrt(np.mean((model.predict(X_val) - y_val) ** 2))

if __name__ == "__main__":
    print(f"training rows = {len(X_train)}, validation rows = {len(X_val)}")
    print("features:", list(model[:-1].get_feature_names_out()))
    print(f"validation RMSE = {rmse:.2f} thousand")
