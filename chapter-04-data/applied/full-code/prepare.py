"""Chapter 4 applied exercise: clean, split, and prepare the listings; then fit a line."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # chapter folder
from messy_data import make_listings  # noqa: E402,F401

from nnbook.preprocessing import (MedianImputer, OneHotEncoder,  # noqa: E402
                                  StandardScaler, train_test_split)

NUMERIC = ["area_m2", "rooms", "year_built"]


# Step 1: one spelling per district.
def normalise_districts(df):
    return df.assign(district=df["district"].str.strip().str.lower())


# Step 2: the same listing entered twice is one listing.
def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


# Step 3: an area above 400 m2 is a units mistake; treat it as missing.
def mark_impossible(df):
    return df.assign(area_m2=df["area_m2"].where(df["area_m2"] <= 400))


# Step 4: split BEFORE any statistic is computed.
def split(df, seed=0):
    train, val = train_test_split(len(df), test_size=0.2, seed=seed)
    return df.iloc[train], df.iloc[val]


# Step 5: learn medians, scaling, and categories from the training rows only.
def fit_preparation(train_df, numeric=NUMERIC):
    values = train_df[numeric].to_numpy(dtype=float)
    imputer = MedianImputer().fit(values)
    scaler = StandardScaler().fit(imputer.transform(values))
    encoder = OneHotEncoder().fit(train_df["district"])
    return imputer, scaler, encoder


def transform(df, preparation, numeric=NUMERIC):
    imputer, scaler, encoder = preparation
    values = scaler.transform(imputer.transform(df[numeric].to_numpy(dtype=float)))
    return np.column_stack([np.ones(len(df)), values, encoder.transform(df["district"])])


# Step 6: least squares (chapter 2) and the validation error.
def fit_linear(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def rmse(X, y, theta):
    return float(np.sqrt(np.mean((X @ theta - y) ** 2)))


def run(numeric=NUMERIC, leak=False):
    """The whole exercise; returns the cleaned table, the preparation, and the validation RMSE."""
    df = mark_impossible(remove_duplicates(normalise_districts(make_listings())))
    if leak:   # Step 7: use a column that is known only after the sale
        numeric = numeric + ["valuation_after_sale_k"]
    train_df, val_df = split(df)
    preparation = fit_preparation(train_df, numeric)
    X_train, X_val = transform(train_df, preparation, numeric), transform(val_df, preparation, numeric)
    y_train, y_val = train_df["price_k"].to_numpy(), val_df["price_k"].to_numpy()
    theta = fit_linear(X_train, y_train)
    return df, preparation, rmse(X_val, y_val, theta)
