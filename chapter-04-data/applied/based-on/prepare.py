"""Chapter 4 applied exercise (starting point): clean, split, and prepare the listings; then fit a line."""

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
    raise NotImplementedError("Step 1: strip spaces and lower-case the district names")


# Step 2: the same listing entered twice is one listing.
def remove_duplicates(df):
    raise NotImplementedError("Step 2: drop exact duplicate rows and reset the index")


# Step 3: an area above 400 m2 is a units mistake; treat it as missing.
def mark_impossible(df):
    raise NotImplementedError("Step 3: replace areas above 400 m2 by NaN")


# Step 4: split BEFORE any statistic is computed.
def split(df, seed=0):
    raise NotImplementedError("Step 4: an 80/20 split with nnbook.preprocessing.train_test_split")


# Step 5: learn medians, scaling, and categories from the training rows only.
def fit_preparation(train_df, numeric=NUMERIC):
    raise NotImplementedError("Step 5: fit MedianImputer, StandardScaler and OneHotEncoder on the training rows")


def transform(df, preparation, numeric=NUMERIC):
    raise NotImplementedError("Step 5: ones column, imputed and scaled numbers, one-hot districts")


# Step 6: least squares (chapter 2) and the validation error.
def fit_linear(X, y):
    raise NotImplementedError("Step 6: least squares, as in chapter 2")


def rmse(X, y, theta):
    raise NotImplementedError("Step 6: root mean squared error")


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
