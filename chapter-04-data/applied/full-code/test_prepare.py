"""Tests for the chapter 4 applied exercise. The same tests ship in based-on/."""

import numpy as np
import pandas as pd


def test_districts_have_one_spelling(load_script):
    m = load_script("prepare")
    df = m.normalise_districts(m.make_listings())
    assert sorted(df["district"].unique()) == ["centre", "east", "north", "south"]


def test_duplicates_and_impossible_areas_are_removed(load_script):
    m = load_script("prepare")
    df = m.mark_impossible(m.remove_duplicates(m.normalise_districts(m.make_listings())))
    assert not df.duplicated().any()
    assert (df["area_m2"].dropna() <= 400).all()


def test_split_is_disjoint_and_complete(load_script):
    m = load_script("prepare")
    df = m.remove_duplicates(m.normalise_districts(m.make_listings()))
    train, val = m.split(df)
    assert len(train) + len(val) == len(df)
    assert not set(train.index) & set(val.index)


def test_preparation_uses_training_rows_only(load_script):
    m = load_script("prepare")
    train = pd.DataFrame({"district": ["a", "b"], "area_m2": [1.0, 3.0],
                          "rooms": [1.0, 1.0], "year_built": [2000.0, np.nan]})
    imputer, scaler, encoder = m.fit_preparation(train)
    assert np.allclose(imputer.median_, [2.0, 1.0, 2000.0])
    assert encoder.categories_ == ["a", "b"]
    X = m.transform(train, (imputer, scaler, encoder))
    assert X.shape == (2, 6) and np.all(X[:, 0] == 1)


def test_the_leak_makes_validation_look_excellent(load_script):
    m = load_script("prepare")
    assert m.run(leak=True)[2] < 0.2 * m.run()[2]
