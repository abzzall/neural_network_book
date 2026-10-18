"""Run every chapter 4 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run(script, *args):
    result = subprocess.run([sys.executable, script, *args], cwd=HERE,
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_messy_data():
    assert run("messy_data.py") == [
        "rows = 515, duplicated rows = 15",
        "missing: {'area_m2': 28, 'year_built': 37}",
        "district spellings = 16",
        "areas above 400 m2 = 12",
    ]


def test_cleaning_demo():
    assert run("cleaning_demo.py") == [
        "start: 8 rows, 6 district spellings",
        "after normalising text: 4 districts",
        "after removing duplicates: 6 rows",
        "median area used for 2 missing values: 56.5",
        "columns: ['area_m2', 'rooms', 'area_missing', 'district_centre', "
        "'district_east', 'district_north', 'district_south']",
    ]


def test_scaling_gd(tmp_path):
    assert run("scaling_gd.py", str(tmp_path)) == [
        "raw           condition number =   178.87, steps = 1327",
        "centred       condition number =     7.72, steps = 64",
        "standardised  condition number =     1.00, steps = 1",
    ]
    assert np.loadtxt(tmp_path / "loss_raw.dat", skiprows=1).shape == (400, 2)


def test_leakage_demo():
    assert run("leakage_demo.py") == [
        "mean validation R^2 over 50 datasets of pure noise:",
        "  features chosen on all data:      0.419",
        "  features chosen on training data: -0.314",
    ]


def test_feature_selection_demo():
    assert run("feature_selection_demo.py") == [
        "linear  |correlation| = 0.712, mutual information = 0.418 nats",
        "square  |correlation| = 0.014, mutual information = 0.272 nats",
        "noise   |correlation| = 0.015, mutual information = 0.000 nats",
    ]


def test_pca_demo():
    assert run("pca_demo.py") == [
        "explained variance ratio: [0.8153 0.1812 0.0035]",
        "matches scikit-learn: True",
        "shape before and after: (300, 3) (300, 2)",
    ]


def test_imbalance_demo():
    assert run("imbalance_demo.py") == [
        "positives: 20 of 2000",
        "accuracy of always predicting the majority: 0.9900",
        "positives in 200 plain 10% test splits: 0 to 6",
        "positives in 200 stratified 10% test splits: 2 to 2",
        "class weights: [ 0.505 50.   ]",
    ]


def test_pipeline_demo():
    assert run("pipeline_demo.py") == [
        "training rows = 400, validation rows = 100",
        "features: ['num__area_m2', 'num__rooms', 'num__year_built', 'cat__district_centre', "
        "'cat__district_east', 'cat__district_north', 'cat__district_south']",
        "validation RMSE = 30.95 thousand",
    ]


def test_kmeans_demo():
    assert run("kmeans_demo.py") == [
        "agreement with the true groups: raw features 0.825, standardised 1.000",
        "inertia 127.79, scikit-learn 127.79",
        "k = 2: inertia  621.29, silhouette 0.519",
        "k = 3: inertia  127.79, silhouette 0.733",
        "k = 4: inertia  111.78, silhouette 0.570",
        "k = 5: inertia   95.80, silhouette 0.450",
        "k = 6: inertia   81.93, silhouette 0.333",
    ]


def test_annotation_agreement():
    assert run("annotation_agreement.py") == [
        "balanced  : raw agreement 0.850, by chance 0.5000, kappa 0.700 (scikit-learn 0.700)",
        "rare class: raw agreement 0.900, by chance 0.8698, kappa 0.232 (scikit-learn 0.232)",
    ]
