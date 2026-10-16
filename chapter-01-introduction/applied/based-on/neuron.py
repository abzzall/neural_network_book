"""Chapter 1 applied exercise (starting point): how the bias moves a neuron's decisions."""

import numpy as np

W = np.array([-3.0, 2.0, -2.5])     # the irrigation neuron of chapter 1

# Eight days of readings: soil moisture, temperature / 40, chance of rain.
DAYS = np.array([[0.20, 0.75, 0.10], [0.80, 0.50, 0.60], [0.30, 0.60, 0.90],
                 [0.10, 0.90, 0.00], [0.50, 0.70, 0.20], [0.60, 0.40, 0.10],
                 [0.40, 0.80, 0.50], [0.90, 0.30, 0.00]])


# Step 1: the weighted sum of every day, before the bias.
def weighted_sums(days=DAYS, w=W):
    raise NotImplementedError("Step 1: days @ w")


# Step 2: irrigate (1) when the weighted sum plus the bias is positive.
def decisions(bias, days=DAYS, w=W):
    raise NotImplementedError("Step 2: 1 where the weighted sum plus the bias is positive")


# Step 3: the bias at which each day's decision flips.
def flip_bias(days=DAYS, w=W):
    raise NotImplementedError("Step 3: the bias that makes the sum exactly zero")


# Step 4: the number of irrigation days for a given bias.
def irrigation_days(bias, days=DAYS, w=W):
    raise NotImplementedError("Step 4: count the ones from decisions")
