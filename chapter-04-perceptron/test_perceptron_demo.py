"""Run the chapter 4 demo exactly as a reader would and check its output."""

import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).with_name("perceptron_demo.py")

EXPECTED = [
    "AND: w = [3. 2.] b = -4.0",
    "AND: predictions = [-1 -1 -1  1]",
    "AND: errors per epoch = [2, 3, 3, 2, 2, 3, 2, 1, 0]",
    "Clouds: accuracy = 1.00, epochs = 2",
]


def test_demo_prints_the_documented_results():
    result = subprocess.run(
        [sys.executable, str(DEMO)], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == EXPECTED
