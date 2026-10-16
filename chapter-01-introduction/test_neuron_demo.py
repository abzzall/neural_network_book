"""Run the chapter 1 demo as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path


def test_neuron_demo():
    out = subprocess.run([sys.executable, str(Path(__file__).with_name("neuron_demo.py"))],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [
        "dry and hot          z = +1.65, sigmoid = 0.839 -> irrigate",
        "wet                  z = -1.90, sigmoid = 0.130 -> do not irrigate",
        "dry, rain expected   z = -0.95, sigmoid = 0.279 -> do not irrigate",
    ]
