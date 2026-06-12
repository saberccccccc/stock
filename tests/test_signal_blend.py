import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from run.forward_frozen_strategy import RawAverageBlendPredictor
from run.blend_alpha_jsonl import blend_percentiles


class FixedPredictor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def predict_alpha(self, sample, valid, regime):
        return self.values


def main():
    raw = FixedPredictor([3.0, 1.0, 2.0])
    average = FixedPredictor([1.0, 3.0, 2.0])
    blend = RawAverageBlendPredictor(raw, average, raw_weight=0.75)
    result = blend.predict_alpha({}, np.ones(3, dtype=bool), "normal")
    expected = np.asarray([0.75, 0.25, 0.5], dtype=np.float32)
    assert np.allclose(result, expected)
    assert int(np.argmax(result)) == 0

    left = np.asarray([1.0, 0.5, 0.0])
    right = np.asarray([0.25, 0.5, 1.0])
    assert np.allclose(
        blend_percentiles(left, right, "weighted", left_weight=0.75),
        [0.8125, 0.5, 0.25],
    )
    assert np.allclose(
        blend_percentiles(left, right, "minimum"),
        [0.25, 0.5, 0.0],
    )
    assert np.allclose(
        blend_percentiles(left, right, "geometric"),
        [0.5, 0.5, 0.0],
    )
    print("test_signal_blend.py: ALL PASSED")


if __name__ == "__main__":
    main()
