"""Foreground metrics pooled across test pixels, matching single-class evaluation."""

from dataclasses import dataclass

import numpy as np

from .io import binary_array


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, prediction, ground_truth):
        prediction, ground_truth = binary_array(prediction), binary_array(ground_truth)
        if prediction.shape != ground_truth.shape:
            raise ValueError("Prediction and ground truth must have identical shapes.")
        self.tp += int(np.count_nonzero(prediction & ground_truth))
        self.fp += int(np.count_nonzero(prediction & ~ground_truth))
        self.fn += int(np.count_nonzero(~prediction & ground_truth))
        self.tn += int(np.count_nonzero(~prediction & ~ground_truth))

    def scores(self):
        # Undefined metrics (zero denominator) are reported as zero.
        return {
            "mIoU": 100.0 * self.tp / max(1, self.tp + self.fp + self.fn),
            "Precision": 100.0 * self.tp / max(1, self.tp + self.fp),
            "Recall": 100.0 * self.tp / max(1, self.tp + self.fn),
            "F1": 200.0 * self.tp / max(1, 2 * self.tp + self.fp + self.fn),
        }


def format_scores(scores):
    return " | ".join(f"{name}={value:.2f}" for name, value in scores.items())
