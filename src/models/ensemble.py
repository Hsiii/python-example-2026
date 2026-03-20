from __future__ import annotations

import numpy as np


def average_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise ValueError("No predictions were provided for ensembling.")
    stacked = np.vstack(predictions)
    return np.mean(stacked, axis=0)


def weighted_average_predictions(predictions: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(predictions) != len(weights):
        raise ValueError("Predictions and weights must have the same length.")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    stacked = np.vstack(predictions)
    return np.average(stacked, axis=0, weights=normalized_weights)
