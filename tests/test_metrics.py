from __future__ import annotations

import numpy as np

from src.evaluation.metrics import concordance_index, integrated_brier_score


def test_concordance_index_bounds() -> None:
    durations = np.asarray([1.0, 2.0, 3.0, 4.0])
    events = np.asarray([1, 1, 0, 1])
    risk_scores = np.asarray([0.9, 0.7, 0.2, 0.1])
    score = concordance_index(durations, risk_scores, events)
    assert 0.0 <= score <= 1.0


def test_integrated_brier_score_non_negative() -> None:
    durations = np.asarray([1.0, 2.0, 3.0, 4.0])
    events = np.asarray([1, 0, 1, 1])
    times = np.asarray([1.5, 2.5, 3.5])
    survival = np.asarray(
        [
            [0.8, 0.6, 0.4],
            [0.9, 0.8, 0.7],
            [0.7, 0.5, 0.3],
            [0.6, 0.4, 0.2],
        ]
    )
    score = integrated_brier_score(durations, events, survival, times)
    assert score >= 0.0
