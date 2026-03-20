from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index as lifelines_concordance_index

try:
    from sksurv.metrics import cumulative_dynamic_auc
except ImportError:  # pragma: no cover - exercised only when scikit-survival is unavailable.
    cumulative_dynamic_auc = None

try:
    from sksurv.util import Surv
except ImportError:  # pragma: no cover - exercised only when scikit-survival is unavailable.
    Surv = None


def concordance_index(event_times: np.ndarray, risk_scores: np.ndarray, event_observed: np.ndarray) -> float:
    return float(lifelines_concordance_index(event_times, -risk_scores, event_observed))


def _censoring_km(event_times: np.ndarray, event_observed: np.ndarray) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    censoring_indicator = 1 - event_observed.astype(int)
    kmf.fit(event_times, event_observed=censoring_indicator)
    return kmf


def integrated_brier_score(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    survival_probabilities: np.ndarray,
    times: np.ndarray,
) -> float:
    if survival_probabilities.shape != (len(event_times), len(times)):
        raise ValueError("survival_probabilities must have shape [n_samples, n_times].")

    kmf = _censoring_km(event_times, event_observed)
    scores = []
    for time_index, time_point in enumerate(times):
        g_t = float(kmf.predict(time_point))
        g_t = max(g_t, 1e-6)
        predictions = survival_probabilities[:, time_index]

        event_before_t = (event_times <= time_point) & (event_observed == 1)
        alive_after_t = event_times > time_point

        event_weights = np.asarray([1.0 / max(float(kmf.predict(max(t - 1e-8, 0.0))), 1e-6) for t in event_times])
        case_term = event_before_t * event_weights * np.square(predictions)
        control_term = alive_after_t * (1.0 / g_t) * np.square(1.0 - predictions)
        scores.append(float(np.mean(case_term + control_term)))

    return float(np.trapezoid(scores, times) / max(times[-1] - times[0], 1e-8))


def time_dependent_auc(
    train_event_times: np.ndarray,
    train_event_observed: np.ndarray,
    test_event_times: np.ndarray,
    test_event_observed: np.ndarray,
    risk_scores: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    if cumulative_dynamic_auc is not None and Surv is not None:
        train_y = Surv.from_arrays(event=train_event_observed.astype(bool), time=train_event_times.astype(np.float64))
        test_y = Surv.from_arrays(event=test_event_observed.astype(bool), time=test_event_times.astype(np.float64))
        auc_values, mean_auc = cumulative_dynamic_auc(train_y, test_y, risk_scores, times)
        return {
            "times": times.tolist(),
            "auc": np.asarray(auc_values, dtype=np.float64).tolist(),
            "mean_auc": float(mean_auc),
        }

    kmf = _censoring_km(test_event_times, test_event_observed)
    auc_values = []
    for time_point in times:
        cases = (test_event_times <= time_point) & (test_event_observed == 1)
        controls = test_event_times > time_point
        if not np.any(cases) or not np.any(controls):
            auc_values.append(np.nan)
            continue

        case_weights = np.asarray([1.0 / max(float(kmf.predict(max(t - 1e-8, 0.0))), 1e-6) for t in test_event_times[cases]])
        control_weights = np.full(np.count_nonzero(controls), 1.0 / max(float(kmf.predict(time_point)), 1e-6))

        case_scores = risk_scores[cases]
        control_scores = risk_scores[controls]
        numerator = 0.0
        denominator = 0.0
        for case_index, case_score in enumerate(case_scores):
            for control_index, control_score in enumerate(control_scores):
                weight = case_weights[case_index] * control_weights[control_index]
                denominator += weight
                if case_score > control_score:
                    numerator += weight
                elif case_score == control_score:
                    numerator += 0.5 * weight
        auc_values.append(float(numerator / denominator) if denominator else np.nan)

    auc_array = np.asarray(auc_values, dtype=np.float64)
    return {
        "times": times.tolist(),
        "auc": auc_array.tolist(),
        "mean_auc": float(np.nanmean(auc_array)),
    }


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predictions: np.ndarray,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.95,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    metric_values = []
    for _ in range(n_bootstrap):
        indices = generator.integers(0, len(event_times), size=len(event_times))
        metric_values.append(
            metric_fn(event_times[indices], event_observed[indices], predictions[indices])
        )

    values = np.asarray(metric_values, dtype=np.float64)
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "mean": float(np.nanmean(values)),
        "lower": float(np.nanquantile(values, lower_q)),
        "upper": float(np.nanquantile(values, upper_q)),
    }


def subgroup_evaluation(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    risk_scores: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for group in pd.Series(groups).dropna().unique():
        mask = groups == group
        if np.count_nonzero(mask) < 3:
            continue
        rows.append(
            {
                "group": group,
                "n": int(np.count_nonzero(mask)),
                "c_index": concordance_index(event_times[mask], risk_scores[mask], event_observed[mask]),
            }
        )
    return pd.DataFrame(rows)
