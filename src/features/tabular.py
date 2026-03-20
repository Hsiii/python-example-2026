from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from helper_code import get_standardized_ethnicity, get_standardized_race, load_age, load_bmi, load_sex


TABULAR_FEATURE_NAMES = [
    "age",
    "bmi",
    "sex_female",
    "sex_male",
    "sex_unknown",
    "race_asian",
    "race_black",
    "race_other",
    "race_unavailable",
    "race_white",
    "ethnicity_hispanic",
    "ethnicity_not_hispanic",
    "ethnicity_unavailable",
]


def extract_demographic_features(data: Mapping[str, Any]) -> np.ndarray:
    age = float(load_age(dict(data)))
    bmi = float(load_bmi(dict(data)))

    sex = load_sex(dict(data))
    sex_features = np.zeros(3, dtype=np.float64)
    if sex == "Female":
        sex_features[0] = 1.0
    elif sex == "Male":
        sex_features[1] = 1.0
    else:
        sex_features[2] = 1.0

    race = get_standardized_race(dict(data)).lower()
    race_features = np.zeros(5, dtype=np.float64)
    race_mapping = {
        "asian": 0,
        "black": 1,
        "others": 2,
        "unavailable": 3,
        "white": 4,
    }
    race_features[race_mapping.get(race, 2)] = 1.0

    ethnicity = get_standardized_ethnicity(dict(data)).lower()
    ethnicity_features = np.zeros(3, dtype=np.float64)
    if ethnicity == "hispanic":
        ethnicity_features[0] = 1.0
    elif ethnicity == "not hispanic":
        ethnicity_features[1] = 1.0
    else:
        ethnicity_features[2] = 1.0

    return np.concatenate([[age, bmi], sex_features, race_features, ethnicity_features]).astype(np.float64)


def build_tabular_matrix(records: list[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    if not records:
        return np.zeros((0, len(TABULAR_FEATURE_NAMES)), dtype=np.float64), list(TABULAR_FEATURE_NAMES)

    features = [extract_demographic_features(record) for record in records]
    matrix = np.vstack(features).astype(np.float64)
    return matrix, list(TABULAR_FEATURE_NAMES)


def tabular_matrix_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    records = frame.to_dict("records")
    return build_tabular_matrix(records)
