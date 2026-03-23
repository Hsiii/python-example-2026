from __future__ import annotations

from src.features.tabular import TABULAR_FEATURE_NAMES, extract_demographic_features


def test_extract_demographic_features_shape() -> None:
    row = {
        "Age": 72,
        "BMI": 28.5,
        "Sex": "Female",
        "Race": "White",
        "Ethnicity": "Not Hispanic",
    }
    features = extract_demographic_features(row)
    assert features.shape == (len(TABULAR_FEATURE_NAMES),)
