from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from helper_code import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    HEADERS,
    HUMAN_ANNOTATIONS_SUBFOLDER,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    load_Time_to_Event,
    load_Time_to_Last_Visit,
    load_demographics,
    load_label,
)
from src.data.loaders import SurvivalRecord, build_dataset_index, dataframe_to_records
from src.features.psg_features import PSGFeatureConfig, PSGFeatureExtractor
from src.features.tabular import build_tabular_matrix
from src.models.classical import CoxPHSurvivalModel


def _choose_prediction_horizon(durations: np.ndarray, events: np.ndarray) -> float:
    event_durations = durations[events > 0.5]
    if event_durations.size > 0:
        return float(np.median(event_durations))
    return float(np.median(durations))


def _choose_probability_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.unique(np.concatenate([[0.5], probabilities])):
        predictions = probabilities >= threshold
        true_positive = np.sum((predictions == 1) & (labels == 1))
        false_positive = np.sum((predictions == 1) & (labels == 0))
        false_negative = np.sum((predictions == 0) & (labels == 1))
        denominator = 2 * true_positive + false_positive + false_negative
        score = (2 * true_positive / denominator) if denominator else 0.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _record_from_challenge_metadata(record: Mapping[str, Any], data_folder: str) -> SurvivalRecord:
    patient_id = str(record[HEADERS["bids_folder"]])
    session_id = str(record[HEADERS["session_id"]])
    site_id = str(record[HEADERS["site_id"]])
    demographics_path = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    demographics = load_demographics(demographics_path, patient_id, session_id)

    label = int(load_label(demographics)) if HEADERS["label"] in demographics else None
    candidates = [
        load_Time_to_Last_Visit(demographics),
        load_Time_to_Event(demographics),
    ]
    if label == 1:
        candidates = [load_Time_to_Event(demographics), load_Time_to_Last_Visit(demographics)]

    time_to_event = None
    for value in candidates:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            time_to_event = number
            break

    return SurvivalRecord(
        record_id=f"{patient_id}__ses-{session_id}",
        patient_id=patient_id,
        session_id=session_id,
        site_id=site_id,
        data_dir=data_folder,
        demographics=demographics,
        raw_psg_path=os.path.join(
            data_folder,
            PHYSIOLOGICAL_DATA_SUBFOLDER,
            site_id,
            f"{patient_id}_ses-{session_id}.edf",
        ),
        algorithmic_path=os.path.join(
            data_folder,
            ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
            site_id,
            f"{patient_id}_ses-{session_id}_caisr_annotations.edf",
        ),
        human_annotations_path=os.path.join(
            data_folder,
            HUMAN_ANNOTATIONS_SUBFOLDER,
            site_id,
            f"{patient_id}_ses-{session_id}_expert_annotations.edf",
        ),
        time_to_event=time_to_event,
        event_observed=label,
    )


def train_challenge_model(data_folder: str, model_folder: str, channel_table_path: str) -> dict[str, Any]:
    frame = build_dataset_index(data_folder)
    labeled = frame.dropna(subset=["time_to_event_survival", "event_observed"]).reset_index(drop=True)
    if labeled.empty:
        raise ValueError("No labeled survival records were found for training.")

    records = dataframe_to_records(labeled, data_folder)
    demographics_matrix, demographic_names = build_tabular_matrix([record.demographics for record in records])

    feature_config = PSGFeatureConfig(
        channel_table_path=channel_table_path,
        cache_dir=str(Path(model_folder) / "feature_cache"),
    )
    psg_extractor = PSGFeatureExtractor(feature_config)
    psg_matrix, psg_feature_names = psg_extractor.extract_matrix(records)

    combined_features = np.concatenate([demographics_matrix, psg_matrix], axis=1)
    durations = labeled["time_to_event_survival"].to_numpy(dtype=np.float64)
    events = labeled["event_observed"].to_numpy(dtype=np.float64)

    model = CoxPHSurvivalModel(penalizer=0.1)
    model.fit({"tabular": combined_features, "duration": durations, "event": events})

    horizon = _choose_prediction_horizon(durations, events)
    survival = model.predict_survival({"tabular": combined_features}, np.asarray([horizon], dtype=np.float64))
    probabilities = 1.0 - survival[:, 0]
    threshold = _choose_probability_threshold(probabilities, events)

    state = {
        "model": model,
        "prediction_horizon": horizon,
        "probability_threshold": threshold,
        "default_probability": float(np.clip(np.mean(probabilities), 0.0, 1.0)),
        "channel_table_path": channel_table_path,
        "demographic_feature_names": demographic_names,
        "psg_feature_names": psg_feature_names,
    }

    Path(model_folder).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": state}, Path(model_folder) / "model.sav", protocol=0)
    return state


def load_challenge_model(model_folder: str) -> dict[str, Any]:
    return joblib.load(Path(model_folder) / "model.sav")


def run_challenge_model(model_bundle: Mapping[str, Any], record: Mapping[str, Any], data_folder: str) -> tuple[int, float]:
    state = model_bundle["model"]
    challenge_record = _record_from_challenge_metadata(record, data_folder)
    feature_config = PSGFeatureConfig(
        channel_table_path=state["channel_table_path"],
        cache_dir=str(Path(data_folder) / ".cache" / "challenge_features"),
    )
    extractor = PSGFeatureExtractor(feature_config)
    psg_feature_map = extractor.extract_record(challenge_record)
    demographic_matrix, _ = build_tabular_matrix([challenge_record.demographics])
    psg_matrix = np.asarray([[psg_feature_map[name] for name in state["psg_feature_names"]]], dtype=np.float64)
    combined = np.concatenate([demographic_matrix, psg_matrix], axis=1)
    survival = state["model"].predict_survival(
        {"tabular": combined},
        np.asarray([state["prediction_horizon"]], dtype=np.float64),
    )
    probability = 1.0 - survival[0, 0] if survival is not None else state["default_probability"]
    probability = float(np.clip(probability, 0.0, 1.0))
    binary_output = int(probability >= state["probability_threshold"])
    return binary_output, probability
