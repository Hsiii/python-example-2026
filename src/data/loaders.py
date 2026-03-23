from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    load_signal_data,
)


SignalMap = dict[str, np.ndarray]
SamplingRateMap = dict[str, float]


@dataclass(frozen=True)
class SurvivalRecord:
    record_id: str
    patient_id: str
    session_id: str
    site_id: str
    data_dir: str
    demographics: dict[str, Any]
    raw_psg_path: str
    algorithmic_path: str
    human_annotations_path: str
    time_to_event: float | None
    event_observed: int | None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(number) and number > 0:
        return number
    return None


def build_dataset_index(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    demographics_path = root / DEMOGRAPHICS_FILE
    if not demographics_path.is_file():
        raise FileNotFoundError(f"Missing demographics file: {demographics_path}")

    frame = pd.read_csv(demographics_path)
    frame[HEADERS["session_id"]] = frame[HEADERS["session_id"]].astype(str)
    frame["record_id"] = (
        frame[HEADERS["bids_folder"]].astype(str)
        + "__ses-"
        + frame[HEADERS["session_id"]].astype(str)
    )
    frame["group_id"] = frame[HEADERS["bids_folder"]].astype(str)

    def _resolve_time(row: pd.Series) -> float | None:
        event = int(load_label(row.to_dict())) if HEADERS["label"] in frame.columns else 0
        if event == 1:
            candidates = [
                _safe_float(load_Time_to_Event(row.to_dict())),
                _safe_float(load_Time_to_Last_Visit(row.to_dict())),
            ]
        else:
            candidates = [
                _safe_float(load_Time_to_Last_Visit(row.to_dict())),
                _safe_float(load_Time_to_Event(row.to_dict())),
            ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    has_labels = HEADERS["label"] in frame.columns
    has_time = (
        HEADERS["time_to_event"] in frame.columns
        or HEADERS["time_to_last_visit"] in frame.columns
    )
    frame["event_observed"] = (
        frame.apply(lambda row: int(load_label(row.to_dict())), axis=1) if has_labels else np.nan
    )
    frame["time_to_event_survival"] = (
        frame.apply(_resolve_time, axis=1) if has_time else np.nan
    )

    frame["raw_psg_path"] = frame.apply(
        lambda row: str(
            root
            / PHYSIOLOGICAL_DATA_SUBFOLDER
            / str(row[HEADERS["site_id"]])
            / f"{row[HEADERS['bids_folder']]}_ses-{row[HEADERS['session_id']]}.edf"
        ),
        axis=1,
    )
    frame["algorithmic_path"] = frame.apply(
        lambda row: str(
            root
            / ALGORITHMIC_ANNOTATIONS_SUBFOLDER
            / str(row[HEADERS["site_id"]])
            / f"{row[HEADERS['bids_folder']]}_ses-{row[HEADERS['session_id']]}_caisr_annotations.edf"
        ),
        axis=1,
    )
    frame["human_annotations_path"] = frame.apply(
        lambda row: str(
            root
            / HUMAN_ANNOTATIONS_SUBFOLDER
            / str(row[HEADERS["site_id"]])
            / f"{row[HEADERS['bids_folder']]}_ses-{row[HEADERS['session_id']]}_expert_annotations.edf"
        ),
        axis=1,
    )
    return frame


def dataframe_to_records(frame: pd.DataFrame, data_dir: str | Path) -> list[SurvivalRecord]:
    demographics_path = Path(data_dir) / DEMOGRAPHICS_FILE
    records: list[SurvivalRecord] = []
    for _, row in frame.iterrows():
        patient_id = str(row[HEADERS["bids_folder"]])
        session_id = str(row[HEADERS["session_id"]])
        demographics = load_demographics(str(demographics_path), patient_id, session_id)
        time_value = row.get("time_to_event_survival")
        event_value = row.get("event_observed")
        records.append(
            SurvivalRecord(
                record_id=str(row["record_id"]),
                patient_id=patient_id,
                session_id=session_id,
                site_id=str(row[HEADERS["site_id"]]),
                data_dir=str(data_dir),
                demographics=demographics,
                raw_psg_path=str(row["raw_psg_path"]),
                algorithmic_path=str(row["algorithmic_path"]),
                human_annotations_path=str(row["human_annotations_path"]),
                time_to_event=_safe_float(time_value),
                event_observed=int(event_value) if pd.notna(event_value) else None,
            )
        )
    return records


def load_signal_bundle(path: str | Path) -> tuple[SignalMap, SamplingRateMap]:
    signal_path = Path(path)
    if not signal_path.is_file():
        return {}, {}
    signals, sampling_rates = load_signal_data(str(signal_path))
    return dict(signals), {str(key): float(value) for key, value in sampling_rates.items()}


def load_record_modalities(record: SurvivalRecord) -> dict[str, Any]:
    raw_psg, raw_psg_fs = load_signal_bundle(record.raw_psg_path)
    algorithmic, algorithmic_fs = load_signal_bundle(record.algorithmic_path)
    human, human_fs = load_signal_bundle(record.human_annotations_path)
    return {
        "record": record,
        "demographics": record.demographics,
        "raw_psg": raw_psg,
        "raw_psg_fs": raw_psg_fs,
        "algorithmic_annotations": algorithmic,
        "algorithmic_fs": algorithmic_fs,
        "human_annotations": human,
        "human_fs": human_fs,
    }
