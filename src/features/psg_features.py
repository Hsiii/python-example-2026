from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from helper_code import (
    derive_bipolar_signal,
    load_rename_rules,
    standardize_channel_names_rename_only,
)
from src.data.loaders import SurvivalRecord, load_record_modalities


PSG_FEATURE_NAMES = [
    "sleep_efficiency",
    "rem_latency_minutes",
    "rem_fraction",
    "n3_fraction",
    "ahi",
    "oxygen_desaturation_index_3",
    "mean_spo2",
    "min_spo2",
    "pct_spo2_below_90",
    "arousal_index",
    "sleep_fragmentation_index",
    "stage_transition_rate",
    "stage_transition_entropy",
    "wake_after_sleep_onset_minutes",
    "num_awakenings",
    "mean_sleep_run_minutes",
    "wake_to_sleep_probability",
    "n2_to_n3_probability",
    "rem_to_wake_probability",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "delta_theta_ratio",
    "delta_alpha_ratio",
]

STAGE_CODE_MAP = {
    "wake": 5,
    "rem": 4,
    "n1": 3,
    "n2": 2,
    "n3": 1,
}


@dataclass(frozen=True)
class PSGFeatureConfig:
    channel_table_path: str
    cache_dir: str
    epoch_length_seconds: float = 30.0
    spo2_desaturation_threshold: float = 3.0
    signal_sample_rate: int = 128
    spectral_window_seconds: int = 4


def standardize_psg_channels(
    raw_signals: dict[str, np.ndarray],
    raw_sampling_rates: dict[str, float],
    channel_table_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not raw_signals:
        return {}, {}

    rename_rules = load_rename_rules(channel_table_path)
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        list(raw_signals.keys()), rename_rules
    )

    processed_signals: dict[str, np.ndarray] = {}
    processed_sampling_rates: dict[str, float] = {}
    for old_name, values in raw_signals.items():
        if old_name in cols_to_drop:
            continue
        new_name = rename_map.get(old_name, old_name.lower())
        processed_signals[new_name] = np.asarray(values, dtype=np.float64)
        processed_sampling_rates[new_name] = float(raw_sampling_rates.get(old_name, 0.0))

    bipolar_configs = [
        ("f3-m2", "f3", ["m2"]),
        ("f4-m1", "f4", ["m1"]),
        ("c3-m2", "c3", ["m2"]),
        ("c4-m1", "c4", ["m1"]),
        ("o1-m2", "o1", ["m2"]),
        ("o2-m1", "o2", ["m1"]),
        ("e1-m2", "e1", ["m2"]),
        ("e2-m1", "e2", ["m1"]),
        ("chin1-chin2", "chin1", ["chin2"]),
    ]
    for target, positive, negatives in bipolar_configs:
        if target in processed_signals or positive not in processed_signals:
            continue
        if not all(name in processed_signals for name in negatives):
            continue
        sampling_rates = [processed_sampling_rates[positive]] + [processed_sampling_rates[name] for name in negatives]
        if len(set(sampling_rates)) != 1:
            continue
        reference = (
            processed_signals[negatives[0]]
            if len(negatives) == 1
            else tuple(processed_signals[name] for name in negatives)
        )
        derived = derive_bipolar_signal(processed_signals[positive], reference)
        if derived is not None:
            processed_signals[target] = np.asarray(derived, dtype=np.float64)
            processed_sampling_rates[target] = float(processed_sampling_rates[positive])

    return processed_signals, processed_sampling_rates


def _count_binary_events(values: np.ndarray, total_hours: float) -> float:
    if values.size == 0 or total_hours <= 0:
        return 0.0
    binary = (values > 0).astype(np.int32)
    starts = np.diff(binary, prepend=0) == 1
    return float(np.count_nonzero(starts) / total_hours)


def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _bandpower(values: np.ndarray, sampling_rate: float, low: float, high: float, window_seconds: int) -> float:
    if values.size < max(int(sampling_rate * window_seconds), 8) or sampling_rate <= 0:
        return 0.0
    freqs, power = signal.welch(values, fs=sampling_rate, nperseg=min(len(values), int(sampling_rate * window_seconds)))
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def _first_available(mapping: dict[str, np.ndarray], candidates: list[str]) -> np.ndarray:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate].size > 0:
            return mapping[candidate]
    return np.asarray([], dtype=np.float64)


def _first_available_fs(mapping: dict[str, float], candidates: list[str]) -> float:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate] > 0:
            return float(mapping[candidate])
    return 0.0


def _stage_features(stage_signal: np.ndarray, epoch_length_seconds: float) -> dict[str, float]:
    if stage_signal.size == 0:
        return {name: 0.0 for name in PSG_FEATURE_NAMES[:19]}

    valid = stage_signal[np.isfinite(stage_signal)]
    valid = valid[valid < 9]
    if valid.size == 0:
        return {name: 0.0 for name in PSG_FEATURE_NAMES[:19]}

    total_epochs = float(valid.size)
    sleep_mask = valid != STAGE_CODE_MAP["wake"]
    sleep_efficiency = float(np.mean(sleep_mask))
    rem_fraction = float(np.mean(valid == STAGE_CODE_MAP["rem"]))
    n3_fraction = float(np.mean(valid == STAGE_CODE_MAP["n3"]))

    sleep_indices = np.where(sleep_mask)[0]
    rem_indices = np.where(valid == STAGE_CODE_MAP["rem"])[0]
    if sleep_indices.size > 0 and rem_indices.size > 0:
        rem_latency_epochs = rem_indices[0] - sleep_indices[0]
        rem_latency_minutes = max(rem_latency_epochs, 0) * epoch_length_seconds / 60.0
    else:
        rem_latency_minutes = 0.0

    transitions = np.column_stack([valid[:-1], valid[1:]]) if valid.size > 1 else np.empty((0, 2), dtype=np.float64)
    transition_count = float(np.count_nonzero(valid[:-1] != valid[1:])) if valid.size > 1 else 0.0
    hours_in_bed = (total_epochs * epoch_length_seconds) / 3600.0
    stage_transition_rate = transition_count / hours_in_bed if hours_in_bed > 0 else 0.0

    if transitions.size > 0:
        unique_pairs, counts = np.unique(transitions, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        stage_transition_entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum())
    else:
        unique_pairs = np.empty((0, 2), dtype=np.float64)
        counts = np.asarray([], dtype=np.float64)
        stage_transition_entropy = 0.0

    after_sleep = valid[sleep_indices[0] :] if sleep_indices.size > 0 else valid
    wake_after_sleep_onset_minutes = (
        float(np.count_nonzero(after_sleep == STAGE_CODE_MAP["wake"])) * epoch_length_seconds / 60.0
    )

    awakenings = 0
    if after_sleep.size > 1:
        awakenings = int(
            np.count_nonzero(
                (after_sleep[:-1] != STAGE_CODE_MAP["wake"]) & (after_sleep[1:] == STAGE_CODE_MAP["wake"])
            )
        )

    sleep_runs: list[int] = []
    current_run = 0
    for stage_code in after_sleep:
        if stage_code != STAGE_CODE_MAP["wake"]:
            current_run += 1
        elif current_run > 0:
            sleep_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        sleep_runs.append(current_run)
    mean_sleep_run_minutes = (
        float(np.mean(sleep_runs)) * epoch_length_seconds / 60.0 if sleep_runs else 0.0
    )

    transition_map = {
        (STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n1"]): 0.0,
        (STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n2"]): 0.0,
        (STAGE_CODE_MAP["n2"], STAGE_CODE_MAP["n3"]): 0.0,
        (STAGE_CODE_MAP["rem"], STAGE_CODE_MAP["wake"]): 0.0,
    }
    if transitions.size > 0:
        count_lookup = {tuple(pair.astype(int)): int(count) for pair, count in zip(unique_pairs, counts, strict=False)}
        wake_out = sum(
            count for pair, count in count_lookup.items() if pair[0] == STAGE_CODE_MAP["wake"]
        )
        n2_out = sum(count for pair, count in count_lookup.items() if pair[0] == STAGE_CODE_MAP["n2"])
        rem_out = sum(count for pair, count in count_lookup.items() if pair[0] == STAGE_CODE_MAP["rem"])
        transition_map[(STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n1"])] = (
            count_lookup.get((STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n1"]), 0) / wake_out if wake_out else 0.0
        )
        transition_map[(STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n2"])] = (
            count_lookup.get((STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n2"]), 0) / wake_out if wake_out else 0.0
        )
        transition_map[(STAGE_CODE_MAP["n2"], STAGE_CODE_MAP["n3"])] = (
            count_lookup.get((STAGE_CODE_MAP["n2"], STAGE_CODE_MAP["n3"]), 0) / n2_out if n2_out else 0.0
        )
        transition_map[(STAGE_CODE_MAP["rem"], STAGE_CODE_MAP["wake"])] = (
            count_lookup.get((STAGE_CODE_MAP["rem"], STAGE_CODE_MAP["wake"]), 0) / rem_out if rem_out else 0.0
        )

    wake_to_sleep_probability = (
        transition_map[(STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n1"])]
        + transition_map[(STAGE_CODE_MAP["wake"], STAGE_CODE_MAP["n2"])]
    )
    sleep_fragmentation_index = (
        awakenings + transition_count
    ) / max(hours_in_bed, 1e-8)

    return {
        "sleep_efficiency": sleep_efficiency,
        "rem_latency_minutes": rem_latency_minutes,
        "rem_fraction": rem_fraction,
        "n3_fraction": n3_fraction,
        "ahi": 0.0,
        "oxygen_desaturation_index_3": 0.0,
        "mean_spo2": 0.0,
        "min_spo2": 0.0,
        "pct_spo2_below_90": 0.0,
        "arousal_index": 0.0,
        "sleep_fragmentation_index": float(sleep_fragmentation_index),
        "stage_transition_rate": float(stage_transition_rate),
        "stage_transition_entropy": float(stage_transition_entropy),
        "wake_after_sleep_onset_minutes": float(wake_after_sleep_onset_minutes),
        "num_awakenings": float(awakenings),
        "mean_sleep_run_minutes": float(mean_sleep_run_minutes),
        "wake_to_sleep_probability": float(wake_to_sleep_probability),
        "n2_to_n3_probability": float(transition_map[(STAGE_CODE_MAP["n2"], STAGE_CODE_MAP["n3"])]),
        "rem_to_wake_probability": float(transition_map[(STAGE_CODE_MAP["rem"], STAGE_CODE_MAP["wake"])]),
    }


def _spo2_features(spo2_signal: np.ndarray, sampling_rate: float, threshold: float) -> dict[str, float]:
    if spo2_signal.size == 0 or sampling_rate <= 0:
        return {
            "oxygen_desaturation_index_3": 0.0,
            "mean_spo2": 0.0,
            "min_spo2": 0.0,
            "pct_spo2_below_90": 0.0,
        }

    clipped = np.clip(spo2_signal.astype(np.float64), 50.0, 100.0)
    baseline = maximum_filter1d(clipped, size=max(int(sampling_rate * 30), 1))
    desaturation = (baseline - clipped) >= threshold
    total_hours = clipped.size / max(sampling_rate * 3600.0, 1e-8)
    odi = _count_binary_events(desaturation.astype(np.int32), total_hours)
    return {
        "oxygen_desaturation_index_3": float(odi),
        "mean_spo2": _safe_nanmean(clipped),
        "min_spo2": float(np.nanmin(clipped)),
        "pct_spo2_below_90": float(np.mean(clipped < 90.0)),
    }


def _spectral_features(
    signals: dict[str, np.ndarray],
    sampling_rates: dict[str, float],
    window_seconds: int,
) -> dict[str, float]:
    eeg_candidates = ["f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1"]
    available = [candidate for candidate in eeg_candidates if candidate in signals and signals[candidate].size > 0]
    if not available:
        return {
            "delta_power": 0.0,
            "theta_power": 0.0,
            "alpha_power": 0.0,
            "beta_power": 0.0,
            "delta_theta_ratio": 0.0,
            "delta_alpha_ratio": 0.0,
        }

    band_values = []
    for candidate in available:
        values = signals[candidate]
        sampling_rate = sampling_rates.get(candidate, 0.0)
        band_values.append(
            [
                _bandpower(values, sampling_rate, 0.5, 4.0, window_seconds),
                _bandpower(values, sampling_rate, 4.0, 8.0, window_seconds),
                _bandpower(values, sampling_rate, 8.0, 12.0, window_seconds),
                _bandpower(values, sampling_rate, 12.0, 30.0, window_seconds),
            ]
        )

    bands = np.asarray(band_values, dtype=np.float64)
    delta_power, theta_power, alpha_power, beta_power = np.mean(bands, axis=0)
    return {
        "delta_power": float(delta_power),
        "theta_power": float(theta_power),
        "alpha_power": float(alpha_power),
        "beta_power": float(beta_power),
        "delta_theta_ratio": float(delta_power / max(theta_power, 1e-8)),
        "delta_alpha_ratio": float(delta_power / max(alpha_power, 1e-8)),
    }


class PSGFeatureExtractor:
    """Cached engineered PSG feature extractor for multimodal benchmarking."""

    def __init__(self, config: PSGFeatureConfig) -> None:
        self.config = config
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def feature_names(self) -> list[str]:
        return list(PSG_FEATURE_NAMES)

    def _cache_path(self, record: SurvivalRecord) -> Path:
        safe_name = record.record_id.replace("/", "_")
        return Path(self.config.cache_dir) / f"{safe_name}.joblib"

    def extract_record(self, record: SurvivalRecord, force_refresh: bool = False) -> dict[str, float]:
        cache_path = self._cache_path(record)
        if cache_path.is_file() and not force_refresh:
            cached = joblib.load(cache_path)
            return {str(key): float(value) for key, value in cached.items()}

        modalities = load_record_modalities(record)
        raw_psg, raw_fs = standardize_psg_channels(
            modalities["raw_psg"],
            modalities["raw_psg_fs"],
            self.config.channel_table_path,
        )

        annotations = modalities["algorithmic_annotations"] or modalities["human_annotations"]
        stage_signal = _first_available(
            annotations,
            ["stage_caisr", "stage_expert", "sleep_stage", "stage"],
        )
        stage_features = _stage_features(stage_signal.astype(np.float64), self.config.epoch_length_seconds)

        total_hours = stage_signal.size * self.config.epoch_length_seconds / 3600.0 if stage_signal.size else 0.0
        respiratory_signal = _first_available(
            annotations,
            ["resp_caisr", "resp_expert", "resp", "respiratory_event"],
        )
        arousal_signal = _first_available(
            annotations,
            ["arousal_caisr", "arousal_expert", "arousal"],
        )
        stage_features["ahi"] = _count_binary_events(respiratory_signal, total_hours)
        stage_features["arousal_index"] = _count_binary_events(arousal_signal, total_hours)

        spo2_signal = _first_available(raw_psg, ["spo2", "sao2"])
        spo2_fs = _first_available_fs(raw_fs, ["spo2", "sao2"])
        spo2_features = _spo2_features(
            spo2_signal,
            spo2_fs,
            self.config.spo2_desaturation_threshold,
        )

        spectral_features = _spectral_features(raw_psg, raw_fs, self.config.spectral_window_seconds)
        feature_map = {**stage_features, **spo2_features, **spectral_features}
        ordered_features = {name: float(feature_map.get(name, 0.0)) for name in PSG_FEATURE_NAMES}
        joblib.dump(ordered_features, cache_path)
        return ordered_features

    def extract_matrix(
        self,
        records: list[SurvivalRecord],
        force_refresh: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        if not records:
            return np.zeros((0, len(PSG_FEATURE_NAMES)), dtype=np.float64), list(PSG_FEATURE_NAMES)

        rows = []
        for record in records:
            feature_map = self.extract_record(record, force_refresh=force_refresh)
            rows.append([feature_map[name] for name in PSG_FEATURE_NAMES])
        return np.asarray(rows, dtype=np.float64), list(PSG_FEATURE_NAMES)