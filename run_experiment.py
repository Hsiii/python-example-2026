#!/usr/bin/env python

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helper_code import HEADERS
from src.data.loaders import build_dataset_index, dataframe_to_records, load_record_modalities
from src.data.splits import cross_validation_splits, train_val_test_split
from src.evaluation.benchmark import save_summary_table
from src.evaluation.metrics import (
    bootstrap_confidence_interval,
    concordance_index,
    integrated_brier_score,
    subgroup_evaluation,
    time_dependent_auc,
)
from src.features.psg_features import PSGFeatureConfig, PSGFeatureExtractor, standardize_psg_channels
from src.features.tabular import build_tabular_matrix
from src.models.factory import build_model
from src.models.signal import prepare_signal_array
from src.utils.config import load_config
from src.utils.io import ensure_directory, save_json, save_yaml
from src.utils.seed import set_global_seed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run survival benchmarking experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON experiment config.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the experiment output directory.")
    return parser


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir(config: dict[str, Any], output_dir_override: str | None) -> Path:
    experiment_name = config.get("experiment", {}).get("name", "benchmark_run")
    base_dir = output_dir_override or config.get("experiment", {}).get("output_dir", "experiments")
    run_dir = Path(base_dir) / f"{experiment_name}_{_timestamp()}"
    ensure_directory(run_dir)
    return run_dir


def _build_age_groups(frame: pd.DataFrame, bins: list[float]) -> np.ndarray:
    age_values = pd.to_numeric(frame[HEADERS["age"]], errors="coerce")
    return pd.cut(age_values, bins=bins, include_lowest=True).astype(str).to_numpy()


def _evaluation_times(train_durations: np.ndarray, num_points: int) -> np.ndarray:
    finite_values = train_durations[np.isfinite(train_durations)]
    lower = float(np.quantile(finite_values, 0.2))
    upper = float(np.quantile(finite_values, 0.8))
    if upper <= lower:
        upper = float(np.max(finite_values))
    return np.linspace(lower, upper, num_points, dtype=np.float64)


def _slice_input(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def _prepare_feature_bundle(config: dict[str, Any], frame: pd.DataFrame, run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    records = dataframe_to_records(frame, config["dataset"]["data_dir"])
    feature_config = config.get("features", {})
    names: dict[str, list[str]] = {}
    bundle: dict[str, np.ndarray] = {
        "duration": frame["time_to_event_survival"].to_numpy(dtype=np.float64),
        "event": frame["event_observed"].to_numpy(dtype=np.float64),
    }

    tabular_parts = []
    if feature_config.get("demographics", True):
        demographics_matrix, demographic_names = build_tabular_matrix([record.demographics for record in records])
        tabular_parts.append(demographics_matrix)
        names["demographics"] = demographic_names

    if feature_config.get("engineered_psg", True):
        extractor = PSGFeatureExtractor(
            PSGFeatureConfig(
                channel_table_path=config["dataset"]["channel_table_path"],
                cache_dir=str(run_dir / "feature_cache"),
            )
        )
        psg_matrix, psg_names = extractor.extract_matrix(records)
        tabular_parts.append(psg_matrix)
        names["engineered_psg"] = psg_names

    if tabular_parts:
        bundle["tabular"] = np.concatenate(tabular_parts, axis=1)

    if feature_config.get("raw_signal", False):
        signal_config = feature_config.get("signal", {})
        signal_rows = []
        for record in records:
            modalities = load_record_modalities(record)
            standardized_signals, standardized_fs = standardize_psg_channels(
                modalities["raw_psg"],
                modalities["raw_psg_fs"],
                config["dataset"]["channel_table_path"],
            )
            signal_rows.append(
                prepare_signal_array(
                    standardized_signals,
                    standardized_fs,
                    target_sample_rate=int(signal_config.get("target_sample_rate", 128)),
                    duration_seconds=int(signal_config.get("duration_seconds", 1800)),
                )
            )
        bundle["signal"] = np.stack(signal_rows, axis=0)
        names["signal"] = []

    return bundle, names


def _evaluate_model(
    model: Any,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    test_frame: pd.DataFrame,
    num_time_points: int,
    seed: int,
) -> dict[str, Any]:
    risk_scores = model.predict_risk(test_data)
    c_index = concordance_index(test_data["duration"], risk_scores, test_data["event"])
    c_index_ci = bootstrap_confidence_interval(
        lambda times, events, predictions: concordance_index(times, predictions, events),
        test_data["duration"],
        test_data["event"],
        risk_scores,
        seed=seed,
    )

    eval_times = _evaluation_times(train_data["duration"], num_time_points)
    survival_probabilities = model.predict_survival(test_data, eval_times)
    ibs = float("nan")
    auc_result = {"times": eval_times.tolist(), "auc": [float("nan")] * len(eval_times), "mean_auc": float("nan")}
    notes = ""
    if survival_probabilities is not None:
        ibs = integrated_brier_score(
            test_data["duration"],
            test_data["event"],
            survival_probabilities,
            eval_times,
        )
        auc_result = time_dependent_auc(
            train_data["duration"],
            train_data["event"],
            test_data["duration"],
            test_data["event"],
            risk_scores,
            eval_times,
        )
    else:
        notes = "Survival probabilities unavailable; IBS and time-dependent AUC were not computed."

    subgroup = subgroup_evaluation(
        test_data["duration"],
        test_data["event"],
        risk_scores,
        _build_age_groups(test_frame, [0, 65, 75, 120]),
    )
    return {
        "c_index": c_index,
        "c_index_ci": c_index_ci,
        "ibs": ibs,
        "auc": auc_result,
        "subgroup": subgroup,
        "notes": notes,
    }


def _run_single_model(
    model_config: dict[str, Any],
    config: dict[str, Any],
    data_bundle: dict[str, np.ndarray],
    frame: pd.DataFrame,
    run_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    split_config = config.get("split", {})
    split_mode = split_config.get("mode", "holdout")
    num_time_points = int(config.get("evaluation", {}).get("num_time_points", 5))
    results: list[dict[str, Any]] = []

    if split_mode == "cv":
        for fold_index, (train_idx, test_idx) in enumerate(
            cross_validation_splits(frame, n_splits=int(split_config.get("n_splits", 5))),
            start=1,
        ):
            train_data = _slice_input(data_bundle, train_idx)
            test_data = _slice_input(data_bundle, test_idx)
            model_run_dir = run_dir / model_config["name"] / f"fold_{fold_index}"
            model = build_model(model_config, str(model_run_dir))
            fit_metrics = model.fit(train_data, None)
            evaluation = _evaluate_model(model, train_data, test_data, frame.iloc[test_idx].reset_index(drop=True), num_time_points, seed)
            fold_result = {
                "model": model_config["name"],
                "fold": fold_index,
                "c_index": evaluation["c_index"],
                "ibs": evaluation["ibs"],
                "mean_auc": evaluation["auc"]["mean_auc"],
                "notes": evaluation["notes"],
                **fit_metrics,
            }
            results.append(fold_result)
            save_json(fold_result, model_run_dir / "metrics.json")
            if not evaluation["subgroup"].empty:
                evaluation["subgroup"].to_csv(model_run_dir / "subgroup_metrics.csv", index=False)
        return results

    split = train_val_test_split(
        frame,
        seed=seed,
        test_size=float(split_config.get("test_size", 0.2)),
        val_size=float(split_config.get("val_size", 0.1)),
    )
    train_data = _slice_input(data_bundle, split.train_idx)
    val_data = _slice_input(data_bundle, split.val_idx)
    test_data = _slice_input(data_bundle, split.test_idx)
    model_run_dir = run_dir / model_config["name"]
    model = build_model(model_config, str(model_run_dir))
    fit_metrics = model.fit(train_data, val_data)
    evaluation = _evaluate_model(model, train_data, test_data, frame.iloc[split.test_idx].reset_index(drop=True), num_time_points, seed)
    result = {
        "model": model_config["name"],
        "fold": 0,
        "c_index": evaluation["c_index"],
        "ibs": evaluation["ibs"],
        "mean_auc": evaluation["auc"]["mean_auc"],
        "notes": evaluation["notes"],
        **fit_metrics,
    }
    save_json(result, model_run_dir / "metrics.json")
    if not evaluation["subgroup"].empty:
        evaluation["subgroup"].to_csv(model_run_dir / "subgroup_metrics.csv", index=False)
    return [result]


def run_experiment(config: dict[str, Any], output_dir_override: str | None = None) -> pd.DataFrame:
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_global_seed(seed)

    run_dir = _make_run_dir(config, output_dir_override)
    save_yaml(config, run_dir / "resolved_config.yaml")

    frame = build_dataset_index(config["dataset"]["data_dir"])
    labeled = frame.dropna(subset=["time_to_event_survival", "event_observed"]).reset_index(drop=True)
    if labeled.empty:
        raise ValueError(
            "No labeled records were found. Populate the training dataset with Time_to_Event/Time_to_Last_Visit and label columns."
        )

    data_bundle, feature_names = _prepare_feature_bundle(config, labeled, run_dir)
    save_json(feature_names, run_dir / "feature_manifest.json")

    benchmark_models = config.get("benchmark", {}).get("models")
    model_configs = benchmark_models if benchmark_models else [config["model"]]

    all_results: list[dict[str, Any]] = []
    for model_config in model_configs:
        all_results.extend(_run_single_model(model_config, config, data_bundle, labeled, run_dir, seed))

    summary = save_summary_table(all_results, str(run_dir / "benchmark_summary.csv"))
    print(summary.to_markdown(index=False))
    return summary


def main() -> None:
    args = get_parser().parse_args()
    config = load_config(args.config)
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
