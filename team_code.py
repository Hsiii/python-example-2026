#!/usr/bin/env python

# pyright: basic, reportMissingTypeStubs=false

import os
from typing import Any, Mapping

from src.utils.challenge_baseline import (
    load_challenge_model,
    run_challenge_model,
    train_challenge_model,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "channel_table.csv")


def train_model(data_folder: str, model_folder: str, verbose: bool, csv_path: str = DEFAULT_CSV_PATH) -> None:
    if verbose:
        print("Training modular Cox baseline with demographic and engineered PSG features...")
    train_challenge_model(data_folder, model_folder, csv_path)
    if verbose:
        print("Done.")


def load_model(model_folder: str, verbose: bool) -> dict[str, Any]:
    if verbose:
        print("Loading modular survival baseline...")
    return load_challenge_model(model_folder)


def run_model(model: Mapping[str, Any], record: Mapping[str, Any], data_folder: str, verbose: bool) -> tuple[int, float]:
    if verbose:
        patient_id = record.get("BidsFolder", "unknown")
        print(f"Scoring record: {patient_id}")
    return run_challenge_model(model, record, data_folder)