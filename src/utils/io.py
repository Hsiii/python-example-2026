from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)


def save_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=False)
