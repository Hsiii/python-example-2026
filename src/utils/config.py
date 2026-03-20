from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(file_handle)
        elif path.suffix.lower() == ".json":
            config = json.load(file_handle)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(config, dict):
        raise ValueError("Experiment config must deserialize to a dictionary.")
    return config


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
