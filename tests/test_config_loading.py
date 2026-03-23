from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config


def test_load_yaml_config() -> None:
    config = load_config(Path("configs") / "cox_baseline.yaml")
    assert config["model"]["name"] == "cox_ph"