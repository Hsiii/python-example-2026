from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.classical import (
    CoxPHSurvivalModel,
    RandomSurvivalForestModel,
    RegularizedCoxSurvivalModel,
    XGBoostSurvivalModel,
)
from src.models.deep import DeepHitStyleModel, DeepSurvModel, DiscreteTimeSurvivalModel
from src.models.multimodal import MultimodalSurvivalModel
from src.models.signal import SignalSurvivalModel


def build_model(model_config: dict[str, Any], run_dir: str) -> Any:
    name = model_config["name"]
    params = dict(model_config.get("params", {}))
    params.setdefault("checkpoint_dir", str(Path(run_dir) / "checkpoints" / name))

    if name == "cox_ph":
        return CoxPHSurvivalModel(**params)
    if name == "regularized_cox":
        return RegularizedCoxSurvivalModel(**params)
    if name == "random_survival_forest":
        return RandomSurvivalForestModel(**params)
    if name == "xgboost_survival":
        return XGBoostSurvivalModel(**params)
    if name == "deep_surv":
        return DeepSurvModel(**params)
    if name == "discrete_time":
        return DiscreteTimeSurvivalModel(**params)
    if name == "deep_hit":
        return DeepHitStyleModel(**params)
    if name == "cnn_survival":
        params.setdefault("encoder_type", "cnn")
        return SignalSurvivalModel(**params)
    if name == "transformer_survival":
        params.setdefault("encoder_type", "transformer")
        return SignalSurvivalModel(**params)
    if name == "multimodal_survival":
        return MultimodalSurvivalModel(**params)
    raise ValueError(f"Unknown model name: {name}")
