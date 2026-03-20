from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelInput, SurvivalModel

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import integrated_brier_score as sksurv_integrated_brier_score
    from sksurv.util import Surv
except ImportError:  # pragma: no cover - exercised only when scikit-survival is unavailable.
    RandomSurvivalForest = None
    sksurv_integrated_brier_score = None
    Surv = None

xgb_import_error: Exception | None = None

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - exercised when xgboost or libomp is unavailable.
    xgb = None
    xgb_import_error = exc


@dataclass
class TabularPreprocessor:
    scale: bool = True

    def __post_init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.imputer.fit_transform(values)
        return self.scaler.fit_transform(transformed) if self.scale else transformed

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.imputer.transform(values)
        return self.scaler.transform(transformed) if self.scale else transformed


class CoxPHSurvivalModel(SurvivalModel):
    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.preprocessor = TabularPreprocessor(scale=True)
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.feature_columns: list[str] = []

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        values = self.preprocessor.fit_transform(train_data["tabular"])
        self.feature_columns = [f"feature_{index}" for index in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=self.feature_columns)
        frame["duration"] = train_data["duration"]
        frame["event"] = train_data["event"]
        self.model.fit(frame, duration_col="duration", event_col="event")
        train_risk = self.predict_risk(train_data)
        return {"train_mean_risk": float(np.mean(train_risk))}

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        values = self.preprocessor.transform(data["tabular"])
        frame = pd.DataFrame(values, columns=self.feature_columns)
        partial_hazard = self.model.predict_partial_hazard(frame)
        return np.asarray(partial_hazard, dtype=np.float64).reshape(-1)

    def predict_survival(self, data: ModelInput, times: np.ndarray) -> np.ndarray:
        values = self.preprocessor.transform(data["tabular"])
        frame = pd.DataFrame(values, columns=self.feature_columns)
        survival_frame = self.model.predict_survival_function(frame, times=times)
        return survival_frame.to_numpy(dtype=np.float64).T

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CoxPHSurvivalModel":
        return joblib.load(path)


class RegularizedCoxSurvivalModel(CoxPHSurvivalModel):
    pass


class RandomSurvivalForestModel(SurvivalModel):
    def __init__(self, n_estimators: int = 200, min_samples_split: int = 10, random_state: int = 42) -> None:
        if RandomSurvivalForest is None or Surv is None:
            raise ImportError("scikit-survival is required for RandomSurvivalForestModel.")

        self.preprocessor = TabularPreprocessor(scale=False)
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        values = self.preprocessor.fit_transform(train_data["tabular"])
        survival_targets = Surv.from_arrays(
            event=train_data["event"].astype(bool),
            time=train_data["duration"].astype(np.float64),
        )
        self.model.fit(values, survival_targets)
        return {"oob_score": float(getattr(self.model, "oob_score_", np.nan))}

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        values = self.preprocessor.transform(data["tabular"])
        return np.asarray(self.model.predict(values), dtype=np.float64)

    def predict_survival(self, data: ModelInput, times: np.ndarray) -> np.ndarray:
        values = self.preprocessor.transform(data["tabular"])
        functions = self.model.predict_survival_function(values)
        predictions = np.zeros((len(functions), len(times)), dtype=np.float64)
        for row_index, survival_fn in enumerate(functions):
            predictions[row_index] = survival_fn(times)
        return predictions

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RandomSurvivalForestModel":
        return joblib.load(path)


class XGBoostSurvivalModel(SurvivalModel):
    def __init__(self, num_boost_round: int = 300, max_depth: int = 4, learning_rate: float = 0.05) -> None:
        if xgb is None:
            message = "xgboost is required for XGBoostSurvivalModel."
            if xgb_import_error is not None:
                message = f"{message} Original import error: {xgb_import_error}"
            raise ImportError(message)

        self.preprocessor = TabularPreprocessor(scale=False)
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any | None = None

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        values = self.preprocessor.fit_transform(train_data["tabular"])
        lower_bound = train_data["duration"].astype(np.float64)
        upper_bound = np.where(
            train_data["event"] > 0.5,
            train_data["duration"].astype(np.float64),
            np.inf,
        )
        dtrain = xgb.DMatrix(values)
        dtrain.set_float_info("label_lower_bound", lower_bound)
        dtrain.set_float_info("label_upper_bound", upper_bound)

        evaluation_sets = [(dtrain, "train")]
        if val_data is not None:
            val_values = self.preprocessor.transform(val_data["tabular"])
            dval = xgb.DMatrix(val_values)
            dval.set_float_info("label_lower_bound", val_data["duration"].astype(np.float64))
            dval.set_float_info(
                "label_upper_bound",
                np.where(val_data["event"] > 0.5, val_data["duration"].astype(np.float64), np.inf),
            )
            evaluation_sets.append((dval, "val"))

        self.model = xgb.train(
            {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "max_depth": self.max_depth,
                "eta": self.learning_rate,
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            },
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evaluation_sets,
            verbose_eval=False,
        )
        return {"train_samples": float(values.shape[0])}

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        values = self.preprocessor.transform(data["tabular"])
        predictions = self.model.predict(xgb.DMatrix(values))
        return -np.asarray(predictions, dtype=np.float64)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostSurvivalModel":
        return joblib.load(path)
