from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.datasets import SurvivalTensorDataset
from src.models.base import ModelInput, SurvivalModel
from src.training.losses import discretize_durations, make_time_bins
from src.training.trainer import TorchSurvivalTrainer

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    nn = None


BaseModule = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for deep survival models.")


class MLPEncoder(BaseModule):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = previous_dim

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        return self.network(values)


class CoxMLP(BaseModule):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims, dropout)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        encoded = self.encoder(batch["tabular"])
        return self.head(encoded)


class DiscreteTimeMLP(BaseModule):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, num_bins: int) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims, dropout)
        self.head = nn.Linear(self.encoder.output_dim, num_bins)

    def forward(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        encoded = self.encoder(batch["tabular"])
        return self.head(encoded)


@dataclass
class TabularNormalizer:
    def __post_init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.imputer.fit_transform(values)
        return self.scaler.fit_transform(transformed)

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.imputer.transform(values)
        return self.scaler.transform(transformed)


class BaseTorchTabularSurvivalModel(SurvivalModel):
    def __init__(
        self,
        *,
        hidden_dims: list[int],
        dropout: float,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        checkpoint_dir: str,
    ) -> None:
        _require_torch()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.normalizer = TabularNormalizer()
        self.network: nn.Module | None = None
        self.history: dict[str, float] = {}

    def _preprocess(self, values: np.ndarray, fit: bool) -> np.ndarray:
        return self.normalizer.fit_transform(values) if fit else self.normalizer.transform(values)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Any:
        return joblib.load(path)


class DeepSurvModel(BaseTorchTabularSurvivalModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        values = self._preprocess(train_data["tabular"], fit=True)
        self.network = CoxMLP(values.shape[1], self.hidden_dims, self.dropout)
        train_dataset = SurvivalTensorDataset(
            tabular=values,
            durations=train_data["duration"],
            events=train_data["event"],
        )

        val_dataset = None
        if val_data is not None:
            val_values = self._preprocess(val_data["tabular"], fit=False)
            val_dataset = SurvivalTensorDataset(
                tabular=val_values,
                durations=val_data["duration"],
                events=val_data["event"],
            )

        trainer = TorchSurvivalTrainer(
            objective="cox",
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            checkpoint_dir=self.checkpoint_dir,
        )
        history = trainer.fit(self.network, train_dataset, val_dataset)
        self.history = {"best_val_loss": history.best_val_loss, "best_epoch": float(history.best_epoch)}
        return dict(self.history)

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        if self.network is None or torch is None:
            raise RuntimeError("Model has not been fitted.")
        self.network.eval()
        values = self._preprocess(data["tabular"], fit=False)
        with torch.no_grad():
            predictions = self.network({"tabular": torch.as_tensor(values, dtype=torch.float32)})
        return predictions.squeeze(-1).cpu().numpy().astype(np.float64)


class DiscreteTimeSurvivalModel(BaseTorchTabularSurvivalModel):
    def __init__(self, *, num_bins: int = 20, ranking_weight: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.ranking_weight = ranking_weight
        self.time_bins: np.ndarray | None = None

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        values = self._preprocess(train_data["tabular"], fit=True)
        self.time_bins = make_time_bins(train_data["duration"], self.num_bins)
        self.network = DiscreteTimeMLP(values.shape[1], self.hidden_dims, self.dropout, len(self.time_bins) + 1)

        train_dataset = SurvivalTensorDataset(
            tabular=values,
            durations=train_data["duration"],
            events=train_data["event"],
            extra_fields={
                "bin_index": discretize_durations(train_data["duration"], self.time_bins),
            },
        )

        val_dataset = None
        if val_data is not None:
            val_values = self._preprocess(val_data["tabular"], fit=False)
            val_dataset = SurvivalTensorDataset(
                tabular=val_values,
                durations=val_data["duration"],
                events=val_data["event"],
                extra_fields={
                    "bin_index": discretize_durations(val_data["duration"], self.time_bins),
                },
            )

        trainer = TorchSurvivalTrainer(
            objective="discrete",
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            checkpoint_dir=self.checkpoint_dir,
            ranking_weight=self.ranking_weight,
        )
        history = trainer.fit(self.network, train_dataset, val_dataset)
        self.history = {"best_val_loss": history.best_val_loss, "best_epoch": float(history.best_epoch)}
        return dict(self.history)

    def _predict_logits(self, data: ModelInput) -> np.ndarray:
        if self.network is None or torch is None:
            raise RuntimeError("Model has not been fitted.")
        self.network.eval()
        values = self._preprocess(data["tabular"], fit=False)
        with torch.no_grad():
            logits = self.network({"tabular": torch.as_tensor(values, dtype=torch.float32)})
        return logits.cpu().numpy().astype(np.float64)

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        logits = self._predict_logits(data)
        hazards = 1.0 / (1.0 + np.exp(-logits))
        return hazards.sum(axis=1)

    def predict_survival(self, data: ModelInput, times: np.ndarray) -> np.ndarray:
        if self.time_bins is None:
            raise RuntimeError("Time bins are unavailable because the model has not been fitted.")
        logits = self._predict_logits(data)
        hazards = 1.0 / (1.0 + np.exp(-logits))
        discrete_survival = np.cumprod(1.0 - hazards, axis=1)
        full_time_grid = np.concatenate([self.time_bins, [max(times.max(), self.time_bins[-1])]])

        interpolated = np.zeros((discrete_survival.shape[0], len(times)), dtype=np.float64)
        for row_index in range(discrete_survival.shape[0]):
            interpolated[row_index] = np.interp(
                times,
                full_time_grid,
                np.concatenate([discrete_survival[row_index], [discrete_survival[row_index, -1]]]),
                left=1.0,
                right=discrete_survival[row_index, -1],
            )
        return interpolated


class DeepHitStyleModel(DiscreteTimeSurvivalModel):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("ranking_weight", 0.1)
        super().__init__(**kwargs)
