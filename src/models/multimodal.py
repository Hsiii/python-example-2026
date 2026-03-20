from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.datasets import SurvivalTensorDataset
from src.models.base import ModelInput, SurvivalModel
from src.models.signal import CNN1DEncoder, TransformerSignalEncoder
from src.training.losses import discretize_durations, make_time_bins
from src.training.trainer import TorchSurvivalTrainer

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    nn = None


BaseModule = nn.Module if nn is not None else object


class _MultimodalNetwork(BaseModule):
    def __init__(
        self,
        *,
        tabular_dim: int,
        signal_channels: int,
        encoder_type: str,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if encoder_type == "cnn":
            self.signal_encoder = CNN1DEncoder(signal_channels, embedding_dim=embedding_dim)
        elif encoder_type == "transformer":
            self.signal_encoder = TransformerSignalEncoder(signal_channels, embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported multimodal encoder: {encoder_type}")

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        signal_embedding = self.signal_encoder(batch["signal"])
        tabular_embedding = self.tabular_encoder(batch["tabular"])
        combined = torch.cat([signal_embedding, tabular_embedding], dim=1)
        return self.head(combined)


class MultimodalSurvivalModel(SurvivalModel):
    def __init__(
        self,
        *,
        encoder_type: str = "cnn",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        objective: str = "cox",
        num_bins: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        max_epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: str,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for MultimodalSurvivalModel.")
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.objective = objective
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.time_bins: np.ndarray | None = None
        self.network: nn.Module | None = None

    def _prepare_tabular(self, values: np.ndarray, fit: bool) -> np.ndarray:
        transformed = self.imputer.fit_transform(values) if fit else self.imputer.transform(values)
        return self.scaler.fit_transform(transformed) if fit else self.scaler.transform(transformed)

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        train_tabular = self._prepare_tabular(train_data["tabular"], fit=True)
        output_dim = 1
        if self.objective == "discrete":
            self.time_bins = make_time_bins(train_data["duration"], self.num_bins)
            output_dim = len(self.time_bins) + 1

        self.network = _MultimodalNetwork(
            tabular_dim=train_tabular.shape[1],
            signal_channels=train_data["signal"].shape[1],
            encoder_type=self.encoder_type,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
        )
        train_dataset = SurvivalTensorDataset(
            tabular=train_tabular,
            signal=train_data["signal"],
            durations=train_data["duration"],
            events=train_data["event"],
            extra_fields=(
                {"bin_index": discretize_durations(train_data["duration"], self.time_bins)}
                if self.objective == "discrete" and self.time_bins is not None
                else None
            ),
        )

        val_dataset = None
        if val_data is not None:
            val_tabular = self._prepare_tabular(val_data["tabular"], fit=False)
            val_dataset = SurvivalTensorDataset(
                tabular=val_tabular,
                signal=val_data["signal"],
                durations=val_data["duration"],
                events=val_data["event"],
                extra_fields=(
                    {"bin_index": discretize_durations(val_data["duration"], self.time_bins)}
                    if self.objective == "discrete" and self.time_bins is not None
                    else None
                ),
            )

        trainer = TorchSurvivalTrainer(
            objective=self.objective,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            checkpoint_dir=self.checkpoint_dir,
        )
        history = trainer.fit(self.network, train_dataset, val_dataset)
        return {"best_val_loss": history.best_val_loss, "best_epoch": float(history.best_epoch)}

    def _predict_outputs(self, data: ModelInput) -> np.ndarray:
        if self.network is None or torch is None:
            raise RuntimeError("Model has not been fitted.")
        self.network.eval()
        tabular = self._prepare_tabular(data["tabular"], fit=False)
        with torch.no_grad():
            outputs = self.network(
                {
                    "tabular": torch.as_tensor(tabular, dtype=torch.float32),
                    "signal": torch.as_tensor(data["signal"], dtype=torch.float32),
                }
            )
        return outputs.cpu().numpy().astype(np.float64)

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        outputs = self._predict_outputs(data)
        if self.objective == "cox":
            return outputs.reshape(-1)
        hazards = 1.0 / (1.0 + np.exp(-outputs))
        return hazards.sum(axis=1)

    def predict_survival(self, data: ModelInput, times: np.ndarray) -> np.ndarray | None:
        if self.objective != "discrete" or self.time_bins is None:
            return None
        outputs = self._predict_outputs(data)
        hazards = 1.0 / (1.0 + np.exp(-outputs))
        discrete_survival = np.cumprod(1.0 - hazards, axis=1)
        full_time_grid = np.concatenate([self.time_bins, [max(times.max(), self.time_bins[-1])]])
        interpolated = np.zeros((outputs.shape[0], len(times)), dtype=np.float64)
        for row_index in range(outputs.shape[0]):
            interpolated[row_index] = np.interp(
                times,
                full_time_grid,
                np.concatenate([discrete_survival[row_index], [discrete_survival[row_index, -1]]]),
                left=1.0,
                right=discrete_survival[row_index, -1],
            )
        return interpolated

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Any:
        return joblib.load(path)
