from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import signal as scipy_signal

from src.data.datasets import SurvivalTensorDataset
from src.models.base import ModelInput, SurvivalModel
from src.training.trainer import TorchSurvivalTrainer

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    nn = None


BaseModule = nn.Module if nn is not None else object


SIGNAL_CHANNEL_ORDER = ["f3-m2", "f4-m1", "c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin1-chin2", "airflow", "abd", "chest", "spo2"]


def prepare_signal_array(
    standardized_signals: dict[str, np.ndarray],
    standardized_sampling_rates: dict[str, float],
    *,
    target_sample_rate: int,
    duration_seconds: int,
    channel_order: list[str] | None = None,
) -> np.ndarray:
    channels = channel_order or SIGNAL_CHANNEL_ORDER
    target_length = target_sample_rate * duration_seconds
    stacked = []
    for channel_name in channels:
        values = standardized_signals.get(channel_name)
        sampling_rate = standardized_sampling_rates.get(channel_name, 0.0)
        if values is None or values.size == 0 or sampling_rate <= 0:
            stacked.append(np.zeros(target_length, dtype=np.float32))
            continue
        resampled = scipy_signal.resample_poly(values.astype(np.float64), target_sample_rate, int(round(sampling_rate)))
        trimmed = resampled[:target_length]
        if trimmed.size < target_length:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: trimmed.size] = trimmed.astype(np.float32)
            stacked.append(padded)
        else:
            stacked.append(trimmed.astype(np.float32))
    return np.stack(stacked, axis=0)


class CNN1DEncoder(BaseModule):
    def __init__(self, in_channels: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(128, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        encoded = self.network(values).squeeze(-1)
        return self.projection(encoded)


class TransformerSignalEncoder(BaseModule):
    def __init__(self, in_channels: int, embedding_dim: int = 128, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, embedding_dim, kernel_size=7, stride=4, padding=3)
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.embedding_dim = embedding_dim

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        projected = self.input_projection(values).transpose(1, 2)
        encoded = self.encoder(projected)
        return encoded.mean(dim=1)


class _SignalSurvivalNetwork(BaseModule):
    def __init__(self, encoder: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        embedding = self.encoder(batch["signal"])
        return self.head(embedding)


class SignalSurvivalModel(SurvivalModel):
    def __init__(
        self,
        *,
        encoder_type: str = "cnn",
        embedding_dim: int = 128,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        max_epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: str,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for SignalSurvivalModel.")
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.network: nn.Module | None = None

    def _build_network(self, input_channels: int) -> nn.Module:
        if self.encoder_type == "cnn":
            encoder = CNN1DEncoder(input_channels, embedding_dim=self.embedding_dim)
        elif self.encoder_type == "transformer":
            encoder = TransformerSignalEncoder(input_channels, embedding_dim=self.embedding_dim)
        else:
            raise ValueError(f"Unsupported signal encoder: {self.encoder_type}")
        return _SignalSurvivalNetwork(encoder, self.embedding_dim)

    def fit(self, train_data: ModelInput, val_data: ModelInput | None = None) -> dict[str, float]:
        signal_values = train_data["signal"]
        self.network = self._build_network(signal_values.shape[1])
        train_dataset = SurvivalTensorDataset(
            signal=signal_values,
            durations=train_data["duration"],
            events=train_data["event"],
        )
        val_dataset = None
        if val_data is not None:
            val_dataset = SurvivalTensorDataset(
                signal=val_data["signal"],
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
        return {"best_val_loss": history.best_val_loss, "best_epoch": float(history.best_epoch)}

    def predict_risk(self, data: ModelInput) -> np.ndarray:
        if self.network is None or torch is None:
            raise RuntimeError("Model has not been fitted.")
        self.network.eval()
        with torch.no_grad():
            predictions = self.network({"signal": torch.as_tensor(data["signal"], dtype=torch.float32)})
        return predictions.squeeze(-1).cpu().numpy().astype(np.float64)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Any:
        return joblib.load(path)
