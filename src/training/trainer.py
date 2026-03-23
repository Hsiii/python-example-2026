from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    GradScaler = None
    autocast = None
    DataLoader = None

from src.training.losses import cox_ph_loss, deephit_ranking_loss, discrete_time_nll


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float


class TorchSurvivalTrainer:
    """Unified trainer for Cox and discrete-time neural survival models."""

    def __init__(
        self,
        *,
        objective: str,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        checkpoint_dir: str,
        use_mixed_precision: bool = True,
        ranking_weight: float = 0.0,
    ) -> None:
        if torch is None or DataLoader is None or GradScaler is None or autocast is None:
            raise ImportError("PyTorch is required for TorchSurvivalTrainer.")

        self.objective = objective
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_mixed_precision = use_mixed_precision
        self.ranking_weight = ranking_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

    def _compute_loss(self, outputs: "torch.Tensor", batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        if self.objective == "cox":
            return cox_ph_loss(outputs.squeeze(-1), batch["duration"], batch["event"])
        if self.objective == "discrete":
            loss = discrete_time_nll(outputs, batch["bin_index"], batch["event"])
            if self.ranking_weight > 0.0:
                loss = loss + self.ranking_weight * deephit_ranking_loss(outputs, batch["duration"], batch["event"])
            return loss
        raise ValueError(f"Unsupported survival objective: {self.objective}")

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, "torch.Tensor"]:
        moved: dict[str, "torch.Tensor"] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device)
        return moved

    def fit(self, model: "torch.nn.Module", train_dataset: Any, val_dataset: Any | None = None) -> TrainingHistory:
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scaler = GradScaler(enabled=self.use_mixed_precision and self.device.type == "cuda")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        train_history: list[float] = []
        val_history: list[float] = []

        for epoch in range(self.max_epochs):
            model.train()
            batch_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch = self._move_batch(batch)
                with autocast(device_type=self.device.type, enabled=self.use_mixed_precision and self.device.type == "cuda"):
                    outputs = model(batch)
                    loss = self._compute_loss(outputs, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
            train_history.append(train_loss)

            if val_loader is None:
                val_loss = train_loss
            else:
                model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self._move_batch(batch)
                        outputs = model(batch)
                        val_losses.append(float(self._compute_loss(outputs, batch).detach().cpu()))
                val_loss = float(np.mean(val_losses)) if val_losses else train_loss

            val_history.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), self.best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if self.best_checkpoint_path.is_file():
            model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))

        return TrainingHistory(
            train_loss=train_history,
            val_loss=val_history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        )
