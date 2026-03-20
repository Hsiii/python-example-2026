from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    Dataset = object  # type: ignore[assignment]


class SurvivalTensorDataset(Dataset):
    """Torch dataset for tabular, signal, or multimodal survival training."""

    def __init__(
        self,
        *,
        durations: np.ndarray,
        events: np.ndarray,
        tabular: np.ndarray | None = None,
        signal: np.ndarray | None = None,
        extra_fields: dict[str, np.ndarray] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to instantiate SurvivalTensorDataset.")

        self.durations = torch.as_tensor(durations, dtype=torch.float32)
        self.events = torch.as_tensor(events, dtype=torch.float32)
        self.tabular = None if tabular is None else torch.as_tensor(tabular, dtype=torch.float32)
        self.signal = None if signal is None else torch.as_tensor(signal, dtype=torch.float32)
        self.extra_fields = {
            key: torch.as_tensor(value) for key, value in (extra_fields or {}).items()
        }

    def __len__(self) -> int:
        return int(self.durations.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "duration": self.durations[index],
            "event": self.events[index],
        }
        if self.tabular is not None:
            batch["tabular"] = self.tabular[index]
        if self.signal is not None:
            batch["signal"] = self.signal[index]
        for key, value in self.extra_fields.items():
            batch[key] = value[index]
        return batch