from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


ModelInput = dict[str, np.ndarray]


class SurvivalModel(ABC):
    """Abstract interface implemented by all benchmarkable survival models."""

    @abstractmethod
    def fit(
        self,
        train_data: ModelInput,
        val_data: ModelInput | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def predict_risk(self, data: ModelInput) -> np.ndarray:
        raise NotImplementedError

    def predict_survival(self, data: ModelInput, times: np.ndarray) -> np.ndarray | None:
        return None

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save().")

    @classmethod
    def load(cls, path: str) -> Any:
        raise NotImplementedError(f"{cls.__name__} does not implement load().")
