from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass(frozen=True)
class DatasetSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def train_val_test_split(
    frame: pd.DataFrame,
    *,
    group_col: str = "group_id",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> DatasetSplit:
    if len(frame) == 0:
        raise ValueError("Cannot split an empty dataset.")

    indices = np.arange(len(frame))
    groups = frame[group_col].astype(str).to_numpy()

    outer_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer_split.split(indices, groups=groups))

    if len(train_val_idx) == 0:
        raise ValueError("Training split is empty after holdout partitioning.")

    train_val_groups = groups[train_val_idx]
    relative_val_size = val_size / max(1.0 - test_size, 1e-8)
    inner_split = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_rel, val_rel = next(inner_split.split(train_val_idx, groups=train_val_groups))

    return DatasetSplit(
        train_idx=np.asarray(train_val_idx[train_rel]),
        val_idx=np.asarray(train_val_idx[val_rel]),
        test_idx=np.asarray(test_idx),
    )


def cross_validation_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_col: str = "group_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = frame[group_col].astype(str).nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Not enough unique groups for {n_splits}-fold CV. Found {unique_groups}."
        )

    groups = frame[group_col].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(frame))
    return [(np.asarray(train_idx), np.asarray(val_idx)) for train_idx, val_idx in splitter.split(indices, groups=groups)]
