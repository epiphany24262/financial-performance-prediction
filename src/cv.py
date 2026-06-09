from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold


def make_groupkfold_splits(groups, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1:
        raise ValueError("groups must be a one-dimensional array-like")
    unique_groups = np.unique(groups_arr)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups, got {len(unique_groups)}")
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(groups_arr), 1))
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(dummy_x, groups=groups_arr)]


def assert_no_group_leakage(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups,
) -> None:
    groups_arr = np.asarray(groups)
    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_groups = set(groups_arr[train_idx])
        valid_groups = set(groups_arr[valid_idx])
        leaked = train_groups & valid_groups
        if leaked:
            raise AssertionError(f"Fold {fold} leaks {len(leaked)} groups between train and validation")

