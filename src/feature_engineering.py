from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import HISTORICAL_QUARTERS, ID_COLUMN, TARGET_COLUMNS
from .validation import stable_row_hash


def replace_inf_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {ID_COLUMN, *TARGET_COLUMNS}
    return [col for col in frame.columns if col not in excluded]


def build_feature_group_id(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> pd.Series:
    if feature_columns is None:
        feature_columns = model_feature_columns(frame)
    features = replace_inf_with_nan(frame.loc[:, list(feature_columns)].copy())
    hashes = stable_row_hash(features)
    # Convert to dense deterministic labels for compact downstream storage.
    labels = pd.factorize(hashes, sort=True)[0]
    return pd.Series(labels, index=frame.index, name="group_id")


def metric_from_target(target: str) -> str:
    if not target.startswith("Q0_"):
        raise ValueError(f"Expected a Q0 target column, got {target!r}")
    return target.removeprefix("Q0_")


def historical_columns_for_target(target: str) -> list[str]:
    metric = metric_from_target(target)
    return [f"{quarter}_{metric}" for quarter in HISTORICAL_QUARTERS]


def historical_matrix(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = historical_columns_for_target(target)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing historical columns for {target}: {missing}")
    return replace_inf_with_nan(frame.loc[:, cols]).apply(pd.to_numeric, errors="coerce")
