from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_SAMPLE_SUBMISSION_SHAPE,
    EXPECTED_TEST_SHAPE,
    EXPECTED_TRAIN_SHAPE,
    HISTORICAL_METRICS,
    HISTORICAL_QUARTERS,
    ID_COLUMN,
    METADATA_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGET_COLUMNS,
)

QUARTER_VALUE_PATTERN = re.compile(r"^Q([0-9]+)_([A-Z0-9_]+)$")
QUARTER_FYE_PATTERN = re.compile(r"^Q([0-9]+)_fiscal_year_end$")


@dataclass(frozen=True)
class SchemaCheckResult:
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    sample_submission_shape: tuple[int, int]
    target_columns: list[str]
    submission_columns: list[str]


def expected_quarter_value_columns(quarter: str) -> list[str]:
    return [f"{quarter}_{metric}" for metric in HISTORICAL_METRICS]


def expected_history_columns() -> list[str]:
    cols: list[str] = []
    for quarter in HISTORICAL_QUARTERS:
        cols.extend(expected_quarter_value_columns(quarter))
        cols.append(f"{quarter}_fiscal_year_end")
    return cols


def expected_test_columns() -> list[str]:
    return [ID_COLUMN, *METADATA_COLUMNS, "Q0_fiscal_year_end", *expected_history_columns()]


def validate_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> SchemaCheckResult:
    if train.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(f"Unexpected train shape: {train.shape}, expected {EXPECTED_TRAIN_SHAPE}")
    if test.shape != EXPECTED_TEST_SHAPE:
        raise ValueError(f"Unexpected test shape: {test.shape}, expected {EXPECTED_TEST_SHAPE}")
    if sample_submission.shape != EXPECTED_SAMPLE_SUBMISSION_SHAPE:
        raise ValueError(
            f"Unexpected sample_submission shape: {sample_submission.shape}, expected {EXPECTED_SAMPLE_SUBMISSION_SHAPE}"
        )

    missing_targets = [col for col in TARGET_COLUMNS if col not in train.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in train: {missing_targets}")

    if list(sample_submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            "sample_submission columns do not match the required order: "
            f"{list(sample_submission.columns)} != {SUBMISSION_COLUMNS}"
        )

    return SchemaCheckResult(
        train_shape=train.shape,
        test_shape=test.shape,
        sample_submission_shape=sample_submission.shape,
        target_columns=TARGET_COLUMNS,
        submission_columns=SUBMISSION_COLUMNS,
    )


def stable_row_hash(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_index(axis=1)
    return pd.util.hash_pandas_object(ordered, index=False)


def safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, eps: float = 1e-9):
    numerator_arr = np.asarray(numerator, dtype="float64")
    denominator_arr = np.asarray(denominator, dtype="float64")
    return np.where(np.abs(denominator_arr) < eps, np.nan, numerator_arr / denominator_arr)


def signed_log1p(values: pd.Series | np.ndarray):
    arr = np.asarray(values, dtype="float64")
    return np.sign(arr) * np.log1p(np.abs(arr))


def split_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    cols = list(columns)
    metadata = [c for c in cols if c in METADATA_COLUMNS]
    targets = [c for c in cols if c in TARGET_COLUMNS]
    id_cols = [c for c in cols if c == ID_COLUMN]
    historical = [c for c in cols if QUARTER_VALUE_PATTERN.match(c) or QUARTER_FYE_PATTERN.match(c)]
    other = [c for c in cols if c not in set(metadata + targets + id_cols + historical)]
    return {
        "id": id_cols,
        "metadata": metadata,
        "historical": historical,
        "targets": targets,
        "other": other,
    }


def quarter_of(column: str) -> str | None:
    match = QUARTER_VALUE_PATTERN.match(column) or QUARTER_FYE_PATTERN.match(column)
    if not match:
        return None
    return f"Q{match.group(1)}"


def metric_of(column: str) -> str | None:
    match = QUARTER_VALUE_PATTERN.match(column)
    if not match:
        return None
    return match.group(2)

