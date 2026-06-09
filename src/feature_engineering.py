from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import HISTORICAL_METRICS, HISTORICAL_QUARTERS, ID_COLUMN, METADATA_COLUMNS, TARGET_COLUMNS
from .validation import safe_ratio, signed_log1p, stable_row_hash


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


def historical_value_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for quarter in HISTORICAL_QUARTERS:
        for metric in HISTORICAL_METRICS:
            col = f"{quarter}_{metric}"
            if col in frame.columns:
                cols.append(col)
    return cols


def historical_fiscal_columns(frame: pd.DataFrame) -> list[str]:
    cols = ["Q0_fiscal_year_end"] if "Q0_fiscal_year_end" in frame.columns else []
    cols.extend([f"{quarter}_fiscal_year_end" for quarter in HISTORICAL_QUARTERS if f"{quarter}_fiscal_year_end" in frame.columns])
    return cols


def raw_history_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [*historical_fiscal_columns(frame), *historical_value_columns(frame)]


def historical_matrix(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = historical_columns_for_target(target)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing historical columns for {target}: {missing}")
    return replace_inf_with_nan(frame.loc[:, cols]).apply(pd.to_numeric, errors="coerce")


def _metric_history(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = [f"{quarter}_{metric}" for quarter in HISTORICAL_QUARTERS if f"{quarter}_{metric}" in frame.columns]
    return replace_inf_with_nan(frame.loc[:, cols]).apply(pd.to_numeric, errors="coerce")


def _row_slope(values: np.ndarray, max_points: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype="float64")
    lags = np.arange(1, values.shape[1] + 1, dtype="float64")
    for row_idx, row in enumerate(values):
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        x = lags[valid][:max_points]
        y = row[valid][:max_points]
        if len(y) < 2:
            continue
        slope, _ = np.polyfit(x, y, deg=1)
        out[row_idx] = slope
    return out


def _add_time_series_features(frame: pd.DataFrame, features: dict[str, object]) -> None:
    for metric in HISTORICAL_METRICS:
        hist = _metric_history(frame, metric)
        if hist.empty:
            continue
        prefix = f"hist_{metric}"
        features[f"{prefix}_last_available"] = hist.bfill(axis=1).iloc[:, 0]
        for n in [2, 4, 8]:
            subset = hist.iloc[:, :n]
            features[f"{prefix}_mean_last_{n}"] = subset.mean(axis=1)
            features[f"{prefix}_non_missing_last_{n}"] = subset.notna().sum(axis=1)
        features[f"{prefix}_median_last_4"] = hist.iloc[:, :4].median(axis=1)
        features[f"{prefix}_std_last_4"] = hist.iloc[:, :4].std(axis=1, ddof=0)
        features[f"{prefix}_std_last_8"] = hist.iloc[:, :8].std(axis=1, ddof=0)
        features[f"{prefix}_min_last_4"] = hist.iloc[:, :4].min(axis=1)
        features[f"{prefix}_max_last_4"] = hist.iloc[:, :4].max(axis=1)
        features[f"{prefix}_slope_last_4"] = _row_slope(hist.iloc[:, :4].to_numpy(dtype="float64"), max_points=4)
        features[f"{prefix}_slope_last_8"] = _row_slope(hist.iloc[:, :8].to_numpy(dtype="float64"), max_points=8)
        features[f"{prefix}_non_missing_count"] = hist.notna().sum(axis=1)
        features[f"{prefix}_missing_count"] = hist.isna().sum(axis=1)


def _add_change_features(frame: pd.DataFrame, features: dict[str, object]) -> None:
    for metric in HISTORICAL_METRICS:
        q1_col = f"Q1_{metric}"
        q2_col = f"Q2_{metric}"
        q3_col = f"Q3_{metric}"
        q5_col = f"Q5_{metric}"
        if q1_col not in frame.columns or q2_col not in frame.columns:
            continue
        q1 = pd.to_numeric(frame[q1_col], errors="coerce")
        q2 = pd.to_numeric(frame[q2_col], errors="coerce")
        q3 = pd.to_numeric(frame[q3_col], errors="coerce") if q3_col in frame.columns else None
        q5 = pd.to_numeric(frame[q5_col], errors="coerce") if q5_col in frame.columns else None
        prefix = f"chg_{metric}"
        features[f"{prefix}_q1_minus_q2"] = q1 - q2
        if q3 is not None:
            features[f"{prefix}_q2_minus_q3"] = q2 - q3
        if q5 is not None:
            features[f"{prefix}_q1_minus_q5"] = q1 - q5
        features[f"{prefix}_q1_over_q2_minus_1"] = safe_ratio(q1, q2) - 1
        if q3 is not None:
            features[f"{prefix}_q2_over_q3_minus_1"] = safe_ratio(q2, q3) - 1
        if q5 is not None:
            features[f"{prefix}_q1_over_q5_minus_1"] = safe_ratio(q1, q5) - 1


def _add_financial_ratio_features(frame: pd.DataFrame, features: dict[str, object]) -> None:
    ratio_specs = {
        "debt_ratio": ("TOTAL_LIABILITIES", "TOTAL_ASSETS"),
        "equity_ratio": ("TOTAL_STOCKHOLDERS_EQUITY", "TOTAL_ASSETS"),
        "current_ratio": ("TOTAL_CURRENT_ASSETS", "TOTAL_CURRENT_LIABILITIES"),
        "noncurrent_assets_ratio": ("TOTAL_NONCURRENT_ASSETS", "TOTAL_ASSETS"),
        "gross_margin": ("GROSS_PROFIT", "REVENUES"),
        "operating_margin": ("OPERATING_INCOME", "REVENUES"),
        "ebitda_margin": ("EBITDA", "REVENUES"),
        "cost_rate": ("COST_OF_REVENUES", "REVENUES"),
        "expense_rate": ("OPERATING_EXPENSES", "REVENUES"),
    }
    for quarter in ["Q1", "Q2", "Q4", "Q5"]:
        for ratio_name, (numerator, denominator) in ratio_specs.items():
            numerator_col = f"{quarter}_{numerator}"
            denominator_col = f"{quarter}_{denominator}"
            if numerator_col not in frame.columns or denominator_col not in frame.columns:
                continue
            features[f"{quarter}_{ratio_name}"] = safe_ratio(
                pd.to_numeric(frame[numerator_col], errors="coerce"),
                pd.to_numeric(frame[denominator_col], errors="coerce"),
            )


def _add_missing_features(frame: pd.DataFrame, features: dict[str, object]) -> None:
    history_cols = raw_history_feature_columns(frame)
    metadata_cols = [col for col in METADATA_COLUMNS if col in frame.columns]
    model_cols = [col for col in model_feature_columns(frame) if col in frame.columns]
    features["row_missing_count"] = frame[model_cols].isna().sum(axis=1)
    features["row_missing_ratio"] = frame[model_cols].isna().mean(axis=1)
    features["history_missing_count"] = frame[history_cols].isna().sum(axis=1)
    features["metadata_missing_count"] = frame[metadata_cols].isna().sum(axis=1) if metadata_cols else 0

    quarter_available = []
    for quarter in HISTORICAL_QUARTERS:
        quarter_cols = [f"{quarter}_{metric}" for metric in HISTORICAL_METRICS if f"{quarter}_{metric}" in frame.columns]
        if quarter_cols:
            quarter_available.append(frame[quarter_cols].notna().any(axis=1).astype(int))
    features["quarter_available_count"] = sum(quarter_available) if quarter_available else 0

    important = [
        "Q1_TOTAL_ASSETS",
        "Q1_TOTAL_LIABILITIES",
        "Q1_TOTAL_STOCKHOLDERS_EQUITY",
        "Q1_REVENUES",
        "Q1_EBITDA",
        "trailingPE",
    ]
    for col in important:
        if col in frame.columns:
            features[f"is_missing_{col}"] = frame[col].isna().astype(int)


def _signed_log_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    generated: dict[str, object] = {}
    numeric_cols = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
    for col in numeric_cols:
        if col.endswith("_fiscal_year_end") or col.startswith("is_missing_"):
            continue
        series = pd.to_numeric(features[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        if series.abs().quantile(0.95) <= 10:
            continue
        generated[f"slog_{col}"] = signed_log1p(series)
    return pd.DataFrame(generated, index=features.index)


def build_feature_frame(
    frame: pd.DataFrame,
    feature_set: str = "history_raw",
) -> pd.DataFrame:
    """Build deterministic features without fitting on validation or test data."""
    source = replace_inf_with_nan(frame.copy())
    if feature_set not in {"history_raw", "history_engineered", "history_metadata_engineered"}:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    base_cols = raw_history_feature_columns(source)
    if feature_set == "history_metadata_engineered":
        base_cols = [*base_cols, *[col for col in METADATA_COLUMNS if col in source.columns]]
    features = source.loc[:, base_cols].copy()

    if feature_set in {"history_engineered", "history_metadata_engineered"}:
        generated: dict[str, object] = {}
        _add_time_series_features(source, generated)
        _add_change_features(source, generated)
        _add_financial_ratio_features(source, generated)
        _add_missing_features(source, generated)
        if generated:
            features = pd.concat([features, pd.DataFrame(generated, index=source.index)], axis=1)
        signed_log_features = _signed_log_feature_frame(features)
        if not signed_log_features.empty:
            features = pd.concat([features, signed_log_features], axis=1)

    features = replace_inf_with_nan(features)
    leaked = [col for col in TARGET_COLUMNS if col in features.columns]
    if leaked:
        raise AssertionError(f"Q0 target columns leaked into features: {leaked}")
    return features
