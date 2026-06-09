from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .constants import HISTORICAL_QUARTERS, TARGET_COLUMNS
from .feature_engineering import historical_columns_for_target, historical_matrix


BASELINE_IDS = ["B0", "B1", "B2", "B3", "B4"]


def _first_available(history: pd.DataFrame, fallback: float) -> pd.Series:
    out = history.bfill(axis=1).iloc[:, 0]
    return out.fillna(fallback).astype(float)


def predict_recent_copy(frame: pd.DataFrame, target: str, fallback: float) -> pd.Series:
    history = historical_matrix(frame, target)
    # B1 requires Q1, then Q2 -> ... -> Q10.
    ordered_cols = historical_columns_for_target(target)
    return _first_available(history.loc[:, ordered_cols], fallback=fallback)


def predict_seasonal_copy(frame: pd.DataFrame, target: str, fallback: float) -> pd.Series:
    history = historical_matrix(frame, target)
    metric = target.removeprefix("Q0_")
    ordered_quarters = ["Q4", "Q3", "Q5", "Q2", "Q6", "Q1", "Q7", "Q8", "Q9", "Q10"]
    ordered_cols = [f"{quarter}_{metric}" for quarter in ordered_quarters]
    return _first_available(history.loc[:, ordered_cols], fallback=fallback)


def predict_short_trend(frame: pd.DataFrame, target: str, fallback: float, max_points: int = 4) -> pd.Series:
    history = historical_matrix(frame, target)
    values = history.to_numpy(dtype="float64")
    preds = np.full(values.shape[0], float(fallback), dtype="float64")
    lags = np.arange(1, len(HISTORICAL_QUARTERS) + 1, dtype="float64")
    for row_idx, row in enumerate(values):
        valid = np.isfinite(row)
        if valid.sum() == 0:
            continue
        chosen_lags = lags[valid][:max_points]
        chosen_values = row[valid][:max_points]
        if len(chosen_values) == 1:
            preds[row_idx] = chosen_values[0]
            continue
        slope, intercept = np.polyfit(chosen_lags, chosen_values, deg=1)
        pred = intercept  # Q0 has lag 0.
        preds[row_idx] = pred if np.isfinite(pred) else fallback
    return pd.Series(preds, index=frame.index, name=target)


def predict_rule(
    rule_id: str,
    frame: pd.DataFrame,
    target: str,
    train_target_values: pd.Series,
) -> pd.Series:
    fallback_median = float(train_target_values.median())
    if rule_id == "B0":
        return pd.Series(float(train_target_values.mean()), index=frame.index, name=target)
    if rule_id == "B1":
        return predict_recent_copy(frame, target, fallback=fallback_median)
    if rule_id == "B2":
        return predict_seasonal_copy(frame, target, fallback=fallback_median)
    if rule_id == "B3":
        return predict_short_trend(frame, target, fallback=fallback_median)
    raise ValueError(f"Unknown baseline rule: {rule_id}")


def simplex_weights(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    weights = []
    for parts in itertools.product(range(units + 1), repeat=n):
        if sum(parts) == units:
            weights.append(tuple(part / units for part in parts))
    return weights


def blend_predictions(
    predictions: Mapping[str, pd.DataFrame],
    target: str,
    members: Sequence[str],
    weights: Sequence[float],
) -> pd.Series:
    if len(members) != len(weights):
        raise ValueError("members and weights must have the same length")
    out = None
    for member, weight in zip(members, weights):
        series = predictions[member][target].astype(float) * float(weight)
        out = series.copy() if out is None else out + series
    if out is None:
        raise ValueError("members must not be empty")
    return out.rename(target)


def _masked_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_values = pd.to_numeric(y_true, errors="coerce")
    pred_values = pd.to_numeric(y_pred, errors="coerce")
    valid = true_values.notna() & pred_values.notna()
    if valid.sum() < 2:
        return math.nan
    return float(r2_score(true_values.loc[valid], pred_values.loc[valid]))


def search_baseline_blend_weights(
    y_true: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    targets: Sequence[str] = TARGET_COLUMNS,
    step: float = 0.1,
) -> tuple[dict[str, dict], pd.DataFrame]:
    candidate_sets = [
        ("B1", ("B1",)),
        ("B2", ("B2",)),
        ("B3", ("B3",)),
        ("B1+B2", ("B1", "B2")),
        ("B1+B3", ("B1", "B3")),
        ("B1+B2+B3", ("B1", "B2", "B3")),
    ]
    weights_by_size = {size: simplex_weights(size, step=step) for size in (1, 2, 3)}
    config: dict[str, dict] = {}
    score_rows = []
    for target in targets:
        best_score = -math.inf
        best_members: tuple[str, ...] | None = None
        best_weights: tuple[float, ...] | None = None
        for candidate_name, members in candidate_sets:
            for weights in weights_by_size[len(members)]:
                pred = blend_predictions(predictions, target, members, weights)
                score = _masked_r2(y_true[target], pred)
                score_rows.append(
                    {
                        "target": target,
                        "candidate": candidate_name,
                        "members": ",".join(members),
                        "weights": ",".join(f"{weight:.1f}" for weight in weights),
                        "r2": score,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_members = members
                    best_weights = weights
        if best_members is None or best_weights is None:
            raise RuntimeError(f"No blend weights found for target {target}")
        config[target] = {
            "members": list(best_members),
            "weights": [float(weight) for weight in best_weights],
            "oof_r2": float(best_score),
            "grid_step": float(step),
        }
    return config, pd.DataFrame(score_rows)
