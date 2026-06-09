from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .constants import TARGET_COLUMNS


def r2_by_target(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLUMNS,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for target in targets:
        true_values = pd.to_numeric(y_true[target], errors="coerce")
        pred_values = pd.to_numeric(y_pred[target], errors="coerce")
        valid = true_values.notna() & pred_values.notna()
        if valid.sum() < 2:
            scores[target] = math.nan
            continue
        scores[target] = float(r2_score(true_values.loc[valid], pred_values.loc[valid]))
    return scores


def mean_target_r2(scores: Mapping[str, float]) -> float:
    values = [value for value in scores.values() if pd.notna(value)]
    return float(np.mean(values)) if values else math.nan


def fold_score_frame(
    fold_predictions: list[dict],
    targets: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    rows = []
    for item in fold_predictions:
        scores = r2_by_target(item["y_true"], item["y_pred"], targets=targets)
        row = {
            "fold": item["fold"],
            "mean_r2": mean_target_r2(scores),
        }
        row.update({f"r2_{target}": score for target, score in scores.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarize_oof_scores(
    experiment_id: str,
    model_name: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    fold_scores: pd.DataFrame,
    runtime_seconds: float,
    targets: Sequence[str] = TARGET_COLUMNS,
    notes: str = "",
) -> dict[str, float | str]:
    target_scores = r2_by_target(y_true, y_pred, targets=targets)
    row: dict[str, float | str] = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "mean_r2": mean_target_r2(target_scores),
        "fold_mean_r2_std": float(fold_scores["mean_r2"].std(ddof=0)) if not fold_scores.empty else math.nan,
        "runtime_seconds": float(runtime_seconds),
        "notes": notes,
    }
    row.update({f"r2_{target}": score for target, score in target_scores.items()})
    return row

