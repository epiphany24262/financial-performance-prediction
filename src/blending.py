from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .constants import TARGET_COLUMNS


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def simplex_weights(n: int, step: float) -> list[tuple[float, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("step must divide 1.0")
    return [tuple(part / units for part in parts) for parts in _compositions(units, n)]


def masked_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_values = pd.to_numeric(y_true, errors="coerce")
    pred_values = pd.to_numeric(y_pred, errors="coerce")
    valid = true_values.notna() & pred_values.notna()
    if valid.sum() < 2:
        return math.nan
    return float(r2_score(true_values.loc[valid], pred_values.loc[valid]))


def blend_target(
    predictions: Mapping[str, pd.DataFrame],
    target: str,
    members: Sequence[str],
    weights: Sequence[float],
) -> pd.Series:
    if len(members) != len(weights):
        raise ValueError("members and weights must have the same length")
    out = None
    for member, weight in zip(members, weights):
        if member not in predictions:
            raise KeyError(f"Unknown blend member: {member}")
        if target not in predictions[member].columns:
            raise KeyError(f"Prediction member {member} missing target {target}")
        weighted = predictions[member][target].astype(float) * float(weight)
        out = weighted.copy() if out is None else out + weighted
    if out is None:
        raise ValueError("At least one blend member is required")
    return out.rename(target)


def apply_blend(
    predictions: Mapping[str, pd.DataFrame],
    config: Mapping[str, Mapping[str, object]],
    targets: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    blended = pd.DataFrame(index=next(iter(predictions.values())).index)
    for target in targets:
        target_config = config[target]
        blended[target] = blend_target(
            predictions,
            target,
            members=list(target_config["members"]),
            weights=list(target_config["weights"]),
        )
    return blended


def _candidate_weights(
    n_members: int,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
) -> list[tuple[str, tuple[float, ...]]]:
    coarse = [("coarse", weights) for weights in simplex_weights(n_members, coarse_step)]
    best_placeholder = tuple([math.nan] * n_members)
    # The fine grid is filtered after the coarse best is known per target.
    return [*coarse, ("fine_placeholder", best_placeholder)]


def search_target_blend_weights(
    y_true: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    targets: Sequence[str] = TARGET_COLUMNS,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    fine_radius: float = 0.05,
) -> tuple[dict[str, dict], pd.DataFrame]:
    members = list(predictions.keys())
    if not members:
        raise ValueError("predictions must contain at least one member")

    coarse_weights = simplex_weights(len(members), coarse_step)
    all_fine_weights = simplex_weights(len(members), fine_step)

    config: dict[str, dict] = {}
    grid_rows = []
    for target in targets:
        best_score = -math.inf
        best_weights: tuple[float, ...] | None = None

        for weights in coarse_weights:
            pred = blend_target(predictions, target, members, weights)
            score = masked_r2(y_true[target], pred)
            grid_rows.append(
                {
                    "target": target,
                    "stage": "coarse",
                    "members": ",".join(members),
                    "weights": ",".join(f"{weight:.2f}" for weight in weights),
                    "r2": score,
                }
            )
            if score > best_score:
                best_score = score
                best_weights = weights

        if best_weights is None:
            raise RuntimeError(f"No coarse blend weights found for {target}")

        for weights in all_fine_weights:
            if any(abs(weight - center) > fine_radius + 1e-12 for weight, center in zip(weights, best_weights)):
                continue
            pred = blend_target(predictions, target, members, weights)
            score = masked_r2(y_true[target], pred)
            grid_rows.append(
                {
                    "target": target,
                    "stage": "fine",
                    "members": ",".join(members),
                    "weights": ",".join(f"{weight:.2f}" for weight in weights),
                    "r2": score,
                }
            )
            if score > best_score:
                best_score = score
                best_weights = weights

        config[target] = {
            "members": members,
            "weights": [float(weight) for weight in best_weights],
            "oof_r2": float(best_score),
            "coarse_step": float(coarse_step),
            "fine_step": float(fine_step),
            "fine_radius": float(fine_radius),
        }

    return config, pd.DataFrame(grid_rows)


def validate_prediction_frame(frame: pd.DataFrame, targets: Sequence[str] = TARGET_COLUMNS) -> None:
    missing = [target for target in targets if target not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    values = frame.loc[:, list(targets)].to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError("Prediction frame contains NaN values")
    if np.isinf(values).any():
        raise ValueError("Prediction frame contains infinite values")
