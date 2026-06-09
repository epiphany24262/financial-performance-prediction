from __future__ import annotations

import pandas as pd

from src.constants import TARGET_COLUMNS
from src.metrics import mean_target_r2, r2_by_target


def test_r2_by_target_returns_one_for_perfect_predictions():
    y_true = pd.DataFrame({target: [1.0, 2.0, 3.0] for target in TARGET_COLUMNS})
    y_pred = y_true.copy()

    scores = r2_by_target(y_true, y_pred)

    assert all(score == 1.0 for score in scores.values())
    assert mean_target_r2(scores) == 1.0


def test_mean_target_r2_ignores_missing_scores():
    scores = {"a": 0.2, "b": float("nan"), "c": 0.6}

    assert mean_target_r2(scores) == 0.4

