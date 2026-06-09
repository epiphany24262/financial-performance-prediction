from __future__ import annotations

import numpy as np
import pandas as pd

from src.accounting_checks import apply_accounting_adjustment, score_accounting_adjustments
from src.blending import apply_blend, search_target_blend_weights, simplex_weights
from src.constants import TARGET_COLUMNS


def test_simplex_weights_sum_to_one():
    weights = simplex_weights(3, step=0.5)

    assert (1.0, 0.0, 0.0) in weights
    assert (0.0, 0.0, 1.0) in weights
    assert all(np.isclose(sum(item), 1.0) for item in weights)


def test_search_target_blend_weights_selects_best_member():
    y_true = pd.DataFrame({target: [1.0, 2.0, 3.0, 4.0] for target in TARGET_COLUMNS})
    good = y_true.copy()
    bad = pd.DataFrame({target: [4.0, 3.0, 2.0, 1.0] for target in TARGET_COLUMNS})

    config, _ = search_target_blend_weights(
        y_true,
        {"good": good, "bad": bad},
        coarse_step=0.5,
        fine_step=0.5,
        fine_radius=0.5,
    )
    blended = apply_blend({"good": good, "bad": bad}, config)

    assert all(config[target]["weights"][0] == 1.0 for target in TARGET_COLUMNS)
    pd.testing.assert_frame_equal(blended, good)


def test_accounting_adjustment_can_be_selected_by_oof_score():
    y_true = pd.DataFrame(
        {
            "Q0_TOTAL_ASSETS": [10.0, 20.0, 30.0],
            "Q0_TOTAL_LIABILITIES": [4.0, 5.0, 6.0],
            "Q0_TOTAL_STOCKHOLDERS_EQUITY": [6.0, 15.0, 24.0],
            "Q0_GROSS_PROFIT": [3.0, 4.0, 5.0],
            "Q0_COST_OF_REVENUES": [7.0, 8.0, 9.0],
            "Q0_REVENUES": [10.0, 12.0, 14.0],
            "Q0_OPERATING_INCOME": [1.0, 2.0, 3.0],
            "Q0_OPERATING_EXPENSES": [2.0, 2.0, 2.0],
            "Q0_EBITDA": [1.5, 2.5, 3.5],
        }
    )
    pred = y_true.copy()
    pred["Q0_TOTAL_STOCKHOLDERS_EQUITY"] = [0.0, 0.0, 0.0]

    adjusted = apply_accounting_adjustment(pred, "balance_sheet_equity")
    selected, scores = score_accounting_adjustments(y_true, pred)

    assert adjusted["Q0_TOTAL_STOCKHOLDERS_EQUITY"].tolist() == [6.0, 15.0, 24.0]
    assert selected in {"balance_sheet_equity", "balance_and_income_identity"}
    assert scores.loc[scores["adjustment"] == selected, "mean_r2"].iloc[0] >= scores.loc[
        scores["adjustment"] == "none", "mean_r2"
    ].iloc[0]
