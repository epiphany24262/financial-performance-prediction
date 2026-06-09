from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMNS
from .metrics import mean_target_r2, r2_by_target


ACCOUNTING_ADJUSTMENTS = [
    "none",
    "balance_sheet_equity",
    "gross_profit_identity",
    "operating_income_identity",
    "income_statement_identity",
    "balance_and_income_identity",
]


def apply_accounting_adjustment(predictions: pd.DataFrame, adjustment: str) -> pd.DataFrame:
    if adjustment not in ACCOUNTING_ADJUSTMENTS:
        raise ValueError(f"Unknown accounting adjustment: {adjustment}")

    adjusted = predictions.copy()
    if adjustment in {"balance_sheet_equity", "balance_and_income_identity"}:
        adjusted["Q0_TOTAL_STOCKHOLDERS_EQUITY"] = (
            adjusted["Q0_TOTAL_ASSETS"] - adjusted["Q0_TOTAL_LIABILITIES"]
        )

    if adjustment in {"gross_profit_identity", "income_statement_identity", "balance_and_income_identity"}:
        adjusted["Q0_GROSS_PROFIT"] = adjusted["Q0_REVENUES"] - adjusted["Q0_COST_OF_REVENUES"]

    if adjustment in {"operating_income_identity", "income_statement_identity", "balance_and_income_identity"}:
        adjusted["Q0_OPERATING_INCOME"] = adjusted["Q0_GROSS_PROFIT"] - adjusted["Q0_OPERATING_EXPENSES"]

    return adjusted


def score_accounting_adjustments(
    y_true: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLUMNS,
) -> tuple[str, pd.DataFrame]:
    rows = []
    best_adjustment = "none"
    best_mean = -np.inf
    for adjustment in ACCOUNTING_ADJUSTMENTS:
        adjusted = apply_accounting_adjustment(predictions, adjustment)
        scores = r2_by_target(y_true, adjusted, targets=targets)
        mean_r2 = mean_target_r2(scores)
        row = {
            "adjustment": adjustment,
            "mean_r2": mean_r2,
        }
        row.update({f"r2_{target}": score for target, score in scores.items()})
        rows.append(row)
        if mean_r2 > best_mean:
            best_mean = mean_r2
            best_adjustment = adjustment
    return best_adjustment, pd.DataFrame(rows)


def identity_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "balance_sheet_assets_minus_liabilities_minus_equity": (
            predictions["Q0_TOTAL_ASSETS"]
            - predictions["Q0_TOTAL_LIABILITIES"]
            - predictions["Q0_TOTAL_STOCKHOLDERS_EQUITY"],
            predictions["Q0_TOTAL_ASSETS"],
        ),
        "gross_profit_revenues_minus_cost_minus_gross_profit": (
            predictions["Q0_REVENUES"]
            - predictions["Q0_COST_OF_REVENUES"]
            - predictions["Q0_GROSS_PROFIT"],
            predictions["Q0_REVENUES"],
        ),
        "operating_income_gross_profit_minus_expenses_minus_operating_income": (
            predictions["Q0_GROSS_PROFIT"]
            - predictions["Q0_OPERATING_EXPENSES"]
            - predictions["Q0_OPERATING_INCOME"],
            predictions["Q0_REVENUES"],
        ),
    }
    rows = []
    for name, (error, denominator) in specs.items():
        denominator = denominator.astype(float).abs().replace(0, np.nan)
        rel_error = (error.astype(float) / denominator).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "identity": name,
                "mean_abs_error": float(error.abs().mean()),
                "median_abs_error": float(error.abs().median()),
                "mean_abs_relative_error": float(rel_error.abs().mean()),
                "median_abs_relative_error": float(rel_error.abs().median()),
                "p95_abs_relative_error": float(rel_error.abs().quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)
