from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import TARGET_COLUMNS
from src.feature_engineering import (
    build_feature_frame,
    build_feature_group_id,
    historical_columns_for_target,
    replace_inf_with_nan,
)


def test_replace_inf_with_nan_converts_non_finite_values():
    frame = pd.DataFrame({"x": [1.0, np.inf, -np.inf]})

    cleaned = replace_inf_with_nan(frame)

    assert cleaned["x"].isna().sum() == 2
    assert cleaned.loc[0, "x"] == 1.0


def test_build_feature_group_id_groups_identical_feature_rows():
    frame = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "industry": ["A", "A", "B"],
            "Q1_REVENUES": [10.0, 10.0, 20.0],
            TARGET_COLUMNS[0]: [100.0, 200.0, 300.0],
        }
    )

    groups = build_feature_group_id(frame, feature_columns=["industry", "Q1_REVENUES"])

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_historical_columns_for_target_uses_q1_to_q10_order():
    cols = historical_columns_for_target("Q0_REVENUES")

    assert cols[0] == "Q1_REVENUES"
    assert cols[-1] == "Q10_REVENUES"
    assert len(cols) == 10


def test_build_feature_frame_excludes_targets_and_replaces_inf():
    frame = pd.read_csv("train.csv")
    frame.loc[0, "Q1_REVENUES"] = np.inf

    features = build_feature_frame(frame, feature_set="history_engineered")

    assert all(target not in features.columns for target in TARGET_COLUMNS)
    assert not np.isinf(features.select_dtypes(include=[np.number]).to_numpy()).any()
    assert "hist_REVENUES_mean_last_4" in features.columns
    assert "chg_REVENUES_q1_minus_q2" in features.columns
