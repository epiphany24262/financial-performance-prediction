from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import PROJECT_ROOT
from src.cv import assert_no_group_leakage, make_groupkfold_splits
from src.feature_engineering import build_feature_group_id, model_feature_columns


def test_groupkfold_splits_do_not_leak_identical_feature_groups():
    train = pd.read_csv(PROJECT_ROOT / "train.csv")
    groups = build_feature_group_id(train, feature_columns=model_feature_columns(train))
    splits = make_groupkfold_splits(groups, n_splits=5)

    assert_no_group_leakage(splits, groups)

    covered = np.concatenate([valid_idx for _, valid_idx in splits])
    assert sorted(covered.tolist()) == list(range(len(train)))

