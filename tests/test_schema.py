from __future__ import annotations

import pandas as pd

from src.constants import PROJECT_ROOT
from src.validation import validate_schema


def test_raw_csv_schema_matches_contract():
    train = pd.read_csv(PROJECT_ROOT / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "test.csv")
    sample_submission = pd.read_csv(PROJECT_ROOT / "sample_submission.csv")

    result = validate_schema(train, test, sample_submission)

    assert result.train_shape == (1624, 212)
    assert result.test_shape == (406, 203)
    assert result.sample_submission_shape == (406, 10)

