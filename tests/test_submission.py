from __future__ import annotations

import pandas as pd

from src.constants import PROJECT_ROOT, SUBMISSION_COLUMNS


def test_sample_submission_column_order_is_preserved_contract():
    sample_submission = pd.read_csv(PROJECT_ROOT / "sample_submission.csv")

    assert list(sample_submission.columns) == SUBMISSION_COLUMNS
    assert sample_submission.shape == (406, 10)
    assert sample_submission["Id"].is_unique

