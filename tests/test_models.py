from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import build_estimator, split_feature_columns


def test_split_feature_columns_detects_numeric_and_categorical():
    frame = pd.DataFrame({"x": [1.0, 2.0], "flag": [True, False], "sector": ["A", "B"]})

    cols = split_feature_columns(frame)

    assert cols.numeric == ["x", "flag"]
    assert cols.categorical == ["sector"]


def test_sklearn_estimators_fit_and_predict_small_mixed_frame():
    n = 40
    X = pd.DataFrame(
        {
            "x1": np.linspace(1.0, 40.0, n),
            "x2": np.linspace(10.0, 400.0, n),
            "sector": ["A", "B", None, "C"] * 10,
        }
    )
    X.loc[3, "x1"] = np.nan
    y = pd.Series(np.linspace(1.0, 5.0, n))

    for model_kind in ["ridge", "hgb"]:
        estimator = build_estimator(model_kind, X)
        estimator.fit(X, y)
        pred = estimator.predict(X)

        assert pred.shape == (n,)
        assert np.isfinite(pred).all()
