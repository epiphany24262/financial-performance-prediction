from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


ModelKind = Literal["ridge", "hgb", "catboost"]


@dataclass(frozen=True)
class FeatureColumns:
    numeric: list[str]
    categorical: list[str]


def split_feature_columns(frame: pd.DataFrame) -> FeatureColumns:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [col for col in frame.columns if col not in numeric]
    return FeatureColumns(numeric=numeric, categorical=categorical)


def prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    cols = split_feature_columns(prepared).categorical
    for col in cols:
        prepared[col] = prepared[col].astype("string").fillna("Missing")
    return prepared, cols


def make_ridge_estimator(frame: pd.DataFrame, alpha: float = 10.0) -> Pipeline:
    cols = split_feature_columns(frame)
    transformers = []
    if cols.numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                cols.numeric,
            )
        )
    if cols.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cols.categorical,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=alpha, solver="svd", tol=1e-4, random_state=42)),
        ]
    )


def make_hgb_estimator(frame: pd.DataFrame) -> Pipeline:
    cols = split_feature_columns(frame)
    transformers = []
    if cols.numeric:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                cols.numeric,
            )
        )
    if cols.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                cols.categorical,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def make_catboost_estimator(
    iterations: int = 1000,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 8.0,
    random_seed: int = 42,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
    )


def build_estimator(model_kind: ModelKind, frame: pd.DataFrame):
    if model_kind == "ridge":
        return make_ridge_estimator(frame)
    if model_kind == "hgb":
        return make_hgb_estimator(frame)
    if model_kind == "catboost":
        return make_catboost_estimator()
    raise ValueError(f"Unknown model kind: {model_kind}")
