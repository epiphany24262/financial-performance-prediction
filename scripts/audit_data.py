from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as _PROJECT_ROOT, HISTORICAL_METRICS, HISTORICAL_QUARTERS, METADATA_COLUMNS, TARGET_COLUMNS
from src.io_utils import write_csv, write_json
from src.plots import (
    plot_accounting_identity_error,
    plot_industry_top20,
    plot_lag_correlation_heatmap,
    plot_missing_by_quarter,
    plot_missing_top20,
    plot_sector_distribution,
    plot_target_correlation_heatmap,
    plot_target_distributions,
)
from src.validation import quarter_of, signed_log1p, stable_row_hash, validate_schema


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(_PROJECT_ROOT / "train.csv")
    test = pd.read_csv(_PROJECT_ROOT / "test.csv")
    sample_submission = pd.read_csv(_PROJECT_ROOT / "sample_submission.csv")
    validate_schema(train, test, sample_submission)
    return train, test, sample_submission


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"Id", *TARGET_COLUMNS}]


def _common_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    test_cols = set(test.columns)
    return [c for c in train.columns if c in test_cols and c not in {"Id", *TARGET_COLUMNS}]


def _schema_summary(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    train_features = _feature_columns(train)
    test_features = _feature_columns(test)
    rows = [
        {
            "dataset": "train",
            "rows": int(train.shape[0]),
            "cols": int(train.shape[1]),
            "id_unique": bool(train["Id"].is_unique),
            "missing_cells": int(train.isna().sum().sum()),
            "missing_rate": float(train.isna().sum().sum() / train.size),
            "numeric_cols": int(train.select_dtypes(include=[np.number]).shape[1]),
            "object_cols": int(train.select_dtypes(include=["object"]).shape[1]),
            "feature_cols": len(train_features),
            "target_cols": len(TARGET_COLUMNS),
        },
        {
            "dataset": "test",
            "rows": int(test.shape[0]),
            "cols": int(test.shape[1]),
            "id_unique": bool(test["Id"].is_unique),
            "missing_cells": int(test.isna().sum().sum()),
            "missing_rate": float(test.isna().sum().sum() / test.size),
            "numeric_cols": int(test.select_dtypes(include=[np.number]).shape[1]),
            "object_cols": int(test.select_dtypes(include=["object"]).shape[1]),
            "feature_cols": len(test_features),
            "target_cols": 0,
        },
        {
            "dataset": "sample_submission",
            "rows": int(sample_submission.shape[0]),
            "cols": int(sample_submission.shape[1]),
            "id_unique": bool(sample_submission["Id"].is_unique),
            "missing_cells": int(sample_submission.isna().sum().sum()),
            "missing_rate": float(sample_submission.isna().sum().sum() / sample_submission.size),
            "numeric_cols": int(sample_submission.select_dtypes(include=[np.number]).shape[1]),
            "object_cols": int(sample_submission.select_dtypes(include=["object"]).shape[1]),
            "feature_cols": 0,
            "target_cols": 0,
        },
    ]
    return pd.DataFrame(rows)


def _dtype_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(train.columns) | set(test.columns))
    rows = []
    for col in cols:
        train_series = train[col] if col in train.columns else pd.Series(dtype="float64")
        test_series = test[col] if col in test.columns else pd.Series(dtype="float64")
        rows.append(
            {
                "column": col,
                "train_dtype": str(train_series.dtype),
                "test_dtype": str(test_series.dtype),
                "train_non_null": int(train_series.notna().sum()) if col in train.columns else 0,
                "test_non_null": int(test_series.notna().sum()) if col in test.columns else 0,
                "train_unique": int(train_series.nunique(dropna=True)) if col in train.columns else 0,
                "test_unique": int(test_series.nunique(dropna=True)) if col in test.columns else 0,
                "train_missing_rate": float(train_series.isna().mean()) if col in train.columns else np.nan,
                "test_missing_rate": float(test_series.isna().mean()) if col in test.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _missing_rate_by_column(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(train.columns) | set(test.columns))
    rows = []
    for col in cols:
        train_rate = float(train[col].isna().mean()) if col in train.columns else np.nan
        test_rate = float(test[col].isna().mean()) if col in test.columns else np.nan
        combined = []
        if col in train.columns:
            combined.append(train[col])
        if col in test.columns:
            combined.append(test[col])
        combined_rate = float(pd.concat(combined, axis=0).isna().mean()) if combined else np.nan
        if col == "Id":
            group = "id"
        elif col in TARGET_COLUMNS:
            group = "target"
        elif col in METADATA_COLUMNS:
            group = "metadata"
        elif quarter_of(col):
            group = "historical"
        else:
            group = "other"
        rows.append(
            {
                "column": col,
                "group": group,
                "in_train": bool(col in train.columns),
                "in_test": bool(col in test.columns),
                "train_missing_rate": train_rate,
                "test_missing_rate": test_rate,
                "combined_missing_rate": combined_rate,
            }
        )
    return pd.DataFrame(rows)


def _missing_rate_by_quarter(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    common = _common_feature_columns(train, test)
    rows = []
    for q in HISTORICAL_QUARTERS:
        q_cols = [c for c in common if c.startswith(f"{q}_")]
        if not q_cols:
            continue
        q_missing = pd.Series(
            {
                c: pd.concat([train[c], test[c]], axis=0).isna().mean()
                for c in q_cols
            }
        )
        rows.append(
            {
                "quarter": q,
                "avg_missing_rate": float(q_missing.mean()),
                "median_missing_rate": float(q_missing.median()),
                "max_missing_rate": float(q_missing.max()),
                "min_missing_rate": float(q_missing.min()),
                "observed_columns": len(q_cols),
            }
        )
    return pd.DataFrame(rows)


def _category_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categorical_cols = sorted(set(train.select_dtypes(include=["object"]).columns) | set(test.select_dtypes(include=["object"]).columns))
    for col in categorical_cols:
        train_series = train[col] if col in train.columns else pd.Series(dtype="object")
        test_series = test[col] if col in test.columns else pd.Series(dtype="object")
        train_values = train_series.dropna().astype(str)
        test_values = test_series.dropna().astype(str)
        train_unique = set(train_values.unique())
        test_unique = set(test_values.unique())
        rows.append(
            {
                "column": col,
                "train_unique": len(train_unique),
                "test_unique": len(test_unique),
                "shared_categories": len(train_unique & test_unique),
                "train_only_categories": len(train_unique - test_unique),
                "test_only_categories": len(test_unique - train_unique),
                "train_missing_rate": float(train_series.isna().mean()) if col in train.columns else np.nan,
                "test_missing_rate": float(test_series.isna().mean()) if col in test.columns else np.nan,
                "train_top_value": train_values.value_counts().index[0] if len(train_values) else None,
                "train_top_freq": int(train_values.value_counts().iloc[0]) if len(train_values) else 0,
                "test_top_value": test_values.value_counts().index[0] if len(test_values) else None,
                "test_top_freq": int(test_values.value_counts().iloc[0]) if len(test_values) else 0,
            }
        )
    return pd.DataFrame(rows)


def _duplicate_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    common_cols = _common_feature_columns(train, test)
    rows = []
    scopes = [
        ("train_non_id", train.drop(columns=["Id"])),
        ("train_common_features", train[common_cols]),
        ("test_common_features", test[common_cols]),
    ]
    for scope_name, frame in scopes:
        hashes = stable_row_hash(frame)
        counts = hashes.value_counts()
        dup_groups = counts[counts > 1]
        rows.append(
            {
                "scope": scope_name,
                "duplicate_rows": int((counts[counts > 1] - 1).sum()),
                "duplicate_groups": int(dup_groups.shape[0]),
                "max_group_size": int(dup_groups.max()) if not dup_groups.empty else 1,
                "notes": "exact duplicate rows within the same dataset scope",
            }
        )

    train_hash = stable_row_hash(train[common_cols])
    test_hash = stable_row_hash(test[common_cols])
    shared_hashes = set(train_hash.unique()) & set(test_hash.unique())
    train_matches = train_hash.isin(shared_hashes).sum()
    test_matches = test_hash.isin(shared_hashes).sum()
    rows.extend(
        [
            {
                "scope": "train_rows_matching_test_common_features",
                "duplicate_rows": int(train_matches),
                "duplicate_groups": int(len(shared_hashes)),
                "max_group_size": int(train_hash[train_hash.isin(shared_hashes)].value_counts().max()) if train_matches else 1,
                "notes": "train rows whose common feature hash also appears in test",
            },
            {
                "scope": "test_rows_matching_train_common_features",
                "duplicate_rows": int(test_matches),
                "duplicate_groups": int(len(shared_hashes)),
                "max_group_size": int(test_hash[test_hash.isin(shared_hashes)].value_counts().max()) if test_matches else 1,
                "notes": "test rows whose common feature hash also appears in train",
            },
        ]
    )
    return pd.DataFrame(rows)


def _target_summary(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in TARGET_COLUMNS:
        s = train[col].astype(float)
        rows.append(
            {
                "target": col,
                "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "positive_rate": float((s > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _numeric_extreme_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in [("train", train), ("test", test)]:
        numeric = frame.select_dtypes(include=[np.number])
        for col in numeric.columns:
            s = pd.to_numeric(numeric[col], errors="coerce")
            arr = s.to_numpy(dtype="float64")
            finite = s[np.isfinite(arr)]
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": col,
                    "count": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "pos_inf_count": int(np.isposinf(arr).sum()),
                    "neg_inf_count": int(np.isneginf(arr).sum()),
                    "negative_count": int((finite < 0).sum()),
                    "negative_rate": float((finite < 0).mean()) if len(finite) else np.nan,
                    "zero_count": int((finite == 0).sum()),
                    "min": float(finite.min()) if len(finite) else np.nan,
                    "p01": float(finite.quantile(0.01)) if len(finite) else np.nan,
                    "p05": float(finite.quantile(0.05)) if len(finite) else np.nan,
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "p95": float(finite.quantile(0.95)) if len(finite) else np.nan,
                    "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                    "max": float(finite.max()) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _metadata_target_correlation(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_metadata = [
        col for col in METADATA_COLUMNS
        if col in train.columns and pd.api.types.is_numeric_dtype(train[col])
    ]
    for feature in numeric_metadata:
        for target in TARGET_COLUMNS:
            pair = train[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = np.nan
            if len(pair) >= 3 and pair[feature].nunique() > 1 and pair[target].nunique() > 1:
                corr = float(pair[feature].corr(pair[target]))
            rows.append(
                {
                    "metadata_column": feature,
                    "target": target,
                    "n_obs": int(len(pair)),
                    "pearson_corr": corr,
                    "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _accounting_identity(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = _common_feature_columns(train, test)
    rows = []
    relative_errors = []
    for q in HISTORICAL_QUARTERS:
        assets_col = f"{q}_TOTAL_ASSETS"
        liabilities_col = f"{q}_TOTAL_LIABILITIES"
        equity_col = f"{q}_TOTAL_STOCKHOLDERS_EQUITY"
        cols = [assets_col, liabilities_col, equity_col]
        if not all(c in common for c in cols):
            continue
        combined = pd.concat([train[cols], test[cols]], axis=0)
        error = combined[assets_col] - combined[liabilities_col] - combined[equity_col]
        rel = error / combined[assets_col].abs().replace(0, np.nan)
        relative_errors.append(rel.rename(q))
        rows.append(
            {
                "quarter": q,
                "n_obs": int(rel.notna().sum()),
                "mean_abs_relative_error": float(rel.abs().mean()),
                "median_abs_relative_error": float(rel.abs().median()),
                "p95_abs_relative_error": float(rel.abs().quantile(0.95)),
                "within_1e-6": float((rel.abs() <= 1e-6).mean()),
                "within_1e-4": float((rel.abs() <= 1e-4).mean()),
                "within_1e-2": float((rel.abs() <= 1e-2).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    combined_rel = pd.concat(relative_errors, axis=0) if relative_errors else pd.Series(dtype=float)
    return summary, combined_rel


def _lag_correlation(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in HISTORICAL_QUARTERS:
        q_cols = [f"{q}_{metric}" for metric in HISTORICAL_METRICS if f"{q}_{metric}" in train.columns]
        row = {"quarter": q}
        for target in TARGET_COLUMNS:
            corrs = []
            for col in q_cols:
                pair = train[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if pair[col].nunique() <= 1 or pair[target].nunique() <= 1:
                    continue
                corrs.append(abs(pair[col].corr(pair[target])))
            row[target] = float(np.nanmean(corrs)) if corrs else np.nan
        rows.append(row)
    matrix = pd.DataFrame(rows).set_index("quarter")
    return matrix


def main() -> None:
    train, test, sample_submission = _load_inputs()

    tables_dir = _PROJECT_ROOT / "results" / "tables"
    figures_dir = _PROJECT_ROOT / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    schema_summary = _schema_summary(train, test, sample_submission)
    dtype_summary = _dtype_summary(train, test)
    missing_rate_by_column = _missing_rate_by_column(train, test)
    missing_rate_by_quarter = _missing_rate_by_quarter(train, test)
    category_summary = _category_summary(train, test)
    duplicate_summary = _duplicate_summary(train, test)
    target_summary = _target_summary(train)
    numeric_extreme_summary = _numeric_extreme_summary(train, test)
    metadata_target_correlation = _metadata_target_correlation(train)
    accounting_summary, accounting_errors = _accounting_identity(train, test)
    lag_corr = _lag_correlation(train)
    target_corr = train[TARGET_COLUMNS].replace([np.inf, -np.inf], np.nan).corr()
    missing_top_series = (
        missing_rate_by_column.loc[
            (~missing_rate_by_column["column"].eq("Id")) & (~missing_rate_by_column["group"].eq("target")),
            ["column", "combined_missing_rate"],
        ]
        .set_index("column")["combined_missing_rate"]
    )

    write_csv(tables_dir / "schema_summary.csv", schema_summary)
    write_csv(tables_dir / "missing_rate_by_column.csv", missing_rate_by_column)
    write_csv(tables_dir / "missing_rate_by_quarter.csv", missing_rate_by_quarter)
    write_csv(tables_dir / "dtype_summary.csv", dtype_summary)
    write_csv(tables_dir / "category_summary.csv", category_summary)
    write_csv(tables_dir / "duplicate_summary.csv", duplicate_summary)
    write_csv(tables_dir / "target_summary.csv", target_summary)
    write_csv(tables_dir / "numeric_extreme_summary.csv", numeric_extreme_summary)
    write_csv(tables_dir / "metadata_target_correlation.csv", metadata_target_correlation)
    write_csv(tables_dir / "accounting_identity_summary.csv", accounting_summary)
    write_csv(tables_dir / "lag_correlation.csv", lag_corr.reset_index())
    write_csv(tables_dir / "target_correlation.csv", target_corr.reset_index().rename(columns={"index": "target"}))

    plot_sector_distribution(train, figures_dir / "fig01_sector_distribution.png")
    plot_industry_top20(train, figures_dir / "fig02_industry_top20.png")
    plot_missing_top20(missing_top_series, figures_dir / "fig03_missing_top20.png")
    plot_missing_by_quarter(missing_rate_by_quarter, figures_dir / "fig04_missing_by_quarter.png")
    plot_target_distributions(train[TARGET_COLUMNS], figures_dir / "fig05_target_distributions.png")
    plot_lag_correlation_heatmap(lag_corr, figures_dir / "fig06_lag_correlation_heatmap.png")
    plot_accounting_identity_error(accounting_errors, figures_dir / "fig07_accounting_identity_error.png")
    plot_target_correlation_heatmap(target_corr, figures_dir / "fig08_target_correlation_heatmap.png")

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample_submission.shape),
        "id_unique_train": bool(train["Id"].is_unique),
        "id_unique_test": bool(test["Id"].is_unique),
        "id_unique_sample_submission": bool(sample_submission["Id"].is_unique),
        "target_columns": TARGET_COLUMNS,
        "submission_column_order_matches": list(sample_submission.columns) == [
            "Id",
            "Q0_REVENUES",
            "Q0_COST_OF_REVENUES",
            "Q0_GROSS_PROFIT",
            "Q0_OPERATING_EXPENSES",
            "Q0_EBITDA",
            "Q0_OPERATING_INCOME",
            "Q0_TOTAL_ASSETS",
            "Q0_TOTAL_LIABILITIES",
            "Q0_TOTAL_STOCKHOLDERS_EQUITY",
        ],
        "missing": {
            "train_missing_cells": int(train.isna().sum().sum()),
            "test_missing_cells": int(test.isna().sum().sum()),
            "train_missing_rate": float(train.isna().sum().sum() / train.size),
            "test_missing_rate": float(test.isna().sum().sum() / test.size),
            "top_missing_columns": missing_rate_by_column.sort_values("combined_missing_rate", ascending=False).head(20)[["column", "combined_missing_rate"]].to_dict(orient="records"),
            "quarter_missing_rate": missing_rate_by_quarter.to_dict(orient="records"),
        },
        "duplicates": duplicate_summary.to_dict(orient="records"),
        "inf_values": (
            numeric_extreme_summary.groupby("dataset")[["pos_inf_count", "neg_inf_count"]]
            .sum()
            .astype(int)
            .to_dict(orient="index")
        ),
        "negative_value_top_columns": (
            numeric_extreme_summary.sort_values("negative_rate", ascending=False)
            .head(20)[["dataset", "column", "negative_count", "negative_rate", "min", "p01"]]
            .to_dict(orient="records")
        ),
        "metadata_target_correlation_top": (
            metadata_target_correlation.sort_values("abs_corr", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
        "constant_columns_train": [c for c in train.columns if train[c].nunique(dropna=True) <= 1],
        "constant_columns_test": [c for c in test.columns if test[c].nunique(dropna=True) <= 1],
        "near_constant_columns_train": [
            c
            for c in train.columns
            if len(train[c].dropna())
            and train[c].dropna().value_counts(normalize=True).iloc[0] >= 0.99
        ],
        "near_constant_columns_test": [
            c
            for c in test.columns
            if len(test[c].dropna())
            and test[c].dropna().value_counts(normalize=True).iloc[0] >= 0.99
        ],
        "accounting_identity": accounting_summary.to_dict(orient="records"),
        "accounting_identity_overall": {
            "n_obs": int(accounting_errors.notna().sum()),
            "mean_abs_relative_error": float(accounting_errors.abs().mean()) if len(accounting_errors) else np.nan,
            "median_abs_relative_error": float(accounting_errors.abs().median()) if len(accounting_errors) else np.nan,
            "p95_abs_relative_error": float(accounting_errors.abs().quantile(0.95)) if len(accounting_errors) else np.nan,
        },
        "target_correlations": target_corr.round(6).to_dict(),
    }
    write_json(_PROJECT_ROOT / "results" / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
