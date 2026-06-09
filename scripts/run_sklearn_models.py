from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ID_COLUMN, PROJECT_ROOT as ROOT, TARGET_COLUMNS
from src.cv import assert_no_group_leakage, make_groupkfold_splits
from src.feature_engineering import build_feature_frame, build_feature_group_id, replace_inf_with_nan
from src.metrics import fold_score_frame, mean_target_r2, r2_by_target
from src.models import build_estimator
from src.io_utils import write_csv, write_json
from src.validation import validate_schema


EXPERIMENTS = [
    {
        "experiment_id": "M1_ridge_history_raw",
        "model_name": "Ridge",
        "model_kind": "ridge",
        "feature_set": "history_raw",
        "target_strategy": "direct",
    },
    {
        "experiment_id": "M2_hgb_history_raw",
        "model_name": "HistGradientBoostingRegressor",
        "model_kind": "hgb",
        "feature_set": "history_raw",
        "target_strategy": "direct",
    },
]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
    validate_schema(train, test, sample_submission)
    return replace_inf_with_nan(train), replace_inf_with_nan(test), sample_submission


def _fit_target_predictions(
    model_kind: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    oof_pred = pd.Series(index=X_train.index, dtype="float64")
    test_pred_parts = []
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splits):
        estimator = build_estimator(model_kind, X_train)
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        valid_mask = y_fold_train.notna()
        estimator.fit(X_fold_train.loc[valid_mask], y_fold_train.loc[valid_mask])
        fold_valid_pred = pd.Series(estimator.predict(X_train.iloc[valid_idx]), index=X_train.index[valid_idx])
        oof_pred.iloc[valid_idx] = fold_valid_pred.values
        fold_rows.append(
            {
                "fold": fold,
                "target": y.name,
                "r2": float(r2_by_target(pd.DataFrame({y.name: y.iloc[valid_idx]}), pd.DataFrame({y.name: fold_valid_pred}), targets=[y.name])[y.name]),
            }
        )

    full_estimator = build_estimator(model_kind, X_train)
    full_mask = y.notna()
    full_estimator.fit(X_train.loc[full_mask], y.loc[full_mask])
    test_pred = pd.Series(full_estimator.predict(X_test), index=X_test.index, name=y.name)
    return oof_pred, test_pred, pd.DataFrame(fold_rows)


def _summarize_experiment(
    experiment_id: str,
    model_name: str,
    feature_set: str,
    target_strategy: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    fold_scores_long: pd.DataFrame,
    runtime_seconds: float,
    notes: str,
) -> dict[str, object]:
    target_scores = r2_by_target(y_true, y_pred)
    fold_mean = fold_scores_long.groupby("fold")["r2"].mean() if not fold_scores_long.empty else pd.Series(dtype=float)
    row: dict[str, object] = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "feature_set": feature_set,
        "target_strategy": target_strategy,
        "mean_r2": mean_target_r2(target_scores),
        "fold_mean_r2_std": float(fold_mean.std(ddof=0)) if not fold_mean.empty else np.nan,
        "runtime_seconds": float(runtime_seconds),
        "notes": notes,
    }
    row.update({f"r2_{target}": score for target, score in target_scores.items()})
    return row


def _append_experiment_log(
    existing_path: Path,
    rows: pd.DataFrame,
    experiment_ids: list[str],
) -> None:
    if existing_path.exists() and existing_path.stat().st_size > 0:
        existing = pd.read_csv(existing_path)
        if not existing.empty and "experiment_id" in existing.columns:
            existing = existing[~existing["experiment_id"].isin(experiment_ids)]
        rows = pd.concat([existing, rows], ignore_index=True)
    write_csv(existing_path, rows)


def run_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment: dict,
) -> dict[str, object]:
    X_train = build_feature_frame(train, feature_set=experiment["feature_set"])
    X_test = build_feature_frame(test, feature_set=experiment["feature_set"])
    y = train[TARGET_COLUMNS].astype(float)

    oof = train[[ID_COLUMN]].copy()
    oof["group_id"] = build_feature_group_id(X_train)
    oof["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splits):
        oof.iloc[valid_idx, oof.columns.get_loc("fold")] = fold

    test_pred = sample_submission[[ID_COLUMN]].copy()
    fold_rows = []
    oof_pred_frame = pd.DataFrame(index=train.index)
    test_pred_frame = pd.DataFrame(index=test.index)

    for target in TARGET_COLUMNS:
        target_y = y[target]
        oof_pred, test_pred_target, target_fold_scores = _fit_target_predictions(
            experiment["model_kind"], X_train, X_test, target_y, splits
        )
        oof_pred_frame[target] = oof_pred
        test_pred_frame[target] = test_pred_target
        target_fold_scores = target_fold_scores.assign(
            experiment_id=experiment["experiment_id"],
            model_name=experiment["model_name"],
            feature_set=experiment["feature_set"],
            target_strategy=experiment["target_strategy"],
        )
        fold_rows.append(target_fold_scores)

    for target in TARGET_COLUMNS:
        oof[f"actual_{target}"] = y[target].values
        oof[f"pred_{target}"] = oof_pred_frame[target].values
        test_pred[target] = test_pred_frame[target].values

    fold_scores_long = pd.concat(fold_rows, ignore_index=True)
    summary = _summarize_experiment(
        experiment_id=experiment["experiment_id"],
        model_name=experiment["model_name"],
        feature_set=experiment["feature_set"],
        target_strategy=experiment["target_strategy"],
        y_true=y,
        y_pred=oof_pred_frame,
        fold_scores_long=fold_scores_long,
        runtime_seconds=float(experiment.get("runtime_seconds", 0.0)),
        notes=experiment["notes"],
    )

    return {
        "summary": summary,
        "fold_scores_long": fold_scores_long,
        "oof": oof,
        "test_pred": test_pred,
        "feature_set": experiment["feature_set"],
    }


def main() -> None:
    start = time.perf_counter()
    train, test, sample_submission = _load_inputs()
    groups = build_feature_group_id(build_feature_frame(train, feature_set="history_raw"))
    splits = make_groupkfold_splits(groups, n_splits=5)
    assert_no_group_leakage(splits, groups)

    all_summaries = []
    all_logs = []

    for experiment in EXPERIMENTS:
        exp_start = time.perf_counter()
        experiment = dict(experiment)
        experiment["notes"] = "GroupKFold OOF sklearn baseline"
        result = run_experiment(train, test, sample_submission, splits, experiment)
        experiment["runtime_seconds"] = time.perf_counter() - exp_start
        result["summary"]["runtime_seconds"] = experiment["runtime_seconds"]
        all_summaries.append(result["summary"])

        experiment_id = experiment["experiment_id"]
        fold_scores_long = result["fold_scores_long"]
        fold_scores_long = fold_scores_long.assign(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mean_r2=result["summary"]["mean_r2"],
            std_r2=result["summary"]["fold_mean_r2_std"],
            runtime_seconds=experiment["runtime_seconds"],
            seed=42,
            git_commit=_git_commit(),
            notes=experiment["notes"],
        )
        all_logs.append(fold_scores_long)

        write_csv(ROOT / "results" / "tables" / f"{experiment_id}_scores.csv", pd.DataFrame([result["summary"]]))
        write_csv(ROOT / "results" / "cv_scores" / f"{experiment_id.lower()}.csv", fold_scores_long)
        write_csv(ROOT / "results" / "oof" / f"{experiment_id.lower()}.csv", result["oof"])
        write_csv(ROOT / "results" / "predictions" / f"{experiment_id.lower()}_test_predictions.csv", result["test_pred"])
        write_json(
            ROOT / "configs" / f"{experiment_id.lower()}.json",
            {
                **experiment,
                "feature_set": experiment["feature_set"],
                "model_kind": experiment["model_kind"],
                "grouping": "feature_hash_groupkfold",
                "n_splits": 5,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    summary_df = pd.DataFrame(all_summaries)
    write_csv(ROOT / "results" / "tables" / "sklearn_model_scores.csv", summary_df)

    log_df = pd.concat(all_logs, ignore_index=True)
    existing_log_path = ROOT / "results" / "experiment_log.csv"
    _append_experiment_log(existing_log_path, log_df, experiment_ids=summary_df["experiment_id"].tolist())

    # Update best config only if these models improve over the current best.
    current_best = json.loads((ROOT / "configs" / "best_config.json").read_text(encoding="utf-8"))
    best_row = summary_df.sort_values("mean_r2", ascending=False).iloc[0]
    if current_best.get("best_mean_oof_r2") is None or float(best_row["mean_r2"]) > float(current_best["best_mean_oof_r2"]):
        write_json(
            ROOT / "configs" / "best_config.json",
            {
                "status": "updated_with_sklearn_model",
                "best_experiment_id": best_row["experiment_id"],
                "best_mean_oof_r2": float(best_row["mean_r2"]),
                "best_fold_mean_r2_std": float(best_row["fold_mean_r2_std"]),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "notes": "Best current result among all evaluated models.",
            },
        )
    else:
        current_best["status"] = "sklearn_evaluated_best_unchanged"
        current_best["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        current_best["notes"] = "Best current result among B0-B4 baselines and M1/M2 sklearn models."
        current_best["latest_evaluated_experiments"] = summary_df[
            ["experiment_id", "mean_r2", "fold_mean_r2_std"]
        ].to_dict(orient="records")
        write_json(ROOT / "configs" / "best_config.json", current_best)

    print(summary_df.to_string(index=False))
    print(f"Saved sklearn experiments in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
