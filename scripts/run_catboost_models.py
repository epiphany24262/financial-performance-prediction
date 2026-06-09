from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ID_COLUMN, PROJECT_ROOT as ROOT, TARGET_COLUMNS
from src.cv import assert_no_group_leakage, make_groupkfold_splits
from src.feature_engineering import build_feature_frame, build_feature_group_id, replace_inf_with_nan
from src.io_utils import write_csv, write_json
from src.metrics import mean_target_r2, r2_by_target
from src.models import prepare_catboost_frame
from src.validation import validate_schema


FEATURE_SET = "history_metadata_engineered"
BASELINE_OOF_PATH = ROOT / "results" / "oof" / "baseline_oof.csv"
BASELINE_TEST_PATH = ROOT / "results" / "predictions" / "baseline_b4_test_predictions.csv"

EXPERIMENTS = [
    {
        "experiment_id": "M3_catboost_direct_history_metadata_engineered",
        "model_name": "CatBoostRegressor",
        "mode": "direct",
        "feature_set": FEATURE_SET,
        "baseline_reference": None,
    },
    {
        "experiment_id": "M4_catboost_residual_history_metadata_engineered",
        "model_name": "CatBoostRegressor",
        "mode": "residual",
        "feature_set": FEATURE_SET,
        "baseline_reference": "B4",
    },
]


CATBOOST_BASE_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 180,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 8.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "od_type": "Iter",
    "od_wait": 20,
    "use_best_model": True,
}


def _selected_experiments() -> list[dict]:
    raw = os.getenv("CATBOOST_EXPERIMENT_IDS", "").strip()
    if not raw:
        return EXPERIMENTS
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    selected = [experiment for experiment in EXPERIMENTS if experiment["experiment_id"] in wanted]
    if not selected:
        raise ValueError(f"No CatBoost experiments matched CATBOOST_EXPERIMENT_IDS={raw!r}")
    return selected


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


def _load_baseline_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_oof = pd.read_csv(BASELINE_OOF_PATH)
    baseline_test = pd.read_csv(BASELINE_TEST_PATH)
    return baseline_oof, baseline_test


def _baseline_test_values(baseline_test: pd.DataFrame, ids: pd.Series, target: str) -> np.ndarray:
    if f"B4_{target}" in baseline_test.columns:
        column = f"B4_{target}"
    elif target in baseline_test.columns:
        column = target
    else:
        raise KeyError(f"Baseline test predictions missing column for {target}")
    return baseline_test[column].reindex(ids).values


def _catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared, cat_cols = prepare_catboost_frame(frame)
    return prepared, cat_cols


def _fit_fold_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    iterations: int,
) -> tuple[CatBoostRegressor, np.ndarray, int]:
    params = dict(CATBOOST_BASE_PARAMS)
    params["iterations"] = int(iterations)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_features)
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        best_iter = int(model.tree_count_ or iterations)
    else:
        best_iter = int(best_iter + 1)
    pred_valid = model.predict(X_valid)
    return model, pred_valid, best_iter


def _fit_full_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    iterations: int,
) -> CatBoostRegressor:
    params = dict(CATBOOST_BASE_PARAMS)
    params["iterations"] = int(iterations)
    params["use_best_model"] = False
    params.pop("od_type", None)
    params.pop("od_wait", None)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, cat_features=cat_features)
    return model


def _run_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    experiment: dict,
) -> dict[str, object]:
    X_train_raw = build_feature_frame(train, feature_set=experiment["feature_set"])
    X_test_raw = build_feature_frame(test, feature_set=experiment["feature_set"])
    X_train, cat_features = _catboost_frame(X_train_raw)
    X_test, _ = _catboost_frame(X_test_raw)
    groups = build_feature_group_id(X_train_raw)
    splits = make_groupkfold_splits(groups, n_splits=5)
    assert_no_group_leakage(splits, groups)

    y = train[TARGET_COLUMNS].astype(float)
    baseline_oof = None
    baseline_test = None
    if experiment["mode"] == "residual":
        baseline_oof, baseline_test = _load_baseline_predictions()
        baseline_oof = baseline_oof.set_index("Id")
        baseline_test = baseline_test.set_index("Id")

    oof_pred_frame = pd.DataFrame(index=train.index)
    test_pred_frame = pd.DataFrame(index=test.index)
    fold_rows = []
    best_iterations = []

    for target in TARGET_COLUMNS:
        target_oof = pd.Series(index=train.index, dtype="float64")
        target_best_iters = []
        if experiment["mode"] == "direct":
            train_target = y[target]
        else:
            train_target = y[target] - baseline_oof[f"B4_{target}"].reindex(train["Id"]).values

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_train = train_target.iloc[train_idx]
            y_fold_valid = train_target.iloc[valid_idx]
            valid_mask = y_fold_train.notna()
            if valid_mask.sum() < 2:
                raise ValueError(f"Not enough valid rows for target {target} fold {fold}")
            _, fold_pred, best_iter = _fit_fold_model(
                X_fold_train.loc[valid_mask],
                y_fold_train.loc[valid_mask],
                X_fold_valid,
                y_fold_valid,
                cat_features,
                iterations=CATBOOST_BASE_PARAMS["iterations"],
            )
            if experiment["mode"] == "residual":
                fold_pred = fold_pred + baseline_oof.loc[train.loc[valid_idx, "Id"], f"B4_{target}"].values
            target_oof.iloc[valid_idx] = fold_pred
            target_best_iters.append(best_iter)
            fold_rows.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "model_name": experiment["model_name"],
                    "feature_set": experiment["feature_set"],
                    "target_strategy": experiment["mode"],
                    "fold": fold,
                    "target": target,
                    "r2": float(r2_by_target(pd.DataFrame({target: y[target].iloc[valid_idx]}), pd.DataFrame({target: fold_pred}), targets=[target])[target]),
                }
            )

        final_iterations = max(100, int(np.median(target_best_iters)))
        if experiment["mode"] == "direct":
            final_model = _fit_full_model(X_train, y[target], cat_features, final_iterations)
            test_pred = final_model.predict(X_test)
            oof_pred = target_oof
        else:
            residual_train = y[target] - baseline_oof[f"B4_{target}"].reindex(train["Id"]).values
            full_residual_model = _fit_full_model(X_train, residual_train, cat_features, final_iterations)
            residual_test_pred = full_residual_model.predict(X_test)
            test_pred = _baseline_test_values(baseline_test, test["Id"], target) + residual_test_pred
            oof_pred = target_oof

        oof_pred_frame[target] = oof_pred
        test_pred_frame[target] = test_pred
        best_iterations.append(final_iterations)

    summary_scores = r2_by_target(y, oof_pred_frame)
    fold_scores_long = pd.DataFrame(fold_rows)
    fold_mean = fold_scores_long.groupby("fold")["r2"].mean()
    summary = {
        "experiment_id": experiment["experiment_id"],
        "model_name": experiment["model_name"],
        "feature_set": experiment["feature_set"],
        "target_strategy": experiment["mode"],
        "mean_r2": mean_target_r2(summary_scores),
        "fold_mean_r2_std": float(fold_mean.std(ddof=0)),
        "runtime_seconds": 0.0,
        "notes": f"CatBoost {experiment['mode']} on {experiment['feature_set']}",
    }
    summary.update({f"r2_{target}": score for target, score in summary_scores.items()})

    oof = train[[ID_COLUMN]].copy()
    oof["group_id"] = groups.values
    oof["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splits):
        oof.iloc[valid_idx, oof.columns.get_loc("fold")] = fold
    for target in TARGET_COLUMNS:
        oof[f"actual_{target}"] = y[target].values
        oof[f"pred_{target}"] = oof_pred_frame[target].values

    test_pred = sample_submission[[ID_COLUMN]].copy()
    for column in sample_submission.columns:
        if column == ID_COLUMN:
            continue
        test_pred[column] = test_pred_frame[column].values

    fold_scores_long = fold_scores_long.assign(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mean_r2=summary["mean_r2"],
        std_r2=summary["fold_mean_r2_std"],
        runtime_seconds=0.0,
        seed=42,
        git_commit=_git_commit(),
        notes=summary["notes"],
    )

    return {
        "summary": summary,
        "fold_scores_long": fold_scores_long,
        "oof": oof,
        "test_pred": test_pred,
        "best_iterations": best_iterations,
    }


def _append_experiment_log(existing_path: Path, rows: pd.DataFrame, experiment_ids: list[str]) -> None:
    if existing_path.exists() and existing_path.stat().st_size > 0:
        existing = pd.read_csv(existing_path)
        if not existing.empty and "experiment_id" in existing.columns:
            existing = existing[~existing["experiment_id"].isin(experiment_ids)]
        rows = pd.concat([existing, rows], ignore_index=True)
    write_csv(existing_path, rows)


def _upsert_summary_table(existing_path: Path, rows: pd.DataFrame, experiment_ids: list[str]) -> None:
    if existing_path.exists() and existing_path.stat().st_size > 0:
        existing = pd.read_csv(existing_path)
        if not existing.empty and "experiment_id" in existing.columns:
            existing = existing[~existing["experiment_id"].isin(experiment_ids)]
        rows = pd.concat([existing, rows], ignore_index=True)
    write_csv(existing_path, rows)


def main() -> None:
    start = time.perf_counter()
    train, test, sample_submission = _load_inputs()

    all_summaries = []
    all_logs = []

    for experiment in _selected_experiments():
        exp_start = time.perf_counter()
        result = _run_experiment(train, test, sample_submission, experiment)
        runtime_seconds = time.perf_counter() - exp_start
        result["summary"]["runtime_seconds"] = runtime_seconds
        result["summary"]["notes"] = f"{result['summary']['notes']} | best_iters={int(np.median(result['best_iterations']))}"
        result["fold_scores_long"]["runtime_seconds"] = runtime_seconds
        result["fold_scores_long"]["notes"] = result["summary"]["notes"]
        result["fold_scores_long"]["mean_r2"] = result["summary"]["mean_r2"]
        result["fold_scores_long"]["std_r2"] = result["summary"]["fold_mean_r2_std"]

        all_summaries.append(result["summary"])
        all_logs.append(result["fold_scores_long"])

        exp_id = experiment["experiment_id"]
        write_csv(ROOT / "results" / "tables" / f"{exp_id}_scores.csv", pd.DataFrame([result["summary"]]))
        write_csv(ROOT / "results" / "cv_scores" / f"{exp_id.lower()}.csv", result["fold_scores_long"])
        write_csv(ROOT / "results" / "oof" / f"{exp_id.lower()}.csv", result["oof"])
        write_csv(ROOT / "results" / "predictions" / f"{exp_id.lower()}_test_predictions.csv", result["test_pred"])
        write_json(
            ROOT / "configs" / f"{exp_id.lower()}.json",
            {
                **experiment,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "catboost_params": {k: v for k, v in CATBOOST_BASE_PARAMS.items() if k not in {"verbose", "allow_writing_files"}},
                "best_iterations": [int(x) for x in result["best_iterations"]],
            },
        )

    summary_df = pd.DataFrame(all_summaries)
    _upsert_summary_table(ROOT / "results" / "tables" / "catboost_model_scores.csv", summary_df, experiment_ids=summary_df["experiment_id"].tolist())

    log_df = pd.concat(all_logs, ignore_index=True)
    _append_experiment_log(ROOT / "results" / "experiment_log.csv", log_df, experiment_ids=summary_df["experiment_id"].tolist())

    current_best = json.loads((ROOT / "configs" / "best_config.json").read_text(encoding="utf-8"))
    best_row = summary_df.sort_values("mean_r2", ascending=False).iloc[0]
    if current_best.get("best_mean_oof_r2") is None or float(best_row["mean_r2"]) > float(current_best["best_mean_oof_r2"]):
        write_json(
            ROOT / "configs" / "best_config.json",
            {
                "status": "updated_with_catboost",
                "best_experiment_id": best_row["experiment_id"],
                "best_mean_oof_r2": float(best_row["mean_r2"]),
                "best_fold_mean_r2_std": float(best_row["fold_mean_r2_std"]),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "notes": "Best current result among all evaluated models.",
            },
        )
    else:
        current_best["status"] = "catboost_evaluated_best_unchanged"
        current_best["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        current_best["notes"] = "Best current result among B0-B4, M1/M2, and M3/M4 CatBoost models."
        current_best["latest_evaluated_experiments"] = summary_df[
            ["experiment_id", "mean_r2", "fold_mean_r2_std"]
        ].to_dict(orient="records")
        write_json(ROOT / "configs" / "best_config.json", current_best)

    print(summary_df.to_string(index=False))
    print(f"Saved CatBoost experiments in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
