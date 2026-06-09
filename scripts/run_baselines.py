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

from src.baselines import BASELINE_IDS, blend_predictions, predict_rule, search_baseline_blend_weights
from src.constants import ID_COLUMN, PROJECT_ROOT as ROOT, TARGET_COLUMNS
from src.cv import assert_no_group_leakage, make_groupkfold_splits
from src.feature_engineering import build_feature_group_id, model_feature_columns, replace_inf_with_nan
from src.io_utils import write_csv, write_json
from src.metrics import fold_score_frame, summarize_oof_scores
from src.validation import validate_schema


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


def _empty_prediction_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=TARGET_COLUMNS, dtype="float64")


def _run_oof_rules(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Series, dict[str, float]]:
    y = train[TARGET_COLUMNS].astype(float)
    predictions = {rule: _empty_prediction_frame(train.index) for rule in ["B0", "B1", "B2", "B3"]}
    fold_assignments = pd.Series(index=train.index, dtype="int64", name="fold")
    fold_predictions: dict[str, list[dict]] = {rule: [] for rule in ["B0", "B1", "B2", "B3"]}
    runtime_by_rule = {rule: 0.0 for rule in ["B0", "B1", "B2", "B3"]}

    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_part = train.iloc[train_idx]
        valid_part = train.iloc[valid_idx]
        y_train = train_part[TARGET_COLUMNS].astype(float)
        y_valid = valid_part[TARGET_COLUMNS].astype(float)
        fold_assignments.iloc[valid_idx] = fold

        for rule in ["B0", "B1", "B2", "B3"]:
            rule_start = time.perf_counter()
            fold_pred = pd.DataFrame(index=valid_part.index, columns=TARGET_COLUMNS, dtype="float64")
            for target in TARGET_COLUMNS:
                fold_pred[target] = predict_rule(rule, valid_part, target, y_train[target])
            predictions[rule].loc[valid_part.index, TARGET_COLUMNS] = fold_pred[TARGET_COLUMNS]
            fold_predictions[rule].append({"fold": fold, "y_true": y_valid, "y_pred": fold_pred})
            runtime_by_rule[rule] += time.perf_counter() - rule_start

    fold_scores = {
        rule: fold_score_frame(items)
        for rule, items in fold_predictions.items()
    }
    return predictions, fold_scores, fold_assignments.astype(int), runtime_by_rule


def _score_blended_oof(
    y: pd.DataFrame,
    base_predictions: dict[str, pd.DataFrame],
    blend_weights: dict[str, dict],
    folds: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    b4_pred = _empty_prediction_frame(y.index)
    for target, config in blend_weights.items():
        b4_pred[target] = blend_predictions(base_predictions, target, config["members"], config["weights"])

    fold_items = []
    for fold in sorted(folds.unique()):
        idx = folds[folds == fold].index
        fold_items.append({"fold": int(fold), "y_true": y.loc[idx, TARGET_COLUMNS], "y_pred": b4_pred.loc[idx, TARGET_COLUMNS]})
    return b4_pred, fold_score_frame(fold_items)


def _predict_test_rules(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y_train = train[TARGET_COLUMNS].astype(float)
    predictions = {rule: _empty_prediction_frame(test.index) for rule in ["B0", "B1", "B2", "B3"]}
    for rule in ["B0", "B1", "B2", "B3"]:
        for target in TARGET_COLUMNS:
            predictions[rule][target] = predict_rule(rule, test, target, y_train[target])
    return predictions


def _build_experiment_log_rows(
    summary: pd.DataFrame,
    fold_scores: dict[str, pd.DataFrame],
    runtime_by_rule: dict[str, float],
) -> pd.DataFrame:
    git_commit = _git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, row in summary.iterrows():
        experiment_id = row["experiment_id"]
        per_fold = fold_scores[experiment_id]
        for _, fold_row in per_fold.iterrows():
            for target in TARGET_COLUMNS:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "timestamp": timestamp,
                        "model_name": row["model_name"],
                        "feature_set": "row_history_only",
                        "target_strategy": "direct_rule",
                        "fold": int(fold_row["fold"]),
                        "target": target,
                        "r2": float(fold_row[f"r2_{target}"]),
                        "mean_r2": float(row["mean_r2"]),
                        "std_r2": float(row["fold_mean_r2_std"]),
                        "runtime_seconds": float(runtime_by_rule.get(experiment_id, row["runtime_seconds"])),
                        "seed": 42,
                        "git_commit": git_commit,
                        "notes": row["notes"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    train, test, sample_submission = _load_inputs()
    y = train[TARGET_COLUMNS].astype(float)
    feature_columns = model_feature_columns(train)
    groups = build_feature_group_id(train, feature_columns=feature_columns)
    splits = make_groupkfold_splits(groups, n_splits=5)
    assert_no_group_leakage(splits, groups)

    base_predictions, fold_scores, folds, runtime_by_rule = _run_oof_rules(train, splits)

    blend_weights, blend_grid = search_baseline_blend_weights(y, base_predictions, targets=TARGET_COLUMNS, step=0.1)
    b4_pred, b4_fold_scores = _score_blended_oof(y, base_predictions, blend_weights, folds)
    all_predictions = dict(base_predictions)
    all_predictions["B4"] = b4_pred
    fold_scores["B4"] = b4_fold_scores

    runtime_total = time.perf_counter() - start
    runtime_by_rule["B4"] = max(0.0, runtime_total - sum(runtime_by_rule.values()))

    summary_rows = []
    for rule in BASELINE_IDS:
        summary_rows.append(
            summarize_oof_scores(
                experiment_id=rule,
                model_name=f"{rule}_baseline",
                y_true=y,
                y_pred=all_predictions[rule],
                fold_scores=fold_scores[rule],
                runtime_seconds=runtime_by_rule[rule],
                notes="GroupKFold OOF baseline",
            )
        )
    summary = pd.DataFrame(summary_rows)

    oof = train[[ID_COLUMN]].copy()
    oof["group_id"] = groups
    oof["fold"] = folds.values
    for target in TARGET_COLUMNS:
        oof[f"actual_{target}"] = y[target].values
    for rule in BASELINE_IDS:
        for target in TARGET_COLUMNS:
            oof[f"{rule}_{target}"] = all_predictions[rule][target].values

    test_rule_predictions = _predict_test_rules(train, test)
    b4_test = _empty_prediction_frame(test.index)
    for target, config in blend_weights.items():
        b4_test[target] = blend_predictions(test_rule_predictions, target, config["members"], config["weights"])

    baseline_submission = sample_submission[[ID_COLUMN]].copy()
    for target in sample_submission.columns:
        if target == ID_COLUMN:
            continue
        baseline_submission[target] = b4_test[target].values

    write_csv(ROOT / "results" / "tables" / "baseline_scores.csv", summary)
    write_csv(ROOT / "results" / "tables" / "baseline_blend_grid.csv", blend_grid)
    write_csv(ROOT / "results" / "oof" / "baseline_oof.csv", oof)
    write_csv(ROOT / "results" / "predictions" / "baseline_b4_test_predictions.csv", baseline_submission)
    for rule, scores in fold_scores.items():
        write_csv(ROOT / "results" / "cv_scores" / f"{rule.lower()}.csv", scores)
    write_json(ROOT / "configs" / "baseline_blend_weights.json", blend_weights)

    experiment_log_rows = _build_experiment_log_rows(summary, fold_scores, runtime_by_rule)
    existing_log_path = ROOT / "results" / "experiment_log.csv"
    if existing_log_path.exists() and existing_log_path.stat().st_size > 0:
        existing_log = pd.read_csv(existing_log_path)
        if not existing_log.empty:
            existing_log = existing_log[~existing_log["experiment_id"].isin(BASELINE_IDS)] if "experiment_id" in existing_log else existing_log
            experiment_log_rows = pd.concat([existing_log, experiment_log_rows], ignore_index=True)
    write_csv(existing_log_path, experiment_log_rows)

    best = summary.sort_values("mean_r2", ascending=False).iloc[0]
    write_json(
        ROOT / "configs" / "best_config.json",
        {
            "status": "baseline_complete",
            "best_experiment_id": best["experiment_id"],
            "best_mean_oof_r2": float(best["mean_r2"]),
            "best_fold_mean_r2_std": float(best["fold_mean_r2_std"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "notes": "Best current result among B0-B4 baselines only.",
        },
    )

    print(summary.to_string(index=False))
    print(f"Saved baseline OOF and scores in {runtime_total:.2f}s")


if __name__ == "__main__":
    main()
