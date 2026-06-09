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

from src.accounting_checks import apply_accounting_adjustment, identity_error_summary, score_accounting_adjustments
from src.blending import apply_blend, search_target_blend_weights, validate_prediction_frame
from src.constants import ID_COLUMN, PROJECT_ROOT as ROOT, SUBMISSION_COLUMNS, TARGET_COLUMNS
from src.io_utils import sha256_file, write_csv, write_json
from src.metrics import mean_target_r2, r2_by_target
from src.validation import validate_schema


OOF_SPECS = {
    "B4": ROOT / "results" / "oof" / "baseline_oof.csv",
    "M3": ROOT / "results" / "oof" / "m3_catboost_direct_history_metadata_engineered.csv",
    "M4": ROOT / "results" / "oof" / "m4_catboost_residual_history_metadata_engineered.csv",
}

TEST_SPECS = {
    "B4": ROOT / "results" / "predictions" / "baseline_b4_test_predictions.csv",
    "M3": ROOT / "results" / "predictions" / "m3_catboost_direct_history_metadata_engineered_test_predictions.csv",
    "M4": ROOT / "results" / "predictions" / "m4_catboost_residual_history_metadata_engineered_test_predictions.csv",
}


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
    return train, test, sample_submission


def _load_oof_member(member: str, train_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(OOF_SPECS[member]).set_index(ID_COLUMN)
    frame = frame.reindex(train_ids)
    if member == "B4":
        pred = pd.DataFrame({target: frame[f"B4_{target}"] for target in TARGET_COLUMNS}, index=train_ids)
    else:
        pred = pd.DataFrame({target: frame[f"pred_{target}"] for target in TARGET_COLUMNS}, index=train_ids)
    meta = frame[["group_id", "fold"]].copy()
    validate_prediction_frame(pred)
    return pred, meta


def _load_test_member(member: str, test_ids: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(TEST_SPECS[member]).set_index(ID_COLUMN)
    frame = frame.reindex(test_ids)
    pred = frame.loc[:, TARGET_COLUMNS].copy()
    validate_prediction_frame(pred)
    return pred


def _submission_from_predictions(sample_submission: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    out = sample_submission[[ID_COLUMN]].copy()
    pred = predictions.reindex(sample_submission[ID_COLUMN])
    for column in sample_submission.columns:
        if column == ID_COLUMN:
            continue
        out[column] = pred[column].values
    if list(out.columns) != SUBMISSION_COLUMNS:
        raise AssertionError("Submission columns do not match sample_submission order")
    values = out.drop(columns=[ID_COLUMN]).to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError("Final submission contains NaN values")
    if np.isinf(values).any():
        raise ValueError("Final submission contains infinite values")
    return out


def _fold_score_rows(
    experiment_id: str,
    model_name: str,
    target_strategy: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    folds: pd.Series,
    runtime_seconds: float,
    notes: str,
) -> pd.DataFrame:
    target_scores = r2_by_target(y_true, y_pred)
    mean_r2 = mean_target_r2(target_scores)
    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for fold in sorted(folds.dropna().unique()):
        mask = folds == fold
        for target in TARGET_COLUMNS:
            score = r2_by_target(
                pd.DataFrame({target: y_true.loc[mask, target]}),
                pd.DataFrame({target: y_pred.loc[mask, target]}),
                targets=[target],
            )[target]
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "timestamp": timestamp,
                    "model_name": model_name,
                    "feature_set": "saved_oof_predictions",
                    "target_strategy": target_strategy,
                    "fold": int(fold),
                    "target": target,
                    "r2": float(score),
                    "mean_r2": mean_r2,
                    "std_r2": np.nan,
                    "runtime_seconds": float(runtime_seconds),
                    "seed": 42,
                    "git_commit": _git_commit(),
                    "notes": notes,
                }
            )
    result = pd.DataFrame(rows)
    fold_mean = result.groupby("fold")["r2"].mean()
    result["std_r2"] = float(fold_mean.std(ddof=0))
    return result


def _summary_row(
    experiment_id: str,
    model_name: str,
    target_strategy: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    folds: pd.Series,
    runtime_seconds: float,
    notes: str,
) -> dict[str, object]:
    scores = r2_by_target(y_true, y_pred)
    fold_means = []
    for fold in sorted(folds.dropna().unique()):
        mask = folds == fold
        fold_scores = r2_by_target(y_true.loc[mask, TARGET_COLUMNS], y_pred.loc[mask, TARGET_COLUMNS])
        fold_means.append(mean_target_r2(fold_scores))
    row: dict[str, object] = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "feature_set": "saved_oof_predictions",
        "target_strategy": target_strategy,
        "mean_r2": mean_target_r2(scores),
        "fold_mean_r2_std": float(np.std(fold_means, ddof=0)),
        "runtime_seconds": float(runtime_seconds),
        "notes": notes,
    }
    row.update({f"r2_{target}": score for target, score in scores.items()})
    return row


def _append_experiment_log(path: Path, rows: pd.DataFrame, experiment_ids: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        if not existing.empty and "experiment_id" in existing.columns:
            existing = existing[~existing["experiment_id"].isin(experiment_ids)]
        rows = pd.concat([existing, rows], ignore_index=True)
    write_csv(path, rows)


def _write_oof(path: Path, ids: pd.Series, meta: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    out = pd.DataFrame({ID_COLUMN: ids.values})
    out["group_id"] = meta["group_id"].values
    out["fold"] = meta["fold"].values
    for target in TARGET_COLUMNS:
        out[f"actual_{target}"] = y_true[target].values
        out[f"pred_{target}"] = y_pred[target].values
    write_csv(path, out)


def _update_best_config(summary_rows: list[dict[str, object]], final_experiment_id: str) -> None:
    current_path = ROOT / "configs" / "best_config.json"
    current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else {}
    candidate = next(row for row in summary_rows if row["experiment_id"] == final_experiment_id)
    if current.get("best_mean_oof_r2") is None or float(candidate["mean_r2"]) >= float(current["best_mean_oof_r2"]):
        write_json(
            current_path,
            {
                "status": "updated_with_final_blend",
                "best_experiment_id": candidate["experiment_id"],
                "best_mean_oof_r2": float(candidate["mean_r2"]),
                "best_fold_mean_r2_std": float(candidate["fold_mean_r2_std"]),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "notes": "Best current result after OOF blend and accounting-postprocess evaluation.",
            },
        )
    else:
        current["status"] = "final_blend_evaluated_best_unchanged"
        current["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        current["latest_evaluated_experiments"] = [
            {
                "experiment_id": row["experiment_id"],
                "mean_r2": float(row["mean_r2"]),
                "fold_mean_r2_std": float(row["fold_mean_r2_std"]),
            }
            for row in summary_rows
        ]
        write_json(current_path, current)


def main() -> None:
    start = time.perf_counter()
    train, test, sample_submission = _load_inputs()
    train_ids = train[ID_COLUMN]
    test_ids = test[ID_COLUMN]
    y_true = train.set_index(ID_COLUMN).loc[train_ids, TARGET_COLUMNS].astype(float)

    oof_predictions: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, pd.DataFrame] = {}
    metas: dict[str, pd.DataFrame] = {}
    for member in OOF_SPECS:
        oof_predictions[member], member_meta = _load_oof_member(member, train_ids)
        test_predictions[member] = _load_test_member(member, test_ids)
        metas[member] = member_meta
    meta = metas["M4"] if "M4" in metas else next(iter(metas.values()))
    folds = meta["fold"]

    blend_config, blend_grid = search_target_blend_weights(
        y_true,
        oof_predictions,
        targets=TARGET_COLUMNS,
        coarse_step=0.05,
        fine_step=0.01,
        fine_radius=0.05,
    )
    blended_oof = apply_blend(oof_predictions, blend_config)
    blended_test = apply_blend(test_predictions, blend_config)
    validate_prediction_frame(blended_oof)
    validate_prediction_frame(blended_test)

    best_adjustment, accounting_scores = score_accounting_adjustments(y_true, blended_oof)
    final_oof = apply_accounting_adjustment(blended_oof, best_adjustment)
    final_test = apply_accounting_adjustment(blended_test, best_adjustment)
    validate_prediction_frame(final_oof)
    validate_prediction_frame(final_test)

    runtime_seconds = time.perf_counter() - start
    m6_notes = "Per-target OOF blend of B4, M3, and M4; weights selected on OOF only."
    m7_notes = f"OOF-selected accounting adjustment: {best_adjustment}."
    m6_summary = _summary_row(
        "M6_oof_blend",
        "OOFWeightedBlend",
        "per_target_oof_blend",
        y_true,
        blended_oof,
        folds,
        runtime_seconds,
        m6_notes,
    )
    m7_summary = _summary_row(
        "M7_accounting_postprocess",
        "AccountingPostprocess",
        best_adjustment,
        y_true,
        final_oof,
        folds,
        runtime_seconds,
        m7_notes,
    )
    summary_rows = [m6_summary, m7_summary]

    m6_logs = _fold_score_rows(
        "M6_oof_blend",
        "OOFWeightedBlend",
        "per_target_oof_blend",
        y_true,
        blended_oof,
        folds,
        runtime_seconds,
        m6_notes,
    )
    m7_logs = _fold_score_rows(
        "M7_accounting_postprocess",
        "AccountingPostprocess",
        best_adjustment,
        y_true,
        final_oof,
        folds,
        runtime_seconds,
        m7_notes,
    )

    blend_table = pd.DataFrame(
        [
            {
                "target": target,
                "members": ",".join(config["members"]),
                "weights": ",".join(f"{weight:.2f}" for weight in config["weights"]),
                "oof_r2": config["oof_r2"],
                "coarse_step": config["coarse_step"],
                "fine_step": config["fine_step"],
                "fine_radius": config["fine_radius"],
            }
            for target, config in blend_config.items()
        ]
    )

    m6_submission = _submission_from_predictions(sample_submission, blended_test)
    final_submission = _submission_from_predictions(sample_submission, final_test)
    write_csv(ROOT / "results" / "tables" / "blend_scores.csv", blend_table)
    write_csv(ROOT / "results" / "tables" / "blend_weight_grid.csv", blend_grid)
    write_csv(ROOT / "results" / "tables" / "accounting_postprocess_scores.csv", accounting_scores)
    write_csv(ROOT / "results" / "tables" / "final_model_scores.csv", pd.DataFrame(summary_rows))
    write_csv(ROOT / "results" / "tables" / "final_prediction_identity_summary.csv", identity_error_summary(final_oof))
    write_csv(ROOT / "results" / "predictions" / "m6_oof_blend_test_predictions.csv", m6_submission)
    write_csv(ROOT / "results" / "predictions" / "final_submission_candidate.csv", final_submission)
    write_csv(ROOT / "deliverables" / "submission.csv", final_submission)
    _write_oof(ROOT / "results" / "oof" / "m6_oof_blend.csv", train_ids, meta, y_true, blended_oof)
    _write_oof(ROOT / "results" / "oof" / "m7_accounting_postprocess.csv", train_ids, meta, y_true, final_oof)
    write_csv(ROOT / "results" / "cv_scores" / "m6_oof_blend.csv", m6_logs)
    write_csv(ROOT / "results" / "cv_scores" / "m7_accounting_postprocess.csv", m7_logs)
    _append_experiment_log(
        ROOT / "results" / "experiment_log.csv",
        pd.concat([m6_logs, m7_logs], ignore_index=True),
        ["M6_oof_blend", "M7_accounting_postprocess"],
    )
    write_json(
        ROOT / "configs" / "blend_weights.json",
        {
            "experiment_id": "M6_oof_blend",
            "members": list(OOF_SPECS.keys()),
            "selection": "per_target_oof_r2",
            "weights": blend_config,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(
        ROOT / "configs" / "accounting_postprocess.json",
        {
            "experiment_id": "M7_accounting_postprocess",
            "selected_adjustment": best_adjustment,
            "selection": "max_mean_oof_r2_among_predefined_accounting_adjustments",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    final_experiment_id = "M7_accounting_postprocess" if best_adjustment != "none" else "M6_oof_blend"
    _update_best_config(summary_rows, final_experiment_id)

    write_json(
        ROOT / "results" / "final_submission_manifest.json",
        {
            "submission_path": "deliverables/submission.csv",
            "submission_sha256": sha256_file(ROOT / "deliverables" / "submission.csv"),
            "row_count": int(final_submission.shape[0]),
            "column_count": int(final_submission.shape[1]),
            "columns": list(final_submission.columns),
            "final_experiment_id": final_experiment_id,
            "best_adjustment": best_adjustment,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Selected accounting adjustment: {best_adjustment}")
    print(f"Saved deliverables/submission.csv in {runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
