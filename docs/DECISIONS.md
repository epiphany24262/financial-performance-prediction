# DECISIONS

## 2026-06-09: Use QuantEnv only

- Automated commands use `conda run -n QuantEnv ...`.
- No `.venv` was created.
- `base` and system Python were not used for project automation.

## 2026-06-09: Freeze raw inputs before analysis

- The five original files were hashed into `results/input_manifest.json`.
- `scripts/bootstrap.py` stops if raw-file hashes differ from the previous manifest.

## 2026-06-09: Preserve sample submission order

- `sample_submission.csv` column order is the contract for final prediction files.
- Metric code may use `TARGET_COLUMNS`, but submission writing must follow `SUBMISSION_COLUMNS`.

## 2026-06-09: Group identical feature records for OOF

- Group IDs are built from non-target model features after converting `+/-inf` to `NaN`.
- All current B0-B4 baselines use `GroupKFold(n_splits=5)`.
- `tests/test_cv_no_leakage.py` verifies that groups do not cross train/validation folds.

## 2026-06-09: Install only immediate test dependencies

- Installed: `pytest`, `pytest-cov`.
- Not installed yet: `catboost`, `xgboost`, `lightgbm`, `optuna`, `jupyter`.
- Environment audit and lock files were regenerated after installation.

## 2026-06-09: Select B4 as current best baseline

- B4 is an OOF-selected per-target blend of B1/B2/B3 rule predictions.
- Weight selection used only GroupKFold OOF predictions and training targets.
- Current B4 OOF mean R2: `0.7830834896750798`.
- The best baseline is not the final model; `Q0_TOTAL_STOCKHOLDERS_EQUITY` remains weak.

