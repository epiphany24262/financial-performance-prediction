# ISSUES

## MAJOR-001: QuantEnv still misses optional comparison and notebook dependencies

- Status: accepted limitation; not a delivery blocker for the current route.
- Found: 2026-06-09
- Evidence: `results/environment_audit.json`
- Current missing packages: `xgboost`, `lightgbm`, `optuna`, `jupyter`
- Resolved part: `pytest`, `pytest-cov`, and `catboost` were installed after the first environment snapshot, then the environment audit and lock files were regenerated.
- Impact: sklearn baselines, tests, CatBoost models, notebook execution through `nbconvert` APIs, Word report generation, and delivery validation all run now. Optional GBDT comparison and Optuna tuning were not used in the final route.
- Handling: leave these packages uninstalled to avoid unnecessary environment churn unless a later extension explicitly requires them.

## MAJOR-002: Exact duplicate feature records must remain grouped during CV

- Status: resolved in current pipeline; keep as an invariant for future model code.
- Found: 2026-06-09
- Evidence: `results/tables/duplicate_summary.csv`, `src/feature_engineering.py`, `src/cv.py`, `tests/test_cv_no_leakage.py`
- Result: train common-feature duplicate rows: 16 rows across 2 groups. Test common-feature duplicate rows: 2 rows across 1 group. Train/test share 2 common-feature hashes.
- Impact: plain KFold would leak identical history records across train/validation folds.
- Handling: all OOF experiments use feature-hash `GroupKFold`; tests and final delivery validation passed.

## MAJOR-003: Numeric columns contain positive infinity values

- Status: resolved in current pipeline; keep as an invariant for future model code.
- Found: 2026-06-09
- Evidence: `results/data_audit.json`, `results/tables/numeric_extreme_summary.csv`, `src/feature_engineering.py`
- Result: train contains 13 `+inf` values and test contains 4 `+inf` values; no `-inf` values found.
- Impact: scalers, linear models, and some tree models can fail unless `inf` is treated as missing.
- Handling: current baseline, feature engineering, sklearn, CatBoost, blend, and submission code convert or reject `+/-inf`; final submission has no `NaN` or `inf`.

## MAJOR-004: Equity target remains weak and sklearn baselines do not beat B4 on mean R2

- Status: resolved by current M6 OOF blend result; keep as historical modeling risk.
- Found: 2026-06-09
- Evidence: `results/tables/baseline_scores.csv`, `results/tables/sklearn_model_scores.csv`, `results/tables/catboost_model_scores.csv`
- Result: best rule baseline B4 reaches mean OOF R2 `0.7830834896750798`. M4 CatBoost residual reaches `0.8252497284371879`. M6 OOF blend reaches `0.8572266051577525`, and `Q0_TOTAL_STOCKHOLDERS_EQUITY` improves to `0.40086690788760915`.
- Impact: M6 is now the final selected model. Target-level variance is documented in the report and target-score heatmap.
- Handling: final submission uses M6 blend weights selected only from OOF results.

## MINOR-001: Accounting identity error has heavy-tailed outliers

- Status: evaluated; no postprocessing applied because OOF did not improve.
- Found: 2026-06-09
- Evidence: `results/tables/accounting_identity_summary.csv`, `figures/fig07_accounting_identity_error.png`
- Result: historical `assets - liabilities - equity` relative error has median near 0 in many quarters, but means are heavily influenced by extreme values.
- Impact: accounting-consistency postprocessing must be evaluated on OOF with robust summaries, not raw mean error alone.
- Handling: `results/tables/accounting_postprocess_scores.csv` shows that `none` has the best OOF mean R2, so the final submission does not force accounting identities.
