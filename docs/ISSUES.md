# ISSUES

## MAJOR-001: QuantEnv still misses modeling and notebook dependencies

- Status: open
- Found: 2026-06-09
- Evidence: `results/environment_audit.json`
- Current missing packages: `catboost`, `xgboost`, `lightgbm`, `optuna`, `jupyter`
- Resolved part: `pytest` and `pytest-cov` were installed after the first environment snapshot, then the environment audit and lock files were regenerated.
- Impact: sklearn baselines and tests can run now. CatBoost, optional GBDT comparison, Optuna tuning, and final Notebook work still require minimal additional installation.
- Handling: install only the next required package set when entering that stage; do not bulk-upgrade the environment.

## MAJOR-002: Exact duplicate feature records must remain grouped during CV

- Status: mitigated in current baseline code; keep open for future model code.
- Found: 2026-06-09
- Evidence: `results/tables/duplicate_summary.csv`, `src/feature_engineering.py`, `src/cv.py`, `tests/test_cv_no_leakage.py`
- Result: train common-feature duplicate rows: 16 rows across 2 groups. Test common-feature duplicate rows: 2 rows across 1 group. Train/test share 2 common-feature hashes.
- Impact: plain KFold would leak identical history records across train/validation folds.
- Handling: all OOF experiments must use feature-hash `GroupKFold`; the current B0-B4 run already does this.

## MAJOR-003: Numeric columns contain positive infinity values

- Status: mitigated in current baseline code; keep open for future model code.
- Found: 2026-06-09
- Evidence: `results/data_audit.json`, `results/tables/numeric_extreme_summary.csv`, `src/feature_engineering.py`
- Result: train contains 13 `+inf` values and test contains 4 `+inf` values; no `-inf` values found.
- Impact: scalers, linear models, and some tree models can fail unless `inf` is treated as missing.
- Handling: current baseline and grouping code converts `+/-inf` to `NaN`. The same entry-point rule must be used for feature-engineered ML models.

## MAJOR-004: Equity target remains weak under rule baselines

- Status: open
- Found: 2026-06-09
- Evidence: `results/tables/baseline_scores.csv`
- Result: best rule baseline B4 reaches mean OOF R2 `0.7830834896750798`, but `Q0_TOTAL_STOCKHOLDERS_EQUITY` remains negative at `-0.05296033455914251`.
- Impact: the final model needs feature engineering and supervised learning for this target; a strong average score can hide this weak target.
- Handling: next modeling stage must report per-target scores and not optimize only the average.

## MINOR-001: Accounting identity error has heavy-tailed outliers

- Status: open
- Found: 2026-06-09
- Evidence: `results/tables/accounting_identity_summary.csv`, `figures/fig07_accounting_identity_error.png`
- Result: historical `assets - liabilities - equity` relative error has median near 0 in many quarters, but means are heavily influenced by extreme values.
- Impact: accounting-consistency postprocessing must be evaluated on OOF with robust summaries, not raw mean error alone.

