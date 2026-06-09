# AGENT_STATE

- Current phase: Baseline stage complete; entering feature engineering and model stage.
- Last successful commands:
  - `conda run -n QuantEnv python -m pytest -q`
  - `conda run -n QuantEnv python scripts/run_baselines.py`
- Current best OOF mean R2: `0.7830834896750798`
- Current best experiment: `B4`
- Open BLOCKER: none.
- Open MAJOR issues:
  - MAJOR-001: modeling/notebook dependencies still missing.
  - MAJOR-002: identical feature rows require GroupKFold grouping.
  - MAJOR-003: numeric `+inf` values must be converted to `NaN` at model entry points.
  - MAJOR-004: `Q0_TOTAL_STOCKHOLDERS_EQUITY` remains weak under rule baselines.
- Next single action: implement fold-safe feature engineering and run M1/M2 sklearn baselines before CatBoost.
- Latest Git commit: use `git log -1 --oneline`.

## Completed Outputs

- Stage 1:
  - `results/environment_audit.json`
  - `results/input_manifest.json`
  - `results/data_audit.json`
  - `results/tables/*.csv`
  - `figures/fig01_sector_distribution.png` to `figures/fig08_target_correlation_heatmap.png`
- Baselines:
  - `results/tables/baseline_scores.csv`
  - `results/tables/baseline_blend_grid.csv`
  - `results/oof/baseline_oof.csv`
  - `results/cv_scores/b0.csv` to `results/cv_scores/b4.csv`
  - `results/predictions/baseline_b4_test_predictions.csv`
  - `configs/baseline_blend_weights.json`
