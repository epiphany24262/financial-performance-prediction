# Financial Performance Prediction

## Project
- Environment: QuantEnv
- Python: Python 3.10.20
- Python executable: D:\Anaconda\envs\QuantEnv\python.exe
- Best experiment: M6_oof_blend
- Best OOF mean R2: 0.857226605158
- Final submission rows/cols: 406 / 10
- Final accounting adjustment: none

## Raw Inputs
- train.csv | fc0d1fab1ed7b597cc900e1e724dc4847814fba7aa04dd722e21fbbb0d0f7049
- test.csv | a1003ff8a6f90a971e17d473781fff84621131f342f0e3d0fb2b0b59b886d7b6
- sample_submission.csv | 874682cab231f9e48b6c038ac4859213ebf70453bf390ce94de31273da8dc1e6
- data_dictionary.txt | ba3697c21555558bb831d33da97b35b420558f2494530dc6763b85eaf30ca128
- 说明.docx | c4e71f313214b7fe2fe1542d53a58ad75621e2f09875d501c18184e247819db6

## Deliverables
- financial_performance_prediction_final.ipynb | 8d17b0faec8b427fc2eb5aede737cf8d23731f213c72535d90133875e1082def
- financial_performance_prediction_report.docx | 35b09c99da5d9495bfd516fb9492412e822f0cf49b74ec89bd45045743f3d089
- financial_performance_prediction_report.pdf | 87f283de61393b52eebc4fb47ff2e441f624a71d8433656c2cd02902640a9d1d
- submission.csv | c8c777fb68ca7d363e218b8ff09d4d7eb2e189f92084523779bc0de9c947e97f

## Reproduction
```bash
conda run -n QuantEnv python scripts/check_environment.py
conda run -n QuantEnv python scripts/bootstrap.py
conda run -n QuantEnv python scripts/audit_data.py
conda run -n QuantEnv python scripts/run_baselines.py
conda run -n QuantEnv python scripts/run_sklearn_models.py
conda run -n QuantEnv python scripts/run_catboost_models.py
conda run -n QuantEnv python scripts/train_final.py
conda run -n QuantEnv python scripts/build_final_figures.py
conda run -n QuantEnv python scripts/build_notebook.py
conda run -n QuantEnv python scripts/build_report.py
conda run -n QuantEnv python scripts/export_report_pdf.py
conda run -n QuantEnv python scripts/package_delivery.py
conda run -n QuantEnv python scripts/validate_delivery.py
```

## Notes
- Notebook kernel name: quantenv
- Notebook cells: 14
- Report manifest generated at: 2026-06-09T15:37:20.398150+00:00
- Submission manifest generated at: 2026-06-09T15:07:14.290730+00:00
