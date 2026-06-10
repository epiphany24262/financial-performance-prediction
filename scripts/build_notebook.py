from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as ROOT


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(text: str):
    return nbf.v4.new_code_cell(text)


BOOTSTRAP_CELL = r"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display


def find_project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        required = ["train.csv", "test.csv", "sample_submission.csv", "data_dictionary.txt", "scripts", "src"]
        if all((candidate / item).exists() for item in required):
            return candidate
    raise FileNotFoundError(
        "无法定位项目根目录。请从 financial-performance-prediction 项目根目录或 notebooks/deliverables 子目录打开本 Notebook。"
    )


ROOT = find_project_root()
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
DELIVERABLES = ROOT / "deliverables"

python_executable = Path(sys.executable).resolve()
if "quantenv" not in str(python_executable).lower():
    raise RuntimeError(
        f"当前 Notebook 内核不是 QuantEnv：{python_executable}\n"
        "请切换到 Jupyter 内核 Python (QuantEnv) 后重新从头运行。"
    )

print("项目根目录:", ROOT)
print("Python 可执行文件:", python_executable)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_outputs(paths: list[str]) -> list[str]:
    return [path for path in paths if not (ROOT / path).exists()]


def run_step(name: str, args: list[str], outputs: list[str]) -> None:
    missing = missing_outputs(outputs)
    if not missing:
        print(f"[跳过] {name}: 关键产物已存在")
        return

    print(f"[运行] {name}: 缺少 {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    start = time.perf_counter()
    completed = subprocess.run(
        [str(python_executable), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout[-6000:])
    if completed.stderr:
        print(completed.stderr[-6000:], file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} 执行失败，返回码 {completed.returncode}")

    still_missing = missing_outputs(outputs)
    if still_missing:
        raise FileNotFoundError(f"{name} 执行后仍缺少产物: {still_missing}")
    print(f"[完成] {name}: {time.perf_counter() - start:.1f}s")


def ensure_artifacts() -> None:
    # Generate every artifact needed by this Notebook when it is missing.
    run_step(
        "环境审计",
        ["scripts/check_environment.py"],
        ["results/environment_audit.json"],
    )
    run_step(
        "原始输入冻结",
        ["scripts/bootstrap.py"],
        ["results/input_manifest.json"],
    )
    run_step(
        "数据审计与 EDA 图表",
        ["scripts/audit_data.py"],
        [
            "results/data_audit.json",
            "results/tables/schema_summary.csv",
            "results/tables/missing_rate_by_column.csv",
            "results/tables/target_summary.csv",
            "figures/fig01_sector_distribution.png",
            "figures/fig08_target_correlation_heatmap.png",
        ],
    )
    run_step(
        "规则基线 B0-B4",
        ["scripts/run_baselines.py"],
        [
            "results/tables/baseline_scores.csv",
            "results/oof/baseline_oof.csv",
            "results/predictions/baseline_b4_test_predictions.csv",
            "configs/baseline_blend_weights.json",
        ],
    )
    run_step(
        "机器学习基线 M1-M2",
        ["scripts/run_sklearn_models.py"],
        [
            "results/tables/sklearn_model_scores.csv",
            "results/oof/m1_ridge_history_raw.csv",
            "results/oof/m2_hgb_history_raw.csv",
        ],
    )
    run_step(
        "CatBoost 直接预测 M3a-M3d",
        [
            "scripts/run_catboost_models.py",
            "M3a_catboost_direct_history_raw,M3b_catboost_direct_history_industry,"
            "M3c_catboost_direct_history_metadata,M3d_catboost_direct_history_metadata_engineered",
        ],
        [
            "results/tables/catboost_model_scores.csv",
            "results/oof/m3a_catboost_direct_history_raw.csv",
            "results/oof/m3b_catboost_direct_history_industry.csv",
            "results/oof/m3c_catboost_direct_history_metadata.csv",
            "results/oof/m3d_catboost_direct_history_metadata_engineered.csv",
            "results/predictions/m3d_catboost_direct_history_metadata_engineered_test_predictions.csv",
        ],
    )
    run_step(
        "CatBoost 残差模型 M4",
        ["scripts/run_catboost_models.py", "M4_catboost_residual_history_metadata_engineered"],
        [
            "results/oof/m4_catboost_residual_history_metadata_engineered.csv",
            "results/predictions/m4_catboost_residual_history_metadata_engineered_test_predictions.csv",
        ],
    )
    run_step(
        "OOF 融合、会计后处理与 submission",
        ["scripts/train_final.py"],
        [
            "results/tables/blend_scores.csv",
            "results/tables/accounting_postprocess_scores.csv",
            "results/tables/final_model_scores.csv",
            "results/oof/m6_oof_blend.csv",
            "results/oof/m7_accounting_postprocess.csv",
            "deliverables/submission.csv",
            "results/final_submission_manifest.json",
        ],
    )
    run_step(
        "最终模型图表",
        ["scripts/build_final_figures.py"],
        [
            "results/tables/all_model_scores.csv",
            "figures/fig09_model_comparison.png",
            "figures/fig10_target_score_heatmap.png",
            "figures/fig11_oof_scatter.png",
            "figures/fig12_residual_distribution.png",
            "figures/fig13_blend_weights.png",
        ],
    )


ensure_artifacts()
print("Notebook 运行所需产物已就绪。")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def show_table(path: Path, n: int = 10) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    display(df.head(n))
    return df


def show_fig(name: str) -> Path:
    path = FIGURES / name
    if not path.exists():
        raise FileNotFoundError(path)
    display(Image(filename=str(path)))
    return path
"""


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (QuantEnv)",
        "language": "python",
        "name": "quantenv",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        _md(
            "# 财务绩效预测：可独立运行 Notebook\n\n"
            "本 Notebook 使用 `Python (QuantEnv)` 内核。首个代码单元会自动定位项目根目录，"
            "检查当前解释器是否来自 `QuantEnv`，并在必要时按顺序运行本项目脚本来补齐缺失的"
            "数据审计、EDA 图表、交叉验证、OOF 融合和 `submission.csv`。如果相关产物已经存在，"
            "该单元会直接跳过对应步骤。所有表格和图形均来自 `results/` 与 `figures/`，避免手工抄写实验数值。"
        ),
        _md(
            "## 0. 独立运行准备\n\n"
            "直接从头运行全部单元即可复现实验展示。若 `results/`、`figures/` 或 `deliverables/submission.csv` "
            "缺失，下面的自举单元会使用当前 Notebook 内核的 Python 解释器重新生成；如果需要从零重建，"
            "CatBoost 交叉验证会耗时较长。"
        ),
        _code(BOOTSTRAP_CELL),
        _md("## 1. 输入文件与环境审计"),
        _code(
            "manifest = read_json(RESULTS / 'input_manifest.json')\n"
            "environment = read_json(RESULTS / 'environment_audit.json')\n"
            "print('环境名称:', environment['conda_environment_name'])\n"
            "print('Python:', environment['python_version'])\n"
            "print('Executable:', environment['python_executable'])\n"
            "print('缺失依赖:', environment['missing_packages'])\n"
            "pd.DataFrame(list(manifest['raw_files'].values())).loc[:, ['name', 'size_bytes', 'sha256']]"
        ),
        _md("## 2. Schema 与数据质量审计"),
        _code(
            "schema = show_table(TABLES / 'schema_summary.csv')\n"
            "missing = show_table(TABLES / 'missing_rate_by_column.csv', n=8)\n"
            "duplicates = show_table(TABLES / 'duplicate_summary.csv')\n"
            "target_summary = show_table(TABLES / 'target_summary.csv')"
        ),
        _md("## 3. EDA 图表"),
        _code(
            "for fig in [\n"
            "    'fig01_sector_distribution.png',\n"
            "    'fig02_industry_top20.png',\n"
            "    'fig03_missing_top20.png',\n"
            "    'fig04_missing_by_quarter.png',\n"
            "    'fig05_target_distributions.png',\n"
            "    'fig06_lag_correlation_heatmap.png',\n"
            "    'fig07_accounting_identity_error.png',\n"
            "    'fig08_target_correlation_heatmap.png',\n"
            "]:\n"
            "    display(Markdown(f'**{fig}**'))\n"
            "    show_fig(fig)"
        ),
        _md("## 4. 交叉验证与模型训练结果"),
        _code(
            "all_scores = show_table(TABLES / 'all_model_scores.csv', n=20)\n"
            "final_scores = show_table(TABLES / 'final_model_scores.csv')\n"
            "show_fig('fig09_model_comparison.png')\n"
            "show_fig('fig10_target_score_heatmap.png')"
        ),
        _md("## 5. OOF 融合与会计后处理"),
        _code(
            "blend_scores = show_table(TABLES / 'blend_scores.csv')\n"
            "accounting_scores = show_table(TABLES / 'accounting_postprocess_scores.csv')\n"
            "show_fig('fig11_oof_scatter.png')\n"
            "show_fig('fig12_residual_distribution.png')\n"
            "show_fig('fig13_blend_weights.png')"
        ),
        _md("## 6. Submission 校验"),
        _code(
            "submission = pd.read_csv(DELIVERABLES / 'submission.csv')\n"
            "sample = pd.read_csv(ROOT / 'sample_submission.csv')\n"
            "assert submission.shape == sample.shape\n"
            "assert list(submission.columns) == list(sample.columns)\n"
            "assert submission['Id'].equals(sample['Id'])\n"
            "values = submission.drop(columns=['Id']).to_numpy(dtype='float64')\n"
            "assert not np.isnan(values).any()\n"
            "assert not np.isinf(values).any()\n"
            "print('submission rows, cols:', submission.shape)\n"
            "print('submission sha256:', sha256_file(DELIVERABLES / 'submission.csv'))\n"
            "submission.head()"
        ),
    ]
    out = ROOT / "notebooks" / "financial_performance_prediction_final.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
