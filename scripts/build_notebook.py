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
            "# Financial Performance Prediction\n\n"
            "本 Notebook 汇总本项目的输入审计、EDA、交叉验证、模型训练结果、OOF 融合、会计后处理和 submission 校验。"
            "所有表格和图形均从 `results/` 与 `figures/` 读取，避免手工抄写实验数值。"
        ),
        _code(
            "from pathlib import Path\n"
            "import hashlib\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "RESULTS = ROOT / 'results'\n"
            "TABLES = RESULTS / 'tables'\n"
            "FIGURES = ROOT / 'figures'\n"
            "DELIVERABLES = ROOT / 'deliverables'\n\n"
            "def show_table(path, n=10):\n"
            "    df = pd.read_csv(path)\n"
            "    display(df.head(n))\n"
            "    return df\n\n"
            "def show_fig(name):\n"
            "    path = FIGURES / name\n"
            "    display(Image(filename=str(path)))\n"
            "    return path\n\n"
            "def sha256_file(path):\n"
            "    h = hashlib.sha256()\n"
            "    with open(path, 'rb') as f:\n"
            "        for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
            "            h.update(chunk)\n"
            "    return h.hexdigest()\n"
        ),
        _md("## 1. 输入文件与环境审计"),
        _code(
            "manifest = json.loads((RESULTS / 'input_manifest.json').read_text(encoding='utf-8'))\n"
            "environment = json.loads((RESULTS / 'environment_audit.json').read_text(encoding='utf-8'))\n"
            "print('Python:', environment['python_version'])\n"
            "print('Executable:', environment['python_executable'])\n"
            "print('Missing packages:', environment['missing_packages'])\n"
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
            "values = submission.drop(columns=['Id'])\n"
            "assert not values.isna().any().any()\n"
            "assert values.applymap(lambda x: x == float('inf') or x == float('-inf')).sum().sum() == 0\n"
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
