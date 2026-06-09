from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ID_COLUMN, PROJECT_ROOT as ROOT
from src.io_utils import sha256_file, write_json


SOURCE_NOTEBOOK = ROOT / "notebooks" / "financial_performance_prediction_final.ipynb"
OUTPUT_NOTEBOOK = ROOT / "deliverables" / "financial_performance_prediction_final.ipynb"
REPORT_PATH = ROOT / "deliverables" / "financial_performance_prediction_report.docx"
REPORT_PDF_PATH = ROOT / "deliverables" / "financial_performance_prediction_report.pdf"
SUBMISSION_PATH = ROOT / "deliverables" / "submission.csv"
README_PATH = ROOT / "deliverables" / "README_delivery.md"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execute_notebook(source: Path, output: Path, kernel_name: str = "quantenv") -> dict:
    nb = nbf.read(str(source), as_version=4)
    ep = ExecutePreprocessor(timeout=1800, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(output))
    return {
        "source_path": str(source),
        "output_path": str(output),
        "kernel_name": kernel_name,
        "output_sha256": sha256_file(output),
        "cell_count": len(nb.cells),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _build_readme() -> str:
    raw_manifest = _read_json(ROOT / "results" / "input_manifest.json")
    env = _read_json(ROOT / "results" / "environment_audit.json")
    final_manifest = _read_json(ROOT / "results" / "final_submission_manifest.json")
    report_manifest = _read_json(ROOT / "results" / "report_manifest.json")
    notebook_manifest = _read_json(ROOT / "results" / "notebook_manifest.json")
    best = _read_json(ROOT / "configs" / "best_config.json")
    lines = [
        "# Financial Performance Prediction",
        "",
        "## Project",
        "- Environment: QuantEnv",
        f"- Python: {env['python_version']}",
        f"- Python executable: {env['python_executable']}",
        f"- Best experiment: {best['best_experiment_id']}",
        f"- Best OOF mean R2: {best['best_mean_oof_r2']:.12f}",
        f"- Final submission rows/cols: {final_manifest['row_count']} / {final_manifest['column_count']}",
        f"- Final accounting adjustment: {final_manifest['best_adjustment']}",
        "",
        "## Raw Inputs",
    ]
    for item in raw_manifest["raw_files"].values():
        lines.append(f"- {item['name']} | {item['sha256']}")
    lines.extend(
        [
            "",
            "## Deliverables",
            f"- {OUTPUT_NOTEBOOK.name} | {notebook_manifest['output_sha256']}",
            f"- {REPORT_PATH.name} | {report_manifest['report_sha256']}",
            (
                f"- {REPORT_PDF_PATH.name} | {report_manifest['report_pdf_sha256']}"
                if REPORT_PDF_PATH.exists() and "report_pdf_sha256" in report_manifest
                else "- report PDF preview | not generated"
            ),
            f"- {SUBMISSION_PATH.name} | {final_manifest['submission_sha256']}",
            "",
            "## Reproduction",
            "```bash",
            "conda run -n QuantEnv python scripts/check_environment.py",
            "conda run -n QuantEnv python scripts/bootstrap.py",
            "conda run -n QuantEnv python scripts/audit_data.py",
            "conda run -n QuantEnv python scripts/run_baselines.py",
            "conda run -n QuantEnv python scripts/run_sklearn_models.py",
            "conda run -n QuantEnv python scripts/run_catboost_models.py",
            "conda run -n QuantEnv python scripts/train_final.py",
            "conda run -n QuantEnv python scripts/build_final_figures.py",
            "conda run -n QuantEnv python scripts/build_notebook.py",
            "conda run -n QuantEnv python scripts/build_report.py",
            "conda run -n QuantEnv python scripts/export_report_pdf.py",
            "conda run -n QuantEnv python scripts/package_delivery.py",
            "conda run -n QuantEnv python scripts/validate_delivery.py",
            "```",
            "",
            "## Notes",
            f"- Notebook kernel name: {notebook_manifest['kernel_name']}",
            f"- Notebook cells: {notebook_manifest['cell_count']}",
            f"- Report manifest generated at: {report_manifest['generated_utc']}",
            f"- Submission manifest generated at: {final_manifest['timestamp_utc']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE_NOTEBOOK.exists():
        raise FileNotFoundError(f"Missing source notebook: {SOURCE_NOTEBOOK}")
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing report: {REPORT_PATH}")
    if not SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"Missing submission: {SUBMISSION_PATH}")

    notebook_manifest = _execute_notebook(SOURCE_NOTEBOOK, OUTPUT_NOTEBOOK, kernel_name="quantenv")
    write_json(ROOT / "results" / "notebook_manifest.json", notebook_manifest)

    readme_text = _build_readme()
    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text(readme_text, encoding="utf-8")

    package_manifest = {
        "notebook": notebook_manifest,
        "report_sha256": sha256_file(REPORT_PATH),
        "report_pdf_sha256": sha256_file(REPORT_PDF_PATH) if REPORT_PDF_PATH.exists() else None,
        "submission_sha256": sha256_file(SUBMISSION_PATH),
        "readme_sha256": sha256_file(README_PATH),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "results" / "package_manifest.json", package_manifest)
    print(f"Saved {OUTPUT_NOTEBOOK}")
    print(f"Saved {README_PATH}")


if __name__ == "__main__":
    main()
