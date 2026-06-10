from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as ROOT
from src.io_utils import sha256_file, write_json


REPORT_PATH = ROOT / "deliverables" / "financial_performance_prediction_report.docx"


def _set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold


def _set_table_style(table) -> None:
    table.style = "Table Grid"
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)


def _add_dataframe_table(doc: Document, df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        doc.add_paragraph(title, style="Heading 3")
    table = doc.add_table(rows=1, cols=len(df.columns))
    _set_table_style(table)
    hdr = table.rows[0].cells
    for idx, col in enumerate(df.columns):
        _set_cell_text(hdr[idx], col, bold=True)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for idx, value in enumerate(row.tolist()):
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = "" if pd.isna(value) else str(value)
            _set_cell_text(cells[idx], text)
    doc.add_paragraph("")


def _add_picture(doc: Document, image_path: Path, caption: str, width_inches: float = 6.3) -> None:
    doc.add_picture(str(image_path), width=Inches(width_inches))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True


def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_summary() -> pd.DataFrame:
    all_scores = pd.read_csv(ROOT / "results" / "tables" / "all_model_scores.csv")
    selected = all_scores[
        all_scores["experiment_id"].isin(
            [
                "B4",
                "M1_ridge_history_raw",
                "M2_hgb_history_raw",
                "M3a_catboost_direct_history_raw",
                "M3b_catboost_direct_history_industry",
                "M3c_catboost_direct_history_metadata",
                "M3d_catboost_direct_history_metadata_engineered",
                "M4_catboost_residual_history_metadata_engineered",
                "M6_oof_blend",
                "M7_accounting_postprocess",
            ]
        )
    ].copy()
    selected = selected.sort_values("mean_r2", ascending=False)
    return selected[
        ["experiment_id", "model_name", "target_strategy", "mean_r2", "fold_mean_r2_std", "runtime_seconds"]
    ]


def _figure_plan() -> list[tuple[str, str]]:
    return [
        ("fig01_sector_distribution.png", "Figure 1. Sector sample distribution."),
        ("fig02_industry_top20.png", "Figure 2. Top-20 industry sample counts."),
        ("fig03_missing_top20.png", "Figure 3. Top-20 missing-rate columns."),
        ("fig04_missing_by_quarter.png", "Figure 4. Mean missing rate by quarter."),
        ("fig05_target_distributions.png", "Figure 5. Signed-log target distributions."),
        ("fig06_lag_correlation_heatmap.png", "Figure 6. Lag correlation heatmap."),
        ("fig07_accounting_identity_error.png", "Figure 7. Accounting identity error distribution."),
        ("fig08_target_correlation_heatmap.png", "Figure 8. Target correlation heatmap."),
        ("fig09_model_comparison.png", "Figure 9. Model OOF mean R2 comparison."),
        ("fig10_target_score_heatmap.png", "Figure 10. Target-level OOF R2 heatmap."),
        ("fig11_oof_scatter.png", "Figure 11. OOF actual-vs-predicted scatter plots."),
        ("fig12_residual_distribution.png", "Figure 12. OOF residual distribution by target."),
        ("fig13_blend_weights.png", "Figure 13. Per-target OOF blend weights."),
    ]


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Financial Performance Prediction Report")
    run.bold = True
    run.font.size = Pt(18)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Final notebook, OOF validation, blend search, accounting postprocessing, and submission packaging.")

    meta = _read_json(ROOT / "results" / "final_submission_manifest.json")
    env = _read_json(ROOT / "results" / "environment_audit.json")
    manifest = _read_json(ROOT / "results" / "input_manifest.json")
    best = _read_json(ROOT / "configs" / "best_config.json")

    _add_heading(doc, "1. Executive Summary", level=1)
    p = doc.add_paragraph()
    p.add_run("Best OOF blend mean R2: ").bold = True
    p.add_run(f"{float(best['best_mean_oof_r2']):.12f}")
    p = doc.add_paragraph()
    p.add_run("Selected accounting adjustment: ").bold = True
    p.add_run(meta["best_adjustment"])
    p = doc.add_paragraph()
    p.add_run("Final experiment: ").bold = True
    p.add_run(meta["final_experiment_id"])
    p = doc.add_paragraph()
    p.add_run("Best baseline / candidate chain: ").bold = True
    p.add_run("B4 -> M3/M4 -> M6 blend -> M7 accounting check")
    p = doc.add_paragraph()
    p.add_run("Submission rows/cols: ").bold = True
    p.add_run(f"{meta['row_count']} / {meta['column_count']}")

    _add_heading(doc, "2. Environment and Raw Input Freeze", level=1)
    doc.add_paragraph(
        f"All automation was run in Conda environment QuantEnv with Python {env['python_version']}. "
        f"Missing optional packages remain: {', '.join(env['missing_packages']) or 'none'}."
    )
    raw_files = pd.DataFrame(list(manifest["raw_files"].values()))[["name", "size_bytes", "sha256"]]
    _add_dataframe_table(doc, raw_files, "Raw file freeze manifest")

    _add_heading(doc, "3. Data Audit and EDA", level=1)
    schema = pd.read_csv(ROOT / "results" / "tables" / "schema_summary.csv")
    _add_dataframe_table(doc, schema, "Schema summary")
    missing = pd.read_csv(ROOT / "results" / "tables" / "missing_rate_by_quarter.csv")
    _add_dataframe_table(doc, missing.head(10), "Quarter-level missing rate summary")

    figures_dir = ROOT / "figures"
    for figure_name, caption in _figure_plan()[:8]:
        _add_picture(doc, figures_dir / figure_name, caption)

    _add_heading(doc, "4. Modeling and Validation", level=1)
    model_summary = _model_summary()
    _add_dataframe_table(doc, model_summary, "Selected experiment summary")
    blend = pd.read_csv(ROOT / "results" / "tables" / "blend_scores.csv")
    _add_dataframe_table(doc, blend, "Per-target OOF blend weights")
    accounting = pd.read_csv(ROOT / "results" / "tables" / "accounting_postprocess_scores.csv")
    _add_dataframe_table(doc, accounting, "Accounting postprocessing candidates")

    for figure_name, caption in _figure_plan()[8:]:
        _add_picture(doc, figures_dir / figure_name, caption)

    _add_heading(doc, "5. Final Submission and Reproducibility", level=1)
    submission = pd.read_csv(ROOT / "deliverables" / "submission.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert submission.shape == sample.shape
    doc.add_paragraph(
        f"The final submission matches the sample template exactly with SHA256 {sha256_file(ROOT / 'deliverables' / 'submission.csv')}. "
        f"The notebook and report are designed to be reproducible from the frozen raw inputs and saved results tables."
    )

    doc.add_paragraph("Key reproducibility command chain:", style="Heading 3")
    for cmd in [
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
    ]:
        doc.add_paragraph(cmd, style="List Bullet")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(REPORT_PATH))

    report_manifest = {
        "best_experiment_id": meta["final_experiment_id"],
        "figure_files": [name for name, _ in _figure_plan()],
        "table_files": [
            "all_model_scores.csv",
            "baseline_scores.csv",
            "sklearn_model_scores.csv",
            "catboost_model_scores.csv",
            "final_model_scores.csv",
            "blend_scores.csv",
            "accounting_postprocess_scores.csv",
        ],
        "report_sha256": sha256_file(REPORT_PATH),
        "submission_sha256": meta["submission_sha256"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "results" / "report_manifest.json", report_manifest)
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
