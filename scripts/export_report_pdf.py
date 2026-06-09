from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pdfplumber
import win32com.client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as ROOT
from src.io_utils import sha256_file, write_json


DOCX_PATH = ROOT / "deliverables" / "financial_performance_prediction_report.docx"
PDF_PATH = ROOT / "deliverables" / "financial_performance_prediction_report.pdf"
PREVIEW_DIR = ROOT / "results" / "report_pdf_preview"


def export_docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    doc = None
    try:
        doc = word.Documents.Open(str(docx_path.resolve()))
        doc.ExportAsFixedFormat(str(pdf_path.resolve()), 17)
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()


def render_preview_pages(pdf_path: Path) -> list[str]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    document = fitz.open(str(pdf_path))
    page_indices = sorted({0, min(2, document.page_count - 1), document.page_count - 1})
    rendered = []
    for page_index in page_indices:
        page = document.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        output = PREVIEW_DIR / f"report_page_{page_index + 1}.png"
        pix.save(str(output))
        rendered.append(str(output.relative_to(ROOT)))
    document.close()
    return rendered


def validate_pdf(pdf_path: Path) -> dict:
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)
        first_text = pdf.pages[0].extract_text() or ""
    return {
        "pdf_path": str(pdf_path.relative_to(ROOT)),
        "pdf_sha256": sha256_file(pdf_path),
        "page_count": page_count,
        "first_page_has_title": "Financial Performance Prediction Report" in first_text,
    }


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(f"Missing report DOCX: {DOCX_PATH}")
    export_docx_to_pdf(DOCX_PATH, PDF_PATH)
    validation = validate_pdf(PDF_PATH)
    validation["preview_images"] = render_preview_pages(PDF_PATH)
    validation["generated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "results" / "report_pdf_validation.json", validation)

    report_manifest_path = ROOT / "results" / "report_manifest.json"
    if report_manifest_path.exists():
        manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
        manifest["report_pdf_sha256"] = validation["pdf_sha256"]
        manifest["report_pdf_page_count"] = validation["page_count"]
        manifest["report_pdf_preview_images"] = validation["preview_images"]
        write_json(report_manifest_path, manifest)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
