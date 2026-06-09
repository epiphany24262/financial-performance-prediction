from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as _PROJECT_ROOT, RAW_FILES
from src.io_utils import file_metadata, read_docx_text, read_text_file, write_json


def _csv_summary(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
    }


def _manifest_path() -> Path:
    return _PROJECT_ROOT / "results" / "input_manifest.json"


def build_manifest() -> dict:
    raw_dir = _PROJECT_ROOT
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_files": {},
        "csv_summary": {},
        "text_summary": {},
        "docx_summary": {},
    }

    for name in RAW_FILES:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path}")
        manifest["raw_files"][name] = file_metadata(path)

    manifest["csv_summary"]["train.csv"] = _csv_summary(raw_dir / "train.csv")
    manifest["csv_summary"]["test.csv"] = _csv_summary(raw_dir / "test.csv")
    manifest["csv_summary"]["sample_submission.csv"] = _csv_summary(raw_dir / "sample_submission.csv")
    manifest["text_summary"]["data_dictionary.txt"] = read_text_file(raw_dir / "data_dictionary.txt")
    manifest["docx_summary"]["说明.docx"] = read_docx_text(raw_dir / "说明.docx")
    return manifest


def verify_against_previous_manifest(current: dict) -> None:
    path = _manifest_path()
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    prev_hashes = {k: v["sha256"] for k, v in previous.get("raw_files", {}).items()}
    curr_hashes = {k: v["sha256"] for k, v in current.get("raw_files", {}).items()}
    if prev_hashes and prev_hashes != curr_hashes:
        raise SystemExit(
            "Raw file hashes changed since the last manifest. Stop here and inspect the raw inputs before continuing."
        )


def main() -> None:
    manifest = build_manifest()
    verify_against_previous_manifest(manifest)
    write_json(_manifest_path(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
