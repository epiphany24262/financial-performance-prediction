from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as _PROJECT_ROOT
from src.io_utils import write_json


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "optuna": "optuna",
    "joblib": "joblib",
    "pyyaml": "yaml",
    "jupyter": "jupyter",
    "nbconvert": "nbconvert",
    "nbformat": "nbformat",
    "ipykernel": "ipykernel",
    "python-docx": "docx",
    "openpyxl": "openpyxl",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
}


def _version_for_module(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def build_environment_audit() -> dict:
    installed_versions = {}
    missing_packages = []
    for package_name, module_name in REQUIRED_PACKAGES.items():
        version = _version_for_module(module_name)
        installed_versions[package_name] = version
        if version is None:
            missing_packages.append(package_name)

    python = str(Path(sys.executable).resolve())
    python_version = _run_command([python, "--version"])
    conda_version = _run_command(["conda", "--version"])
    pip_version = _run_command([python, "-m", "pip", "--version"])
    conda_env_list = _run_command(["conda", "env", "list"]) or ""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conda_environment_name": "QuantEnv",
        "python_executable": python,
        "python_version": python_version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "conda_version": conda_version,
        "pip_version": pip_version,
        "required_packages": list(REQUIRED_PACKAGES.keys()),
        "installed_versions": installed_versions,
        "missing_packages": missing_packages,
        "conda_env_list_contains_quantenv": "QuantEnv" in conda_env_list,
        "active_path_hint_contains_quantenv": "quantenv" in python.lower(),
    }


def main() -> None:
    audit = build_environment_audit()
    write_json(_PROJECT_ROOT / "results" / "environment_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
