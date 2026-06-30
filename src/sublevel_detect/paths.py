from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "argon" / "FHdata.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main_scan_dir(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "main" / "fullscan"


def main_report_dir(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "main"


def ablation_dir(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "ablation"


def robustness_dir(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "robustness"


def sensitivity_dir(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "sensitivity"


def default_input_text() -> str:
    return DEFAULT_INPUT.relative_to(PROJECT_ROOT).as_posix()


def default_output_text() -> str:
    return DEFAULT_OUTPUT.relative_to(PROJECT_ROOT).as_posix()
