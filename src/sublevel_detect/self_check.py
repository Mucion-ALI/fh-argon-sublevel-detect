from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from . import paths


def _blocked_terms() -> list[str]:
    return [
        "argon_" + "kneutral_" + ("v" + "25"),
        "kneutral_" + ("v" + "25"),
        "V" + "25",
        "v" + "25",
        "V" + "24",
        "v" + "24",
        "V" + "23",
        "v" + "23",
        "FEATURE_" + "VERSION",
        "feature_" + "version",
    ]


def _iter_text_files(root: Path) -> Iterable[Path]:
    ignored = {
        "data",
        "output",
        "source_data_package",
        "__pycache__",
        ".pytest_cache",
        ".git",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv"}:
            yield path


def check_project(root: Path | None = None) -> Dict[str, Any]:
    root = paths.PROJECT_ROOT if root is None else Path(root)
    missing = []
    for rel in [
        "run.py",
        "src/sublevel_detect/model.py",
        "src/sublevel_detect/cli.py",
        "src/sublevel_detect/main_pipeline.py",
        "src/sublevel_detect/ablation_pipeline.py",
        "data/argon/FHdata.xlsx",
        "README.md",
        "LICENSE",
    ]:
        if not (root / rel).exists():
            missing.append(rel)
    hits = []
    blocked = _blocked_terms()
    for path in _iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in blocked:
            if term in text:
                hits.append({"path": str(path.relative_to(root)), "term": term})
        if re.search(r"\b[Vv][0-9]+\b", text):
            hits.append({"path": str(path.relative_to(root)), "term": "version_numeric_label"})
    return {"ok": not missing and not hits, "missing": missing, "blocked_term_hits": hits}


def main(argv: Sequence[str] | None = None) -> int:
    result = check_project()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1
