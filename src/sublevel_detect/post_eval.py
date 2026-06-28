from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from . import model, paths


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_paper_summary(*, sweep_dir: Path, output_root: Path) -> Dict[str, Any]:
    decision_path = sweep_dir / "decision.json"
    decision = _load_json(decision_path)
    scan_rows = _read_csv_rows(sweep_dir / "scan_summary.csv")
    selection_rows = _read_csv_rows(sweep_dir / "model_selection_table.csv")
    report_dir = paths.main_report_dir(output_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "selected_k": decision.get("selected_k"),
        "best_k_by_structure": decision.get("best_k_by_structure"),
        "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
        "best_k_by_composite": decision.get("best_k_by_composite_kneutral"),
        "level_count": len(scan_rows),
        "selection_table_rows": len(selection_rows),
        "evidence_files": {
            "decision": str(decision_path),
            "scan_summary": str(sweep_dir / "scan_summary.csv"),
            "model_selection_table": str(sweep_dir / "model_selection_table.csv"),
            "physical_response": str(sweep_dir / "vr_physical_response.csv"),
            "structure_metrics": str(sweep_dir / "structure_metrics.csv"),
        },
    }
    json_path = report_dir / "paper_summary.json"
    md_path = report_dir / "paper_summary.md"
    json_path.write_text(json.dumps(model.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Paper Evidence Summary",
        "",
        f"- Selected K: {payload['selected_k']}",
        f"- Best K by structure: {payload['best_k_by_structure']}",
        f"- Best K by physical response: {payload['best_k_by_physical_response']}",
        f"- Main scan rows: {payload['level_count']}",
        "",
        "Primary evidence files:",
    ]
    for name, value in payload["evidence_files"].items():
        lines.append(f"- {name}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["json_path"] = json_path
    payload["markdown_path"] = md_path
    return payload
