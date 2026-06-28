from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from . import main_pipeline, model, paths


def _json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _config_field_names() -> set[str]:
    return {field.name for field in fields(model.Config)}


def _config_from_dict(payload: Dict[str, Any]) -> model.Config:
    names = _config_field_names()
    return model.Config(**{key: value for key, value in payload.items() if key in names})


def hyperopt_enabled(*, mode: str, exclude: Sequence[str]) -> bool:
    return mode == "fullscan" and "hpopt" not in set(exclude)


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return value
    return value


def load_summary_rows(sweep_dir: Path) -> List[Dict[str, Any]]:
    payload = _json_load(sweep_dir / "scan_summary.json", {})
    if isinstance(payload, dict) and isinstance(payload.get("levels"), list):
        return [dict(row) for row in payload["levels"]]
    csv_path = sweep_dir / "scan_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing scan summary under {sweep_dir}")
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: _parse_literal(value) for key, value in row.items()})
    return rows


def load_forward_evidence(sweep_dir: Path) -> Dict[str, Any] | None:
    evidence = _json_load(sweep_dir / "forward_evidence.json", None)
    return evidence if isinstance(evidence, dict) else None


def run_no_forward_anchor_gap(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    exclude: Sequence[str],
) -> Path:
    out_dir = paths.ablation_dir(output_root) / "no_forward_anchor_gap" / "fullscan"
    base_payload = _json_load(main_sweep_dir / "config_used.json", {})
    if not isinstance(base_payload, dict):
        base_payload = {}
    defaults = main_pipeline.profile_defaults(mode)
    cpu_workers = int(defaults.pop("cpu_workers"))
    selected_device = "cuda" if device == "auto" and model.torch.cuda.is_available() else device
    dispatch_strategy = "cpu_4" if selected_device == "cpu" and cpu_workers > 1 else "single"
    base_payload.update(
        {
            "data_path": str(paths.resolve_project_path(input_path)),
            "out_dir": str(out_dir),
            "device": device,
            "selected_device": selected_device,
            "dispatch_strategy": dispatch_strategy,
            "cpu_workers": cpu_workers,
            "retain_promoted_level_copies": False,
            "launch_monitor": False,
            "resume_mode": "auto" if mode == "fullscan" else "off",
            "forward_prior_mode": "off",
            "forward_evidence_path": "",
            "forward_main_spacing": 0.0,
            "forward_spacing_std": 0.0,
            "forward_confidence": 0.0,
            "w_prior_anchor": 0.0,
            "w_prior_gap": 0.0,
            "w_extra_level_sparsity": 0.0,
            "hyperopt_enabled": hyperopt_enabled(mode=mode, exclude=exclude),
            **defaults,
        }
    )
    cfg = _config_from_dict(base_payload)
    model.run_level_scan(cfg)
    return out_dir


def select_from_rows(
    rows: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any] | None,
    *,
    use_weight_degeneracy: bool = True,
    use_energy_cluster_degeneracy: bool = True,
    metric_bic_aic_only: bool = False,
) -> Dict[str, Any]:
    return model.select_k_neutral_level_decision(
        rows,
        forward_evidence=evidence,
        cluster_tolerance_eV=0.02,
        use_weight_degeneracy=use_weight_degeneracy,
        use_energy_cluster_degeneracy=use_energy_cluster_degeneracy,
        metric_bic_aic_only=metric_bic_aic_only,
    )


def analyze_group(name: str, sweep_dir: Path, selector_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rows = load_summary_rows(sweep_dir)
    evidence = load_forward_evidence(sweep_dir)
    decision = select_from_rows(rows, evidence, **(selector_kwargs or {}))
    source_decision = _json_load(sweep_dir / "decision.json", {})
    return {
        "name": name,
        "sweep_dir": str(sweep_dir),
        "source_decision": source_decision,
        "reference_free_decision": decision,
        "rows": rows,
    }


def flatten_group_summary(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        decision = group["reference_free_decision"]
        rows.append(
            {
                "group": group["name"],
                "selected_k": decision.get("selected_k"),
                "best_k_by_fit": decision.get("best_k_by_fit"),
                "best_k_by_cv": decision.get("best_k_by_cv"),
                "best_k_by_d1_rmse": decision.get("best_k_by_d1_rmse"),
                "best_k_by_d2_rmse": decision.get("best_k_by_d2_rmse"),
                "best_k_by_structure": decision.get("best_k_by_structure"),
                "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
                "best_k_by_bic": decision.get("best_k_by_bic"),
                "best_k_by_aic": decision.get("best_k_by_aic"),
                "best_k_by_composite": decision.get("best_k_by_composite_kneutral"),
                "energy_cluster_negative_count": len(decision.get("energy_cluster_negative_evidence", [])),
                "selector_flags": json.dumps(decision.get("selector_ablation_flags", {}), ensure_ascii=False),
            }
        )
    return rows


def write_outputs(groups: Sequence[Dict[str, Any]], output_root: str | Path) -> Dict[str, Path]:
    root = paths.ablation_dir(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_rows = flatten_group_summary(groups)
    json_path = root / "ablation_summary.json"
    csv_path = root / "ablation_summary.csv"
    report_path = root / "ablation_report.md"
    _json_dump({"groups": groups, "summary": summary_rows}, json_path)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    metric_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    selector_payload: Dict[str, Any] = {}
    for group in groups:
        name = group["name"]
        decision = group["reference_free_decision"]
        selector_payload[name] = decision
        for row in decision.get("scored_levels", []):
            metric_rows.append({"group": name, **row})
        for row in decision.get("energy_cluster_metric_summary", []):
            cluster_rows.append({"group": name, **row})
    pd.DataFrame(metric_rows).to_csv(root / "ablation_metric_table.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(root / "energy_cluster_table.csv", index=False)
    _json_dump(selector_payload, root / "selector_ablation_decision.json")
    lines = [
        "# Ablation Baseline Report",
        "",
        "| group | selected K | fit | CV | d1 | d2 | structure | physical | BIC | AIC | clustered K count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {selected_k} | {best_k_by_fit} | {best_k_by_cv} | {best_k_by_d1_rmse} | "
            "{best_k_by_d2_rmse} | {best_k_by_structure} | {best_k_by_physical_response} | "
            "{best_k_by_bic} | {best_k_by_aic} | {energy_cluster_negative_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The ablation line compares the main baseline, selector-only variants, and the no-forward-anchor-gap retrain.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "report": report_path}


def run(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: str | Path,
    exclude: Sequence[str],
) -> Dict[str, Any]:
    main_sweep = Path(main_sweep_dir)
    if not main_sweep.exists():
        raise FileNotFoundError(f"Missing main baseline sweep: {main_sweep}")
    no_prior_dir = run_no_forward_anchor_gap(
        mode=mode,
        input_path=input_path,
        output_root=output_root,
        device=device,
        main_sweep_dir=main_sweep,
        exclude=exclude,
    )
    groups = [
        analyze_group("main_baseline", main_sweep),
        analyze_group("no_forward_anchor_gap", no_prior_dir),
        analyze_group("selector_no_weight_degeneracy", main_sweep, {"use_weight_degeneracy": False}),
        analyze_group("selector_no_energy_cluster", main_sweep, {"use_energy_cluster_degeneracy": False}),
        analyze_group("selector_metric_bic_aic_only", main_sweep, {"metric_bic_aic_only": True}),
    ]
    outputs = write_outputs(groups, output_root)
    return {
        "ok": True,
        "mode": mode,
        "summary": str(outputs["csv"]),
        "report": str(outputs["report"]),
        "groups": [group["name"] for group in groups],
    }
