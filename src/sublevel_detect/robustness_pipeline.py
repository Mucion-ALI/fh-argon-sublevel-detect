from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from . import ablation_pipeline, main_pipeline, model, paths


def _json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _config_from_dict(payload: Dict[str, Any]) -> model.Config:
    names = {field.name for field in fields(model.Config)}
    return model.Config(**{key: value for key, value in dict(payload).items() if key in names})


def _scenario_weights(name: str, updates: Dict[str, float]) -> Dict[str, Any]:
    weights = dict(model.DEFAULT_SELECTOR_RANK_WEIGHTS)
    weights.update({key: float(value) for key, value in updates.items()})
    return {"scenario": name, "rank_weights": weights}


def selector_weight_scenarios() -> List[Dict[str, Any]]:
    scenarios = [_scenario_weights("baseline", {})]
    for key, value in model.DEFAULT_SELECTOR_RANK_WEIGHTS.items():
        scenarios.append(_scenario_weights(f"{key}_minus_25pct", {key: value * 0.75}))
        scenarios.append(_scenario_weights(f"{key}_plus_25pct", {key: value * 1.25}))
    scenarios.extend(
        [
            _scenario_weights(
                "fit_heavy",
                {"rmse": 1.5, "summary": 1.5, "d1": 1.25, "d2": 1.25, "structure": 1.0, "physical": 0.75},
            ),
            _scenario_weights(
                "structure_heavy",
                {"rmse": 0.75, "summary": 0.75, "d1": 1.25, "d2": 1.25, "structure": 1.75},
            ),
            _scenario_weights(
                "physical_heavy",
                {"rmse": 0.75, "summary": 0.75, "structure": 1.0, "physical": 1.5},
            ),
            _scenario_weights(
                "complexity_heavy",
                {"rmse": 0.75, "summary": 0.75, "bic": 1.5, "aic": 0.75, "degeneracy": 1.5},
            ),
        ]
    )
    return scenarios


def _decision_row(name: str, decision: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    selected_k = int(decision.get("selected_k", -1))
    selected = {}
    for row in decision.get("scored_levels", []):
        if int(row.get("n_levels", -1)) == selected_k:
            selected = row
            break
    return {
        "scenario": name,
        "selected_k": selected_k,
        "selected_composite_rank_score": model.safe_float(selected.get("composite_rank_score")),
        "best_k_by_fit": decision.get("best_k_by_fit"),
        "best_k_by_d1_rmse": decision.get("best_k_by_d1_rmse"),
        "best_k_by_d2_rmse": decision.get("best_k_by_d2_rmse"),
        "best_k_by_structure": decision.get("best_k_by_structure"),
        "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
        "best_k_by_bic": decision.get("best_k_by_bic"),
        "best_k_by_aic": decision.get("best_k_by_aic"),
        **{f"weight_{key}": float(value) for key, value in weights.items()},
    }


def analyze_selector_weight_perturbation(main_sweep_dir: Path, output_root: Path) -> Dict[str, Any]:
    rows = ablation_pipeline.load_summary_rows(main_sweep_dir)
    evidence = ablation_pipeline.load_forward_evidence(main_sweep_dir)
    decisions: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []
    for scenario in selector_weight_scenarios():
        name = str(scenario["scenario"])
        weights = dict(scenario["rank_weights"])
        decision = model.select_k_neutral_level_decision(
            rows,
            forward_evidence=evidence,
            cluster_tolerance_eV=0.02,
            rank_weights=weights,
        )
        decisions[name] = decision
        table_rows.append(_decision_row(name, decision, weights))
    counts = pd.DataFrame(table_rows)["selected_k"].value_counts().sort_index()
    summary_rows = [
        {"selected_k": int(k), "scenario_count": int(v), "scenario_fraction": float(v / max(len(table_rows), 1))}
        for k, v in counts.items()
    ]
    root = paths.robustness_dir(output_root)
    _write_csv(root / "selector_weight_perturbation.csv", table_rows)
    _write_csv(root / "selector_weight_perturbation_summary.csv", summary_rows)
    _json_dump({"decisions": decisions, "rows": table_rows}, root / "selector_weight_perturbation.json")
    _json_dump({"summary": summary_rows}, root / "selector_weight_perturbation_summary.json")
    lines = [
        "# Selector Weight Perturbation",
        "",
        "| selected K | scenario count | fraction |",
        "|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append("| {selected_k} | {scenario_count} | {scenario_fraction:.3f} |".format(**row))
    (root / "selector_weight_perturbation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": table_rows, "summary": summary_rows}


def _available_vr_values(input_path: str | Path) -> List[float]:
    frame = model.read_excel_or_csv(paths.resolve_project_path(input_path))
    c_vr = model.pick_col(frame, ("Vr", "vr", "V_r", "Ur"))
    return [float(x) for x in sorted(frame[c_vr].astype(float).unique().tolist())]


def _loo_config(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    excluded_vr: float,
) -> model.Config:
    base_payload = _json_load(main_sweep_dir / "config_used.json", {})
    if not isinstance(base_payload, dict):
        base_payload = {}
    defaults = main_pipeline.profile_defaults(mode)
    cpu_workers = int(base_payload.get("cpu_workers", defaults.get("cpu_workers", 1)))
    selected_device = "cuda" if device == "cuda" and model.torch.cuda.is_available() else "cpu"
    dispatch_strategy = "cpu_4" if selected_device == "cpu" and cpu_workers > 1 else "single"
    label = str(excluded_vr).replace("-", "minus_").replace(".", "p")
    base_payload.update(
        {
            "data_path": str(paths.resolve_project_path(input_path)),
            "out_dir": str(paths.robustness_dir(output_root) / "leave_one_vr_out" / f"vr_{label}" / "fullscan"),
            "device": device,
            "selected_device": selected_device,
            "dispatch_strategy": dispatch_strategy,
            "cpu_workers": cpu_workers,
            "retain_promoted_level_copies": False,
            "launch_monitor": False,
            "resume_mode": "auto" if mode == "fullscan" else "off",
            "hyperopt_enabled": False,
            "exclude_vr_values": str(excluded_vr),
        }
    )
    return _config_from_dict(base_payload)


def run_leave_one_vr_out(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
) -> Dict[str, Any]:
    root = paths.robustness_dir(output_root)
    vr_values = _available_vr_values(input_path)
    rows: List[Dict[str, Any]] = []
    fold_payloads: List[Dict[str, Any]] = []
    for vr in vr_values:
        cfg = _loo_config(
            mode=mode,
            input_path=input_path,
            output_root=output_root,
            device=device,
            main_sweep_dir=main_sweep_dir,
            excluded_vr=vr,
        )
        result = model.run_level_scan(cfg)
        decision = result["decision"]
        selected_k = int(decision.get("selected_k", -1))
        selected = {}
        for row in decision.get("scored_levels", []):
            if int(row.get("n_levels", -1)) == selected_k:
                selected = row
                break
        row = {
            "excluded_vr": float(vr),
            "selected_k": selected_k,
            "selected_composite_rank_score": model.safe_float(selected.get("composite_rank_score")),
            "selected_rmse_mean": model.safe_float(selected.get("rmse_mean")),
            "selected_d1_rmse_mean": model.safe_float(selected.get("d1_rmse_mean")),
            "selected_d2_rmse_mean": model.safe_float(selected.get("d2_rmse_mean")),
            "selected_bic": model.safe_float(selected.get("bic")),
            "selected_aic": model.safe_float(selected.get("aic")),
            "selected_structure_score": model.safe_float(selected.get("structure_score")),
            "selected_physical_score": model.safe_float(selected.get("vr_physical_response_score")),
            "best_k_by_fit": decision.get("best_k_by_fit"),
            "best_k_by_d1_rmse": decision.get("best_k_by_d1_rmse"),
            "best_k_by_d2_rmse": decision.get("best_k_by_d2_rmse"),
            "best_k_by_structure": decision.get("best_k_by_structure"),
            "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
            "best_k_by_bic": decision.get("best_k_by_bic"),
            "best_k_by_aic": decision.get("best_k_by_aic"),
            "sweep_dir": result.get("scan_dir"),
        }
        rows.append(row)
        fold_payloads.append({"excluded_vr": float(vr), "decision": decision, "sweep_dir": result.get("scan_dir")})
    _write_csv(root / "leave_one_vr_out_summary.csv", rows)
    _json_dump({"folds": fold_payloads, "summary": rows}, root / "leave_one_vr_out_summary.json")
    lines = [
        "# Leave-One-Vr-Out Robustness",
        "",
        "| excluded Vr | selected K | composite rank | RMSE | d1 RMSE | d2 RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {excluded_vr:.1f} | {selected_k} | {selected_composite_rank_score:.3f} | "
            "{selected_rmse_mean:.6f} | {selected_d1_rmse_mean:.6f} | {selected_d2_rmse_mean:.6f} |".format(**row)
        )
    (root / "leave_one_vr_out_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "folds": fold_payloads}


def write_summary(selector: Dict[str, Any], loo: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    root = paths.robustness_dir(output_root)
    selector_rows = list(selector.get("rows", []))
    loo_rows = list(loo.get("rows", []))
    selector_k4_count = sum(1 for row in selector_rows if int(row.get("selected_k", -1)) == 4)
    loo_k4_count = sum(1 for row in loo_rows if int(row.get("selected_k", -1)) == 4)
    payload = {
        "selector_scenario_count": int(len(selector_rows)),
        "selector_k4_count": int(selector_k4_count),
        "selector_k4_fraction": float(selector_k4_count / max(len(selector_rows), 1)),
        "leave_one_vr_fold_count": int(len(loo_rows)),
        "leave_one_vr_k4_count": int(loo_k4_count),
        "leave_one_vr_k4_fraction": float(loo_k4_count / max(len(loo_rows), 1)),
        "selector_summary": selector.get("summary", []),
        "leave_one_vr_summary": loo_rows,
    }
    _json_dump(payload, root / "robustness_summary.json")
    lines = [
        "# Robustness Summary",
        "",
        f"- Selector perturbation K=4 count: {selector_k4_count}/{len(selector_rows)}",
        f"- Leave-one-Vr-out K=4 count: {loo_k4_count}/{len(loo_rows)}",
        "",
        "Primary evidence files:",
        "- `selector_weight_perturbation.csv`",
        "- `selector_weight_perturbation_summary.csv`",
        "- `leave_one_vr_out_summary.csv`",
    ]
    (root / "robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: str | Path,
) -> Dict[str, Any]:
    main_sweep = Path(main_sweep_dir)
    if not main_sweep.exists():
        raise FileNotFoundError(f"Missing main baseline sweep: {main_sweep}")
    output = paths.resolve_project_path(output_root)
    selector = analyze_selector_weight_perturbation(main_sweep, output)
    loo = run_leave_one_vr_out(
        mode=mode,
        input_path=input_path,
        output_root=output,
        device=device,
        main_sweep_dir=main_sweep,
    )
    summary = write_summary(selector, loo, output)
    return {
        "ok": True,
        "mode": mode,
        "summary": str(paths.robustness_dir(output) / "robustness_summary.json"),
        "selector_k4_count": summary["selector_k4_count"],
        "selector_scenario_count": summary["selector_scenario_count"],
        "leave_one_vr_k4_count": summary["leave_one_vr_k4_count"],
        "leave_one_vr_fold_count": summary["leave_one_vr_fold_count"],
    }
