from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

from . import model, paths, post_eval


def _json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _selected_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "cuda":
        return "cuda" if model.torch.cuda.is_available() else "cpu"
    if requested == "auto":
        return "cpu"
    return requested


def profile_defaults(mode: str) -> Dict[str, Any]:
    if mode == "smoke":
        return {
            "epochs": 2,
            "scan_seeds": "0",
            "level_scan_min": 1,
            "level_scan_max": 2,
            "early_stop_min_epochs": 1,
            "early_stop_warmup": 1,
            "early_stop_patience": 10,
            "hyperopt_short_epochs": 1,
            "hyperopt_trials": 2,
            "hyperopt_stage1_epochs": 1,
            "hyperopt_stage2_epochs": 1,
            "hyperopt_stage3_epochs": 2,
            "hyperopt_stage1_top_k": 1,
            "hyperopt_stage2_top_k": 1,
            "hyperopt_stage1_levels": "1,2",
            "hyperopt_stage2_levels": "1,2",
            "hyperopt_stage3_levels": "1,2",
            "hyperopt_stage1_seeds": "0",
            "hyperopt_stage2_seeds": "0",
            "hyperopt_stage3_seeds": "0",
            "cpu_workers": 1,
        }
    return {
        "epochs": 3500,
        "scan_seeds": "0,1,2",
        "level_scan_min": 1,
        "level_scan_max": 8,
        "early_stop_min_epochs": 300,
        "early_stop_warmup": 300,
        "early_stop_patience": 45,
        "hyperopt_short_epochs": 80,
        "hyperopt_trials": 14,
        "hyperopt_stage1_epochs": 80,
        "hyperopt_stage2_epochs": 240,
        "hyperopt_stage3_epochs": 300,
        "hyperopt_stage1_top_k": 4,
        "hyperopt_stage2_top_k": 2,
        "hyperopt_stage1_levels": "1,2,4,6,8",
        "hyperopt_stage2_levels": "2,4,6,8",
        "hyperopt_stage3_levels": "1,2,3,4,5,6,7,8",
        "hyperopt_stage1_seeds": "0",
        "hyperopt_stage2_seeds": "0,1",
        "hyperopt_stage3_seeds": "0,1",
        "cpu_workers": 4,
    }


def build_config(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    exclude: Sequence[str],
) -> model.Config:
    defaults = profile_defaults(mode)
    data_path = paths.resolve_project_path(input_path)
    out_dir = paths.main_scan_dir(output_root)
    selected_device = _selected_device(device)
    cpu_workers = int(defaults.pop("cpu_workers"))
    dispatch_strategy = "cpu_4" if selected_device == "cpu" and cpu_workers > 1 else "single"
    cfg = model.Config(
        data_path=str(data_path),
        out_dir=str(out_dir),
        profile=mode,
        optimizer="muon_hybrid",
        device=device,
        selected_device=selected_device,
        fixed_levels=0,
        full_train_all_levels=True,
        dispatch_strategy=dispatch_strategy,
        cpu_workers=cpu_workers,
        resume_mode="auto" if mode == "fullscan" else "off",
        launch_monitor=False,
        retain_promoted_level_copies=True,
        hyperopt_enabled=(mode == "fullscan" and "hpopt" not in set(exclude)),
        w_extra_level_sparsity=0.0,
        **defaults,
    )
    return cfg


def prepare_forward_prior(cfg: model.Config) -> tuple[model.Config, Dict[str, Any]]:
    sweep_dir = Path(cfg.out_dir)
    evidence = model.build_forward_evidence(cfg)
    next_cfg, priors = apply_forward_prior_with_strength(cfg, evidence, strength_factor=1.0)
    next_cfg.forward_prior_mode = "auto"
    next_cfg.forward_evidence_path = str(sweep_dir / "forward_evidence.json")
    model.safe_json_dump(evidence, sweep_dir / "forward_evidence.json")
    model.safe_json_dump(priors, sweep_dir / "forward_priors.json")
    return next_cfg, {
        "mode": "auto",
        "soft_constraint_only": True,
        "forward_evidence": evidence,
        "forward_priors": priors,
    }


def apply_forward_prior_with_strength(
    cfg: model.Config,
    evidence: Dict[str, Any],
    *,
    strength_factor: float,
) -> tuple[model.Config, Dict[str, Any]]:
    factor = max(0.0, float(strength_factor))
    priors = model.compile_forward_priors(evidence, cfg)
    updates = dict(priors.get("config_updates", {}))
    if factor <= 0.0:
        updates.update(
            {
                "forward_prior_mode": "off",
                "forward_main_spacing": 0.0,
                "forward_spacing_std": 0.0,
                "forward_confidence": 0.0,
                "w_prior_anchor": 0.0,
                "w_prior_gap": 0.0,
            }
        )
    else:
        updates["w_prior_anchor"] = float(model.safe_float(updates.get("w_prior_anchor"), cfg.w_prior_anchor) * factor)
        updates["w_prior_gap"] = float(model.safe_float(updates.get("w_prior_gap"), cfg.w_prior_gap) * factor)
    scaled_priors = dict(priors)
    scaled_priors["strength_factor"] = factor
    scaled_priors["config_updates"] = updates
    return model.apply_forward_priors_to_config(cfg, scaled_priors), scaled_priors


def run(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    exclude: Sequence[str],
) -> Dict[str, Any]:
    cfg = build_config(mode=mode, input_path=input_path, output_root=output_root, device=device, exclude=exclude)
    cfg, forward_prior = prepare_forward_prior(cfg)
    sweep = model.run_level_scan(cfg)
    sweep_dir = Path(cfg.out_dir)
    production_summary = {
        "mode": mode,
        "hyperopt_enabled": bool(cfg.hyperopt_enabled),
        "input": str(Path(cfg.data_path)),
        "sweep_dir": str(sweep_dir),
        "forward_prior": forward_prior,
        "device": {"requested": device, "selected": cfg.selected_device},
        "config": asdict(cfg),
        "decision": sweep["decision"],
    }
    _json_dump(production_summary, sweep_dir / "production_run_summary.json")
    paper_summary = post_eval.write_paper_summary(sweep_dir=sweep_dir, output_root=paths.resolve_project_path(output_root))
    return {
        "ok": True,
        "mode": mode,
        "sweep_dir": str(sweep_dir),
        "paper_summary": str(paper_summary["json_path"]),
        "selected_k": sweep["decision"].get("selected_k"),
        "hyperopt_enabled": bool(cfg.hyperopt_enabled),
    }
