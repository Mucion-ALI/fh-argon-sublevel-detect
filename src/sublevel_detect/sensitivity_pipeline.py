from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from . import main_pipeline, model, paths


@dataclass(frozen=True)
class SensitivityProfile:
    prior_factors: List[float]
    prior_seeds: List[int]
    seed_jitter_seeds: List[int]
    bootstrap_replicates: int
    bootstrap_seeds: List[int]
    noise_replicates: int
    noise_seeds: List[int]
    window_radii: List[float]
    window_seeds: List[int]


def sensitivity_profile(mode: str) -> SensitivityProfile:
    if str(mode) == "smoke":
        return SensitivityProfile(
            prior_factors=[0.0, 1.0],
            prior_seeds=[0],
            seed_jitter_seeds=[0, 1],
            bootstrap_replicates=1,
            bootstrap_seeds=[0],
            noise_replicates=1,
            noise_seeds=[0],
            window_radii=[1.0, 1.5],
            window_seeds=[0],
        )
    return SensitivityProfile(
        prior_factors=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        prior_seeds=list(range(5)),
        seed_jitter_seeds=list(range(10)),
        bootstrap_replicates=30,
        bootstrap_seeds=[0, 1, 2],
        noise_replicates=30,
        noise_seeds=[0, 1, 2],
        window_radii=[1.0, 1.25, 1.5, 1.75, 2.0],
        window_seeds=list(range(5)),
    )


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
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config_from_dict(payload: Dict[str, Any]) -> model.Config:
    names = {field.name for field in fields(model.Config)}
    return model.Config(**{key: value for key, value in dict(payload).items() if key in names})


def _base_config(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    out_dir: Path,
    scan_seeds: Sequence[int],
    init_jitter_scale: float,
) -> model.Config:
    base_payload = _json_load(main_sweep_dir / "config_used.json", {})
    if not isinstance(base_payload, dict):
        base_payload = {}
    defaults = main_pipeline.profile_defaults(mode)
    cpu_workers = int(base_payload.get("cpu_workers", defaults.get("cpu_workers", 1)))
    selected_device = "cuda" if device == "cuda" and model.torch.cuda.is_available() else "cpu"
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
            "hyperopt_enabled": False,
            "scan_seeds": ",".join(str(int(seed)) for seed in scan_seeds),
            "init_jitter_scale": float(init_jitter_scale),
        }
    )
    if mode == "smoke":
        base_payload.update(main_pipeline.profile_defaults("smoke"))
        base_payload.update(
            {
                "data_path": str(paths.resolve_project_path(input_path)),
                "out_dir": str(out_dir),
                "device": device,
                "selected_device": selected_device,
                "dispatch_strategy": "single",
                "cpu_workers": 1,
                "retain_promoted_level_copies": False,
                "launch_monitor": False,
                "resume_mode": "off",
                "hyperopt_enabled": False,
                "scan_seeds": ",".join(str(int(seed)) for seed in scan_seeds),
                "init_jitter_scale": float(init_jitter_scale),
            }
        )
    return _config_from_dict(base_payload)


def _factor_label(value: float) -> str:
    text = f"{float(value):g}".replace("-", "minus_").replace(".", "p")
    return f"prior_{text}x"


def _production_prior_updates(main_sweep_dir: Path) -> Dict[str, Any]:
    config_payload = _json_load(main_sweep_dir / "config_used.json", {})
    if not isinstance(config_payload, dict):
        raise FileNotFoundError(f"Missing production config_used.json in {main_sweep_dir}")
    prior_payload = _json_load(main_sweep_dir / "forward_priors.json", {})
    updates: Dict[str, Any] = {}
    if isinstance(prior_payload, dict):
        updates.update(dict(prior_payload.get("config_updates", {}) or {}))
    for key in (
        "forward_prior_mode",
        "forward_main_spacing",
        "forward_spacing_std",
        "forward_confidence",
        "forward_anchor_step",
        "w_prior_anchor",
        "w_prior_gap",
    ):
        if key in config_payload:
            updates[key] = config_payload[key]
    missing = [key for key in ("w_prior_anchor", "w_prior_gap") if key not in updates]
    if missing:
        raise ValueError(f"Production prior weights missing from {main_sweep_dir / 'config_used.json'}: {missing}")
    return updates


def _production_forward_evidence(main_sweep_dir: Path) -> Dict[str, Any]:
    evidence = _json_load(main_sweep_dir / "forward_evidence.json", {})
    if isinstance(evidence, dict) and evidence:
        return evidence
    cfg_payload = _json_load(main_sweep_dir / "config_used.json", {})
    if not isinstance(cfg_payload, dict):
        raise FileNotFoundError(f"Missing production config_used.json in {main_sweep_dir}")
    cfg = _config_from_dict(cfg_payload)
    loaded = model.load_forward_evidence_for_config(cfg)
    return loaded if loaded is not None else model.build_forward_evidence(cfg)


def _apply_production_prior_strength(
    cfg: model.Config,
    main_sweep_dir: Path,
    strength_factor: float,
) -> tuple[model.Config, Dict[str, Any]]:
    production = _production_prior_updates(main_sweep_dir)
    factor = max(0.0, float(strength_factor))
    if factor <= 0.0:
        updates = dict(production)
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
        updates = dict(production)
        updates["forward_prior_mode"] = "auto"
        updates["w_prior_anchor"] = float(model.safe_float(production.get("w_prior_anchor"), 0.0) * factor)
        updates["w_prior_gap"] = float(model.safe_float(production.get("w_prior_gap"), 0.0) * factor)
    priors = {
        "method": "production_prior_strength_scaling",
        "soft_constraint_only": True,
        "strength_factor": factor,
        "source_config": str(main_sweep_dir / "config_used.json"),
        "config_updates": updates,
    }
    return model.apply_forward_priors_to_config(cfg, priors), priors


def _collect_scorecard_rows(scan_dir: Path, *, analysis: str, condition: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scorecard_path in sorted(scan_dir.glob("levels/L*/seeds/seed_*/full/scorecard.json")):
        scorecard = _json_load(scorecard_path, {})
        params = scorecard.get("params", {}) if isinstance(scorecard, dict) else {}
        metrics = scorecard.get("metrics_curve", {}) if isinstance(scorecard, dict) else {}
        level = int(scorecard.get("n_levels", 0))
        seed = int(scorecard.get("seed", 0))
        for channel in range(1, level + 1):
            rows.append(
                {
                    "analysis": analysis,
                    "condition": condition,
                    "n_levels": level,
                    "seed": seed,
                    "channel": channel,
                    "energy_v": model.safe_float(params.get(f"level_energy_{channel:02d}")),
                    "weight": model.safe_float(params.get(f"level_weight_{channel:02d}")),
                    "rmse_mean": model.safe_float(metrics.get("rmse_mean")),
                    "scorecard_path": str(scorecard_path),
                }
            )
    return rows


def _expected_levels(cfg: model.Config) -> List[int]:
    if int(cfg.fixed_levels) > 0:
        return [int(cfg.fixed_levels)]
    return [level for level in range(int(cfg.level_scan_min), int(cfg.level_scan_max) + 1) if 1 <= level <= 8]


def _scan_complete(scan_dir: Path, cfg: model.Config, seeds: Sequence[int]) -> bool:
    if not (scan_dir / "decision.json").exists():
        return False
    for level in _expected_levels(cfg):
        for seed in seeds:
            scorecard = scan_dir / "levels" / f"L{int(level):02d}" / "seeds" / f"seed_{int(seed):03d}" / "full" / "scorecard.json"
            if not scorecard.exists():
                return False
    return True


def _decision_selected_row(decision: Dict[str, Any]) -> Dict[str, Any]:
    selected_k = int(decision.get("selected_k", -1))
    for row in decision.get("scored_levels", []):
        if int(row.get("n_levels", -1)) == selected_k:
            return dict(row)
    return {}


def run_prior_strength_scan(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    profile: SensitivityProfile,
) -> Dict[str, Any]:
    root = paths.sensitivity_dir(output_root) / "prior_strength"
    evidence = _production_forward_evidence(main_sweep_dir)
    selection_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    decisions: Dict[str, Any] = {}
    for factor in profile.prior_factors:
        label = _factor_label(factor)
        out_dir = root / label / "fullscan"
        cfg = _base_config(
            mode=mode,
            input_path=input_path,
            output_root=output_root,
            device=device,
            main_sweep_dir=main_sweep_dir,
            out_dir=out_dir,
            scan_seeds=profile.prior_seeds,
            init_jitter_scale=0.05,
        )
        cfg, priors = _apply_production_prior_strength(cfg, main_sweep_dir, factor)
        cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
        model.safe_json_dump(evidence, out_dir / "forward_evidence.json")
        model.safe_json_dump(priors, out_dir / "forward_priors.json")
        if _scan_complete(out_dir, cfg, profile.prior_seeds):
            decision = _json_load(out_dir / "decision.json", {})
        else:
            result = model.run_level_scan(cfg)
            decision = result["decision"]
        selected = _decision_selected_row(decision)
        decisions[label] = decision
        selection_rows.append(
            {
                "prior_strength": float(factor),
                "condition": label,
                "selected_k": int(decision.get("selected_k", -1)),
                "selected_composite_rank_score": model.safe_float(selected.get("composite_rank_score")),
                "best_k_by_fit": decision.get("best_k_by_fit"),
                "best_k_by_d1_rmse": decision.get("best_k_by_d1_rmse"),
                "best_k_by_d2_rmse": decision.get("best_k_by_d2_rmse"),
                "best_k_by_structure": decision.get("best_k_by_structure"),
                "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
                "best_k_by_bic": decision.get("best_k_by_bic"),
                "best_k_by_aic": decision.get("best_k_by_aic"),
                "w_prior_anchor": float(cfg.w_prior_anchor),
                "w_prior_gap": float(cfg.w_prior_gap),
                "sweep_dir": str(out_dir),
            }
        )
        for row in _collect_scorecard_rows(out_dir, analysis="prior_strength", condition=label):
            row["prior_strength"] = float(factor)
            channel_rows.append(row)
    _write_csv(root / "prior_strength_selection.csv", selection_rows)
    _write_csv(root / "prior_strength_channel_drift.csv", channel_rows)
    _json_dump({"decisions": decisions, "selection": selection_rows}, root / "prior_strength_summary.json")
    return {"selection": selection_rows, "channels": channel_rows, "summary": str(root / "prior_strength_summary.json")}


def _read_input_frame(input_path: str | Path) -> tuple[pd.DataFrame, Dict[str, str]]:
    frame = model.read_excel_or_csv(paths.resolve_project_path(input_path))
    cols = {
        "curve": model.pick_col(frame, ("curve_id", "curveID", "curve", "id")),
        "vr": model.pick_col(frame, ("Vr", "vr", "V_r", "Ur")),
        "va": model.pick_col(frame, ("Va", "va", "V_a", "Ua")),
        "ip": model.pick_col(frame, ("IuA", "Ip", "ip", "I", "I_meas", "current", "Ip_uA", "I_uA")),
    }
    return frame.copy(), cols


def _baseline_residuals_by_curve(main_sweep_dir: Path) -> Dict[int, np.ndarray]:
    residuals: Dict[int, List[float]] = {}
    points_path = main_sweep_dir.parent / "k_selected_full" / "prediction_points.csv"
    if not points_path.exists():
        raise FileNotFoundError(
            f"Missing prediction_points.csv for residual bootstrap: {points_path}. "
            "Run the main pipeline first so selected-model residuals are exported."
        )
    pred = pd.read_csv(points_path)
    curve_col = "curve_id" if "curve_id" in pred.columns else "curve_idx"
    if curve_col not in pred.columns:
        raise ValueError(f"prediction_points.csv is missing curve_id/curve_idx: {points_path}")
    if "residual" in pred.columns:
        residual_col = "residual"
        for curve_id, sub in pred.groupby(curve_col):
            values = sub[residual_col].astype(float).to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            residuals[int(curve_id)] = values.tolist()
    else:
        obs_col = "observed" if "observed" in pred.columns else "Ip"
        fit_col = "predicted" if "predicted" in pred.columns else "fit"
        if obs_col not in pred.columns or fit_col not in pred.columns:
            raise ValueError(f"prediction_points.csv is missing observed/predicted residual columns: {points_path}")
        for curve_id, sub in pred.groupby(curve_col):
            values = (sub[obs_col].astype(float) - sub[fit_col].astype(float)).to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            residuals[int(curve_id)] = values.tolist()
    if not residuals or all(len(values) == 0 for values in residuals.values()):
        raise ValueError(f"prediction_points.csv contains no finite residuals: {points_path}")
    return {key: np.asarray(value, dtype=np.float64) for key, value in residuals.items()}


def _write_perturbed_input(
    *,
    source_frame: pd.DataFrame,
    cols: Dict[str, str],
    path: Path,
    rng: np.random.Generator,
    mode: str,
    residuals: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    frame = source_frame.copy()
    current = frame[cols["ip"]].astype(float).to_numpy(dtype=np.float64)
    curve_ids = frame[cols["curve"]].astype(int).to_numpy()
    if mode == "residual_bootstrap":
        noise = np.zeros_like(current)
        for curve_id in sorted(set(curve_ids.tolist())):
            mask = curve_ids == curve_id
            pool = residuals.get(int(curve_id))
            if pool is None or pool.size == 0:
                raise ValueError(f"Missing production residual pool for curve_id={int(curve_id)}")
            noise[mask] = rng.choice(pool, size=int(np.sum(mask)), replace=True)
        frame[cols["ip"]] = current + noise
    elif mode == "noise_perturbation":
        noise = np.zeros_like(current)
        for curve_id in sorted(set(curve_ids.tolist())):
            mask = curve_ids == curve_id
            values = current[mask]
            residual = values - model.moving_average_np(values, 7)
            sigma = float(np.std(residual))
            noise[mask] = rng.normal(0.0, sigma, size=int(np.sum(mask)))
        frame[cols["ip"]] = current + noise
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {"path": str(path), "sha256": digest, "row_count": int(len(frame))}


def _run_uncertainty_condition(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    analysis: str,
    condition: str,
    scan_seeds: Sequence[int],
    init_jitter_scale: float,
    peak_window_radius: float | None = None,
) -> Dict[str, Any]:
    out_dir = paths.sensitivity_dir(output_root) / "uncertainty" / analysis / condition / "fullscan"
    cfg = _base_config(
        mode=mode,
        input_path=input_path,
        output_root=output_root,
        device=device,
        main_sweep_dir=main_sweep_dir,
        out_dir=out_dir,
        scan_seeds=scan_seeds,
        init_jitter_scale=init_jitter_scale,
    )
    evidence = _production_forward_evidence(main_sweep_dir)
    cfg, priors = _apply_production_prior_strength(cfg, main_sweep_dir, 1.0)
    if peak_window_radius is not None:
        cfg.peak_window_radius = float(peak_window_radius)
    cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
    model.safe_json_dump(evidence, out_dir / "forward_evidence.json")
    model.safe_json_dump(priors, out_dir / "forward_priors.json")
    if _scan_complete(out_dir, cfg, scan_seeds):
        decision = _json_load(out_dir / "decision.json", {})
    else:
        result = model.run_level_scan(cfg)
        decision = result["decision"]
    return {"scan_dir": out_dir, "decision": decision}


def summarize_channel_uncertainty(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return {"channel_summary": [], "gap_summary": [], "selection_summary": []}
    if "n_levels" in frame.columns:
        selected = frame[frame["n_levels"].astype(int) == 4].copy()
    elif "selected_k" in frame.columns:
        selected = frame[frame["selected_k"].astype(int) == 4].copy()
    else:
        selected = frame.copy()
    channel_summary: List[Dict[str, Any]] = []
    for channel, sub in selected.groupby("channel"):
        energies = sub["energy_v"].astype(float).to_numpy(dtype=np.float64)
        weights = sub["weight"].astype(float).to_numpy(dtype=np.float64)
        energies = energies[np.isfinite(energies)]
        weights = weights[np.isfinite(weights)]
        if energies.size == 0:
            continue
        channel_summary.append(
            {
                "channel": int(channel),
                "sample_count": int(len(sub)),
                "energy_mean": float(np.mean(energies)),
                "energy_sd": float(np.std(energies, ddof=1)) if energies.size > 1 else 0.0,
                "energy_ci95_low": float(np.percentile(energies, 2.5)),
                "energy_ci95_high": float(np.percentile(energies, 97.5)),
                "weight_mean": float(np.mean(weights)) if weights.size else float("nan"),
                "weight_sd": float(np.std(weights, ddof=1)) if weights.size > 1 else 0.0,
                "weight_ci95_low": float(np.percentile(weights, 2.5)) if weights.size else float("nan"),
                "weight_ci95_high": float(np.percentile(weights, 97.5)) if weights.size else float("nan"),
            }
        )
    gap_values: List[float] = []
    group_cols = [col for col in ("analysis", "condition", "seed", "replicate") if col in selected.columns]
    if group_cols:
        for _, sub in selected.groupby(group_cols, dropna=False):
            by_channel = {int(row["channel"]): float(row["energy_v"]) for _, row in sub.iterrows()}
            if 1 in by_channel and 2 in by_channel:
                gap_values.append(by_channel[2] - by_channel[1])
    gaps = np.asarray(gap_values, dtype=np.float64)
    gap_summary = []
    if gaps.size:
        gap_summary.append(
            {
                "gap_name": "E2_minus_E1",
                "sample_count": int(gaps.size),
                "gap_mean_v": float(np.mean(gaps)),
                "gap_sd_v": float(np.std(gaps, ddof=1)) if gaps.size > 1 else 0.0,
                "gap_ci95_low_v": float(np.percentile(gaps, 2.5)),
                "gap_ci95_high_v": float(np.percentile(gaps, 97.5)),
            }
        )
    if "analysis" in frame.columns and "selected_k" in frame.columns:
        selection_cols = [col for col in ("analysis", "condition", "selected_k") if col in frame.columns]
        selection = (
            frame[selection_cols]
            .drop_duplicates()
            .groupby(["analysis", "selected_k"], dropna=False)
            .size()
            .reset_index(name="scenario_count")
        )
        total = selection.groupby("analysis")["scenario_count"].transform("sum")
        selection["scenario_fraction"] = selection["scenario_count"] / total
        selection_summary = selection.to_dict("records")
    else:
        selection_summary = []
    return {"channel_summary": channel_summary, "gap_summary": gap_summary, "selection_summary": selection_summary}


def _production_k4_anchors(main_sweep_dir: Path) -> List[Dict[str, Any]]:
    selected_scorecard = main_sweep_dir.parent / "k_selected_full" / "scorecard.json"
    scorecard = _json_load(selected_scorecard, {})
    if not isinstance(scorecard, dict) or int(scorecard.get("n_levels", 0) or 0) != 4:
        decision = _json_load(main_sweep_dir / "decision.json", {})
        if int((decision or {}).get("selected_k", -1)) != 4:
            return []
        candidates: List[tuple[float, Path]] = []
        for path in sorted((main_sweep_dir / "levels" / "L04" / "seeds").glob("seed_*/full/scorecard.json")):
            payload = _json_load(path, {})
            metrics = payload.get("metrics_curve", {}) if isinstance(payload, dict) else {}
            candidates.append((model.safe_float(metrics.get("rmse_mean"), float("inf")), path))
        if not candidates:
            return []
        _, best_path = min(candidates, key=lambda item: item[0])
        scorecard = _json_load(best_path, {})
    params = scorecard.get("params", {}) if isinstance(scorecard, dict) else {}
    anchors: List[Dict[str, Any]] = []
    for channel in range(1, 5):
        energy = model.safe_float(params.get(f"level_energy_{channel:02d}"), float("nan"))
        weight = model.safe_float(params.get(f"level_weight_{channel:02d}"), float("nan"))
        if np.isfinite(energy):
            anchors.append(
                {
                    "anchor_channel": int(channel),
                    "anchor_energy_v": float(energy),
                    "anchor_weight": float(weight),
                }
            )
    return anchors if len(anchors) == 4 else []


def _best_anchor_assignment(
    anchors: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    anchor_list = list(anchors)
    candidate_list = list(candidates)
    if len(anchor_list) != len(candidate_list):
        return []
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(len(candidate_list))):
        cost = 0.0
        for anchor_idx, candidate_idx in enumerate(perm):
            cost += abs(
                model.safe_float(candidate_list[candidate_idx].get("energy_v"), float("nan"))
                - model.safe_float(anchor_list[anchor_idx].get("anchor_energy_v"), float("nan"))
            )
        if np.isfinite(cost) and cost < best_cost:
            best_cost = float(cost)
            best_perm = tuple(perm)
    if best_perm is None:
        return []
    return [(anchor_list[idx], candidate_list[candidate_idx]) for idx, candidate_idx in enumerate(best_perm)]


def summarize_anchor_matched_k4_uncertainty(
    rows: Sequence[Dict[str, Any]],
    anchors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    frame = pd.DataFrame([dict(row) for row in rows])
    anchor_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(anchors, start=1):
        anchor_rows.append(
            {
                "anchor_channel": int(row.get("anchor_channel", row.get("channel", idx))),
                "anchor_energy_v": model.safe_float(row.get("anchor_energy_v", row.get("energy_v")), float("nan")),
                "anchor_weight": model.safe_float(row.get("anchor_weight", row.get("weight")), float("nan")),
            }
        )
    if frame.empty or len(anchor_rows) != 4:
        return {"samples": [], "channel_summary": []}
    selected = frame[frame["n_levels"].astype(int) == 4].copy() if "n_levels" in frame.columns else frame.copy()
    group_cols = [col for col in ("analysis", "condition", "seed", "replicate") if col in selected.columns]
    if not group_cols:
        selected["_anchor_group"] = 0
        group_cols = ["_anchor_group"]
    sample_rows: List[Dict[str, Any]] = []
    for _, sub in selected.groupby(group_cols, dropna=False):
        candidates: List[Dict[str, Any]] = []
        for _, row in sub.iterrows():
            energy = model.safe_float(row.get("energy_v"), float("nan"))
            if not np.isfinite(energy):
                continue
            candidates.append({key: row[key] for key in sub.columns if key != "_anchor_group"})
        if len(candidates) != len(anchor_rows):
            continue
        for anchor, candidate in _best_anchor_assignment(anchor_rows, candidates):
            energy = model.safe_float(candidate.get("energy_v"), float("nan"))
            sample_rows.append(
                {
                    **candidate,
                    "anchor_channel": int(anchor["anchor_channel"]),
                    "anchor_energy_v": float(anchor["anchor_energy_v"]),
                    "anchor_weight": model.safe_float(anchor.get("anchor_weight"), float("nan")),
                    "matched_channel": int(candidate.get("channel", -1)),
                    "energy_delta_v": float(energy - float(anchor["anchor_energy_v"])),
                    "matching_method": "minimum_total_absolute_energy_delta",
                }
            )
    if not sample_rows:
        return {"samples": [], "channel_summary": []}
    matched = pd.DataFrame(sample_rows)
    channel_summary: List[Dict[str, Any]] = []
    for anchor_channel, sub in matched.groupby("anchor_channel"):
        energies = sub["energy_v"].astype(float).to_numpy(dtype=np.float64)
        weights = sub["weight"].astype(float).to_numpy(dtype=np.float64) if "weight" in sub.columns else np.asarray([], dtype=np.float64)
        deltas = sub["energy_delta_v"].astype(float).to_numpy(dtype=np.float64)
        energies = energies[np.isfinite(energies)]
        weights = weights[np.isfinite(weights)]
        deltas = deltas[np.isfinite(deltas)]
        if energies.size == 0:
            continue
        channel_summary.append(
            {
                "anchor_channel": int(anchor_channel),
                "anchor_energy_v": float(sub["anchor_energy_v"].iloc[0]),
                "anchor_weight": model.safe_float(sub["anchor_weight"].iloc[0], float("nan")),
                "sample_count": int(len(sub)),
                "energy_mean": float(np.mean(energies)),
                "energy_sd": float(np.std(energies, ddof=1)) if energies.size > 1 else 0.0,
                "energy_ci95_low": float(np.percentile(energies, 2.5)),
                "energy_ci95_high": float(np.percentile(energies, 97.5)),
                "energy_delta_mean_v": float(np.mean(deltas)) if deltas.size else float("nan"),
                "energy_delta_sd_v": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "energy_delta_ci95_low_v": float(np.percentile(deltas, 2.5)) if deltas.size else float("nan"),
                "energy_delta_ci95_high_v": float(np.percentile(deltas, 97.5)) if deltas.size else float("nan"),
                "weight_mean": float(np.mean(weights)) if weights.size else float("nan"),
                "weight_sd": float(np.std(weights, ddof=1)) if weights.size > 1 else 0.0,
                "weight_ci95_low": float(np.percentile(weights, 2.5)) if weights.size else float("nan"),
                "weight_ci95_high": float(np.percentile(weights, 97.5)) if weights.size else float("nan"),
            }
        )
    return {"samples": sample_rows, "channel_summary": channel_summary}


def run_uncertainty_scan(
    *,
    mode: str,
    input_path: str | Path,
    output_root: str | Path,
    device: str,
    main_sweep_dir: Path,
    profile: SensitivityProfile,
) -> Dict[str, Any]:
    root = paths.sensitivity_dir(output_root) / "uncertainty"
    source_frame, cols = _read_input_frame(input_path)
    residuals = _baseline_residuals_by_curve(main_sweep_dir)
    sample_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    conditions: List[tuple[str, str, str | Path, Sequence[int], float, float | None, int | None]] = [
        ("seed_jitter", "original", input_path, profile.seed_jitter_seeds, 0.05, None, None)
    ]
    for radius in profile.window_radii:
        conditions.append(("window_radius", f"radius_{str(radius).replace('.', 'p')}", input_path, profile.window_seeds, 0.05, float(radius), None))
    perturbed_root = root / "perturbed_inputs"
    for idx in range(profile.bootstrap_replicates):
        input_file = perturbed_root / "residual_bootstrap" / f"replicate_{idx:03d}.csv"
        meta = _write_perturbed_input(
            source_frame=source_frame,
            cols=cols,
            path=input_file,
            rng=np.random.default_rng(idx),
            mode="residual_bootstrap",
            residuals=residuals,
        )
        manifest_rows.append({"analysis": "residual_bootstrap", "replicate": idx, **meta})
        conditions.append(("residual_bootstrap", f"replicate_{idx:03d}", input_file, profile.bootstrap_seeds, 0.05, None, idx))
    for idx in range(profile.noise_replicates):
        input_file = perturbed_root / "noise_perturbation" / f"replicate_{idx:03d}.csv"
        meta = _write_perturbed_input(
            source_frame=source_frame,
            cols=cols,
            path=input_file,
            rng=np.random.default_rng(10_000 + idx),
            mode="noise_perturbation",
            residuals=residuals,
        )
        manifest_rows.append({"analysis": "noise_perturbation", "replicate": idx, **meta})
        conditions.append(("noise_perturbation", f"replicate_{idx:03d}", input_file, profile.noise_seeds, 0.05, None, idx))

    for analysis, condition, condition_input, seeds, jitter, radius, replicate in conditions:
        result = _run_uncertainty_condition(
            mode=mode,
            input_path=condition_input,
            output_root=output_root,
            device=device,
            main_sweep_dir=main_sweep_dir,
            analysis=analysis,
            condition=condition,
            scan_seeds=seeds,
            init_jitter_scale=jitter,
            peak_window_radius=radius,
        )
        selected_k = int(result["decision"].get("selected_k", -1))
        decisions.append({"analysis": analysis, "condition": condition, "replicate": replicate, "selected_k": selected_k, "sweep_dir": str(result["scan_dir"])})
        for row in _collect_scorecard_rows(Path(result["scan_dir"]), analysis=analysis, condition=condition):
            row["selected_k"] = selected_k
            row["replicate"] = replicate
            row["peak_window_radius"] = radius
            sample_rows.append(row)

    conditional_summary = summarize_channel_uncertainty(sample_rows)
    production_anchors = _production_k4_anchors(main_sweep_dir)
    anchor_summary = summarize_anchor_matched_k4_uncertainty(sample_rows, production_anchors)
    _write_csv(root / "channel_uncertainty_samples.csv", sample_rows)
    _write_csv(root / "channel_uncertainty_conditional_k4.csv", conditional_summary["channel_summary"] + conditional_summary["gap_summary"])
    _write_csv(root / "channel_uncertainty_summary.csv", conditional_summary["channel_summary"] + conditional_summary["gap_summary"])
    _write_csv(root / "channel_uncertainty_anchor_matched_samples.csv", anchor_summary["samples"])
    _write_csv(root / "channel_uncertainty_anchor_matched.csv", anchor_summary["channel_summary"])
    _write_csv(root / "uncertainty_selection_summary.csv", conditional_summary["selection_summary"])
    _write_csv(root / "perturbed_input_manifest.csv", manifest_rows)
    _json_dump(
        {
            "decisions": decisions,
            "conditional_k4_all_fits": conditional_summary,
            "production_k4_anchors": production_anchors,
            "production_anchor_matched_k4": anchor_summary,
            "perturbed_inputs": manifest_rows,
        },
        root / "channel_uncertainty_summary.json",
    )
    return {
        "samples": sample_rows,
        "conditional_summary": conditional_summary,
        "anchor_summary": anchor_summary,
        "decisions": decisions,
    }


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
    profile = sensitivity_profile(mode)
    prior = run_prior_strength_scan(
        mode=mode,
        input_path=input_path,
        output_root=output,
        device=device,
        main_sweep_dir=main_sweep,
        profile=profile,
    )
    uncertainty = run_uncertainty_scan(
        mode=mode,
        input_path=input_path,
        output_root=output,
        device=device,
        main_sweep_dir=main_sweep,
        profile=profile,
    )
    root = paths.sensitivity_dir(output)
    summary = {
        "ok": True,
        "mode": mode,
        "prior_strength_rows": len(prior.get("selection", [])),
        "uncertainty_sample_rows": len(uncertainty.get("samples", [])),
        "prior_strength_selection": str(root / "prior_strength" / "prior_strength_selection.csv"),
        "channel_uncertainty_conditional_k4": str(root / "uncertainty" / "channel_uncertainty_conditional_k4.csv"),
        "channel_uncertainty_anchor_matched": str(root / "uncertainty" / "channel_uncertainty_anchor_matched.csv"),
        "channel_uncertainty_summary": str(root / "uncertainty" / "channel_uncertainty_summary.csv"),
        "uncertainty_selection_summary": str(root / "uncertainty" / "uncertainty_selection_summary.csv"),
    }
    _json_dump(summary, root / "sensitivity_summary.json")
    return summary
