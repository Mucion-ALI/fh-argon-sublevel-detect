from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DATA_PATH = "data/argon/FHdata.xlsx"
DEFAULT_OUT_DIR = "outputs/main/fullscan"
MODEL_SCHEMA = "sublevel_detect_formal"
SUPPORTED_SPACING_STATUSES = {
    "stable_main_period_supported",
    "weak_drift_supported",
    "structured_perturbation_supported",
}


__all__ = [
    "Config",
    "EarlyStopState",
    "EarlyStopper",
    "HybridOptimizer",
    "PoissonRateFHCoreMultiLevel",
    "build_optimizer",
    "compute_structure_metrics",
    "compute_vr_physical_response_metrics",
    "compute_energy_cluster_diagnostics",
    "compute_channel_degeneracy_diagnostics",
    "compute_weight_diagnostics",
    "eval_metrics",
    "load_curves_and_init",
    "model_predict",
    "run_level_fit",
    "run_level_scan",
    "select_adaptive_level_decision",
    "select_k_neutral_level_decision",
    "train_multilevel",
]


@dataclass
class Config:
    data_path: str = DEFAULT_DATA_PATH
    out_dir: str = DEFAULT_OUT_DIR
    device: str = "auto"
    seed: int = 0
    epochs: int = 3500
    scan_short_epochs: int = 60
    cv_epochs: int = 60
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    optimizer: str = "muon_hybrid"
    level_scan_min: int = 1
    level_scan_max: int = 8
    fixed_levels: int = 0
    scan_seeds: str = "0,1,2"
    full_train_all_levels: bool = True
    retain_promoted_level_copies: bool = True
    n_max: int = 16
    min_level_gap: float = 0.04
    level_weight_entropy: float = 1.0e-3
    w_raw: float = 1.0
    w_d1: float = 0.04
    w_d2: float = 0.02
    w_smooth: float = 0.01
    w_peak_window: float = 0.08
    w_vr_late_bias: float = 0.04
    w_vr_amplitude_ratio: float = 0.04
    w_high_vr_valley_depth: float = 0.06
    w_reg: float = 1.0e-4
    w_prior_gap: float = 2.0e-4
    w_prior_anchor: float = 2.0e-4
    w_extra_level_sparsity: float = 0.0
    early_stop_min_epochs: int = 300
    early_stop_warmup: int = 300
    early_stop_patience: int = 50
    early_stop_min_delta_rel: float = 2.0e-4
    early_stop_smoothing: int = 7
    checkpoint_min_interval: int = 10
    low_weight_quantile: float = 0.25
    low_weight_floor: float = 0.015
    concentration_quantile: float = 0.90
    concentration_floor: float = 0.78
    reference_k: int = 4
    hyperopt_trials: int = 14
    profile: str = "production"
    cpu_workers: int = 4
    cuda_workers: int = 1
    selected_device: str = "auto"
    dispatch_strategy: str = "cpu_4"
    resume_mode: str = "auto"
    launch_monitor: bool = True
    monitor_interval_seconds: int = 15
    hyperopt_enabled: bool = True
    hyperopt_short_epochs: int = 80
    hyperopt_top_k: int = 1
    hyperopt_levels: str = "1,2,4,6,8"
    hyperopt_stage1_epochs: int = 80
    hyperopt_stage2_epochs: int = 240
    hyperopt_stage3_epochs: int = 300
    hyperopt_stage1_top_k: int = 4
    hyperopt_stage2_top_k: int = 2
    hyperopt_stage1_levels: str = "1,2,4,6,8"
    hyperopt_stage2_levels: str = "2,4,6,8"
    hyperopt_stage3_levels: str = "1,2,3,4,5,6,7,8"
    hyperopt_stage1_seeds: str = "0"
    hyperopt_stage2_seeds: str = "0,1"
    hyperopt_stage3_seeds: str = "0,1"
    run_stage: str = "full_sweep"
    hyperopt_stage: str = ""
    peak_window_radius: float = 1.5
    flatline_min_osc_ratio: float = 0.25
    vr_late_va_min: float = 60.0
    vr_late_bias_abs_threshold: float = 0.10
    high_vr_threshold: float = 8.0
    high_vr_valley_lift_threshold: float = 0.12
    kernel_mode: str = "k_neutral"
    cluster_tolerance_eV: float = 0.02
    forward_prior_mode: str = "off"
    forward_evidence_path: str = ""
    forward_main_spacing: float = 0.0
    forward_spacing_std: float = 0.0
    forward_confidence: float = 0.0
    forward_anchor_step: float = 0.25


@dataclass
class EarlyStopState:
    epoch: int
    value: float
    smoothed_value: float
    best_value: float
    best_epoch: int
    wait_count: int
    should_stop: bool


class EarlyStopper:
    def __init__(
        self,
        warmup_epochs: int = 30,
        min_epochs: int = 0,
        patience: int = 50,
        min_delta_rel: float = 2.0e-4,
        smoothing: int = 7,
    ) -> None:
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_epochs = max(0, int(min_epochs))
        self.patience = max(0, int(patience))
        self.min_delta_rel = max(0.0, float(min_delta_rel))
        self.smoothing = max(1, int(smoothing))
        self.values: List[float] = []
        self.best_value = float("inf")
        self.best_epoch = 0
        self.wait_count = 0

    def update(self, epoch: int, value: float) -> EarlyStopState:
        epoch = int(epoch)
        value = float(value)
        self.values.append(value)
        window = self.values[-self.smoothing :]
        smoothed = float(np.mean(window))
        threshold = abs(self.best_value) * self.min_delta_rel if np.isfinite(self.best_value) else 0.0
        improved = smoothed < self.best_value - threshold
        if improved:
            self.best_value = smoothed
            self.best_epoch = epoch
            self.wait_count = 0
        elif epoch > self.warmup_epochs:
            self.wait_count += 1
        stop_allowed = epoch > max(self.warmup_epochs, self.min_epochs)
        should_stop = bool(stop_allowed and self.wait_count > self.patience)
        return EarlyStopState(
            epoch=epoch,
            value=value,
            smoothed_value=smoothed,
            best_value=float(self.best_value),
            best_epoch=int(self.best_epoch),
            wait_count=int(self.wait_count),
            should_stop=should_stop,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_epochs": int(self.warmup_epochs),
            "min_epochs": int(self.min_epochs),
            "patience": int(self.patience),
            "min_delta_rel": float(self.min_delta_rel),
            "smoothing": int(self.smoothing),
            "values": [float(x) for x in self.values],
            "best_value": float(self.best_value),
            "best_epoch": int(self.best_epoch),
            "wait_count": int(self.wait_count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.min_epochs = int(state.get("min_epochs", self.min_epochs))
        self.values = [float(x) for x in state.get("values", [])]
        self.best_value = safe_float(state.get("best_value"), float("inf"))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.wait_count = int(state.get("wait_count", 0))


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str | Sequence[int]) -> List[int]:
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values or [0]


def resolve_device(name: str) -> torch.device:
    value = str(name).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out


def safe_nanmean(values: Sequence[float], default: float = float("nan")) -> float:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_ready(value.item())
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def safe_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(node.text or "" for node in item.findall(".//a:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(".//a:sheets/a:sheet", ns)
        sheet_name = first_sheet.attrib.get("name", "sheet1") if first_sheet is not None else "sheet1"
        sheet_path = "xl/worksheets/sheet1.xml"
        sheet = ET.fromstring(archive.read(sheet_path))
        rows: List[List[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values: List[str] = []
            for cell in row.findall("a:c", ns):
                raw = cell.find("a:v", ns)
                value = "" if raw is None else str(raw.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)
    if not rows:
        raise ValueError(f"Workbook has no rows: {path}")
    headers = [str(x) for x in rows[0]]
    data = rows[1:]
    frame = pd.DataFrame(data, columns=headers)
    frame.attrs["sheet_name"] = sheet_name
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if not converted.isna().all():
            frame[column] = converted
    return frame


def read_excel_or_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if suffix == ".xlsx":
                return _read_xlsx_without_openpyxl(path)
            raise
    raise ValueError(f"Unsupported data format: {path}")


def pick_col(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return str(candidate)
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(frame.columns)}")


def moving_average_np(values: np.ndarray, win: int = 7) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    win = int(win)
    if win <= 1 or arr.size < 3:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(padded, kernel, mode="valid")


def estimate_peak_spacing(va: np.ndarray, current: np.ndarray) -> Optional[float]:
    va = np.asarray(va, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    mask = (va >= 30.0) & (va <= 80.0)
    if int(np.sum(mask)) < 8:
        return None
    x = va[mask]
    y = moving_average_np(current[mask], 9)
    if y.size < 8:
        return None
    scale = float(np.max(y) - np.min(y))
    if scale <= 1e-12:
        return None
    peaks: List[int] = []
    for idx in range(1, y.size - 1):
        if y[idx] > y[idx - 1] and y[idx] >= y[idx + 1]:
            local = y[max(0, idx - 8) : min(y.size, idx + 9)]
            if y[idx] - float(np.min(local)) >= 0.04 * scale:
                peaks.append(idx)
    chosen: List[int] = []
    for idx in sorted(peaks, key=lambda i: y[i], reverse=True):
        if all(abs(idx - old) >= 6 for old in chosen):
            chosen.append(idx)
    chosen = sorted(chosen)
    if len(chosen) < 2:
        return None
    return float(np.mean(np.diff(x[np.asarray(chosen, dtype=int)])))


def _turning_points(va: np.ndarray, current: np.ndarray, kind: str) -> Dict[str, Any]:
    va = np.asarray(va, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    mask = (va >= 30.0) & (va <= 80.0)
    if int(np.sum(mask)) < 8:
        return {"positions": [], "values": [], "scale": 0.0, "noise": float("nan"), "relative_amplitude": 0.0}
    x = va[mask]
    raw = current[mask]
    y = moving_average_np(raw, 9)
    scale = float(np.max(y) - np.min(y)) if y.size else 0.0
    baseline = max(abs(float(np.median(y))) if y.size else 0.0, 1e-6)
    residual = raw - moving_average_np(raw, 7)
    noise = float(np.std(residual)) if residual.size else float("nan")
    if y.size < 8 or scale <= 1e-12:
        return {"positions": [], "values": [], "scale": scale, "noise": noise, "relative_amplitude": 0.0}
    points: List[int] = []
    for idx in range(1, y.size - 1):
        local = y[max(0, idx - 8) : min(y.size, idx + 9)]
        if str(kind) == "valley":
            is_turn = y[idx] < y[idx - 1] and y[idx] <= y[idx + 1]
            prominence = float(np.max(local) - y[idx])
        else:
            is_turn = y[idx] > y[idx - 1] and y[idx] >= y[idx + 1]
            prominence = float(y[idx] - np.min(local))
        if is_turn and prominence >= 0.04 * scale:
            points.append(idx)
    chosen: List[int] = []
    for idx in sorted(points, key=lambda i: abs(float(y[i] - np.median(y))), reverse=True):
        if all(abs(idx - old) >= 5 for old in chosen):
            chosen.append(idx)
    chosen = sorted(chosen)
    return {
        "positions": [float(x[idx]) for idx in chosen],
        "values": [float(y[idx]) for idx in chosen],
        "scale": scale,
        "noise": noise,
        "relative_amplitude": float(scale / baseline),
    }


def compute_forward_evidence(curves: Sequence[Dict[str, Any]], reference_k: int = 4) -> Dict[str, Any]:
    curve_rows: List[Dict[str, Any]] = []
    all_intervals: List[float] = []
    amplitude_scores: List[float] = []
    envelope_slopes: List[float] = []
    for curve in curves:
        va = np.asarray(curve["Va"], dtype=np.float64)
        ip = np.asarray(curve["Ip"], dtype=np.float64)
        peaks = _turning_points(va, ip, "peak")
        valleys = _turning_points(va, ip, "valley")
        intervals: List[float] = []
        for positions in (peaks["positions"], valleys["positions"]):
            if len(positions) >= 2:
                intervals.extend(float(x) for x in np.diff(np.asarray(positions, dtype=np.float64)))
        intervals = [x for x in intervals if 4.0 <= x <= 20.0 and np.isfinite(x)]
        all_intervals.extend(intervals)
        relative_amp = max(float(peaks["relative_amplitude"]), float(valleys["relative_amplitude"]))
        amplitude_scores.append(relative_amp)
        if len(peaks["positions"]) >= 2:
            peak_x = np.asarray(peaks["positions"], dtype=np.float64)
            peak_y = np.asarray(peaks["values"], dtype=np.float64)
            slope = float(np.polyfit(peak_x, peak_y, 1)[0]) if peak_x.size >= 2 else float("nan")
            if np.isfinite(slope):
                envelope_slopes.append(slope)
        curve_rows.append(
            {
                "curve_id": int(curve.get("curve_id", len(curve_rows))),
                "n_peaks": int(len(peaks["positions"])),
                "n_valleys": int(len(valleys["positions"])),
                "peak_positions": peaks["positions"],
                "valley_positions": valleys["positions"],
                "spacing_intervals": intervals,
                "relative_amplitude": relative_amp,
                "noise_estimate": float(peaks["noise"]) if np.isfinite(float(peaks["noise"])) else None,
            }
        )

    interval_arr = np.asarray(all_intervals, dtype=np.float64)
    main_spacing = float(np.median(interval_arr)) if interval_arr.size else float("nan")
    spacing_std = float(np.std(interval_arr)) if interval_arr.size else float("nan")
    spacing_iqr = (
        float(np.percentile(interval_arr, 75) - np.percentile(interval_arr, 25))
        if interval_arr.size
        else float("nan")
    )
    amp = float(np.median(np.asarray(amplitude_scores, dtype=np.float64))) if amplitude_scores else 0.0
    amp_strength = float(np.clip((amp - 0.01) / 0.08, 0.0, 1.0))
    if np.isfinite(main_spacing) and main_spacing > 0 and np.isfinite(spacing_std):
        stability = float(np.clip(1.0 - spacing_std / max(0.25 * main_spacing, 1e-6), 0.0, 1.0))
    else:
        stability = 0.0
    count_score = float(np.clip(interval_arr.size / max(2.0 * max(int(reference_k), 1), 1.0), 0.0, 1.0))
    confidence = float(amp_strength * (0.45 + 0.35 * stability + 0.20 * count_score))
    if confidence >= 0.55:
        status = "forward_spacing_supported"
    elif confidence >= 0.35:
        status = "forward_spacing_weak"
    else:
        status = "low_forward_confidence"
    anchor_start = main_spacing if np.isfinite(main_spacing) else 11.55
    anchors = [float(anchor_start + 0.25 * idx) for idx in range(8)]
    k_support = [
        {
            "k": int(k),
            "support_score": float(confidence * math.exp(-abs(int(k) - int(reference_k)) / 3.0)),
        }
        for k in range(1, 9)
    ]
    return {
        "method": "forward_evidence_soft_prior",
        "soft_constraint_only": True,
        "reference_k": int(reference_k),
        "curves": curve_rows,
        "global": {
            "main_spacing": main_spacing,
            "spacing_std": spacing_std,
            "spacing_iqr": spacing_iqr,
            "n_spacing_intervals": int(interval_arr.size),
            "relative_amplitude_median": amp,
            "confidence": confidence,
            "diagnostic_status": status,
            "candidate_level_anchors": anchors,
            "candidate_k_support": k_support,
            "envelope_slope": float(np.median(envelope_slopes)) if envelope_slopes else None,
        },
    }


def build_forward_evidence(cfg: Config) -> Dict[str, Any]:
    curves, _ = load_curves_and_init(cfg)
    return compute_forward_evidence(curves, reference_k=int(cfg.reference_k))


def compile_forward_priors(evidence: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    global_ev = evidence.get("global", {}) if isinstance(evidence, dict) else {}
    main_spacing = safe_float(global_ev.get("main_spacing"), 11.55)
    if not np.isfinite(main_spacing) or main_spacing <= 0:
        main_spacing = 11.55
    main_spacing = float(np.clip(main_spacing, 9.0, 14.5))
    spacing_std = safe_float(global_ev.get("spacing_std"), 0.0)
    if not np.isfinite(spacing_std) or spacing_std < 0:
        spacing_std = 0.0
    confidence = float(np.clip(safe_float(global_ev.get("confidence"), 0.0), 0.0, 1.0))
    anchor_step = 0.25
    updates = {
        "forward_prior_mode": "auto",
        "forward_main_spacing": main_spacing,
        "forward_spacing_std": float(spacing_std),
        "forward_confidence": confidence,
        "forward_anchor_step": anchor_step,
        "w_prior_anchor": float(max(cfg.w_prior_anchor, cfg.w_prior_anchor * (1.0 + 2.0 * confidence))),
        "w_prior_gap": float(max(cfg.w_prior_gap, cfg.w_prior_gap * (1.0 + confidence))),
        "early_stop_min_delta_rel": float(max(cfg.early_stop_min_delta_rel, 1.0e-4 * (1.0 + confidence))),
    }
    return {
        "method": "forward_prior_compiler",
        "soft_constraint_only": True,
        "candidate_k": list(range(1, 9)),
        "hard_gates": [],
        "config_updates": updates,
        "basis": {
            "main_spacing": main_spacing,
            "spacing_std": float(spacing_std),
            "confidence": confidence,
            "diagnostic_status": str(global_ev.get("diagnostic_status", "missing")),
        },
    }


def apply_forward_priors_to_config(cfg: Config, priors: Dict[str, Any]) -> Config:
    values = asdict(cfg)
    for key, value in dict(priors.get("config_updates", {})).items():
        if key in values:
            values[key] = value
    return Config(**values)


def load_forward_evidence_for_config(cfg: Config) -> Optional[Dict[str, Any]]:
    path_text = str(getattr(cfg, "forward_evidence_path", "") or "").strip()
    if path_text:
        path = Path(path_text)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    main_spacing = safe_float(getattr(cfg, "forward_main_spacing", 0.0), 0.0)
    confidence = safe_float(getattr(cfg, "forward_confidence", 0.0), 0.0)
    if str(getattr(cfg, "forward_prior_mode", "off")) != "off" and main_spacing > 0:
        return {
            "method": "config_forward_prior",
            "soft_constraint_only": True,
            "reference_k": int(cfg.reference_k),
            "global": {
                "main_spacing": main_spacing,
                "spacing_std": safe_float(getattr(cfg, "forward_spacing_std", 0.0), 0.0),
                "confidence": confidence,
                "diagnostic_status": "config_forward_prior",
                "candidate_level_anchors": [float(main_spacing + float(cfg.forward_anchor_step) * idx) for idx in range(8)],
                "envelope_slope": None,
            },
        }
    return None


def compute_forward_reverse_consistency(row: Dict[str, Any], forward_evidence: Dict[str, Any]) -> Dict[str, Any]:
    global_ev = forward_evidence.get("global", {}) if isinstance(forward_evidence, dict) else {}
    forward_spacing = safe_float(global_ev.get("main_spacing"), float("nan"))
    forward_std = safe_float(global_ev.get("spacing_std"), 0.0)
    confidence = float(np.clip(safe_float(global_ev.get("confidence"), 0.0), 0.0, 1.0))
    learned_spacing = safe_float(row.get("V_exc_weighted", row.get("V_exc")), float("nan"))
    abs_delta = abs(learned_spacing - forward_spacing) if np.isfinite(learned_spacing) and np.isfinite(forward_spacing) else float("nan")
    rel_delta = abs_delta / max(abs(forward_spacing), 1e-6) if np.isfinite(abs_delta) else float("nan")
    anchors = [safe_float(x) for x in global_ev.get("candidate_level_anchors", [])]
    energies = [
        safe_float(row.get(f"level_energy_{idx:02d}"))
        for idx in range(1, int(row.get("n_levels", 0)) + 1)
        if f"level_energy_{idx:02d}" in row
    ]
    pair_count = min(len(anchors), len(energies))
    if pair_count:
        level_delta = float(np.mean(np.abs(np.asarray(energies[:pair_count]) - np.asarray(anchors[:pair_count]))))
    else:
        level_delta = float("nan")
    envelope_slope = global_ev.get("envelope_slope")
    damping = safe_float(row.get("damping"), float("nan"))
    if envelope_slope is None or not np.isfinite(damping):
        envelope_status = "not_evaluated"
    elif damping >= 0.0:
        envelope_status = "compatible"
    else:
        envelope_status = "tension"
    tolerance = max(0.20, 2.0 * max(float(forward_std), 0.0) / max(abs(forward_spacing), 1e-6)) if np.isfinite(forward_spacing) else 0.20
    if confidence < 0.35:
        status = "low_forward_confidence"
    elif np.isfinite(rel_delta) and rel_delta <= tolerance:
        status = "consistent"
    else:
        status = "tension"
    return {
        "n_levels": int(row.get("n_levels", 0)),
        "forward_main_spacing": forward_spacing,
        "learned_effective_spacing": learned_spacing,
        "spacing_abs_delta": abs_delta,
        "spacing_rel_delta": rel_delta,
        "level_anchor_mean_abs_delta": level_delta,
        "forward_confidence": confidence,
        "spacing_tolerance_rel": float(tolerance),
        "envelope_consistency_status": envelope_status,
        "status": status,
        "soft_constraint_only": True,
    }


def load_curves_and_init(cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        alt = Path(__file__).resolve().parents[2] / "data" / "argon" / "FHdata.xlsx"
        if data_path.name == "FHdata.xlsx" and alt.exists():
            data_path = alt
        else:
            raise FileNotFoundError(f"Dataset not found: {cfg.data_path}")
    frame = read_excel_or_csv(data_path)
    c_curve = pick_col(frame, ("curve_id", "curveID", "curve", "id"))
    c_vr = pick_col(frame, ("Vr", "vr", "V_r", "Ur"))
    c_va = pick_col(frame, ("Va", "va", "V_a", "Ua"))
    c_ip = pick_col(frame, ("IuA", "Ip", "ip", "I", "I_meas", "current", "Ip_uA", "I_uA"))

    curves: List[Dict[str, Any]] = []
    spacings: List[float] = []
    for curve_idx, curve_id in enumerate(sorted(frame[c_curve].unique().tolist())):
        sub = frame[frame[c_curve] == curve_id].copy().sort_values(c_va)
        va = sub[c_va].to_numpy(dtype=np.float32)
        vr_values = sub[c_vr].to_numpy(dtype=np.float32)
        current = sub[c_ip].to_numpy(dtype=np.float32)
        vr0 = float(vr_values[0]) if vr_values.size else 0.0
        vr_is_vector = bool(np.max(np.abs(vr_values - vr0)) > 1e-6) if vr_values.size else False
        spacing = estimate_peak_spacing(va, current)
        if spacing is not None and np.isfinite(spacing):
            spacings.append(float(spacing))
        curves.append(
            {
                "curve_id": int(curve_id),
                "curve_idx": int(curve_idx),
                "Va": va,
                "Vr": vr_values if vr_is_vector else vr0,
                "Vr_is_vector": vr_is_vector,
                "Ip": current,
            }
        )
    forward_main = safe_float(getattr(cfg, "forward_main_spacing", 0.0), 0.0)
    if str(getattr(cfg, "forward_prior_mode", "off")) != "off" and 5.0 <= forward_main <= 20.0:
        v_exc = float(forward_main)
        init_source = "forward_soft_prior"
    else:
        v_exc = float(np.mean(spacings)) if spacings else 11.5
        init_source = "raw_peak_spacing"
    return curves, {
        "V_exc_init": v_exc,
        "V_exc_std": float(np.std(spacings)) if spacings else 0.0,
        "V_exc_init_source": init_source,
        "forward_confidence": safe_float(getattr(cfg, "forward_confidence", 0.0), 0.0),
        "n_curves": float(len(curves)),
        "n_rows": float(len(frame)),
    }


def inv_bounded(value: float, lo: float, hi: float) -> float:
    value = float(np.clip(value, lo + 1e-6, hi - 1e-6))
    ratio = (value - lo) / (hi - lo)
    ratio = float(np.clip(ratio, 1e-6, 1.0 - 1e-6))
    return math.log(ratio / (1.0 - ratio))


class PoissonRateFHCoreMultiLevel(nn.Module):
    def __init__(
        self,
        n_curves: int,
        n_max: int = 16,
        n_levels: int = 4,
        min_level_gap: float = 0.04,
        device: Optional[torch.device] = None,
        V_exc_init: float = 11.5,
    ) -> None:
        super().__init__()
        self.n_curves = int(n_curves)
        self.n_max = int(n_max)
        self.n_levels = int(max(1, min(8, n_levels)))
        self.min_level_gap = float(min_level_gap)
        self.device_hint = str(device or "cpu")

        self.raw_E1 = nn.Parameter(torch.tensor(inv_bounded(float(V_exc_init), 9.0, 14.5), dtype=torch.float32))
        if self.n_levels > 1:
            init_gaps = torch.full((self.n_levels - 1,), -2.5, dtype=torch.float32)
            self.raw_dE = nn.Parameter(init_gaps)
        else:
            self.raw_dE = nn.Parameter(torch.empty(0, dtype=torch.float32))
        init_logits = torch.zeros(self.n_levels, dtype=torch.float32)
        self.raw_level_logits = nn.Parameter(init_logits)

        self.raw_amp = nn.Parameter(torch.tensor(inv_bounded(1.2, 0.05, 5.0), dtype=torch.float32))
        self.raw_offset = nn.Parameter(torch.tensor(inv_bounded(0.02, -0.5, 0.8), dtype=torch.float32))
        self.raw_v_emit = nn.Parameter(torch.tensor(inv_bounded(4.0, 0.0, 18.0), dtype=torch.float32))
        self.raw_power = nn.Parameter(torch.tensor(inv_bounded(1.25, 0.7, 2.4), dtype=torch.float32))
        self.raw_osc_amp = nn.Parameter(torch.tensor(inv_bounded(0.28, 0.0, 0.90), dtype=torch.float32))
        self.raw_width = nn.Parameter(torch.tensor(inv_bounded(1.25, 0.25, 5.0), dtype=torch.float32))
        self.raw_damping = nn.Parameter(torch.tensor(inv_bounded(0.010, 0.0, 0.060), dtype=torch.float32))
        self.raw_phase = nn.Parameter(torch.tensor(inv_bounded(0.0, -6.0, 6.0), dtype=torch.float32))
        self.raw_vr_scale = nn.Parameter(torch.tensor(inv_bounded(1.0, 0.80, 1.20), dtype=torch.float32))
        self.raw_slope = nn.Parameter(torch.tensor(inv_bounded(0.0, -0.02, 0.04), dtype=torch.float32))
        self.raw_collector_threshold = nn.Parameter(torch.tensor(inv_bounded(2.5, -4.0, 10.0), dtype=torch.float32))
        self.raw_collector_width = nn.Parameter(torch.tensor(inv_bounded(5.0, 0.8, 14.0), dtype=torch.float32))
        self.raw_collector_floor = nn.Parameter(torch.tensor(inv_bounded(0.72, 0.20, 0.98), dtype=torch.float32))
        self.raw_vr_contrast = nn.Parameter(torch.tensor(inv_bounded(0.0, -0.25, 0.35), dtype=torch.float32))
        self.raw_late_contrast = nn.Parameter(torch.tensor(inv_bounded(0.0, -0.35, 0.25), dtype=torch.float32))
        self.raw_vr_late_contrast = nn.Parameter(torch.tensor(inv_bounded(0.08, -0.20, 0.45), dtype=torch.float32))
        self.raw_late_onset = nn.Parameter(torch.tensor(inv_bounded(58.0, 45.0, 75.0), dtype=torch.float32))
        self.raw_late_width = nn.Parameter(torch.tensor(inv_bounded(7.0, 2.0, 18.0), dtype=torch.float32))
        self.raw_high_energy_loss_strength = nn.Parameter(torch.tensor(inv_bounded(0.04, 0.0, 0.45), dtype=torch.float32))
        self.raw_high_energy_loss_onset = nn.Parameter(torch.tensor(inv_bounded(68.0, 52.0, 88.0), dtype=torch.float32))
        self.raw_high_energy_loss_width = nn.Parameter(torch.tensor(inv_bounded(7.0, 2.0, 20.0), dtype=torch.float32))
        self.raw_vr_late_baseline = nn.Parameter(torch.tensor(inv_bounded(0.0, -0.050, 0.030), dtype=torch.float32))

        self.raw_curve_gain = nn.Parameter(torch.zeros(self.n_curves, dtype=torch.float32))
        self.raw_curve_bias = nn.Parameter(torch.zeros(self.n_curves, dtype=torch.float32))
        self.raw_curve_dva = nn.Parameter(torch.zeros(self.n_curves, dtype=torch.float32))

    @staticmethod
    def bounded(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    def level_params(self) -> Dict[str, torch.Tensor]:
        e1 = self.bounded(self.raw_E1, 9.0, 14.5)
        if self.n_levels > 1:
            gaps = self.min_level_gap + F.softplus(self.raw_dE)
            energies = torch.cat([e1.view(1), e1 + torch.cumsum(gaps, dim=0)])
        else:
            gaps = torch.empty(0, device=e1.device, dtype=e1.dtype)
            energies = e1.view(1)
        weights = F.softmax(self.raw_level_logits, dim=0)
        return {"energies": energies, "weights": weights, "gaps": gaps}

    def phys_params(self) -> Dict[str, torch.Tensor]:
        return {
            "amp": self.bounded(self.raw_amp, 0.05, 5.0),
            "offset": self.bounded(self.raw_offset, -0.5, 0.8),
            "v_emit": self.bounded(self.raw_v_emit, 0.0, 18.0),
            "power": self.bounded(self.raw_power, 0.7, 2.4),
            "osc_amp": self.bounded(self.raw_osc_amp, 0.0, 0.90),
            "width": self.bounded(self.raw_width, 0.25, 5.0),
            "damping": self.bounded(self.raw_damping, 0.0, 0.060),
            "phase": self.bounded(self.raw_phase, -6.0, 6.0),
            "vr_scale": self.bounded(self.raw_vr_scale, 0.80, 1.20),
            "slope": self.bounded(self.raw_slope, -0.02, 0.04),
            "collector_threshold": self.bounded(self.raw_collector_threshold, -4.0, 10.0),
            "collector_width": self.bounded(self.raw_collector_width, 0.8, 14.0),
            "collector_floor": self.bounded(self.raw_collector_floor, 0.20, 0.98),
            "vr_contrast": self.bounded(self.raw_vr_contrast, -0.25, 0.35),
            "late_contrast": self.bounded(self.raw_late_contrast, -0.35, 0.25),
            "vr_late_contrast": self.bounded(self.raw_vr_late_contrast, -0.20, 0.45),
            "late_onset": self.bounded(self.raw_late_onset, 45.0, 75.0),
            "late_width": self.bounded(self.raw_late_width, 2.0, 18.0),
            "high_energy_loss_strength": self.bounded(self.raw_high_energy_loss_strength, 0.0, 0.45),
            "high_energy_loss_onset": self.bounded(self.raw_high_energy_loss_onset, 52.0, 88.0),
            "high_energy_loss_width": self.bounded(self.raw_high_energy_loss_width, 2.0, 20.0),
            "vr_late_baseline": self.bounded(self.raw_vr_late_baseline, -0.050, 0.030),
        }

    def nuisance_values(self, curve_idx: Optional[int | torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if curve_idx is None:
            idx = torch.tensor(0, device=self.raw_curve_gain.device, dtype=torch.long)
        elif isinstance(curve_idx, torch.Tensor):
            idx = curve_idx.to(device=self.raw_curve_gain.device, dtype=torch.long)
        else:
            idx = torch.tensor(int(curve_idx), device=self.raw_curve_gain.device, dtype=torch.long)
        gain = 1.0 + 0.12 * torch.tanh(self.raw_curve_gain[idx])
        bias = 0.12 * torch.tanh(self.raw_curve_bias[idx])
        dva = 1.5 * torch.tanh(self.raw_curve_dva[idx])
        return gain, bias, dva

    def kernel_components(
        self,
        Va: torch.Tensor,
        Vr: torch.Tensor,
        curve_idx: Optional[int | torch.Tensor] = None,
        nuisance_mode: str = "neutral",
    ) -> Dict[str, torch.Tensor]:
        p = self.phys_params()
        levels = self.level_params()
        va = Va.float()
        vr = Vr.float()
        if str(nuisance_mode) == "curve":
            _, _, dva = self.nuisance_values(curve_idx)
            va = va + dva
        e_collision = torch.clamp(va, min=0.0)
        e_collect = va - p["vr_scale"] * vr
        drive = F.softplus(e_collision - p["v_emit"], beta=2.0)
        norm = torch.pow(torch.clamp(torch.max(e_collision).detach(), min=1.0) + 1.0, p["power"])
        envelope = p["amp"] * torch.pow(drive + 1e-4, p["power"]) / norm
        envelope = envelope + p["offset"] + p["slope"] * e_collision

        collector_arg = (e_collect - p["collector_threshold"]) / (p["collector_width"] + 1e-6)
        collector_gate = torch.sigmoid(collector_arg)
        collector_transmission = p["collector_floor"] + (1.0 - p["collector_floor"]) * collector_gate

        late_gate = torch.sigmoid((e_collision - p["late_onset"]) / (p["late_width"] + 1e-6))
        high_vr_gate = torch.sigmoid((vr - 6.0) / 2.0)
        vr_norm = torch.clamp((vr - 5.0) / 5.0, min=-1.5, max=1.5)
        contrast_log = (
            p["vr_contrast"] * vr_norm
            + p["late_contrast"] * late_gate
            + p["vr_late_contrast"] * vr_norm * late_gate
        )
        contrast_scale = torch.exp(torch.clamp(contrast_log, min=-0.65, max=0.65))
        relative_collision = torch.clamp(e_collision / (torch.clamp(torch.max(e_collision).detach(), min=1.0) + 1e-6), 0.0, 1.0)
        envelope = envelope + p["vr_late_baseline"] * high_vr_gate * late_gate * relative_collision

        high_energy_gate = torch.sigmoid((e_collision - p["high_energy_loss_onset"]) / (p["high_energy_loss_width"] + 1e-6))
        high_energy_loss = 1.0 - p["high_energy_loss_strength"] * high_energy_gate * (0.35 + 0.65 * high_vr_gate)

        energies = levels["energies"].view(1, -1).to(device=e_collision.device, dtype=e_collision.dtype)
        weights = levels["weights"].view(1, -1).to(device=e_collision.device, dtype=e_collision.dtype)
        phase = (e_collision.view(-1, 1) + p["phase"]) / (energies + 1e-6)
        nearest = torch.round(phase)
        residual = phase - nearest
        dips = torch.exp(-0.5 * torch.square(residual * energies / (p["width"] + 1e-6)))
        weighted_dip = torch.sum(weights * dips, dim=1)
        decay = torch.exp(-p["damping"] * e_collision)
        modulation = torch.clamp(1.0 - p["osc_amp"] * contrast_scale * weighted_dip * decay, min=0.03, max=1.15)
        pred = envelope * collector_transmission * high_energy_loss * modulation
        return {
            "prediction": torch.clamp(pred, min=0.0),
            "e_collision": e_collision,
            "e_collect": e_collect,
            "collector_transmission": collector_transmission,
            "collector_gate": collector_gate,
            "late_gate": late_gate,
            "high_vr_gate": high_vr_gate,
            "contrast_scale": contrast_scale,
            "high_energy_loss": high_energy_loss,
            "envelope": envelope,
            "weighted_dip": weighted_dip,
            "modulation": modulation,
        }

    def forward_core(
        self,
        Va: torch.Tensor,
        Vr: torch.Tensor,
        curve_idx: Optional[int | torch.Tensor] = None,
        nuisance_mode: str = "neutral",
    ) -> torch.Tensor:
        components = self.kernel_components(Va, Vr, curve_idx=curve_idx, nuisance_mode=nuisance_mode)
        pred = components["prediction"]
        return torch.clamp(pred, min=0.0)

    def forward(
        self,
        Va: torch.Tensor,
        Vr: torch.Tensor,
        curve_idx: Optional[int | torch.Tensor] = None,
        nuisance_mode: str = "curve",
    ) -> torch.Tensor:
        pred = self.forward_core(Va, Vr, curve_idx=curve_idx, nuisance_mode=nuisance_mode)
        if str(nuisance_mode) == "curve":
            gain, bias, _ = self.nuisance_values(curve_idx)
            pred = gain * pred + bias
        return torch.clamp(pred, min=0.0)

    def regularization_loss(self, cfg: Config) -> torch.Tensor:
        loss = torch.zeros((), device=self.raw_E1.device)
        for name, param in self.named_parameters():
            if param.numel() == 0:
                continue
            if name.startswith("raw_curve"):
                loss = loss + 0.25 * torch.mean(torch.square(param))
            else:
                loss = loss + torch.mean(torch.square(param))
        levels = self.level_params()
        weights = levels["weights"]
        entropy = -torch.sum(weights * torch.log(weights + 1e-12))
        max_entropy = math.log(max(int(self.n_levels), 2))
        entropy_penalty = (max_entropy - entropy) / max_entropy
        if levels["gaps"].numel() > 0:
            gap_penalty = torch.mean(torch.relu(float(cfg.min_level_gap) - levels["gaps"]) ** 2)
        else:
            gap_penalty = torch.zeros((), device=self.raw_E1.device)
        anchor_start = safe_float(getattr(cfg, "forward_main_spacing", 0.0), 0.0)
        if str(getattr(cfg, "forward_prior_mode", "off")) == "off" or anchor_start <= 0.0:
            anchor_start = 11.55
        anchor_step = safe_float(getattr(cfg, "forward_anchor_step", 0.25), 0.25)
        anchors = torch.linspace(
            float(anchor_start),
            float(anchor_start + anchor_step * max(self.n_levels - 1, 0)),
            self.n_levels,
            device=self.raw_E1.device,
        )
        anchor_penalty = torch.mean(torch.square(levels["energies"] - anchors))
        return (
            float(cfg.w_reg) * loss
            + float(cfg.level_weight_entropy) * entropy_penalty
            + float(cfg.w_prior_gap) * gap_penalty
            + float(cfg.w_prior_anchor) * anchor_penalty
        )


def model_predict(
    model: PoissonRateFHCoreMultiLevel,
    Va: torch.Tensor,
    Vr: torch.Tensor,
    curve_idx: Optional[int | torch.Tensor] = None,
    nuisance_mode: str = "curve",
) -> torch.Tensor:
    return model(Va, Vr, curve_idx=curve_idx, nuisance_mode=nuisance_mode)


def curve_to_tensors(curve: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    va_np = np.asarray(curve["Va"], dtype=np.float32)
    ip_np = np.asarray(curve["Ip"], dtype=np.float32)
    if bool(curve["Vr_is_vector"]):
        vr_np = np.asarray(curve["Vr"], dtype=np.float32)
    else:
        vr_np = np.full_like(va_np, float(curve["Vr"]), dtype=np.float32)
    va_t = torch.tensor(va_np, device=device)
    ip_t = torch.tensor(ip_np, device=device)
    d1_t = finite_diff(ip_t, va_t)
    d2_t = finite_diff(d1_t, va_t)
    weight_np = structure_window_weights_np(va_np, ip_np)
    valley_weight_np = valley_window_weights_np(va_np, ip_np)
    return {
        "curve_idx": torch.tensor(int(curve["curve_idx"]), device=device, dtype=torch.long),
        "Va": va_t,
        "Vr": torch.tensor(vr_np, device=device),
        "Ip": ip_t,
        "Ip_d1": d1_t,
        "Ip_d2": d2_t,
        "structure_weight": torch.tensor(weight_np, device=device),
        "valley_weight": torch.tensor(valley_weight_np, device=device),
    }


def finite_diff(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    dy = values[1:] - values[:-1]
    dx = torch.clamp(x[1:] - x[:-1], min=1e-6)
    d = dy / dx
    return torch.cat([d, d[-1:]], dim=0)


def finite_diff_np(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if values.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    dy = values[1:] - values[:-1]
    dx = np.maximum(x[1:] - x[:-1], 1e-6)
    d = dy / dx
    return np.concatenate([d, d[-1:]])


def structure_window_weights_np(va: np.ndarray, current: np.ndarray, radius: float = 1.5) -> np.ndarray:
    va = np.asarray(va, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    weight = np.ones_like(va, dtype=np.float32)
    peaks = _turning_points(va, current, "peak")["positions"]
    valleys = _turning_points(va, current, "valley")["positions"]
    for pos in list(peaks) + list(valleys):
        mask = np.abs(va - float(pos)) <= float(radius)
        weight[mask] = 2.0
    return weight


def valley_window_weights_np(va: np.ndarray, current: np.ndarray, radius: float = 1.5) -> np.ndarray:
    va = np.asarray(va, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    weight = np.zeros_like(va, dtype=np.float32)
    valleys = _turning_points(va, current, "valley")["positions"]
    for pos in valleys:
        mask = np.abs(va - float(pos)) <= float(radius)
        weight[mask] = 1.0
    return weight


def _interp_at(x: np.ndarray, y: np.ndarray, pos: float) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.interp(float(pos), x, y))


def _nearest_position_delta(target_pos: float, candidate_positions: Sequence[float]) -> float:
    if not candidate_positions:
        return float("nan")
    arr = np.asarray(candidate_positions, dtype=np.float64)
    return float(arr[int(np.argmin(np.abs(arr - float(target_pos))))] - float(target_pos))


def compute_vr_physical_response_metrics(
    curves: Sequence[Dict[str, Any]],
    predictions: Sequence[np.ndarray | Sequence[float]],
    late_va_min: float = 60.0,
    late_bias_abs_threshold: float = 0.10,
    high_vr_threshold: float = 8.0,
    high_vr_valley_lift_threshold: float = 0.12,
) -> Dict[str, Any]:
    per_curve: List[Dict[str, Any]] = []
    for curve, pred_values in zip(curves, predictions):
        va = np.asarray(curve["Va"], dtype=np.float64)
        target = np.asarray(curve["Ip"], dtype=np.float64)
        pred = np.asarray(pred_values, dtype=np.float64)
        vr_class = float(np.mean(np.asarray(curve["Vr"], dtype=np.float64))) if bool(curve.get("Vr_is_vector", False)) else float(curve.get("Vr", 0.0))
        late_mask = va >= float(late_va_min)
        if int(np.sum(late_mask)) < 2:
            continue
        late_va = va[late_mask]
        late_target = target[late_mask]
        late_pred = pred[late_mask]
        target_range = float(np.max(late_target) - np.min(late_target))
        pred_range = float(np.max(late_pred) - np.min(late_pred))
        bias = float(np.mean(late_pred - late_target))
        valleys = _turning_points(late_va, late_target, "valley")["positions"]
        valley_lifts: List[float] = []
        for pos in valleys:
            idx = int(np.argmin(np.abs(va - float(pos))))
            valley_lifts.append(max(0.0, float(pred[idx] - target[idx])))
        valley_lift = safe_nanmean(valley_lifts, default=0.0)
        per_curve.append(
            {
                "curve_id": int(curve.get("curve_id", len(per_curve) + 1)),
                "vr_class": vr_class,
                "late_va_min": float(late_va_min),
                "late_bias_fit_minus_target": bias,
                "late_target_range": target_range,
                "late_pred_range": pred_range,
                "late_valley_count": int(len(valleys)),
                "late_valley_lift_mean": valley_lift,
                "is_high_vr": bool(vr_class >= float(high_vr_threshold)),
            }
        )

    per_class: List[Dict[str, Any]] = []
    for vr_value in sorted({row["vr_class"] for row in per_curve}):
        group = [row for row in per_curve if row["vr_class"] == vr_value]
        per_class.append(
            {
                "vr_class": float(vr_value),
                "curve_count": int(len(group)),
                "late_bias_fit_minus_target": safe_nanmean([row["late_bias_fit_minus_target"] for row in group]),
                "late_abs_bias": safe_nanmean([abs(row["late_bias_fit_minus_target"]) for row in group]),
                "late_valley_lift_mean": safe_nanmean([row["late_valley_lift_mean"] for row in group], default=0.0),
                "is_high_vr": bool(float(vr_value) >= float(high_vr_threshold)),
            }
        )

    finite_bias = [abs(row["late_bias_fit_minus_target"]) for row in per_curve if np.isfinite(row["late_bias_fit_minus_target"])]
    high_vr_lifts = [row["late_valley_lift_mean"] for row in per_curve if bool(row["is_high_vr"]) and np.isfinite(row["late_valley_lift_mean"])]
    max_abs_bias = max(finite_bias) if finite_bias else float("inf")
    max_high_vr_valley_lift = max(high_vr_lifts) if high_vr_lifts else 0.0
    vr_late_bias_pass = bool(max_abs_bias <= float(late_bias_abs_threshold))
    high_vr_valley_depth_pass = bool(max_high_vr_valley_lift <= float(high_vr_valley_lift_threshold))
    response_score = (
        min(max_abs_bias / max(float(late_bias_abs_threshold), 1.0e-9), 10.0)
        + min(max_high_vr_valley_lift / max(float(high_vr_valley_lift_threshold), 1.0e-9), 10.0)
    ) / 2.0
    summary = {
        "vr_late_bias_pass": vr_late_bias_pass,
        "high_vr_valley_depth_pass": high_vr_valley_depth_pass,
        "late_va_min": float(late_va_min),
        "late_bias_abs_threshold": float(late_bias_abs_threshold),
        "high_vr_threshold": float(high_vr_threshold),
        "high_vr_valley_lift_threshold": float(high_vr_valley_lift_threshold),
        "max_abs_late_bias": float(max_abs_bias),
        "max_high_vr_valley_lift": float(max_high_vr_valley_lift),
        "vr_physical_response_score": float(response_score),
    }
    return {"per_curve": per_curve, "per_class": per_class, "summary": summary}


def compute_structure_metrics(
    curves: Sequence[Dict[str, Any]],
    predictions: Sequence[np.ndarray | Sequence[float]],
    window_radius: float = 1.5,
    flatline_min_osc_ratio: float = 0.25,
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = []
    per_curve: List[Dict[str, Any]] = []
    for curve, pred_values in zip(curves, predictions):
        va = np.asarray(curve["Va"], dtype=np.float64)
        target = np.asarray(curve["Ip"], dtype=np.float64)
        pred = np.asarray(pred_values, dtype=np.float64)
        target_d1 = finite_diff_np(target, va)
        pred_d1 = finite_diff_np(pred, va)
        target_d2 = finite_diff_np(target_d1, va)
        pred_d2 = finite_diff_np(pred_d1, va)
        target_peaks = _turning_points(va, target, "peak")["positions"]
        target_valleys = _turning_points(va, target, "valley")["positions"]
        pred_peaks = _turning_points(va, pred, "peak")["positions"]
        pred_valleys = _turning_points(va, pred, "valley")["positions"]
        target_amp = max(float(np.max(target) - np.min(target)), 1e-12)
        pred_amp = max(float(np.max(pred) - np.min(pred)), 0.0)
        curve_rows: List[Dict[str, Any]] = []
        for kind, positions, pred_positions in (
            ("peak", target_peaks, pred_peaks),
            ("valley", target_valleys, pred_valleys),
        ):
            for ordinal, pos in enumerate(positions, start=1):
                mask = np.abs(va - float(pos)) <= float(window_radius)
                if int(np.sum(mask)) < 2:
                    continue
                err = pred[mask] - target[mask]
                d1_err = pred_d1[mask] - target_d1[mask]
                d2_err = pred_d2[mask] - target_d2[mask]
                pos_delta = _nearest_position_delta(float(pos), pred_positions)
                amp_delta = _interp_at(va, pred, float(pos)) - _interp_at(va, target, float(pos))
                row = {
                    "curve_id": int(curve.get("curve_id", len(per_curve))),
                    "curve_idx": int(curve.get("curve_idx", len(per_curve))),
                    "vr_class": float(np.mean(np.asarray(curve["Vr"], dtype=np.float64))) if bool(curve.get("Vr_is_vector", False)) else float(curve.get("Vr", 0.0)),
                    "kind": kind,
                    "ordinal": int(ordinal),
                    "center_va": float(pos),
                    "segment_rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "segment_d1_rmse": float(np.sqrt(np.mean(np.square(d1_err)))),
                    "segment_d2_rmse": float(np.sqrt(np.mean(np.square(d2_err)))),
                    "extremum_position_error": float(pos_delta),
                    "extremum_amplitude_error": float(amp_delta),
                    "contrast_error": float(abs(pred_amp - target_amp)),
                }
                segments.append(row)
                curve_rows.append(row)
        if curve_rows:
            per_curve.append(
                {
                    "curve_id": int(curve.get("curve_id", len(per_curve))),
                    "vr_class": float(np.mean(np.asarray(curve["Vr"], dtype=np.float64))) if bool(curve.get("Vr_is_vector", False)) else float(curve.get("Vr", 0.0)),
                    "segment_count": int(len(curve_rows)),
                    "segment_rmse_mean": float(np.mean([r["segment_rmse"] for r in curve_rows])),
                    "segment_d1_rmse_mean": float(np.mean([r["segment_d1_rmse"] for r in curve_rows])),
                    "segment_d2_rmse_mean": float(np.mean([r["segment_d2_rmse"] for r in curve_rows])),
                    "position_error_abs_mean": safe_nanmean([abs(r["extremum_position_error"]) for r in curve_rows]),
                    "amplitude_error_abs_mean": safe_nanmean([abs(r["extremum_amplitude_error"]) for r in curve_rows]),
                    "target_oscillation_range": float(target_amp),
                    "pred_oscillation_range": float(pred_amp),
                    "oscillation_ratio": float(pred_amp / target_amp),
                    "flatline_guard_pass": bool(pred_amp / target_amp >= float(flatline_min_osc_ratio)),
                }
            )
    if segments:
        summary = {
            "segment_count": int(len(segments)),
            "segment_rmse_mean": float(np.mean([r["segment_rmse"] for r in segments])),
            "segment_d1_rmse_mean": float(np.mean([r["segment_d1_rmse"] for r in segments])),
            "segment_d2_rmse_mean": float(np.mean([r["segment_d2_rmse"] for r in segments])),
            "position_error_abs_mean": safe_nanmean([abs(r["extremum_position_error"]) for r in segments]),
            "amplitude_error_abs_mean": safe_nanmean([abs(r["extremum_amplitude_error"]) for r in segments]),
            "contrast_error_mean": safe_nanmean([r["contrast_error"] for r in segments]),
            "flatline_guard_pass": bool(all(row["flatline_guard_pass"] for row in per_curve)),
        }
    else:
        summary = {
            "segment_count": 0,
            "segment_rmse_mean": float("inf"),
            "segment_d1_rmse_mean": float("inf"),
            "segment_d2_rmse_mean": float("inf"),
            "position_error_abs_mean": float("inf"),
            "amplitude_error_abs_mean": float("inf"),
            "contrast_error_mean": float("inf"),
            "flatline_guard_pass": False,
        }
    class_rows: List[Dict[str, Any]] = []
    for vr_value in sorted({row["vr_class"] for row in per_curve}):
        group = [row for row in per_curve if row["vr_class"] == vr_value]
        class_rows.append(
            {
                "vr_class": float(vr_value),
                "curve_count": int(len(group)),
                "segment_rmse_mean": float(np.mean([r["segment_rmse_mean"] for r in group])),
                "segment_d1_rmse_mean": float(np.mean([r["segment_d1_rmse_mean"] for r in group])),
                "segment_d2_rmse_mean": float(np.mean([r["segment_d2_rmse_mean"] for r in group])),
                "flatline_guard_pass": bool(all(r["flatline_guard_pass"] for r in group)),
            }
        )
    return {"segments": segments, "per_curve": per_curve, "per_class": class_rows, "summary": summary}


def compute_losses_multilevel(
    cfg: Config,
    model: PoissonRateFHCoreMultiLevel,
    tensor_curves: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    raw_losses: List[torch.Tensor] = []
    d1_losses: List[torch.Tensor] = []
    d2_losses: List[torch.Tensor] = []
    peak_window_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []
    vr_late_bias_losses: List[torch.Tensor] = []
    vr_amplitude_ratio_losses: List[torch.Tensor] = []
    high_vr_valley_losses: List[torch.Tensor] = []
    for curve in tensor_curves:
        va = curve["Va"]
        target = curve["Ip"]
        pred = model(va, curve["Vr"], curve_idx=curve["curve_idx"], nuisance_mode="curve")
        pred_d1 = finite_diff(pred, va)
        pred_d2 = finite_diff(pred_d1, va)
        raw_losses.append(torch.mean(torch.square(pred - target)))
        d1_losses.append(torch.mean(torch.square(pred_d1 - curve["Ip_d1"])))
        d2_losses.append(torch.mean(torch.square(pred_d2 - curve["Ip_d2"])))
        weights = curve.get("structure_weight")
        if weights is not None:
            peak_window_losses.append(torch.sum(weights * torch.square(pred - target)) / torch.clamp(torch.sum(weights), min=1.0))
        else:
            peak_window_losses.append(torch.mean(torch.square(pred - target)))
        if pred.numel() >= 3:
            smooth_losses.append(torch.mean(torch.square(pred[2:] - 2.0 * pred[1:-1] + pred[:-2])))
        else:
            smooth_losses.append(torch.zeros((), device=pred.device))
        late_mask = va >= float(cfg.vr_late_va_min)
        if int(torch.sum(late_mask).detach().cpu()) >= 2:
            pred_late = pred[late_mask]
            target_late = target[late_mask]
            late_bias = torch.mean(pred_late - target_late)
            target_range = torch.clamp(torch.max(target_late) - torch.min(target_late), min=1.0e-4)
            pred_range = torch.clamp(torch.max(pred_late) - torch.min(pred_late), min=1.0e-4)
            vr_late_bias_losses.append(torch.square(late_bias))
            vr_amplitude_ratio_losses.append(torch.square(torch.log(pred_range / target_range)))
            mean_vr = torch.mean(curve["Vr"].float())
            valley_weight = curve.get("valley_weight")
            if valley_weight is not None and float(mean_vr.detach().cpu()) >= float(cfg.high_vr_threshold):
                valley_mask = valley_weight[late_mask]
                denom = torch.clamp(torch.sum(valley_mask), min=1.0)
                high_vr_valley_losses.append(torch.sum(valley_mask * torch.square(torch.relu(pred_late - target_late))) / denom)
            else:
                high_vr_valley_losses.append(torch.zeros((), device=pred.device))
        else:
            zero = torch.zeros((), device=pred.device)
            vr_late_bias_losses.append(zero)
            vr_amplitude_ratio_losses.append(zero)
            high_vr_valley_losses.append(zero)
    raw = torch.mean(torch.stack(raw_losses))
    d1 = torch.mean(torch.stack(d1_losses))
    d2 = torch.mean(torch.stack(d2_losses))
    peak_window = torch.mean(torch.stack(peak_window_losses))
    smooth = torch.mean(torch.stack(smooth_losses))
    vr_late_bias = torch.mean(torch.stack(vr_late_bias_losses))
    vr_amplitude_ratio = torch.mean(torch.stack(vr_amplitude_ratio_losses))
    high_vr_valley_depth = torch.mean(torch.stack(high_vr_valley_losses))
    reg = model.regularization_loss(cfg)
    total = (
        float(cfg.w_raw) * raw
        + float(cfg.w_d1) * d1
        + float(cfg.w_d2) * d2
        + float(cfg.w_peak_window) * peak_window
        + float(cfg.w_smooth) * smooth
        + float(cfg.w_vr_late_bias) * vr_late_bias
        + float(cfg.w_vr_amplitude_ratio) * vr_amplitude_ratio
        + float(cfg.w_high_vr_valley_depth) * high_vr_valley_depth
        + reg
    )
    monitor = total + 0.25 * d1 + 0.10 * d2 + 0.25 * peak_window + 0.20 * vr_late_bias + 0.10 * vr_amplitude_ratio + 0.20 * high_vr_valley_depth
    return {
        "loss_total": total,
        "loss_monitor": monitor,
        "loss_raw": raw,
        "loss_d1": d1,
        "loss_d2": d2,
        "loss_peak_window": peak_window,
        "loss_smooth": smooth,
        "loss_vr_late_bias": vr_late_bias,
        "loss_vr_amplitude_ratio": vr_amplitude_ratio,
        "loss_high_vr_valley_depth": high_vr_valley_depth,
        "loss_reg": reg,
    }


class _LocalMuon(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float, momentum: float = 0.95) -> None:
        defaults = {"lr": float(lr), "weight_decay": float(weight_decay), "momentum": float(momentum)}
        super().__init__(list(params), defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            momentum = float(group["momentum"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if wd:
                    grad = grad.add(param, alpha=wd)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                denom = torch.sqrt(torch.mean(torch.square(buf))).clamp_min(1e-8)
                param.add_(buf / denom, alpha=-lr)
        return loss


class HybridOptimizer:
    def __init__(self, muon: Optional[_LocalMuon], adamw: Optional[torch.optim.AdamW], summary_payload: Dict[str, Any]) -> None:
        self.muon = muon
        self.adamw = adamw
        self._summary = dict(summary_payload)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.muon is not None and state.get("muon") is not None:
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state.get("adamw") is not None:
            self.adamw.load_state_dict(state["adamw"])

    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)


def build_optimizer(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    optimizer_name: str = "muon_hybrid",
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
) -> HybridOptimizer:
    trainable = [(name, param) for name, param in named_parameters if param.requires_grad]
    name = str(optimizer_name).strip().lower()
    if name not in {"muon_hybrid", "adamw"}:
        name = "muon_hybrid"
    if name == "adamw":
        params = [param for _, param in trainable]
        adamw = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay)) if params else None
        return HybridOptimizer(
            None,
            adamw,
            {
                "optimizer_name": "adamw",
                "muon_param_count": 0,
                "adamw_fallback_param_count": len(params),
                "muon_parameter_names": [],
                "adamw_fallback_parameter_names": [n for n, _ in trainable],
            },
        )

    muon_items = [(n, p) for n, p in trainable if p.ndim >= 1 and p.numel() > 1]
    adamw_items = [(n, p) for n, p in trainable if not (p.ndim >= 1 and p.numel() > 1)]
    muon = _LocalMuon([p for _, p in muon_items], lr=float(lr), weight_decay=float(weight_decay)) if muon_items else None
    adamw = (
        torch.optim.AdamW([p for _, p in adamw_items], lr=float(lr), weight_decay=float(weight_decay))
        if adamw_items
        else None
    )
    return HybridOptimizer(
        muon,
        adamw,
        {
            "optimizer_name": "muon_hybrid",
            "muon_param_count": len(muon_items),
            "adamw_fallback_param_count": len(adamw_items),
            "muon_parameter_names": [n for n, _ in muon_items],
            "adamw_fallback_parameter_names": [n for n, _ in adamw_items],
        },
    )


@torch.no_grad()
def eval_metrics(
    model: PoissonRateFHCoreMultiLevel,
    curves: Sequence[Dict[str, Any]],
    device: torch.device,
    nuisance_mode: str = "curve",
    cfg: Optional[Config] = None,
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    predictions: List[np.ndarray] = []
    for curve in curves:
        tensors = curve_to_tensors(curve, device)
        pred = model_predict(model, tensors["Va"], tensors["Vr"], tensors["curve_idx"], nuisance_mode=nuisance_mode)
        target = tensors["Ip"]
        err = pred - target
        pred_np = pred.detach().cpu().numpy()
        predictions.append(pred_np)
        d1_err = finite_diff(pred, tensors["Va"]) - tensors["Ip_d1"]
        d2_err = finite_diff(finite_diff(pred, tensors["Va"]), tensors["Va"]) - tensors["Ip_d2"]
        rows.append(
            {
                "curve_id": int(curve["curve_id"]),
                "rmse": float(torch.sqrt(torch.mean(torch.square(err))).detach().cpu()),
                "mae": float(torch.mean(torch.abs(err)).detach().cpu()),
                "d1_rmse": float(torch.sqrt(torch.mean(torch.square(d1_err))).detach().cpu()),
                "d2_rmse": float(torch.sqrt(torch.mean(torch.square(d2_err))).detach().cpu()),
                "sse": float(torch.sum(torch.square(err)).detach().cpu()),
                "n_points": int(err.numel()),
            }
        )
    rmse_mean = float(np.mean([r["rmse"] for r in rows])) if rows else float("nan")
    mae_mean = float(np.mean([r["mae"] for r in rows])) if rows else float("nan")
    d1_rmse_mean = float(np.mean([r["d1_rmse"] for r in rows])) if rows else float("nan")
    d2_rmse_mean = float(np.mean([r["d2_rmse"] for r in rows])) if rows else float("nan")
    sse_total = float(np.sum([r["sse"] for r in rows])) if rows else float("nan")
    n_points = int(np.sum([r["n_points"] for r in rows])) if rows else 0
    n_params = int(sum(param.numel() for param in model.parameters()))
    sigma2 = max(sse_total / max(n_points, 1), 1e-12)
    aic = float(n_points * math.log(sigma2) + 2.0 * n_params)
    bic = float(n_points * math.log(sigma2) + n_params * math.log(max(n_points, 1)))
    structure = compute_structure_metrics(curves, predictions)
    if cfg is None:
        vr_response = compute_vr_physical_response_metrics(curves, predictions)
    else:
        vr_response = compute_vr_physical_response_metrics(
            curves,
            predictions,
            late_va_min=float(cfg.vr_late_va_min),
            late_bias_abs_threshold=float(cfg.vr_late_bias_abs_threshold),
            high_vr_threshold=float(cfg.high_vr_threshold),
            high_vr_valley_lift_threshold=float(cfg.high_vr_valley_lift_threshold),
        )
    vr_summary = vr_response["summary"]
    return {
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "d1_rmse_mean": d1_rmse_mean,
        "d2_rmse_mean": d2_rmse_mean,
        "sse_total": sse_total,
        "n_points": n_points,
        "n_params": n_params,
        "aic": aic,
        "bic": bic,
        "structure_score": float(
            structure["summary"]["segment_rmse_mean"]
            + 0.10 * structure["summary"]["segment_d1_rmse_mean"]
            + 0.02 * structure["summary"]["segment_d2_rmse_mean"]
        ),
        "flatline_guard_pass": bool(structure["summary"]["flatline_guard_pass"]),
        "structure_metrics": structure,
        "vr_physical_response": vr_response,
        "vr_physical_response_score": float(vr_summary["vr_physical_response_score"]),
        "vr_late_bias_pass": bool(vr_summary["vr_late_bias_pass"]),
        "high_vr_valley_depth_pass": bool(vr_summary["high_vr_valley_depth_pass"]),
        "per_curve": rows,
    }


def extract_params(model: PoissonRateFHCoreMultiLevel) -> Dict[str, Any]:
    with torch.no_grad():
        phys = model.phys_params()
        levels = model.level_params()
        energies = [float(x) for x in levels["energies"].detach().cpu().tolist()]
        weights = [float(x) for x in levels["weights"].detach().cpu().tolist()]
        payload: Dict[str, Any] = {
            "n_levels": int(model.n_levels),
            "V_exc": float(energies[0]) if energies else float("nan"),
            "V_exc_weighted": float(np.sum(np.asarray(energies) * np.asarray(weights))) if energies else float("nan"),
            "effective_excitation_mean": float(np.sum(np.asarray(energies) * np.asarray(weights))) if energies else float("nan"),
            "effective_excitation_std": float(np.sqrt(np.sum(np.asarray(weights) * np.square(np.asarray(energies) - np.sum(np.asarray(energies) * np.asarray(weights)))))) if energies else float("nan"),
        }
        for key, value in phys.items():
            payload[key] = float(value.detach().cpu())
        for idx, value in enumerate(energies, start=1):
            payload[f"level_energy_{idx:02d}"] = float(value)
        for idx, value in enumerate(weights, start=1):
            payload[f"level_weight_{idx:02d}"] = float(value)
        return payload


def config_hash(cfg: Config) -> str:
    payload = json.dumps(json_ready(asdict(cfg)), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_last_checkpoint(
    path: Path,
    model: PoissonRateFHCoreMultiLevel,
    optimizer: HybridOptimizer,
    epoch: int,
    best_state: Optional[Dict[str, torch.Tensor]],
    best_loss: float,
    best_epoch: int,
    stopper: EarlyStopper,
    cfg: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_state": best_state,
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "early_stopper": stopper.state_dict(),
            "config_hash": config_hash(cfg),
        },
        path,
    )


def write_status(
    out_dir: Path,
    cfg: Config,
    n_levels: int,
    epoch: int,
    epochs: int,
    losses: Dict[str, torch.Tensor],
    best_epoch: int,
    early_stop: bool,
    start_time: Optional[float] = None,
) -> None:
    elapsed = float(time.perf_counter() - start_time) if start_time is not None else 0.0
    status = "early_stopped" if bool(early_stop) else ("completed" if int(epoch) >= int(epochs) else "running")
    eta_hint = "complete" if status in {"completed", "early_stopped"} else "unknown"
    payload = {
        "n_levels": int(n_levels),
        "seed": int(cfg.seed),
        "status": status,
        "stage": str(getattr(cfg, "run_stage", "full_sweep")),
        "hyperopt_stage": str(getattr(cfg, "hyperopt_stage", "")),
        "epoch": int(epoch),
        "epochs": int(epochs),
        "loss_total": float(losses["loss_total"].detach().cpu()),
        "loss_monitor": float(losses["loss_monitor"].detach().cpu()),
        "loss_raw": float(losses["loss_raw"].detach().cpu()),
        "loss_d1": float(losses["loss_d1"].detach().cpu()),
        "loss_d2": float(losses["loss_d2"].detach().cpu()),
        "loss_peak_window": float(losses["loss_peak_window"].detach().cpu()),
        "loss_vr_late_bias": float(losses["loss_vr_late_bias"].detach().cpu()),
        "loss_vr_amplitude_ratio": float(losses["loss_vr_amplitude_ratio"].detach().cpu()),
        "loss_high_vr_valley_depth": float(losses["loss_high_vr_valley_depth"].detach().cpu()),
        "structure_monitor": float(
            (
                losses["loss_peak_window"]
                + losses["loss_vr_late_bias"]
                + losses["loss_vr_amplitude_ratio"]
                + losses["loss_high_vr_valley_depth"]
            )
            .detach()
            .cpu()
        ),
        "best_epoch": int(best_epoch),
        "early_stop": bool(early_stop),
        "heartbeat": float(time.time()),
        "elapsed_seconds": elapsed,
        "eta_hint": eta_hint,
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_best": "checkpoint_best.pt",
    }
    safe_json_dump(payload, out_dir / "status.json")


def train_multilevel(
    cfg: Config,
    model: PoissonRateFHCoreMultiLevel,
    curves: Sequence[Dict[str, Any]],
    device: torch.device,
    out_dir: Path,
    epochs: int,
    resume_ckpt: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_ckpt is not None and Path(resume_ckpt).exists():
        loaded = torch.load(Path(resume_ckpt), map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "model_state" in loaded:
            resume_payload = loaded
            model.load_state_dict(loaded["model_state"], strict=False)
        else:
            model.load_state_dict(loaded, strict=False)
    tensor_curves = [curve_to_tensors(curve, device) for curve in curves]
    optimizer = build_optimizer(model.named_parameters(), cfg.optimizer, cfg.lr, cfg.weight_decay)
    stopper = EarlyStopper(
        warmup_epochs=int(cfg.early_stop_warmup),
        min_epochs=int(cfg.early_stop_min_epochs),
        patience=int(cfg.early_stop_patience),
        min_delta_rel=float(cfg.early_stop_min_delta_rel),
        smoothing=int(cfg.early_stop_smoothing),
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    start_epoch = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload.get("optimizer_state", {}))
        stopper.load_state_dict(resume_payload.get("early_stopper", {}))
        best_state = resume_payload.get("best_state")
        best_loss = safe_float(resume_payload.get("best_loss"), float("inf"))
        best_epoch = int(resume_payload.get("best_epoch", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
    log_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    model.train()
    for epoch in range(start_epoch + 1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses_multilevel(cfg, model, tensor_curves)
        loss = losses["loss_total"]
        loss.backward()
        if float(cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        monitor_value = float(losses["loss_monitor"].detach().cpu())
        state = stopper.update(epoch, monitor_value)
        if monitor_value < best_loss:
            best_loss = monitor_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(cfg.checkpoint_min_interval)) == 0 or state.should_stop:
            row = {
                "epoch": int(epoch),
                "loss_total": loss_value,
                "loss_monitor": monitor_value,
                "loss_raw": float(losses["loss_raw"].detach().cpu()),
                "loss_d1": float(losses["loss_d1"].detach().cpu()),
                "loss_d2": float(losses["loss_d2"].detach().cpu()),
                "loss_peak_window": float(losses["loss_peak_window"].detach().cpu()),
                "loss_smooth": float(losses["loss_smooth"].detach().cpu()),
                "loss_vr_late_bias": float(losses["loss_vr_late_bias"].detach().cpu()),
                "loss_vr_amplitude_ratio": float(losses["loss_vr_amplitude_ratio"].detach().cpu()),
                "loss_high_vr_valley_depth": float(losses["loss_high_vr_valley_depth"].detach().cpu()),
                "loss_reg": float(losses["loss_reg"].detach().cpu()),
                "best_epoch": int(best_epoch),
                "early_stop_wait": int(state.wait_count),
                "early_stop": bool(state.should_stop),
            }
            log_rows.append(row)
            save_last_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, epoch, best_state, best_loss, best_epoch, stopper, cfg)
            write_status(out_dir, cfg, int(model.n_levels), epoch, int(epochs), losses, best_epoch, bool(state.should_stop), start)
        if state.should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    metrics = eval_metrics(model, curves, device, nuisance_mode="curve", cfg=cfg)
    params = extract_params(model)
    weights = [params[f"level_weight_{idx:02d}"] for idx in range(1, int(model.n_levels) + 1)]
    diagnostics = compute_weight_diagnostics(weights, n_levels=int(model.n_levels), reference_k=int(cfg.reference_k))
    forward_evidence = load_forward_evidence_for_config(cfg)
    forward_consistency = (
        compute_forward_reverse_consistency(params, forward_evidence)
        if forward_evidence is not None
        else None
    )
    elapsed = float(time.perf_counter() - start)
    scorecard = {
        "n_levels": int(model.n_levels),
        "model_schema": MODEL_SCHEMA,
        "config_hash": config_hash(cfg),
        "seed": int(cfg.seed),
        "epochs_requested": int(epochs),
        "epochs_completed": int(log_rows[-1]["epoch"] if log_rows else start_epoch),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "optimizer_summary": optimizer.summary(),
        "metrics_curve": metrics,
        "params": params,
        "weight_diagnostics": diagnostics,
        "forward_reverse_consistency": forward_consistency,
        "early_stop": {
            "triggered": bool(log_rows[-1]["early_stop"] if log_rows else False),
            "min_epochs": int(stopper.min_epochs),
            "warmup_epochs": int(stopper.warmup_epochs),
            "patience": int(stopper.patience),
            "min_delta_rel": float(stopper.min_delta_rel),
            "smoothing": int(stopper.smoothing),
        },
    }
    if not (out_dir / "checkpoint_last.pt").exists():
        save_last_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, int(start_epoch), best_state, best_loss, best_epoch, stopper, cfg)
    torch.save(model.state_dict(), out_dir / "checkpoint_best.pt")
    safe_json_dump(params, out_dir / "params_best.json")
    safe_json_dump(metrics, out_dir / "metrics_best.json")
    safe_json_dump(scorecard, out_dir / "scorecard.json")
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    return scorecard


def run_level_fit(
    cfg: Config,
    n_levels: int,
    seed: int = 0,
    out_dir: str | Path | None = None,
    epochs: Optional[int] = None,
    device: str | torch.device | None = None,
    resume_ckpt: Optional[str | Path] = None,
) -> Dict[str, Any]:
    seed = int(seed)
    local_cfg = Config(**asdict(cfg))
    local_cfg.seed = seed
    set_seed(seed)
    curves, init = load_curves_and_init(local_cfg)
    torch_device = device if isinstance(device, torch.device) else resolve_device(str(device or local_cfg.selected_device or local_cfg.device))
    model = PoissonRateFHCoreMultiLevel(
        n_curves=len(curves),
        n_max=int(local_cfg.n_max),
        n_levels=int(n_levels),
        min_level_gap=float(local_cfg.min_level_gap),
        device=torch_device,
        V_exc_init=float(init["V_exc_init"]),
    ).to(torch_device)
    target_dir = Path(out_dir if out_dir is not None else local_cfg.out_dir)
    scorecard = train_multilevel(
        local_cfg,
        model,
        curves,
        torch_device,
        target_dir,
        int(epochs if epochs is not None else local_cfg.epochs),
        Path(resume_ckpt) if resume_ckpt is not None else None,
    )
    scorecard["data_init"] = init
    safe_json_dump(scorecard, target_dir / "scorecard.json")
    return scorecard


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray([max(0.0, float(x)) for x in weights], dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(max(len(arr), 1), 1.0 / max(len(arr), 1), dtype=np.float64)
    return arr / total


def compute_weight_diagnostics(
    weights: Sequence[float],
    n_levels: Optional[int] = None,
    reference_k: int = 4,
    low_weight_threshold: Optional[float] = None,
    concentration_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    arr = normalize_weights(weights)
    n = int(n_levels if n_levels is not None else len(arr))
    reference_k = int(reference_k)
    low_thr = float(low_weight_threshold if low_weight_threshold is not None else max(0.015, 0.35 / max(n, 1)))
    concentration_thr = float(concentration_threshold if concentration_threshold is not None else max(0.78, 1.0 - 0.60 / max(n, 1)))
    hhi = float(np.sum(np.square(arr)))
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)))
    entropy_effective = float(math.exp(entropy))
    inv_hhi = float(1.0 / max(hhi, 1e-12))
    sorted_desc = np.sort(arr)[::-1]
    low_count = int(np.sum(arr < low_thr))
    extra = arr[reference_k:] if n > reference_k else np.asarray([], dtype=np.float64)
    n_extra_low = int(np.sum(extra < low_thr)) if extra.size else 0
    status = "balanced"
    if float(sorted_desc[0]) >= concentration_thr:
        status = "top_heavy"
    if n > reference_k and extra.size and (n_extra_low >= 1 or float(np.sum(extra)) < low_thr * max(1, extra.size)):
        status = "extra_level_degenerate"
    return {
        "n_levels": int(n),
        "weights": [float(x) for x in arr.tolist()],
        "min_weight": float(np.min(arr)),
        "max_weight": float(np.max(arr)),
        "top1_mass": float(sorted_desc[0]),
        "top2_mass": float(np.sum(sorted_desc[:2])),
        "tail_mass_excluding_top1": float(1.0 - sorted_desc[0]),
        "hhi": hhi,
        "effective_n_inverse_hhi": inv_hhi,
        "entropy": entropy,
        "entropy_effective_n": entropy_effective,
        "normalized_entropy": float(entropy / math.log(max(n, 2))),
        "low_weight_threshold": low_thr,
        "concentration_threshold": concentration_thr,
        "low_weight_count": low_count,
        "extra_weight_mass_beyond_k4": float(np.sum(extra)) if extra.size else 0.0,
        "max_extra_level_weight": float(np.max(extra)) if extra.size else 0.0,
        "n_extra_levels_under_threshold": n_extra_low,
        "concentration_status": status,
    }


def compute_energy_cluster_diagnostics(
    energies: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    tolerance_eV: float = 0.02,
) -> Dict[str, Any]:
    finite: List[Tuple[int, float]] = []
    for idx, value in enumerate(energies, start=1):
        energy = safe_float(value, float("nan"))
        if np.isfinite(energy):
            finite.append((idx, energy))
    n = len(finite)
    if n == 0:
        return {
            "energy_cluster_status": "missing",
            "cluster_count": 0,
            "clustered_level_indices": [],
            "cluster_centers_eV": [],
            "cluster_spans_eV": [],
            "cluster_weight_mass": [],
            "clustered_level_count": 0,
            "effective_distinct_level_count": 0,
            "cluster_tolerance_eV": float(tolerance_eV),
        }

    if weights is None:
        weight_arr = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    else:
        weight_arr = normalize_weights(weights)
        if len(weight_arr) < n:
            weight_arr = np.pad(weight_arr, (0, n - len(weight_arr)), mode="constant")
            weight_arr = normalize_weights(weight_arr)
    by_index_weight = {idx: float(weight_arr[pos]) for pos, (idx, _) in enumerate(finite)}
    sorted_levels = sorted(finite, key=lambda item: item[1])
    max_span = 2.0 * float(tolerance_eV)
    clusters: List[List[Tuple[int, float]]] = []
    used: set[int] = set()
    i = 0
    while i < len(sorted_levels):
        if sorted_levels[i][0] in used:
            i += 1
            continue
        best_end = i
        for j in range(i + 1, len(sorted_levels)):
            if sorted_levels[j][1] - sorted_levels[i][1] <= max_span + 1e-12:
                best_end = j
            else:
                break
        group = [item for item in sorted_levels[i : best_end + 1] if item[0] not in used]
        if len(group) >= 2:
            clusters.append(group)
            used.update(idx for idx, _ in group)
            i = best_end + 1
        else:
            i += 1

    index_groups = [[int(idx) for idx, _ in group] for group in clusters]
    centers = [float(np.mean([energy for _, energy in group])) for group in clusters]
    spans = [float(max(energy for _, energy in group) - min(energy for _, energy in group)) for group in clusters]
    weight_mass = [float(sum(by_index_weight.get(idx, 0.0) for idx, _ in group)) for group in clusters]
    effective_distinct = n - sum(max(0, len(group) - 1) for group in clusters)
    return {
        "energy_cluster_status": "clustered" if clusters else "separated",
        "cluster_count": int(len(clusters)),
        "clustered_level_indices": index_groups,
        "cluster_centers_eV": centers,
        "cluster_spans_eV": spans,
        "cluster_weight_mass": weight_mass,
        "clustered_level_count": int(sum(len(group) for group in clusters)),
        "effective_distinct_level_count": int(effective_distinct),
        "cluster_tolerance_eV": float(tolerance_eV),
    }


def compute_channel_degeneracy_diagnostics(
    weights: Sequence[float],
    energies: Optional[Sequence[float]] = None,
    n_levels: Optional[int] = None,
    low_weight_threshold: Optional[float] = None,
    concentration_threshold: Optional[float] = None,
    cluster_tolerance_eV: float = 0.02,
) -> Dict[str, Any]:
    arr = normalize_weights(weights)
    n = int(n_levels if n_levels is not None else len(arr))
    low_thr = float(low_weight_threshold if low_weight_threshold is not None else max(0.015, 0.35 / max(n, 1)))
    concentration_thr = float(concentration_threshold if concentration_threshold is not None else max(0.78, 1.0 - 0.60 / max(n, 1)))
    hhi = float(np.sum(np.square(arr)))
    entropy = float(-np.sum(arr * np.log(arr + 1e-12)))
    sorted_desc = np.sort(arr)[::-1]
    low_count = int(np.sum(arr < low_thr))
    reasons: List[str] = []
    if sorted_desc.size and float(sorted_desc[0]) >= concentration_thr:
        reasons.append("top_heavy")
    if low_count > 0:
        reasons.append("low_weight")
    energy_diag = compute_energy_cluster_diagnostics(
        energies or [],
        arr.tolist(),
        tolerance_eV=float(cluster_tolerance_eV),
    )
    if energy_diag["energy_cluster_status"] == "clustered":
        reasons.append("energy_clustered")
    return {
        "n_levels": int(n),
        "weights": [float(x) for x in arr.tolist()],
        "min_weight": float(np.min(arr)),
        "max_weight": float(np.max(arr)),
        "top1_mass": float(sorted_desc[0]) if sorted_desc.size else 0.0,
        "top2_mass": float(np.sum(sorted_desc[:2])) if sorted_desc.size else 0.0,
        "hhi": hhi,
        "effective_n_inverse_hhi": float(1.0 / max(hhi, 1e-12)),
        "entropy": entropy,
        "entropy_effective_n": float(math.exp(entropy)),
        "normalized_entropy": float(entropy / math.log(max(n, 2))),
        "low_weight_threshold": low_thr,
        "concentration_threshold": concentration_thr,
        "low_weight_count": low_count,
        "channel_degeneracy_status": "degenerate" if reasons else "balanced",
        "channel_degeneracy_reasons": reasons,
        **energy_diag,
    }


def _row_weights(row: Dict[str, Any]) -> List[float]:
    if "weights" in row and row["weights"] is not None:
        return [float(x) for x in row["weights"]]
    weights: List[float] = []
    for idx in range(1, int(row.get("n_levels", 0)) + 1):
        key = f"level_weight_{idx:02d}"
        if key in row:
            weights.append(float(row[key]))
    return weights or [1.0]


def _row_energies(row: Dict[str, Any]) -> List[float]:
    if "energies" in row and row["energies"] is not None:
        return [float(x) for x in row["energies"]]
    energies: List[float] = []
    for idx in range(1, int(row.get("n_levels", 0)) + 1):
        key = f"level_energy_{idx:02d}"
        if key in row:
            energies.append(float(row[key]))
    return energies


def _rank_map(rows: Sequence[Dict[str, Any]], key: str, default: float = float("inf")) -> Dict[int, float]:
    values = [(int(row["n_levels"]), safe_float(row.get(key), default)) for row in rows]
    finite = [(k, value) for k, value in values if np.isfinite(value)]
    if not finite:
        return {int(row["n_levels"]): float(len(rows)) for row in rows}
    ordered = sorted(finite, key=lambda item: (item[1], item[0]))
    ranks: Dict[int, float] = {}
    last_value: Optional[float] = None
    last_rank = 0
    for pos, (k, value) in enumerate(ordered, start=1):
        if last_value is None or abs(value - last_value) > 1e-12:
            last_rank = pos
            last_value = value
        ranks[k] = float(last_rank)
    fallback = float(len(rows) + 1)
    for k, _ in values:
        ranks.setdefault(k, fallback)
    return ranks


def _best_k(rows: Sequence[Dict[str, Any]], key: str, default: float = float("inf")) -> int:
    return int(min(rows, key=lambda row: (safe_float(row.get(key), default), int(row["n_levels"])))["n_levels"])


def select_k_neutral_level_decision(
    rows: Sequence[Dict[str, Any]],
    forward_evidence: Optional[Dict[str, Any]] = None,
    cluster_tolerance_eV: float = 0.02,
    use_weight_degeneracy: bool = True,
    use_energy_cluster_degeneracy: bool = True,
    metric_bic_aic_only: bool = False,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No rows supplied for level decision")
    clean: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["n_levels"] = int(item.get("n_levels", item.get("level", 0)))
        weights = _row_weights(item)
        energies = _row_energies(item)
        channel_diag = compute_channel_degeneracy_diagnostics(
            weights,
            energies,
            n_levels=int(item["n_levels"]),
            cluster_tolerance_eV=float(cluster_tolerance_eV),
        )
        energy_diag = compute_energy_cluster_diagnostics(
            energies,
            weights,
            tolerance_eV=float(cluster_tolerance_eV),
        )
        item["channel_degeneracy"] = channel_diag
        item["energy_cluster_diagnostics"] = energy_diag
        clean.append(item)
        channel_rows.append({"n_levels": int(item["n_levels"]), **channel_diag})
        energy_rows.append({"n_levels": int(item["n_levels"]), **energy_diag})

    rank_rmse = _rank_map(clean, "rmse_mean")
    rank_cv = _rank_map(clean, "cv_rmse_mean")
    rank_d1 = _rank_map(clean, "d1_rmse_mean")
    rank_d2 = _rank_map(clean, "d2_rmse_mean")
    rank_structure = _rank_map(clean, "structure_score")
    rank_physical = _rank_map(clean, "vr_physical_response_score")
    rank_bic = _rank_map(clean, "bic")
    rank_aic = _rank_map(clean, "aic")

    scored: List[Dict[str, Any]] = []
    for row in clean:
        k = int(row["n_levels"])
        channel_diag = row["channel_degeneracy"]
        reasons = set(str(x) for x in channel_diag.get("channel_degeneracy_reasons", []))
        weight_degenerate = bool(reasons.intersection({"top_heavy", "low_weight"}))
        energy_clustered = str(channel_diag.get("energy_cluster_status")) == "clustered"
        degeneracy_penalty = 0.0
        if use_weight_degeneracy and weight_degenerate:
            degeneracy_penalty += 1.0
        if use_energy_cluster_degeneracy and energy_clustered:
            degeneracy_penalty += 1.0
        if bool(row.get("flatline_guard_pass")) is False and "flatline_guard_pass" in row:
            degeneracy_penalty += 0.5
        if metric_bic_aic_only:
            composite = (
                rank_rmse[k]
                + rank_cv[k]
                + 1.25 * rank_structure[k]
                + rank_physical[k]
                + rank_bic[k]
                + 0.5 * rank_aic[k]
            )
            applied_penalty = 0.0
        else:
            composite = (
                rank_rmse[k]
                + rank_cv[k]
                + rank_d1[k]
                + rank_d2[k]
                + 1.25 * rank_structure[k]
                + rank_physical[k]
                + rank_bic[k]
                + 0.5 * rank_aic[k]
                + degeneracy_penalty
            )
            applied_penalty = degeneracy_penalty
        scored.append(
            {
                "n_levels": k,
                "composite_rank_score": float(composite),
                "rmse_rank": rank_rmse[k],
                "cv_rank": rank_cv[k],
                "d1_rank": rank_d1[k],
                "d2_rank": rank_d2[k],
                "structure_rank": rank_structure[k],
                "physical_rank": rank_physical[k],
                "bic_rank": rank_bic[k],
                "aic_rank": rank_aic[k],
                "degeneracy_penalty": float(applied_penalty),
                "rmse_mean": safe_float(row.get("rmse_mean"), float("inf")),
                "cv_rmse_mean": safe_float(row.get("cv_rmse_mean", row.get("rmse_mean")), float("inf")),
                "d1_rmse_mean": safe_float(row.get("d1_rmse_mean"), float("inf")),
                "d2_rmse_mean": safe_float(row.get("d2_rmse_mean"), float("inf")),
                "structure_score": safe_float(row.get("structure_score"), float("inf")),
                "vr_physical_response_score": safe_float(row.get("vr_physical_response_score"), float("inf")),
                "bic": safe_float(row.get("bic"), float("inf")),
                "aic": safe_float(row.get("aic"), float("inf")),
                "channel_degeneracy_status": str(channel_diag.get("channel_degeneracy_status")),
                "channel_degeneracy_reasons": list(channel_diag.get("channel_degeneracy_reasons", [])),
                "energy_cluster_status": str(channel_diag.get("energy_cluster_status")),
                "effective_distinct_level_count": int(channel_diag.get("effective_distinct_level_count", k)),
            }
        )
    selected_row = min(scored, key=lambda row: (row["composite_rank_score"], row["n_levels"]))
    selected_k = int(selected_row["n_levels"])
    payload = {
        "decision": "select",
        "selection_policy": "k_neutral_rank_aggregation_bic_aic_complexity",
        "evaluated_levels": sorted(int(row["n_levels"]) for row in clean),
        "selected_k": selected_k,
        "best_k_by_fit": _best_k(clean, "rmse_mean"),
        "best_k_by_cv": _best_k(clean, "cv_rmse_mean"),
        "best_k_by_d1_rmse": _best_k(clean, "d1_rmse_mean"),
        "best_k_by_d2_rmse": _best_k(clean, "d2_rmse_mean"),
        "best_k_by_structure": _best_k(clean, "structure_score"),
        "best_k_by_physical_response": _best_k(clean, "vr_physical_response_score"),
        "best_k_by_bic": _best_k(clean, "bic"),
        "best_k_by_aic": _best_k(clean, "aic"),
        "best_k_by_composite_kneutral": selected_k,
        "best_k_by_composite": selected_k,
        "flatline_guard_pass": bool(all(bool(row.get("flatline_guard_pass", True)) for row in clean)),
        "vr_late_bias_pass": bool(all(bool(row.get("vr_late_bias_pass", True)) for row in clean)),
        "high_vr_valley_depth_pass": bool(all(bool(row.get("high_vr_valley_depth_pass", True)) for row in clean)),
        "scored_levels": scored,
        "channel_degeneracy_summary": channel_rows,
        "energy_cluster_metric_summary": energy_rows,
        "energy_cluster_negative_evidence": [
            row for row in energy_rows if str(row.get("energy_cluster_status")) == "clustered"
        ],
        "structure_metric_summary": [
            {
                "n_levels": int(row["n_levels"]),
                "structure_score": safe_float(row.get("structure_score"), float("inf")),
                "segment_rmse_mean": safe_float(row.get("segment_rmse_mean"), float("inf")),
                "segment_d1_rmse_mean": safe_float(row.get("segment_d1_rmse_mean"), float("inf")),
                "segment_d2_rmse_mean": safe_float(row.get("segment_d2_rmse_mean"), float("inf")),
                "flatline_guard_pass": bool(row.get("flatline_guard_pass", True)),
            }
            for row in clean
        ],
        "physical_response_metric_summary": [
            {
                "n_levels": int(row["n_levels"]),
                "vr_physical_response_score": safe_float(row.get("vr_physical_response_score"), float("inf")),
                "vr_late_bias_pass": bool(row.get("vr_late_bias_pass", True)),
                "high_vr_valley_depth_pass": bool(row.get("high_vr_valley_depth_pass", True)),
            }
            for row in clean
        ],
        "selector_ablation_flags": {
            "use_weight_degeneracy": bool(use_weight_degeneracy),
            "use_energy_cluster_degeneracy": bool(use_energy_cluster_degeneracy),
            "metric_bic_aic_only": bool(metric_bic_aic_only),
        },
    }
    if forward_evidence is not None:
        consistency_rows = [compute_forward_reverse_consistency(row, forward_evidence) for row in clean]
        statuses = [str(row.get("status")) for row in consistency_rows]
        payload["forward_reverse_consistency"] = consistency_rows
        payload["forward_reverse_consistency_summary"] = {
            "soft_constraint_only": True,
            "status_counts": {status: int(statuses.count(status)) for status in sorted(set(statuses))},
            "forward_diagnostic_status": str((forward_evidence.get("global") or {}).get("diagnostic_status", "missing")),
            "forward_confidence": safe_float((forward_evidence.get("global") or {}).get("confidence"), 0.0),
            "forward_main_spacing": safe_float((forward_evidence.get("global") or {}).get("main_spacing"), float("nan")),
        }
    return payload


def select_adaptive_level_decision(
    rows: Sequence[Dict[str, Any]],
    reference_k: int = 4,
    forward_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No rows supplied for level decision")
    clean = [dict(row) for row in rows]
    rmses = np.asarray([safe_float(row.get("rmse_mean")) for row in clean], dtype=np.float64)
    structure_scores = np.asarray([safe_float(row.get("structure_score")) for row in clean], dtype=np.float64)
    finite_rmse = rmses[np.isfinite(rmses)]
    finite_structure = structure_scores[np.isfinite(structure_scores)]
    low_thr = 0.04
    if finite_rmse.size >= 2:
        rmse_span = max(float(np.max(finite_rmse) - np.min(finite_rmse)), 1e-12)
        marginal_floor = max(0.002, 0.05 * rmse_span)
    else:
        marginal_floor = 0.002
    if finite_structure.size >= 2:
        structure_span = max(float(np.max(finite_structure) - np.min(finite_structure)), 1e-12)
        structure_floor = max(0.002, 0.05 * structure_span)
    else:
        structure_floor = 0.002

    diagnostics: List[Dict[str, Any]] = []
    by_k: Dict[int, Dict[str, Any]] = {}
    for row in clean:
        k = int(row.get("n_levels", row.get("level", 0)))
        diag = compute_weight_diagnostics(
            _row_weights(row),
            n_levels=k,
            reference_k=reference_k,
            low_weight_threshold=low_thr,
        )
        row["n_levels"] = k
        row["weight_diagnostics"] = diag
        by_k[k] = row
        diagnostics.append({"n_levels": k, **diag})

    best_fit = min(clean, key=lambda r: safe_float(r.get("rmse_mean"), float("inf")))
    best_cv = min(clean, key=lambda r: safe_float(r.get("cv_rmse_mean", r.get("rmse_mean")), float("inf")))
    best_bic = min(clean, key=lambda r: safe_float(r.get("bic"), float("inf")))
    best_structure = min(clean, key=lambda r: safe_float(r.get("structure_score"), float("inf")))
    best_physical = min(clean, key=lambda r: safe_float(r.get("vr_physical_response_score"), float("inf")))
    ref = by_k.get(int(reference_k))
    ref_rmse = safe_float((ref or {}).get("rmse_mean"), float("inf"))
    ref_cv = safe_float((ref or {}).get("cv_rmse_mean", (ref or {}).get("rmse_mean")), float("inf"))
    ref_structure = safe_float((ref or {}).get("structure_score"), float("inf"))
    ref_diag = (ref or {}).get("weight_diagnostics", {})
    k_gt_ref: List[Dict[str, Any]] = []
    for row in clean:
        k = int(row["n_levels"])
        if k <= reference_k:
            continue
        diag = row["weight_diagnostics"]
        fit_delta_vs_ref = ref_rmse - safe_float(row.get("rmse_mean"), float("inf"))
        cv_delta_vs_ref = ref_cv - safe_float(row.get("cv_rmse_mean", row.get("rmse_mean")), float("inf"))
        structure_delta_vs_ref = ref_structure - safe_float(row.get("structure_score"), float("inf"))
        negative = bool(
            diag["concentration_status"] == "extra_level_degenerate"
            and fit_delta_vs_ref <= marginal_floor
            and cv_delta_vs_ref <= marginal_floor
        )
        structure_negative = bool(negative and structure_delta_vs_ref <= structure_floor)
        k_gt_ref.append(
            {
                "n_levels": k,
                "fit_delta_vs_k4": float(fit_delta_vs_ref),
                "cv_delta_vs_k4": float(cv_delta_vs_ref),
                "structure_delta_vs_k4": float(structure_delta_vs_ref),
                "bic_delta_vs_k4": float(safe_float(row.get("bic"), float("inf")) - safe_float((ref or {}).get("bic"), float("inf"))),
                "negative_evidence_label": "extra_level_unsupported" if negative else "not_negative",
                "structure_negative_evidence_label": "extra_level_structure_unsupported" if structure_negative else "structure_improved_or_not_degenerate",
                **diag,
            }
        )
    negative_pass = bool(k_gt_ref and all(item["negative_evidence_label"] == "extra_level_unsupported" for item in k_gt_ref))
    structure_negative_pass = bool(k_gt_ref and all(item["structure_negative_evidence_label"] == "extra_level_structure_unsupported" for item in k_gt_ref))

    scored: List[Dict[str, Any]] = []
    for row in clean:
        k = int(row["n_levels"])
        rmse = safe_float(row.get("rmse_mean"), float("inf"))
        cv = safe_float(row.get("cv_rmse_mean", row.get("rmse_mean")), float("inf"))
        bic = safe_float(row.get("bic"), float("inf"))
        structure_score = safe_float(row.get("structure_score"), float("inf"))
        physical_score = safe_float(row.get("vr_physical_response_score"), float("inf"))
        diag = row["weight_diagnostics"]
        complexity_penalty = 0.01 * abs(k - reference_k)
        degeneracy_penalty = 0.25 if diag["concentration_status"] in {"top_heavy", "extra_level_degenerate"} else 0.0
        if k > reference_k and negative_pass:
            degeneracy_penalty += 0.25
        structure_term = 0.10 * structure_score if np.isfinite(structure_score) else 0.0
        physical_term = 0.08 * physical_score if np.isfinite(physical_score) else 0.0
        flatline_penalty = 0.35 if row.get("flatline_guard_pass") is False else 0.0
        score = rmse + 0.35 * cv + 0.0005 * bic + structure_term + physical_term + flatline_penalty + complexity_penalty + degeneracy_penalty
        scored.append({"n_levels": k, "composite_score": float(score), "rmse_mean": rmse, "cv_rmse_mean": cv, "bic": bic, "structure_score": structure_score, "vr_physical_response_score": physical_score})
    best_composite = min(scored, key=lambda row: row["composite_score"])
    if ref is not None and negative_pass and str(ref_diag.get("concentration_status")) == "balanced":
        best_k = int(reference_k)
    else:
        best_k = int(best_composite["n_levels"])
    decision = "promote" if best_k == int(reference_k) and negative_pass else "review"
    payload = {
        "decision": decision,
        "evaluated_levels": sorted(int(row["n_levels"]) for row in clean),
        "best_k_by_fit": int(best_fit["n_levels"]),
        "best_k_by_cv": int(best_cv["n_levels"]),
        "best_k_by_bic": int(best_bic["n_levels"]),
        "best_k_by_structure": int(best_structure["n_levels"]),
        "best_k_by_physical_response": int(best_physical["n_levels"]),
        "best_k_by_composite": best_k,
        "adaptive_marginal_rmse_floor": float(marginal_floor),
        "adaptive_structure_floor": float(structure_floor),
        "k4_concentration_status": str(ref_diag.get("concentration_status", "missing")),
        "k4_structure_status": "structure_preserved" if bool((ref or {}).get("flatline_guard_pass", False)) else "flatline_or_missing",
        "k_gt4_negative_evidence_pass": negative_pass,
        "k_gt4_structure_negative_evidence_pass": structure_negative_pass,
        "flatline_guard_pass": bool(all(bool(row.get("flatline_guard_pass", False)) for row in clean)),
        "vr_late_bias_pass": bool(all(bool(row.get("vr_late_bias_pass", False)) for row in clean)),
        "high_vr_valley_depth_pass": bool(all(bool(row.get("high_vr_valley_depth_pass", False)) for row in clean)),
        "kernel_ablation_improvement": "not_evaluated",
        "structure_metric_summary": [
            {
                "n_levels": int(row["n_levels"]),
                "structure_score": safe_float(row.get("structure_score"), float("inf")),
                "segment_rmse_mean": safe_float(row.get("segment_rmse_mean"), float("inf")),
                "segment_d1_rmse_mean": safe_float(row.get("segment_d1_rmse_mean"), float("inf")),
                "segment_d2_rmse_mean": safe_float(row.get("segment_d2_rmse_mean"), float("inf")),
                "flatline_guard_pass": bool(row.get("flatline_guard_pass", False)),
            }
            for row in clean
        ],
        "physical_response_metric_summary": [
            {
                "n_levels": int(row["n_levels"]),
                "vr_physical_response_score": safe_float(row.get("vr_physical_response_score"), float("inf")),
                "vr_late_bias_pass": bool(row.get("vr_late_bias_pass", False)),
                "high_vr_valley_depth_pass": bool(row.get("high_vr_valley_depth_pass", False)),
            }
            for row in clean
        ],
        "scored_levels": scored,
        "level_weight_diagnostics": diagnostics,
        "extra_level_negative_evidence": k_gt_ref,
    }
    if forward_evidence is not None:
        consistency_rows = [compute_forward_reverse_consistency(row, forward_evidence) for row in clean]
        by_level = {int(row["n_levels"]): row for row in consistency_rows}
        statuses = [str(row.get("status")) for row in consistency_rows]
        payload["forward_reverse_consistency"] = consistency_rows
        payload["k4_forward_reverse_status"] = str(by_level.get(int(reference_k), {}).get("status", "missing"))
        payload["forward_reverse_consistency_summary"] = {
            "soft_constraint_only": True,
            "status_counts": {status: int(statuses.count(status)) for status in sorted(set(statuses))},
            "forward_diagnostic_status": str((forward_evidence.get("global") or {}).get("diagnostic_status", "missing")),
            "forward_confidence": safe_float((forward_evidence.get("global") or {}).get("confidence"), 0.0),
            "forward_main_spacing": safe_float((forward_evidence.get("global") or {}).get("main_spacing"), float("nan")),
        }
        if payload["decision"] == "promote" and payload["k4_forward_reverse_status"] == "low_forward_confidence":
            payload["decision_note"] = "reverse_model_promotes_k4_but_forward_evidence_is_low_confidence"
    return payload


def summarize_seed_outputs(level_dir: Path, level: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed_dir in sorted((level_dir / "seeds").glob("seed_*")):
        full_dir = seed_dir / "full"
        scorecard_path = full_dir / "scorecard.json"
        if not scorecard_path.exists():
            continue
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        metrics = scorecard.get("metrics_curve", {})
        structure_summary = ((metrics.get("structure_metrics") or {}).get("summary") or {})
        params = scorecard.get("params", {})
        consistency = scorecard.get("forward_reverse_consistency") or {}
        weights = [
            safe_float(params.get(f"level_weight_{idx:02d}"))
            for idx in range(1, int(level) + 1)
            if f"level_weight_{idx:02d}" in params
        ]
        energies = [
            safe_float(params.get(f"level_energy_{idx:02d}"))
            for idx in range(1, int(level) + 1)
            if f"level_energy_{idx:02d}" in params
        ]
        energy_diag = compute_energy_cluster_diagnostics(energies, weights)
        channel_diag = compute_channel_degeneracy_diagnostics(weights, energies, n_levels=int(level))
        row = {
            "n_levels": int(level),
            "seed": int(seed_dir.name.split("_")[-1]),
            "rmse_mean": safe_float(metrics.get("rmse_mean")),
            "mae_mean": safe_float(metrics.get("mae_mean")),
            "d1_rmse_mean": safe_float(metrics.get("d1_rmse_mean")),
            "d2_rmse_mean": safe_float(metrics.get("d2_rmse_mean")),
            "aic": safe_float(metrics.get("aic")),
            "bic": safe_float(metrics.get("bic")),
            "structure_score": safe_float(metrics.get("structure_score"), float("inf")),
            "vr_physical_response_score": safe_float(metrics.get("vr_physical_response_score"), float("inf")),
            "vr_late_bias_pass": bool(metrics.get("vr_late_bias_pass", False)),
            "high_vr_valley_depth_pass": bool(metrics.get("high_vr_valley_depth_pass", False)),
            "flatline_guard_pass": bool(metrics.get("flatline_guard_pass", False)),
            "segment_rmse_mean": safe_float(structure_summary.get("segment_rmse_mean"), float("inf")),
            "segment_d1_rmse_mean": safe_float(structure_summary.get("segment_d1_rmse_mean"), float("inf")),
            "segment_d2_rmse_mean": safe_float(structure_summary.get("segment_d2_rmse_mean"), float("inf")),
            "structure_segment_count": int(structure_summary.get("segment_count", 0)),
            "n_params": int(metrics.get("n_params", 0)),
            "sse_total": safe_float(metrics.get("sse_total")),
            "n_points": int(metrics.get("n_points", 0)),
            "weights": weights,
            "energies": energies,
            "energy_cluster_status": str(energy_diag.get("energy_cluster_status")),
            "cluster_count": int(energy_diag.get("cluster_count", 0)),
            "effective_distinct_level_count": int(energy_diag.get("effective_distinct_level_count", int(level))),
            "channel_degeneracy_status": str(channel_diag.get("channel_degeneracy_status")),
            "channel_degeneracy_reasons": list(channel_diag.get("channel_degeneracy_reasons", [])),
            "scorecard_path": str(scorecard_path),
            "V_exc": safe_float(params.get("V_exc")),
            "V_exc_weighted": safe_float(params.get("V_exc_weighted")),
            "effective_excitation_mean": safe_float(params.get("effective_excitation_mean")),
            "effective_excitation_std": safe_float(params.get("effective_excitation_std")),
            "damping": safe_float(params.get("damping")),
            "osc_amp": safe_float(params.get("osc_amp")),
            "phase": safe_float(params.get("phase")),
            "forward_reverse_status": str(consistency.get("status", "not_evaluated")),
        }
        for idx in range(1, int(level) + 1):
            key = f"level_energy_{idx:02d}"
            if key in params:
                row[key] = safe_float(params.get(key))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No seed outputs for L{level:02d}")
    best = min(rows, key=lambda row: row["rmse_mean"])
    rmse_values = [row["rmse_mean"] for row in rows if np.isfinite(row["rmse_mean"])]
    return {
        **best,
        "seed_count": int(len(rows)),
        "rmse_seed_mean": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "rmse_seed_std": float(np.std(rmse_values)) if len(rmse_values) > 1 else 0.0,
        "rmse_seed_iqr": float(np.percentile(rmse_values, 75) - np.percentile(rmse_values, 25)) if rmse_values else float("nan"),
    }


def write_scan_tables(out_dir: Path, summaries: Sequence[Dict[str, Any]], decision: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    weight_rows = []
    structure_summary_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    vr_response_rows: List[Dict[str, Any]] = []
    vr_response_class_rows: List[Dict[str, Any]] = []
    for row in summaries:
        diag = compute_weight_diagnostics(row.get("weights", []), n_levels=int(row["n_levels"]))
        channel_diag = compute_channel_degeneracy_diagnostics(
            row.get("weights", []),
            row.get("energies", []),
            n_levels=int(row["n_levels"]),
        )
        summary_rows.append({k: v for k, v in row.items() if k not in {"weights", "energies"}})
        weight_rows.append({"n_levels": int(row["n_levels"]), "seed": int(row["seed"]), **diag})
    for scorecard_path in sorted((out_dir / "levels").glob("L*/seeds/seed_*/full/scorecard.json")):
        try:
            level = int(scorecard_path.parents[3].name.replace("L", ""))
            seed = int(scorecard_path.parents[1].name.replace("seed_", ""))
            scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        except (IndexError, ValueError, json.JSONDecodeError, OSError):
            continue
        structure = ((scorecard.get("metrics_curve") or {}).get("structure_metrics") or {})
        vr_response = ((scorecard.get("metrics_curve") or {}).get("vr_physical_response") or {})
        if structure.get("summary"):
            structure_summary_rows.append({"n_levels": level, "seed": seed, **structure.get("summary", {})})
        for segment in structure.get("segments", []):
            segment_rows.append({"n_levels": level, "seed": seed, **segment})
        for curve in structure.get("per_curve", []):
            curve_rows.append({"n_levels": level, "seed": seed, **curve})
        per_class = structure.get("per_class") or []
        if isinstance(per_class, dict):
            for curve_class, stats in per_class.items():
                class_rows.append({"n_levels": level, "seed": seed, "curve_class": curve_class, **stats})
        else:
            for stats in per_class:
                if isinstance(stats, dict):
                    class_rows.append({"n_levels": level, "seed": seed, **stats})
        for row in vr_response.get("per_curve", []):
            if isinstance(row, dict):
                vr_response_rows.append({"n_levels": level, "seed": seed, **row})
        for row in vr_response.get("per_class", []):
            if isinstance(row, dict):
                vr_response_class_rows.append({"n_levels": level, "seed": seed, **row})
    pd.DataFrame(summary_rows).to_csv(out_dir / "scan_summary.csv", index=False)
    pd.DataFrame(decision.get("scored_levels", [])).to_csv(out_dir / "model_selection_table.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out_dir / "level_weight_diagnostics.csv", index=False)
    pd.DataFrame(decision.get("channel_degeneracy_summary", [])).to_csv(out_dir / "channel_degeneracy_summary.csv", index=False)
    pd.DataFrame(decision.get("energy_cluster_metric_summary", [])).to_csv(out_dir / "energy_cluster_table.csv", index=False)
    pd.DataFrame(decision.get("extra_level_negative_evidence", [])).to_csv(out_dir / "extra_level_negative_evidence.csv", index=False)
    pd.DataFrame(decision.get("forward_reverse_consistency", [])).to_csv(out_dir / "forward_reverse_consistency.csv", index=False)
    pd.DataFrame(structure_summary_rows).to_csv(out_dir / "structure_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(out_dir / "peak_valley_segments.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "curve_structure_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / "class_structure_summary.csv", index=False)
    pd.DataFrame(vr_response_rows).to_csv(out_dir / "vr_physical_response.csv", index=False)
    safe_json_dump({"levels": list(summaries)}, out_dir / "scan_summary.json")
    safe_json_dump(decision, out_dir / "decision.json")
    safe_json_dump({"weight_diagnostics": weight_rows}, out_dir / "level_weight_diagnostics.json")
    safe_json_dump({"channel_degeneracy_summary": decision.get("channel_degeneracy_summary", [])}, out_dir / "channel_degeneracy_summary.json")
    safe_json_dump({"energy_cluster_metric_summary": decision.get("energy_cluster_metric_summary", [])}, out_dir / "energy_cluster_table.json")
    safe_json_dump({"extra_level_negative_evidence": decision.get("extra_level_negative_evidence", [])}, out_dir / "extra_level_negative_evidence.json")
    safe_json_dump({"forward_reverse_consistency": decision.get("forward_reverse_consistency", [])}, out_dir / "forward_reverse_consistency.json")
    safe_json_dump(
        {
            "structure_metric_summary": decision.get("structure_metric_summary", []),
            "all_seed_structure_summary": structure_summary_rows,
            "curve_structure_summary": curve_rows,
            "class_structure_summary": class_rows,
            "vr_physical_response_summary": decision.get("physical_response_metric_summary", []),
            "vr_physical_response_curve_rows": vr_response_rows,
            "vr_physical_response_class_rows": vr_response_class_rows,
        },
        out_dir / "structure_metrics.json",
    )
    safe_json_dump(
        {
            "physical_response_metric_summary": decision.get("physical_response_metric_summary", []),
            "per_curve": vr_response_rows,
            "per_class": vr_response_class_rows,
        },
        out_dir / "vr_physical_response.json",
    )


def copy_best_level_artifact(scan_dir: Path, level: int, target_dir: Path) -> None:
    source_level = scan_dir / "levels" / f"L{level:02d}" / "seeds"
    candidates: List[Tuple[float, Path]] = []
    for seed_dir in sorted(source_level.glob("seed_*")):
        full = seed_dir / "full"
        metrics_path = full / "metrics_best.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            candidates.append((safe_float(metrics.get("rmse_mean"), float("inf")), full))
    if not candidates:
        return
    _, best_dir = min(candidates, key=lambda item: item[0])
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(best_dir, target_dir)


def build_hyperopt_candidates(cfg: Config) -> List[Dict[str, Any]]:
    return [
        {"name": "baseline", "updates": {}},
        {
            "name": "structure_guard",
            "updates": {
                "w_d1": float(cfg.w_d1) * 1.25,
                "w_d2": max(float(cfg.w_d2), 0.03),
                "w_peak_window": max(float(cfg.w_peak_window), 0.12),
                "w_smooth": float(cfg.w_smooth) * 0.5,
                "w_vr_late_bias": max(float(cfg.w_vr_late_bias), 0.06),
                "w_high_vr_valley_depth": max(float(cfg.w_high_vr_valley_depth), 0.08),
            },
        },
        {
            "name": "low_smooth_peak",
            "updates": {
                "w_d1": float(cfg.w_d1),
                "w_d2": max(float(cfg.w_d2), 0.015),
                "w_peak_window": max(float(cfg.w_peak_window), 0.10),
                "w_smooth": float(cfg.w_smooth) * 0.25,
                "lr": float(cfg.lr) * 0.85,
            },
        },
        {
            "name": "d2_stronger",
            "updates": {
                "w_d2": max(float(cfg.w_d2) * 1.75, 0.035),
                "w_peak_window": max(float(cfg.w_peak_window), 0.10),
            },
        },
        {
            "name": "d1_d2_balanced",
            "updates": {
                "w_d1": float(cfg.w_d1) * 1.15,
                "w_d2": max(float(cfg.w_d2) * 1.25, 0.025),
                "w_smooth": float(cfg.w_smooth) * 0.6,
            },
        },
        {
            "name": "low_smooth_high_peak",
            "updates": {
                "w_peak_window": max(float(cfg.w_peak_window) * 1.75, 0.14),
                "w_smooth": float(cfg.w_smooth) * 0.15,
                "lr": float(cfg.lr) * 0.75,
            },
        },
        {
            "name": "vr_response_guard",
            "updates": {
                "w_vr_late_bias": max(float(cfg.w_vr_late_bias) * 1.75, 0.08),
                "w_vr_amplitude_ratio": max(float(cfg.w_vr_amplitude_ratio) * 1.75, 0.08),
                "w_high_vr_valley_depth": max(float(cfg.w_high_vr_valley_depth) * 1.75, 0.10),
                "w_smooth": float(cfg.w_smooth) * 0.4,
            },
        },
        {
            "name": "vr_bias_stronger",
            "updates": {
                "w_vr_late_bias": max(float(cfg.w_vr_late_bias) * 2.4, 0.10),
                "w_vr_amplitude_ratio": max(float(cfg.w_vr_amplitude_ratio), 0.05),
                "lr": float(cfg.lr) * 0.85,
            },
        },
        {
            "name": "prior_anchor_stronger",
            "updates": {
                "w_prior_anchor": float(cfg.w_prior_anchor) * 1.8,
                "w_prior_gap": float(cfg.w_prior_gap) * 1.4,
            },
        },
        {
            "name": "prior_gap_stronger",
            "updates": {
                "w_prior_gap": float(cfg.w_prior_gap) * 2.2,
                "min_level_gap": max(float(cfg.min_level_gap), 0.05),
            },
        },
        {
            "name": "lr_lower_patience",
            "updates": {
                "lr": float(cfg.lr) * 0.65,
                "early_stop_patience": int(max(int(cfg.early_stop_patience), 70)),
            },
        },
        {
            "name": "lr_higher_guarded",
            "updates": {
                "lr": float(cfg.lr) * 1.25,
                "grad_clip": max(float(cfg.grad_clip), 4.0),
                "w_smooth": float(cfg.w_smooth) * 0.5,
            },
        },
    ][: max(1, int(cfg.hyperopt_trials))]


def apply_config_updates(cfg: Config, updates: Dict[str, Any]) -> Config:
    values = asdict(cfg)
    for key, value in updates.items():
        if key in values:
            values[key] = value
    return Config(**values)


def rank_hyperopt_candidates(candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        scores = [
            safe_float(row.get("score"), float("nan"))
            for row in candidate.get("stage_scores", [])
            if np.isfinite(safe_float(row.get("score"), float("nan")))
        ]
        if scores:
            mean_score = float(np.mean(scores))
        else:
            mean_score = safe_float(candidate.get("mean_score"), float("inf"))
        payload = dict(candidate)
        payload["mean_score"] = mean_score
        ranked.append(payload)
    ranked.sort(key=lambda row: (safe_float(row.get("mean_score"), float("inf")), str(row.get("name", ""))))
    return ranked[: max(1, int(top_k))]


def hyperopt_score_from_metrics(metrics: Dict[str, Any]) -> float:
    score = safe_float(metrics.get("rmse_mean"), float("inf"))
    score += 0.10 * safe_float(metrics.get("structure_score"), 0.0)
    score += 0.08 * safe_float(metrics.get("vr_physical_response_score"), 0.0)
    score += 0.03 * safe_float(metrics.get("d1_rmse_mean"), 0.0)
    score += 0.03 * safe_float(metrics.get("d2_rmse_mean"), 0.0)
    if not bool(metrics.get("flatline_guard_pass", False)):
        score += 0.35
    return float(score)


def run_hyperopt_stage(
    cfg: Config,
    tmp_root: Path,
    stage_name: str,
    candidates: Sequence[Dict[str, Any]],
    levels: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        candidate_cfg = apply_config_updates(cfg, dict(candidate.get("updates", {})))
        candidate_cfg.hyperopt_enabled = False
        candidate_cfg.retain_promoted_level_copies = False
        candidate_cfg.epochs = int(epochs)
        candidate_cfg.scan_seeds = ",".join(str(int(seed)) for seed in seeds)
        candidate_cfg.device = "cpu"
        candidate_cfg.selected_device = "cpu"
        candidate_cfg.dispatch_strategy = "single"
        candidate_cfg.run_stage = "hyperopt"
        candidate_cfg.hyperopt_stage = str(stage_name)
        rows: List[Dict[str, Any]] = []
        for level in levels:
            for seed in seeds:
                trial_dir = tmp_root / stage_name / f"trial_{idx:02d}_{candidate['name']}" / f"L{int(level):02d}" / f"seed_{int(seed):03d}"
                scorecard = run_level_fit(
                    candidate_cfg,
                    n_levels=int(level),
                    seed=int(seed),
                    out_dir=trial_dir,
                    epochs=int(epochs),
                    device="cpu",
                )
                metrics = scorecard.get("metrics_curve", {})
                score = hyperopt_score_from_metrics(metrics)
                rows.append(
                    {
                        "stage": str(stage_name),
                        "candidate": str(candidate["name"]),
                        "n_levels": int(level),
                        "seed": int(seed),
                        "score": float(score),
                        "rmse_mean": safe_float(metrics.get("rmse_mean")),
                        "structure_score": safe_float(metrics.get("structure_score")),
                        "vr_physical_response_score": safe_float(metrics.get("vr_physical_response_score")),
                        "d1_rmse_mean": safe_float(metrics.get("d1_rmse_mean")),
                        "d2_rmse_mean": safe_float(metrics.get("d2_rmse_mean")),
                        "flatline_guard_pass": bool(metrics.get("flatline_guard_pass", False)),
                        "epochs_completed": int(scorecard.get("epochs_completed", 0)),
                    }
                )
        next_candidate = dict(candidate)
        next_candidate.setdefault("stage_scores", [])
        next_candidate.setdefault("stage_results", [])
        next_candidate["stage_scores"] = list(next_candidate["stage_scores"]) + rows
        next_candidate["stage_results"] = list(next_candidate["stage_results"]) + [
            {
                "stage": str(stage_name),
                "epochs": int(epochs),
                "levels": [int(level) for level in levels],
                "seeds": [int(seed) for seed in seeds],
                "mean_score": float(np.mean([row["score"] for row in rows])) if rows else float("inf"),
                "rows": rows,
            }
        ]
        completed.append(next_candidate)
    ranked = rank_hyperopt_candidates(completed, top_k=top_k)
    return ranked, completed, {
        "stage": str(stage_name),
        "epochs": int(epochs),
        "levels": [int(level) for level in levels],
        "seeds": [int(seed) for seed in seeds],
        "top_k": int(top_k),
        "ranked": [
            {
                "name": str(row.get("name")),
                "mean_score": safe_float(row.get("mean_score")),
                "updates": dict(row.get("updates", {})),
            }
            for row in rank_hyperopt_candidates(completed, top_k=len(completed))
        ],
    }


def run_hyperopt(cfg: Config, out_dir: Path) -> Tuple[Config, Dict[str, Any]]:
    if not bool(cfg.hyperopt_enabled):
        summary = {"enabled": False, "selected": {"name": "disabled", "updates": {}}, "trials": []}
        safe_json_dump(summary, out_dir / "hyperopt_summary.json")
        pd.DataFrame([]).to_csv(out_dir / "hyperopt_summary.csv", index=False)
        return cfg, summary

    tmp_root = out_dir / "_hyperopt_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"name": str(candidate["name"]), "updates": dict(candidate.get("updates", {})), "stage_scores": [], "stage_results": []}
        for candidate in build_hyperopt_candidates(cfg)
    ]
    stage_payloads: List[Dict[str, Any]] = []
    all_by_name: Dict[str, Dict[str, Any]] = {str(candidate["name"]): dict(candidate) for candidate in candidates}
    try:
        active, completed, stage_payload = run_hyperopt_stage(
            cfg,
            tmp_root,
            "stage1_short",
            candidates,
            [level for level in parse_int_list(cfg.hyperopt_stage1_levels) if 1 <= level <= 8],
            parse_int_list(cfg.hyperopt_stage1_seeds),
            int(cfg.hyperopt_stage1_epochs),
            int(cfg.hyperopt_stage1_top_k),
        )
        stage_payloads.append(stage_payload)
        all_by_name.update({str(candidate["name"]): candidate for candidate in completed})
        active, completed, stage_payload = run_hyperopt_stage(
            cfg,
            tmp_root,
            "stage2_mid",
            active,
            [level for level in parse_int_list(cfg.hyperopt_stage2_levels) if 1 <= level <= 8],
            parse_int_list(cfg.hyperopt_stage2_seeds),
            int(cfg.hyperopt_stage2_epochs),
            int(cfg.hyperopt_stage2_top_k),
        )
        stage_payloads.append(stage_payload)
        all_by_name.update({str(candidate["name"]): candidate for candidate in completed})
        active, completed, stage_payload = run_hyperopt_stage(
            cfg,
            tmp_root,
            "stage3_confirm",
            active,
            [level for level in parse_int_list(cfg.hyperopt_stage3_levels) if 1 <= level <= 8],
            parse_int_list(cfg.hyperopt_stage3_seeds),
            int(cfg.hyperopt_stage3_epochs),
            max(1, int(cfg.hyperopt_top_k)),
        )
        stage_payloads.append(stage_payload)
        all_by_name.update({str(candidate["name"]): candidate for candidate in completed})
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    trials = rank_hyperopt_candidates(list(all_by_name.values()), top_k=len(all_by_name))
    selected = rank_hyperopt_candidates(active, top_k=1)[0] if active else {"name": "baseline", "updates": {}}
    selected_cfg = apply_config_updates(cfg, dict(selected.get("updates", {})))
    summary = {
        "enabled": True,
        "method": "successive_halving_production",
        "short_epochs": int(cfg.hyperopt_stage1_epochs),
        "stages": stage_payloads,
        "selected": selected,
        "trials": trials,
    }
    safe_json_dump(summary, out_dir / "hyperopt_summary.json")
    pd.DataFrame(
        [
            {
                "name": t["name"],
                "mean_score": t["mean_score"],
                "stage_count": len(t.get("stage_results", [])),
                "score_count": len(t.get("stage_scores", [])),
                **{f"update_{k}": v for k, v in t.get("updates", {}).items()},
            }
            for t in trials
        ]
    ).to_csv(out_dir / "hyperopt_summary.csv", index=False)
    return selected_cfg, summary


def _scorecard_is_current(full_dir: Path, cfg: Config) -> bool:
    scorecard_path = full_dir / "scorecard.json"
    if not scorecard_path.exists() or not (full_dir / "checkpoint_best.pt").exists():
        return False
    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if scorecard.get("model_schema") != MODEL_SCHEMA:
        return False
    if int(scorecard.get("epochs_requested", -1)) != int(cfg.epochs):
        return False
    early_stop = scorecard.get("early_stop") or {}
    if int(early_stop.get("min_epochs", -1)) != int(cfg.early_stop_min_epochs):
        return False
    if str(scorecard.get("config_hash", "")) != config_hash(cfg):
        return False
    return True


def checkpoint_matches_config(path: Path, cfg: Config) -> bool:
    if not Path(path).exists():
        return False
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return str(payload.get("config_hash", "")) == config_hash(cfg)


def _run_level_fit_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    cfg = Config(**payload["cfg"])
    out_dir = Path(payload["out_dir"])
    resume_ckpt = Path(payload["resume_ckpt"]) if payload.get("resume_ckpt") else None
    result = run_level_fit(
        cfg,
        n_levels=int(payload["level"]),
        seed=int(payload["seed"]),
        out_dir=out_dir,
        epochs=int(payload["epochs"]),
        device=str(payload["device"]),
        resume_ckpt=resume_ckpt,
    )
    return {"n_levels": int(payload["level"]), "seed": int(payload["seed"]), "out_dir": str(out_dir), "rmse_mean": safe_float((result.get("metrics_curve") or {}).get("rmse_mean"))}


def run_level_scan(cfg: Config) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_evidence = load_forward_evidence_for_config(cfg)
    if str(getattr(cfg, "forward_prior_mode", "off")) == "auto" and forward_evidence is None:
        forward_evidence = build_forward_evidence(cfg)
        priors = compile_forward_priors(forward_evidence, cfg)
        cfg = apply_forward_priors_to_config(cfg, priors)
        cfg.forward_evidence_path = str(out_dir / "forward_evidence.json")
        safe_json_dump(forward_evidence, out_dir / "forward_evidence.json")
        safe_json_dump(priors, out_dir / "forward_priors.json")
    cfg, hyperopt_summary = run_hyperopt(cfg, out_dir)
    levels = [int(cfg.fixed_levels)] if int(cfg.fixed_levels) > 0 else list(range(int(cfg.level_scan_min), int(cfg.level_scan_max) + 1))
    levels = [level for level in levels if 1 <= level <= 8]
    seeds = parse_int_list(cfg.scan_seeds)
    device = resolve_device(cfg.selected_device if cfg.selected_device != "auto" else cfg.device)
    config_payload = asdict(cfg)
    config_payload["resolved_device"] = str(device)
    safe_json_dump(config_payload, out_dir / "config_used.json")

    summaries: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    for level in levels:
        level_dir = out_dir / "levels" / f"L{level:02d}"
        for seed in seeds:
            full_dir = level_dir / "seeds" / f"seed_{seed:03d}" / "full"
            checkpoint_last = full_dir / "checkpoint_last.pt"
            resume_ckpt = checkpoint_last if str(cfg.resume_mode) == "auto" and checkpoint_matches_config(checkpoint_last, cfg) else None
            if str(cfg.resume_mode) == "auto" and _scorecard_is_current(full_dir, cfg):
                continue
            jobs.append(
                {
                    "cfg": asdict(cfg),
                    "level": int(level),
                    "seed": int(seed),
                    "out_dir": str(full_dir),
                    "epochs": int(cfg.epochs),
                    "device": str(device),
                    "resume_ckpt": str(resume_ckpt) if resume_ckpt is not None else "",
                }
            )
    use_parallel = (
        str(cfg.dispatch_strategy) == "cpu_4"
        and str(device) == "cpu"
        and len(jobs) > 1
        and int(cfg.cpu_workers) > 1
    )
    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(cfg.cpu_workers))) as pool:
            for _ in pool.map(_run_level_fit_worker, jobs):
                pass
    else:
        for job in jobs:
            _run_level_fit_worker(job)

    for level in levels:
        level_dir = out_dir / "levels" / f"L{level:02d}"
        summaries.append(summarize_seed_outputs(level_dir, level))
    for row in summaries:
        row["cv_rmse_mean"] = row["rmse_seed_mean"]
    decision = select_k_neutral_level_decision(
        summaries,
        forward_evidence=forward_evidence,
        cluster_tolerance_eV=float(cfg.cluster_tolerance_eV),
    )
    write_scan_tables(out_dir, summaries, decision)
    is_retained_sweep = out_dir.name == "fullscan" and out_dir.parent.name == "main"
    if bool(cfg.retain_promoted_level_copies) and is_retained_sweep:
        copy_best_level_artifact(out_dir, 1, out_dir.parent / "k1_full")
        copy_best_level_artifact(out_dir, int(decision.get("selected_k", 1)), out_dir.parent / "k_selected_full")
    runtime = {
        "decision": decision.get("decision"),
        "evaluated_levels": decision.get("evaluated_levels"),
        "selected_k": decision.get("selected_k"),
        "best_k_by_composite": decision.get("best_k_by_composite"),
        "best_k_by_composite_kneutral": decision.get("best_k_by_composite_kneutral"),
        "best_k_by_bic": decision.get("best_k_by_bic"),
        "best_k_by_aic": decision.get("best_k_by_aic"),
        "energy_cluster_negative_evidence": decision.get("energy_cluster_negative_evidence"),
        "channel_degeneracy_summary": decision.get("channel_degeneracy_summary"),
        "forward_reverse_consistency_summary": decision.get("forward_reverse_consistency_summary"),
        "best_k_by_structure": decision.get("best_k_by_structure"),
        "best_k_by_physical_response": decision.get("best_k_by_physical_response"),
        "best_k_by_d1_rmse": decision.get("best_k_by_d1_rmse"),
        "best_k_by_d2_rmse": decision.get("best_k_by_d2_rmse"),
        "flatline_guard_pass": decision.get("flatline_guard_pass"),
        "vr_late_bias_pass": decision.get("vr_late_bias_pass"),
        "high_vr_valley_depth_pass": decision.get("high_vr_valley_depth_pass"),
        "hyperopt_selected_config": hyperopt_summary.get("selected"),
        "hyperopt_method": hyperopt_summary.get("method"),
        "hyperopt_stage_count": len(hyperopt_summary.get("stages", [])),
        "training_epochs": int(cfg.epochs),
        "early_stop_min_epochs": int(cfg.early_stop_min_epochs),
        "model_selected": bool(decision.get("decision") == "select"),
        "evidence_confirmed": bool(decision.get("selected_k") == decision.get("best_k_by_composite_kneutral")),
        "source_sweep": "outputs/main/fullscan/decision.json",
    }
    confirm_dir = out_dir.parent / "confirm_round"
    confirm_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(runtime, confirm_dir / "iteration_summary.json")
    safe_json_dump(runtime, confirm_dir / "metrics_compare.json")
    safe_json_dump({"retained_roots": ["confirm_round", "k1_full", "k_sweep_1_8", "k_selected_full"]}, confirm_dir / "runtime_summary.json")
    decision["hyperopt_selected_config"] = hyperopt_summary.get("selected")
    decision["hyperopt_method"] = hyperopt_summary.get("method")
    decision["hyperopt_stage_count"] = len(hyperopt_summary.get("stages", []))
    decision["training_epochs"] = int(cfg.epochs)
    decision["early_stop_min_epochs"] = int(cfg.early_stop_min_epochs)
    safe_json_dump(decision, out_dir / "decision.json")
    return {"scan_dir": str(out_dir), "decision": decision, "summaries": summaries, "hyperopt": hyperopt_summary}
