from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import ablation_pipeline, main_pipeline, model, paths, robustness_pipeline, sensitivity_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Frank-Hertz sublevel detection reproduction pipeline."
    )
    parser.add_argument("--mode", choices=["fullscan", "smoke"], default="fullscan")
    parser.add_argument(
        "--exclude",
        action="append",
        choices=["hpopt"],
        default=[],
        help="Skip an optional stage. Currently supported: hpopt.",
    )
    parser.add_argument("--ablation", action="store_true", help="Run the ablation baseline after the main baseline.")
    parser.add_argument("--robustness", action="store_true", help="Run selector perturbation and leave-one-Vr-out robustness.")
    parser.add_argument("--sensitivity", action="store_true", help="Run forward-prior and channel-uncertainty sensitivity scans.")
    parser.add_argument("--input", default=paths.default_input_text(), help="Input data file.")
    parser.add_argument("--output", default=paths.default_output_text(), help="Output root directory.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser


def hyperopt_enabled(args: argparse.Namespace) -> bool:
    return str(args.mode) == "fullscan" and "hpopt" not in set(args.exclude or [])


def existing_main_result(args: argparse.Namespace) -> dict | None:
    if not bool(getattr(args, "sensitivity", False)) or str(args.mode) != "fullscan":
        return None
    sweep_dir = paths.main_scan_dir(args.output)
    decision_path = sweep_dir / "decision.json"
    if not decision_path.exists():
        return None
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    paper_summary = paths.main_report_dir(args.output) / "paper_summary.json"
    cfg_path = sweep_dir / "config_used.json"
    hyperopt = None
    if cfg_path.exists():
        try:
            hyperopt = bool(json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("hyperopt_enabled", False))
        except json.JSONDecodeError:
            hyperopt = None
    return {
        "ok": True,
        "mode": str(args.mode),
        "sweep_dir": str(Path(sweep_dir)),
        "paper_summary": str(paper_summary) if paper_summary.exists() else "",
        "selected_k": decision.get("selected_k"),
        "hyperopt_enabled": hyperopt,
        "reused_existing_main": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    main_result = existing_main_result(args) or main_pipeline.run(
        mode=str(args.mode),
        input_path=args.input,
        output_root=args.output,
        device=str(args.device),
        exclude=args.exclude or [],
    )
    result = {"main": main_result}
    if bool(args.ablation):
        result["ablation"] = ablation_pipeline.run(
            mode=str(args.mode),
            input_path=args.input,
            output_root=args.output,
            device=str(args.device),
            main_sweep_dir=main_result["sweep_dir"],
            exclude=args.exclude or [],
        )
    if bool(args.robustness):
        result["robustness"] = robustness_pipeline.run(
            mode=str(args.mode),
            input_path=args.input,
            output_root=args.output,
            device=str(args.device),
            main_sweep_dir=main_result["sweep_dir"],
        )
    if bool(args.sensitivity):
        result["sensitivity"] = sensitivity_pipeline.run(
            mode=str(args.mode),
            input_path=args.input,
            output_root=args.output,
            device=str(args.device),
            main_sweep_dir=main_result["sweep_dir"],
        )
    print(json.dumps(model.json_ready(result), ensure_ascii=False, indent=2))
    return 0
