from __future__ import annotations

import argparse
import json
from typing import Sequence

from . import ablation_pipeline, main_pipeline, model, paths


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
    parser.add_argument("--input", default=paths.default_input_text(), help="Input data file.")
    parser.add_argument("--output", default=paths.default_output_text(), help="Output root directory.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser


def hyperopt_enabled(args: argparse.Namespace) -> bool:
    return str(args.mode) == "fullscan" and "hpopt" not in set(args.exclude or [])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    main_result = main_pipeline.run(
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
    print(json.dumps(model.json_ready(result), ensure_ascii=False, indent=2))
    return 0
