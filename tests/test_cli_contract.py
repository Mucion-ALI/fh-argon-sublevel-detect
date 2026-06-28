from __future__ import annotations

from pathlib import Path

from sublevel_detect import ablation_pipeline, cli, main_pipeline, model, paths, self_check


def test_excluding_hpopt_disables_hyperparameter_optimization() -> None:
    args = cli.build_parser().parse_args(["--mode", "fullscan", "--exclude", "hpopt"])
    assert cli.hyperopt_enabled(args) is False
    cfg = main_pipeline.build_config(
        mode=args.mode,
        input_path=args.input,
        output_root=args.output,
        device=args.device,
        exclude=args.exclude,
    )
    assert cfg.hyperopt_enabled is False
    assert ablation_pipeline.hyperopt_enabled(mode=args.mode, exclude=args.exclude) is False


def test_fullscan_defaults_enable_hyperparameter_optimization() -> None:
    args = cli.build_parser().parse_args(["--mode", "fullscan"])
    assert cli.hyperopt_enabled(args) is True
    cfg = main_pipeline.build_config(
        mode=args.mode,
        input_path=args.input,
        output_root=args.output,
        device=args.device,
        exclude=args.exclude,
    )
    assert cfg.hyperopt_enabled is True
    assert Path(cfg.data_path) == paths.DEFAULT_INPUT
    assert Path(cfg.out_dir) == paths.DEFAULT_OUTPUT / "main" / "fullscan"


def test_external_input_and_output_paths_can_be_overridden() -> None:
    input_path = paths.PROJECT_ROOT / "data" / "argon" / "FHdata.xlsx"
    output_root = paths.PROJECT_ROOT / "_tmp_contract_output"
    cfg = main_pipeline.build_config(
        mode="smoke",
        input_path=str(input_path),
        output_root=str(output_root),
        device="cpu",
        exclude=["hpopt"],
    )
    assert Path(cfg.data_path) == input_path
    assert Path(cfg.out_dir) == output_root / "main" / "fullscan"
    assert cfg.hyperopt_enabled is False


def test_formal_project_has_no_historical_version_labels() -> None:
    result = self_check.check_project(paths.PROJECT_ROOT)
    assert result["missing"] == []
    assert result["blocked_term_hits"] == []


def test_model_is_library_only_and_run_py_is_the_workflow_entry() -> None:
    exported = set(getattr(model, "__all__", []))
    assert "main" not in exported
    assert not hasattr(model, "main")


def test_run_py_is_the_only_executable_entrypoint() -> None:
    entrypoint = 'if __name__ == "__main__"'
    offenders = []
    for path in (paths.PROJECT_ROOT / "src" / "sublevel_detect").glob("*.py"):
        if entrypoint in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
    assert entrypoint in (paths.PROJECT_ROOT / "run.py").read_text(encoding="utf-8")
