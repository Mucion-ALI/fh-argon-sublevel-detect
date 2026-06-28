from __future__ import annotations

from pathlib import Path

import json

from sublevel_detect import ablation_pipeline, cli, main_pipeline, model, paths, robustness_pipeline, self_check


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


def test_cli_defaults_to_cpu_and_robustness_off() -> None:
    args = cli.build_parser().parse_args([])
    assert args.device == "cpu"
    assert args.robustness is False


def test_cli_does_not_run_robustness_unless_requested(monkeypatch) -> None:
    calls = {"robustness": 0}

    def fake_main_run(**kwargs):
        return {"sweep_dir": str(paths.PROJECT_ROOT / "_tmp_contract_output" / "main" / "fullscan")}

    def fake_robustness_run(**kwargs):
        calls["robustness"] += 1
        return {}

    monkeypatch.setattr(cli.main_pipeline, "run", fake_main_run)
    monkeypatch.setattr(cli.robustness_pipeline, "run", fake_robustness_run)

    assert cli.main(["--mode", "smoke"]) == 0
    assert calls["robustness"] == 0

    assert cli.main(["--mode", "smoke", "--robustness"]) == 0
    assert calls["robustness"] == 1


def test_auto_device_uses_cpu_dispatch_unless_cuda_is_explicit() -> None:
    default_model_cfg = model.Config()
    assert default_model_cfg.device == "cpu"
    assert default_model_cfg.selected_device == "cpu"

    auto_cfg = main_pipeline.build_config(
        mode="fullscan",
        input_path=paths.DEFAULT_INPUT,
        output_root=paths.PROJECT_ROOT / "_tmp_contract_output",
        device="auto",
        exclude=[],
    )
    assert auto_cfg.selected_device == "cpu"
    assert auto_cfg.dispatch_strategy == "cpu_4"
    assert str(model.resolve_device("auto")) == "cpu"


def test_explicit_cuda_is_the_only_cuda_dispatch_path(monkeypatch) -> None:
    monkeypatch.setattr(main_pipeline.model.torch.cuda, "is_available", lambda: True)
    cuda_cfg = main_pipeline.build_config(
        mode="fullscan",
        input_path=paths.DEFAULT_INPUT,
        output_root=paths.PROJECT_ROOT / "_tmp_contract_output",
        device="cuda",
        exclude=[],
    )
    auto_cfg = main_pipeline.build_config(
        mode="fullscan",
        input_path=paths.DEFAULT_INPUT,
        output_root=paths.PROJECT_ROOT / "_tmp_contract_output",
        device="auto",
        exclude=[],
    )
    assert cuda_cfg.selected_device == "cuda"
    assert cuda_cfg.dispatch_strategy == "single"
    assert auto_cfg.selected_device == "cpu"
    assert auto_cfg.dispatch_strategy == "cpu_4"


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


def test_cli_accepts_explicit_robustness_flag() -> None:
    args = cli.build_parser().parse_args(["--mode", "fullscan", "--robustness"])
    assert args.robustness is True


def test_selector_rank_weights_preserve_default_and_allow_perturbation() -> None:
    rows = [
        {
            "n_levels": 1,
            "rmse_mean": 3.0,
            "cv_rmse_mean": 1.0,
            "d1_rmse_mean": 1.0,
            "d2_rmse_mean": 1.0,
            "structure_score": 1.0,
            "vr_physical_response_score": 1.0,
            "bic": 1.0,
            "aic": 1.0,
            "weights": [1.0],
            "energies": [11.5],
        },
        {
            "n_levels": 2,
            "rmse_mean": 1.0,
            "cv_rmse_mean": 3.0,
            "d1_rmse_mean": 3.0,
            "d2_rmse_mean": 3.0,
            "structure_score": 3.0,
            "vr_physical_response_score": 3.0,
            "bic": 3.0,
            "aic": 3.0,
            "weights": [0.5, 0.5],
            "energies": [11.5, 12.0],
        },
    ]
    default_decision = model.select_k_neutral_level_decision(rows)
    explicit_default = model.select_k_neutral_level_decision(
        rows,
        rank_weights={
            "rmse": 1.0,
            "summary": 1.0,
            "d1": 1.0,
            "d2": 1.0,
            "structure": 1.25,
            "physical": 1.0,
            "bic": 1.0,
            "aic": 0.5,
            "degeneracy": 1.0,
        },
    )
    fit_heavy = model.select_k_neutral_level_decision(
        rows,
        rank_weights={
            "rmse": 20.0,
            "summary": 0.0,
            "d1": 0.0,
            "d2": 0.0,
            "structure": 0.0,
            "physical": 0.0,
            "bic": 0.0,
            "aic": 0.0,
            "degeneracy": 0.0,
        },
    )
    assert explicit_default["selected_k"] == default_decision["selected_k"]
    assert fit_heavy["selected_k"] == 2


def test_leave_one_vr_config_preserves_main_sweep_training_settings(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "main" / "fullscan"
    sweep_dir.mkdir(parents=True)
    payload = {
        "data_path": str(paths.DEFAULT_INPUT),
        "out_dir": str(sweep_dir),
        "epochs": 9,
        "scan_seeds": "2",
        "level_scan_min": 2,
        "level_scan_max": 3,
        "early_stop_patience": 7,
        "hyperopt_enabled": True,
    }
    (sweep_dir / "config_used.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg = robustness_pipeline._loo_config(
        mode="fullscan",
        input_path=paths.DEFAULT_INPUT,
        output_root=tmp_path / "outputs",
        device="cpu",
        main_sweep_dir=sweep_dir,
        excluded_vr=4.0,
    )

    assert cfg.epochs == 9
    assert cfg.scan_seeds == "2"
    assert cfg.level_scan_min == 2
    assert cfg.level_scan_max == 3
    assert cfg.early_stop_patience == 7
    assert cfg.hyperopt_enabled is False
    assert cfg.exclude_vr_values == "4.0"


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
