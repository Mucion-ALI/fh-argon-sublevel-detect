from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import json

from sublevel_detect import cli, main_pipeline, model, paths, sensitivity_pipeline


def test_cli_sensitivity_flag_defaults_off_and_can_be_enabled() -> None:
    default_args = cli.build_parser().parse_args([])
    enabled_args = cli.build_parser().parse_args(["--sensitivity"])

    assert default_args.sensitivity is False
    assert enabled_args.sensitivity is True


def test_cli_reuses_existing_main_scan_for_full_sensitivity(monkeypatch, tmp_path: Path) -> None:
    sweep = tmp_path / "main" / "fullscan"
    sweep.mkdir(parents=True)
    (sweep / "decision.json").write_text(json.dumps({"selected_k": 4}), encoding="utf-8")
    (sweep / "config_used.json").write_text(json.dumps({"hyperopt_enabled": True}), encoding="utf-8")
    (tmp_path / "main" / "paper_summary.json").write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(["--mode", "fullscan", "--output", str(tmp_path), "--sensitivity"])

    def fail_main_run(**kwargs):
        raise AssertionError("main pipeline should be reused, not rerun")

    monkeypatch.setattr(cli.main_pipeline, "run", fail_main_run)
    result = cli.existing_main_result(args)

    assert result is not None
    assert result["selected_k"] == 4
    assert result["hyperopt_enabled"] is True
    assert result["reused_existing_main"] is True


def test_prior_strength_factor_scales_only_forward_prior_weights(tmp_path: Path) -> None:
    cfg = model.Config(out_dir=str(tmp_path / "main" / "fullscan"))
    evidence = {
        "global": {
            "main_spacing": 11.7,
            "spacing_std": 0.2,
            "confidence": 0.5,
        }
    }
    prior_cfg, priors = main_pipeline.apply_forward_prior_with_strength(cfg, evidence, strength_factor=1.0)
    no_prior_cfg, _ = main_pipeline.apply_forward_prior_with_strength(cfg, evidence, strength_factor=0.0)
    double_cfg, _ = main_pipeline.apply_forward_prior_with_strength(cfg, evidence, strength_factor=2.0)

    assert prior_cfg.forward_prior_mode == "auto"
    assert prior_cfg.forward_main_spacing == pytest.approx(11.7)
    assert prior_cfg.w_prior_anchor == pytest.approx(priors["config_updates"]["w_prior_anchor"])
    assert prior_cfg.w_prior_gap == pytest.approx(priors["config_updates"]["w_prior_gap"])

    assert no_prior_cfg.forward_prior_mode == "off"
    assert no_prior_cfg.w_prior_anchor == 0.0
    assert no_prior_cfg.w_prior_gap == 0.0

    assert double_cfg.forward_prior_mode == "auto"
    assert double_cfg.w_prior_anchor == pytest.approx(prior_cfg.w_prior_anchor * 2.0)
    assert double_cfg.w_prior_gap == pytest.approx(prior_cfg.w_prior_gap * 2.0)
    assert double_cfg.forward_main_spacing == pytest.approx(prior_cfg.forward_main_spacing)


def test_sensitivity_prior_strength_one_x_uses_production_prior_without_recompile(tmp_path: Path) -> None:
    main_sweep = tmp_path / "main" / "fullscan"
    main_sweep.mkdir(parents=True)
    production_cfg = {
        "forward_prior_mode": "auto",
        "forward_main_spacing": 11.72,
        "forward_spacing_std": 0.18,
        "forward_confidence": 0.64,
        "forward_anchor_step": 0.25,
        "w_prior_anchor": 0.0012,
        "w_prior_gap": 0.0008,
    }
    (main_sweep / "config_used.json").write_text(json.dumps(production_cfg), encoding="utf-8")

    cfg = model.Config(w_prior_anchor=9.0, w_prior_gap=7.0)
    one_x, priors = sensitivity_pipeline._apply_production_prior_strength(cfg, main_sweep, 1.0)
    two_x, _ = sensitivity_pipeline._apply_production_prior_strength(cfg, main_sweep, 2.0)
    zero_x, _ = sensitivity_pipeline._apply_production_prior_strength(cfg, main_sweep, 0.0)

    assert one_x.forward_prior_mode == "auto"
    assert one_x.forward_main_spacing == pytest.approx(11.72)
    assert one_x.forward_spacing_std == pytest.approx(0.18)
    assert one_x.forward_confidence == pytest.approx(0.64)
    assert one_x.w_prior_anchor == pytest.approx(production_cfg["w_prior_anchor"])
    assert one_x.w_prior_gap == pytest.approx(production_cfg["w_prior_gap"])
    assert priors["config_updates"]["w_prior_anchor"] == pytest.approx(production_cfg["w_prior_anchor"])
    assert priors["config_updates"]["w_prior_gap"] == pytest.approx(production_cfg["w_prior_gap"])

    assert two_x.w_prior_anchor == pytest.approx(production_cfg["w_prior_anchor"] * 2.0)
    assert two_x.w_prior_gap == pytest.approx(production_cfg["w_prior_gap"] * 2.0)
    assert zero_x.forward_prior_mode == "off"
    assert zero_x.w_prior_anchor == 0.0
    assert zero_x.w_prior_gap == 0.0


def test_init_jitter_scale_is_deterministic_per_seed_and_changes_across_seeds() -> None:
    base_kwargs = {
        "n_curves": 2,
        "n_levels": 4,
        "n_max": 16,
        "min_level_gap": 0.04,
        "V_exc_init": 11.5,
    }
    a = model.PoissonRateFHCoreMultiLevel(**base_kwargs, init_jitter_scale=0.05, init_seed=3)
    b = model.PoissonRateFHCoreMultiLevel(**base_kwargs, init_jitter_scale=0.05, init_seed=3)
    c = model.PoissonRateFHCoreMultiLevel(**base_kwargs, init_jitter_scale=0.05, init_seed=4)
    d = model.PoissonRateFHCoreMultiLevel(**base_kwargs, init_jitter_scale=0.0, init_seed=4)
    e = model.PoissonRateFHCoreMultiLevel(**base_kwargs, init_jitter_scale=0.0, init_seed=9)

    assert torch.equal(a.raw_dE, b.raw_dE)
    assert torch.equal(a.raw_level_logits, b.raw_level_logits)
    assert not torch.equal(a.raw_dE, c.raw_dE)
    assert not torch.equal(a.raw_level_logits, c.raw_level_logits)
    assert torch.equal(d.raw_dE, e.raw_dE)
    assert torch.equal(d.raw_level_logits, e.raw_level_logits)


def test_curve_to_tensors_uses_configured_peak_window_radius() -> None:
    va = np.arange(30.0, 80.0, 1.0, dtype=np.float32)
    ip = np.sin((va - 30.0) / 3.0).astype(np.float32)
    curve = {"curve_id": 0, "curve_idx": 0, "Va": va, "Vr": 0.0, "Vr_is_vector": False, "Ip": ip}

    narrow = model.curve_to_tensors(curve, torch.device("cpu"), peak_window_radius=0.1)
    wide = model.curve_to_tensors(curve, torch.device("cpu"), peak_window_radius=2.0)

    assert float(torch.sum(wide["structure_weight"])) > float(torch.sum(narrow["structure_weight"]))
    assert float(torch.sum(wide["valley_weight"])) >= float(torch.sum(narrow["valley_weight"]))


def test_uncertainty_summary_reports_channel_intervals_and_gap() -> None:
    rows = [
        {"analysis": "seed_jitter", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 1, "energy_v": 11.40, "weight": 0.30},
        {"analysis": "seed_jitter", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 2, "energy_v": 11.70, "weight": 0.40},
        {"analysis": "seed_jitter", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 1, "energy_v": 11.50, "weight": 0.35},
        {"analysis": "seed_jitter", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 2, "energy_v": 11.80, "weight": 0.38},
        {"analysis": "seed_jitter", "selected_k": 4, "n_levels": 3, "seed": 0, "channel": 1, "energy_v": 11.20, "weight": 1.00},
    ]

    summary = sensitivity_pipeline.summarize_channel_uncertainty(rows)
    channel_rows = [row for row in summary["channel_summary"] if row.get("channel") in {1, 2}]
    gap_row = summary["gap_summary"][0]

    assert {int(row["channel"]) for row in channel_rows} == {1, 2}
    assert all(row["energy_ci95_low"] <= row["energy_mean"] <= row["energy_ci95_high"] for row in channel_rows)
    assert all(row["weight_ci95_low"] <= row["weight_mean"] <= row["weight_ci95_high"] for row in channel_rows)
    assert gap_row["gap_name"] == "E2_minus_E1"
    assert gap_row["gap_mean_v"] == pytest.approx(0.30)
    assert gap_row["gap_ci95_low_v"] <= gap_row["gap_mean_v"] <= gap_row["gap_ci95_high_v"]


def test_residual_bootstrap_requires_prediction_residual_table(tmp_path: Path) -> None:
    main_sweep = tmp_path / "main" / "fullscan"
    main_sweep.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="prediction_points.csv"):
        sensitivity_pipeline._baseline_residuals_by_curve(main_sweep)


def test_residual_bootstrap_noise_scale_comes_from_prediction_residuals(tmp_path: Path) -> None:
    main_sweep = tmp_path / "main" / "fullscan"
    selected_dir = tmp_path / "main" / "k_selected_full"
    selected_dir.mkdir(parents=True)
    rows = []
    for curve_id in (1, 2):
        for idx in range(60):
            observed = float(idx)
            residual = 0.05 if idx % 2 else -0.05
            rows.append(
                {
                    "curve_id": curve_id,
                    "Vr": float(curve_id),
                    "Va": float(idx),
                    "observed": observed,
                    "predicted": observed - residual,
                    "residual": residual,
                }
            )
    pd = pytest.importorskip("pandas")
    pd.DataFrame(rows).to_csv(selected_dir / "prediction_points.csv", index=False)

    residuals = sensitivity_pipeline._baseline_residuals_by_curve(main_sweep)
    source = pd.DataFrame(
        {
            "curve_id": [1] * 20 + [2] * 20,
            "Vr": [1.0] * 20 + [2.0] * 20,
            "Va": list(range(20)) + list(range(20)),
            "IuA": np.linspace(0.0, 20.0, 40),
        }
    )
    out_path = tmp_path / "bootstrap.csv"
    sensitivity_pipeline._write_perturbed_input(
        source_frame=source,
        cols={"curve": "curve_id", "vr": "Vr", "va": "Va", "ip": "IuA"},
        path=out_path,
        rng=np.random.default_rng(3),
        mode="residual_bootstrap",
        residuals=residuals,
    )

    boot = pd.read_csv(out_path)
    noise = boot["IuA"].to_numpy(dtype=float) - source["IuA"].to_numpy(dtype=float)
    assert float(np.std(noise)) < 0.08
    assert float(np.std(source["IuA"].to_numpy(dtype=float))) > 5.0


def test_anchor_matched_k4_summary_uses_production_channels_as_fixed_anchors() -> None:
    anchors = [
        {"channel": 1, "energy_v": 11.50, "weight": 0.36},
        {"channel": 2, "energy_v": 11.74, "weight": 0.40},
        {"channel": 3, "energy_v": 12.59, "weight": 0.09},
        {"channel": 4, "energy_v": 13.96, "weight": 0.15},
    ]
    rows = [
        {"analysis": "seed_jitter", "condition": "a", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 1, "energy_v": 11.49, "weight": 0.35},
        {"analysis": "seed_jitter", "condition": "a", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 2, "energy_v": 11.76, "weight": 0.41},
        {"analysis": "seed_jitter", "condition": "a", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 3, "energy_v": 12.61, "weight": 0.08},
        {"analysis": "seed_jitter", "condition": "a", "selected_k": 8, "n_levels": 4, "seed": 0, "channel": 4, "energy_v": 13.97, "weight": 0.16},
        {"analysis": "seed_jitter", "condition": "b", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 1, "energy_v": 13.95, "weight": 0.16},
        {"analysis": "seed_jitter", "condition": "b", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 2, "energy_v": 12.58, "weight": 0.08},
        {"analysis": "seed_jitter", "condition": "b", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 3, "energy_v": 11.75, "weight": 0.40},
        {"analysis": "seed_jitter", "condition": "b", "selected_k": 8, "n_levels": 4, "seed": 1, "channel": 4, "energy_v": 11.51, "weight": 0.36},
    ]

    matched = sensitivity_pipeline.summarize_anchor_matched_k4_uncertainty(rows, anchors)
    summary = matched["channel_summary"]
    samples = matched["samples"]
    by_anchor = {int(row["anchor_channel"]): row for row in summary}

    assert sorted(by_anchor) == [1, 2, 3, 4]
    assert by_anchor[1]["anchor_energy_v"] == pytest.approx(11.50)
    assert by_anchor[1]["energy_mean"] == pytest.approx(11.50)
    assert by_anchor[2]["energy_mean"] == pytest.approx(11.755)
    assert all(int(row["sample_count"]) == 2 for row in summary)
    assert len(samples) == 8


def test_sensitivity_smoke_profile_uses_reduced_replicates() -> None:
    cfg = sensitivity_pipeline.sensitivity_profile("smoke")

    assert cfg.prior_factors == [0.0, 1.0]
    assert cfg.prior_seeds == [0]
    assert cfg.bootstrap_replicates == 1
    assert cfg.noise_replicates == 1
    assert cfg.window_radii == [1.0, 1.5]
