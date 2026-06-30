from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_make_figures_module():
    script = Path(__file__).resolve().parents[2] / "ESSAY" / "ajp_argon_sublevel_manuscript" / "visualization" / "scripts" / "make_figures.py"
    if not script.exists():
        pytest.skip("external manuscript figure script is not part of this source repository")
    spec = importlib.util.spec_from_file_location("fhargon_make_figures", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except RuntimeError as exc:
        pytest.skip(f"external manuscript figure script is unavailable: {exc}")
    return module


def test_channel_uncertainty_figure_uses_bars_with_interval_labels(monkeypatch, tmp_path: Path) -> None:
    module = load_make_figures_module()
    source_dir = tmp_path / "source_data"
    figure_dir = tmp_path / "figures"
    source_dir.mkdir()
    figure_dir.mkdir()
    pd.DataFrame(
        [
            {
                "channel": 1,
                "sample_count": 12,
                "energy_mean": 11.50,
                "energy_ci95_low": 11.47,
                "energy_ci95_high": 11.53,
                "weight_mean": 0.36,
                "weight_ci95_low": 0.01,
                "weight_ci95_high": 0.58,
            },
            {
                "channel": 2,
                "sample_count": 12,
                "energy_mean": 11.74,
                "energy_ci95_low": 11.70,
                "energy_ci95_high": 11.78,
                "weight_mean": 0.40,
                "weight_ci95_low": 0.05,
                "weight_ci95_high": 0.66,
            },
        ]
    ).to_csv(source_dir / "channel_uncertainty_summary.csv", index=False)
    captured = {}

    def inspect_figure(fig, stem, *, svg=True):
        captured["patch_counts"] = [len(ax.patches) for ax in fig.axes]
        captured["bar_widths"] = [patch.get_width() for ax in fig.axes for patch in ax.patches]
        captured["bar_centers"] = [
            patch.get_x() + patch.get_width() / 2.0 for ax in fig.axes for patch in ax.patches
        ]
        captured["texts"] = [text.get_text() for ax in fig.axes for text in ax.texts]
        return [str(Path(stem).with_suffix(".png"))]

    monkeypatch.setattr(module, "SOURCE_DATA", source_dir)
    monkeypatch.setattr(module, "FIG_MAIN", figure_dir)
    monkeypatch.setattr(module, "save_figure", inspect_figure)

    created = module.plot_channel_uncertainty()

    assert created == [str((figure_dir / "fig11_channel_uncertainty").with_suffix(".png"))]
    assert captured["patch_counts"] == [2, 2]
    assert max(captured["bar_widths"]) <= 0.42
    assert captured["bar_centers"][1] - captured["bar_centers"][0] >= 1.1
    assert "11.50" in captured["texts"]
    assert "0.36" in captured["texts"]
