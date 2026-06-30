from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


OUTPUT = _env_path("SUBLEVEL_OUTPUT", REPO_ROOT / "output")
PACKAGE = _env_path("SUBLEVEL_PACKAGE", REPO_ROOT / "source_data_package")
MANUSCRIPT = _env_path(
    "SUBLEVEL_MANUSCRIPT",
    WORKSPACE_ROOT / "ESSAY" / "ajp_argon_sublevel_manuscript",
)
SOURCE_TABLES = _env_path("SUBLEVEL_SOURCE_TABLES", MANUSCRIPT / "visualization" / "source_data")
FIGURES = _env_path("SUBLEVEL_FIGURES", MANUSCRIPT / "visualization" / "figures")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}


@dataclass(frozen=True)
class PackageFile:
    package_path: Path
    source_path: Path
    category: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(path: Path, base: Path) -> str | None:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def relative(path: Path) -> str:
    path = path.resolve()
    mappings = (
        (OUTPUT, "output"),
        (PACKAGE, "source_data_package"),
        (SOURCE_TABLES, "manuscript_visualization/source_data"),
        (FIGURES, "manuscript_visualization/figures"),
        (REPO_ROOT, ""),
        (WORKSPACE_ROOT, ".."),
    )
    for base, prefix in mappings:
        rel = _relative_to(path, base)
        if rel is None:
            continue
        return f"{prefix}/{rel}".strip("/")
    return path.name


def _replace_path(text: str, absolute_base: Path, logical_base: str) -> str:
    absolute = absolute_base.resolve()
    replacements = {
        str(absolute): logical_base,
        absolute.as_posix(): logical_base,
        absolute.as_posix().replace("/", "//"): logical_base,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\\", "/")


def _normalize_text_payload(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8-sig")
    for base, logical in (
        (OUTPUT, "output"),
        (PACKAGE, "source_data_package"),
        (REPO_ROOT, "."),
        (WORKSPACE_ROOT / "SubLevel_Detect", "."),
        (WORKSPACE_ROOT, ".."),
    ):
        text = _replace_path(text, base, logical)
    while "//" in text:
        text = text.replace("//", "/")
    path.write_text(text, encoding="utf-8", newline="")


def normalize_text_payloads(records: list[PackageFile]) -> None:
    for record in records:
        _normalize_text_payload(PACKAGE / record.package_path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_files(files: Iterable[tuple[Path, Path, str]], records: list[PackageFile]) -> None:
    for src, dst, category in files:
        if not src.exists():
            raise FileNotFoundError(src)
        copy_file(src, dst)
        records.append(PackageFile(dst.relative_to(PACKAGE), src, category))


def nonrecursive_files(src_dir: Path, dst_dir: Path, category: str) -> Iterable[tuple[Path, Path, str]]:
    for src in sorted(p for p in src_dir.iterdir() if p.is_file()):
        yield src, dst_dir / src.name, category


def selected_fullscan_files(src_dir: Path, dst_dir: Path, category: str) -> Iterable[tuple[Path, Path, str]]:
    names = {
        "channel_degeneracy_summary.csv",
        "channel_degeneracy_summary.json",
        "class_structure_summary.csv",
        "config_used.json",
        "curve_structure_summary.csv",
        "decision.json",
        "energy_cluster_table.csv",
        "energy_cluster_table.json",
        "forward_evidence.json",
        "forward_priors.json",
        "forward_reverse_consistency.csv",
        "forward_reverse_consistency.json",
        "hyperopt_summary.csv",
        "hyperopt_summary.json",
        "level_weight_diagnostics.csv",
        "level_weight_diagnostics.json",
        "model_selection_table.csv",
        "peak_valley_segments.csv",
        "production_run_summary.json",
        "scan_summary.csv",
        "scan_summary.json",
        "structure_metrics.csv",
        "structure_metrics.json",
        "vr_physical_response.csv",
        "vr_physical_response.json",
    }
    for name in sorted(names):
        src = src_dir / name
        if src.exists():
            yield src, dst_dir / name, category


def copy_source_tables(records: list[PackageFile]) -> None:
    files = (
        (src, PACKAGE / "manuscript_source_tables" / src.name, "manuscript_source_table")
        for src in sorted(SOURCE_TABLES.iterdir())
        if src.is_file() and src.suffix.lower() in {".csv", ".json"}
    )
    copy_files(files, records)


def copy_figures(records: list[PackageFile]) -> None:
    files = []
    for src in sorted(p for p in FIGURES.rglob("*") if p.is_file()):
        rel = src.relative_to(FIGURES)
        if rel.name.startswith("fig_supp_error_reduction_heatmap."):
            continue
        files.append((src, PACKAGE / "figures" / rel, f"figure_{rel.parts[0]}"))
    copy_files(files, records)


def copy_run_records(records: list[PackageFile]) -> None:
    for name in ("k1_full", "k_selected_full"):
        src_dir = OUTPUT / "main" / name
        dst_dir = PACKAGE / "run_records" / name
        copy_files(nonrecursive_files(src_dir, dst_dir, f"run_record_{name}"), records)


def copy_output_results(records: list[PackageFile]) -> None:
    copy_files(nonrecursive_files(OUTPUT / "main", PACKAGE / "output_results" / "main", "output_main"), records)
    copy_files(
        selected_fullscan_files(
            OUTPUT / "main" / "fullscan",
            PACKAGE / "output_results" / "main" / "fullscan",
            "output_main_fullscan",
        ),
        records,
    )
    for name in ("k1_full", "k_selected_full"):
        src = OUTPUT / "main" / name / "prediction_points.csv"
        if src.exists():
            copy_files(
                [(src, PACKAGE / "output_results" / "main" / name / src.name, f"output_main_{name}")],
                records,
            )

    copy_files(nonrecursive_files(OUTPUT / "ablation", PACKAGE / "output_results" / "ablation", "output_ablation"), records)
    copy_files(
        selected_fullscan_files(
            OUTPUT / "ablation" / "no_forward_anchor_gap" / "fullscan",
            PACKAGE / "output_results" / "ablation" / "no_forward_anchor_gap" / "fullscan",
            "output_ablation_fullscan",
        ),
        records,
    )
    confirm_dir = OUTPUT / "ablation" / "no_forward_anchor_gap" / "confirm_round"
    if confirm_dir.exists():
        copy_files(
            nonrecursive_files(
                confirm_dir,
                PACKAGE / "output_results" / "ablation" / "no_forward_anchor_gap" / "confirm_round",
                "output_ablation_confirm_round",
            ),
            records,
        )

    copy_files(
        nonrecursive_files(OUTPUT / "robustness", PACKAGE / "output_results" / "robustness", "output_robustness"),
        records,
    )
    loo_root = OUTPUT / "robustness" / "leave_one_vr_out"
    if loo_root.exists():
        for fullscan in sorted(loo_root.glob("*/fullscan")):
            dst = PACKAGE / "output_results" / "robustness" / "leave_one_vr_out" / fullscan.parent.name / "fullscan"
            copy_files(selected_fullscan_files(fullscan, dst, "output_robustness_fullscan"), records)

    sensitivity_files = [
        OUTPUT / "sensitivity" / "sensitivity_summary.json",
        OUTPUT / "sensitivity" / "residual_bootstrap_scale_check.csv",
    ]
    copy_files(
        (
            (src, PACKAGE / "output_results" / "sensitivity" / src.name, "output_sensitivity")
            for src in sensitivity_files
            if src.exists()
        ),
        records,
    )
    for subdir in ("prior_strength", "uncertainty"):
        src_dir = OUTPUT / "sensitivity" / subdir
        if src_dir.exists():
            files = (
                (src, PACKAGE / "output_results" / "sensitivity" / subdir / src.name, f"output_sensitivity_{subdir}")
                for src in sorted(p for p in src_dir.iterdir() if p.is_file())
                if src.suffix.lower() in {".csv", ".json"}
            )
            copy_files(files, records)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_package(records: list[PackageFile]) -> dict[str, object]:
    prior_rows = read_csv_rows(PACKAGE / "manuscript_source_tables" / "prior_strength_selection.csv")
    prior_1x = next(row for row in prior_rows if float(row["prior_strength"]) == 1.0)
    anchor_rows = read_csv_rows(PACKAGE / "manuscript_source_tables" / "channel_uncertainty_anchor_matched.csv")
    residual_rows = read_csv_rows(PACKAGE / "manuscript_source_tables" / "residual_bootstrap_scale_check.csv")
    output_residual = next(row for row in residual_rows if row["source"] == "output")
    residual_delta = abs(
        float(output_residual["bootstrap_noise_std"]) - float(output_residual["prediction_residual_std"])
    )
    anchors = [float(row["anchor_energy_v"]) for row in anchor_rows]
    expected_anchors = [
        11.500251770019531,
        11.739255905151367,
        12.59369945526123,
        13.96471881866455,
    ]
    table_count = sum(1 for r in records if r.category == "manuscript_source_table")
    figure_count = sum(1 for r in records if r.category.startswith("figure_"))
    hash_failures = []
    for record in records:
        path = PACKAGE / record.package_path
        if not path.exists() or sha256(path) == "":
            hash_failures.append(record.package_path.as_posix())

    checks = {
        "manuscript_source_table_count_is_29": table_count == 29,
        "figure_asset_count_is_71": figure_count == 71,
        "prior_1x_selected_k_is_4": int(prior_1x["selected_k"]) == 4,
        "anchor_energies_match_expected": all(
            abs(actual - expected) < 1e-9 for actual, expected in zip(anchors, expected_anchors)
        ),
        "residual_bootstrap_std_close_to_prediction_residual_std_lt_0p01_uA": residual_delta < 0.01,
        "hashes_present_for_all_files": not hash_failures,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "package_files": len(records),
            "manuscript_source_tables": table_count,
            "figure_assets": figure_count,
            "run_record_files": sum(1 for r in records if r.category.startswith("run_record_")),
            "output_result_files": sum(1 for r in records if r.category.startswith("output_")),
        },
        "prior_1x": {
            "selected_k": int(prior_1x["selected_k"]),
            "w_prior_anchor": float(prior_1x["w_prior_anchor"]),
            "w_prior_gap": float(prior_1x["w_prior_gap"]),
        },
        "anchor_energies_v": anchors,
        "residual_bootstrap": {
            "bootstrap_noise_std": float(output_residual["bootstrap_noise_std"]),
            "prediction_residual_std": float(output_residual["prediction_residual_std"]),
            "std_abs_delta_uA": residual_delta,
        },
        "hash_failures": hash_failures,
    }


def write_index(records: list[PackageFile]) -> list[dict[str, object]]:
    rows = []
    for record in sorted(records, key=lambda item: item.package_path.as_posix()):
        package_file = PACKAGE / record.package_path
        rows.append(
            {
                "package_path": record.package_path.as_posix(),
                "source_path": relative(record.source_path),
                "category": record.category,
                "size_bytes": package_file.stat().st_size,
                "sha256": sha256(package_file),
            }
        )
    with (PACKAGE / "FILE_INDEX.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (PACKAGE / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['package_path']}\n")
    return rows


def write_readme(validation: dict[str, object]) -> None:
    text = f"""# Source Data Package

This package is the manuscript-facing, curated source-data bundle for the argon
Franck-Hertz sublevel-detection analysis. It is rebuilt from
`SubLevel_Detect/output/` and the manuscript visualization workspace.

## Contents

- `manuscript_source_tables/`: 29 CSV/JSON source tables used by the manuscript.
- `figures/main/`: 35 main-text figure assets.
- `figures/supplementary/`: 36 supplementary figure assets.
- `run_records/`: minimal `K=1` and production `K=4` run records.
- `output_results/`: selected `output/` decisions, summaries, prediction points,
  robustness summaries, and sensitivity summaries.
- `FILE_INDEX.csv`: package index with SHA256 hashes.
- `SHA256SUMS.txt`: checksum list for every packaged file.
- `validation_report.json`: machine-readable consistency checks.

The package intentionally excludes full perturbation training trees, full
seed-level checkpoint forests, and large intermediate outputs. These are
regenerated from `data/argon/FHdata.xlsx` with:

```powershell
python run.py --mode fullscan --ablation --robustness
python run.py --mode fullscan --exclude hpopt --sensitivity --device cpu
```

Validation status: `{validation["status"]}`.
"""
    (PACKAGE / "README.md").write_text(text, encoding="utf-8")


def write_manifest(rows: list[dict[str, object]], validation: dict[str, object]) -> None:
    lines = [
        "# Source Data Package Manifest",
        "",
        "This manifest indexes the curated manuscript source-data package.",
        "All paths are relative to `SubLevel_Detect/source_data_package/`.",
        "",
        f"- Validation status: `{validation['status']}`",
        f"- Package files: {validation['counts']['package_files']}",
        f"- Manuscript source tables: {validation['counts']['manuscript_source_tables']}",
        f"- Figure assets: {validation['counts']['figure_assets']}",
        f"- Run-record files: {validation['counts']['run_record_files']}",
        f"- Selected output-result files: {validation['counts']['output_result_files']}",
        "",
        "## Files",
        "",
        "| Package path | Category | Size bytes | SHA256 |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['package_path']}` | `{row['category']}` | {row['size_bytes']} | `{row['sha256']}` |"
        )
    lines.append("")
    (PACKAGE / "source_data_package_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not OUTPUT.exists():
        raise FileNotFoundError(f"Expected recomputed output directory: {OUTPUT}")
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)

    records: list[PackageFile] = []
    copy_source_tables(records)
    copy_figures(records)
    copy_run_records(records)
    copy_output_results(records)
    normalize_text_payloads(records)
    rows = write_index(records)
    validation = validate_package(records)
    (PACKAGE / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(validation)
    write_manifest(rows, validation)
    if validation["status"] != "pass":
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
