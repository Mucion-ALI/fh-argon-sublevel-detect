# SubLevel Detect

[中文 README](README.zh-CN.md)

Frank-Hertz argon sublevel detection source project for paper reproduction.

This repository is code-first: it contains source code, one input spreadsheet, tests, and documentation. Checkpoints and generated analysis products are not committed. Running the pipeline recreates all results under `outputs/`.

## What This Project Reproduces

The code fits and evaluates a multi-level Frank-Hertz argon model. It is organized around two reproducible baselines:

- Main baseline: forward-evidence preparation, optional hyperparameter optimization, training over candidate level counts, and automatic post-evaluation.
- Ablation baseline: selector-only ablations plus a no-forward-anchor-gap retraining run to test how much the final decision depends on forward anchors.

The retained physical response audit uses two gates: late-bias and high-retarding-voltage valley-depth.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## If You Are an AI Agent

Use this sequence to reproduce the study without guessing project layout:

1. Inspect `README.md`, `README.zh-CN.md`, `docs/reproduction.md`, and `docs/outputs.md`.
2. Confirm that `data/argon/FHdata.xlsx` exists and that no `outputs/` directory is being reused from an earlier run.
3. Run a functional smoke check:

```powershell
python run.py --mode smoke --exclude hpopt --ablation
```

4. Treat smoke outputs as runtime validation only. Delete or ignore smoke outputs before paper-facing analysis.
5. Run the main paper baseline:

```powershell
python run.py --mode fullscan
```

6. Run the full baseline with ablation evidence:

```powershell
python run.py --mode fullscan --ablation
```

7. For a faster but non-hyperoptimized reproduction, use:

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

8. Summarize conclusions only from `outputs/main/fullscan/decision.json`, `outputs/main/fullscan/model_selection_table.csv`, `outputs/main/paper_summary.json`, and `outputs/ablation/ablation_summary.csv`.

## Experiment Design

Input data:

- Default file: `data/argon/FHdata.xlsx`
- Expected content: Frank-Hertz argon current-voltage curves with accelerating voltage, retarding voltage, curve identifier, and measured current columns.
- Loader behavior: the code resolves common column labels and can read Excel directly; it also includes a fallback reader for `.xlsx` files when the usual Excel dependency is unavailable.

Main baseline workflow:

- The command `python run.py --mode fullscan` builds forward evidence from the observed oscillatory structure.
- Hyperparameter optimization is enabled by default for full scans and can be skipped with `--exclude hpopt`.
- Candidate level counts are scanned across the configured K range.
- Each candidate writes per-level metrics, checkpoints, scorecards, scan tables, and selector diagnostics under `outputs/main/fullscan/`.
- Automatic post-evaluation writes paper-facing summaries under `outputs/main/`.

Ablation workflow:

- `--ablation` runs the main baseline first, then launches ablation analysis.
- Selector ablations recompute decisions while removing individual selector components.
- The no-forward-anchor-gap condition retrains with forward anchor priors disabled.
- Ablation outputs are written under `outputs/ablation/`.

Device and path controls:

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output outputs --device cpu
```

`--device` accepts `cpu`, `cuda`, or `auto`.

## Data Analysis

Primary decision files:

- `outputs/main/fullscan/decision.json`: selected level count and decision diagnostics.
- `outputs/main/fullscan/model_selection_table.csv`: scored candidate K table used for model selection.
- `outputs/main/fullscan/scan_summary.csv`: fit, cross-validation, structure, and physical-response metrics by candidate.
- `outputs/main/paper_summary.json`: compact paper-facing summary of the selected result.

Structure analysis:

- `structure_metrics.csv` summarizes peak-valley preservation and flatline guarding.
- `peak_valley_segments.csv`, `curve_structure_summary.csv`, and `class_structure_summary.csv` provide segment-level and grouped diagnostics.

Physical response analysis:

- `vr_physical_response.csv` contains per-curve physical-response diagnostics.
- `vr_physical_response.json` stores physical-response summaries and grouped rows.
- The reported physical caveats are late-bias and high-retarding-voltage valley-depth.

Ablation analysis:

- `outputs/ablation/ablation_summary.csv` compares the main baseline, no-forward-anchor-gap retrain, and selector-only variants.
- `outputs/ablation/selector_ablation_decision.json` stores detailed selector decisions for each ablation group.
- `outputs/ablation/ablation_report.md` is the human-readable ablation report.

For manuscript writing, cite the generated JSON/CSV files rather than intermediate checkpoints. Checkpoints are runtime artifacts and are intentionally ignored by git.

## Commands

Smoke check:

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode smoke --exclude hpopt --ablation
```

Full main baseline:

```powershell
python run.py --mode fullscan
```

Full main baseline without hyperparameter optimization:

```powershell
python run.py --mode fullscan --exclude hpopt
```

Full main baseline plus ablation:

```powershell
python run.py --mode fullscan --ablation
```

## Outputs

Main baseline evidence:

- `outputs/main/fullscan/decision.json`
- `outputs/main/fullscan/model_selection_table.csv`
- `outputs/main/fullscan/scan_summary.csv`
- `outputs/main/fullscan/vr_physical_response.csv`
- `outputs/main/paper_summary.json`
- `outputs/main/paper_summary.md`

Ablation baseline evidence:

- `outputs/ablation/ablation_summary.csv`
- `outputs/ablation/ablation_report.md`
- `outputs/ablation/selector_ablation_decision.json`

## Tests

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

The smoke test creates a temporary output directory inside this repository and removes it before finishing.
