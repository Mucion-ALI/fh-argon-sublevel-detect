# SubLevel Detect

Frank-Hertz argon sublevel detection source project for paper reproduction.

This repository is code-first: it contains source code, the input spreadsheet, tests, and documentation. Training checkpoints and generated analysis products are not committed. Running the pipeline creates all results under `outputs/`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick Check

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode smoke --exclude hpopt --ablation
```

Smoke mode is only for functional validation.

## Main Baseline

Run the full baseline with hyperparameter optimization, K scan, training, and post-evaluation:

```powershell
python run.py --mode fullscan
```

Skip hyperparameter optimization and use default training settings:

```powershell
python run.py --mode fullscan --exclude hpopt
```

Run the main baseline and then the complete ablation baseline:

```powershell
python run.py --mode fullscan --ablation
```

Common overrides:

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output outputs --device cpu
```

`--device` accepts `cpu`, `cuda`, or `auto`.

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

The physical response audit uses two retained gates: late-bias and high-retarding-voltage valley-depth.

## Tests

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

The smoke test creates a temporary output directory inside this repository and removes it before finishing.
