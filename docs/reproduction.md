# Reproduction

## Baseline Lines

This project has two parallel lines.

Main baseline:

1. Prepare forward evidence from the input curves.
2. Optionally run hyperparameter optimization.
3. Train the K scan.
4. Generate automatic post-evaluation tables and paper summary files.

Ablation baseline:

1. Reuse the main baseline sweep as the production reference.
2. Re-run selector variants.
3. Retrain the no-forward-anchor-gap condition.
4. Write ablation summary tables and a Markdown report.

## Commands

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode fullscan
python run.py --mode fullscan --exclude hpopt
python run.py --mode fullscan --ablation
```

Use `--input` for a different spreadsheet and `--output` for a different output root.

## Interpretation

Fullscan outputs are intended for paper claims. Smoke outputs are functional checks only.
