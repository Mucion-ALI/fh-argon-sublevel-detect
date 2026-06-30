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

Robustness baseline:

1. Reuse the main baseline sweep as the production reference.
2. Recompute selector decisions under a fixed rank-weight perturbation grid.
3. Retrain K candidates while leaving out one retarding-voltage curve at a time.
4. Write selector perturbation, leave-one-Vr-out, and combined robustness summaries.

Sensitivity baseline:

1. Reuse the main baseline sweep as the production reference.
2. Scale the recorded production forward prior weights directly for prior-strength factors.
3. Export and reuse selected-model point residuals for residual bootstrap.
4. Write both conditional K=4 stress-test drift and production-anchor matched K=4 uncertainty tables.

## Commands

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

Use `--input` for a different spreadsheet and `--output` for a different output root.

## Interpretation

Fullscan outputs are intended for paper claims. Smoke outputs are functional checks only.

The leave-one-Vr-out analysis fixes the main baseline hyperparameter configuration and retrains each excluded-Vr fold. It is a robustness check for the workflow-level selected K, not a separate hyperparameter search and not an independent holdout prediction benchmark.

For sensitivity, `channel_uncertainty_conditional_k4.csv` is not a production confidence interval. Use `channel_uncertainty_anchor_matched.csv` when the production selected model is K=4 and uncertainty language needs fixed production-channel anchors.

