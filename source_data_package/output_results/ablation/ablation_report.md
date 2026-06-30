# Ablation Baseline Report

| group | selected K | fit | CV | d1 | d2 | structure | physical | BIC | AIC | clustered K count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_baseline | 4 | 4 | 4 | 8 | 8 | 8 | 4 | 4 | 4 | 0 |
| no_forward_anchor_gap | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 5 | 7 | 0 |
| selector_no_weight_degeneracy | 4 | 4 | 4 | 8 | 8 | 8 | 4 | 4 | 4 | 0 |
| selector_no_energy_cluster | 4 | 4 | 4 | 8 | 8 | 8 | 4 | 4 | 4 | 0 |
| selector_metric_bic_aic_only | 4 | 4 | 4 | 8 | 8 | 8 | 4 | 4 | 4 | 0 |

The ablation line compares the main baseline, selector-only variants, and the no-forward-anchor-gap retrain.
