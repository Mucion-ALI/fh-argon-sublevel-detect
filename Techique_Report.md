# Technical Report

This report records the implementation details of the effective multichannel response analysis workflow for argon Franck-Hertz teaching-laboratory data. It is synchronized with the current source tree and the curated source-data package. The analysis decomposes measured current-voltage curves into effective response channels; it does not claim line-by-line atomic term assignment.

## 1. Repository Scope and Source Map

The public reproduction target is this repository. Runtime outputs are written to `output/` and are ignored by git. The committed `source_data_package/` is a curated, manuscript-facing archive built from the current recomputed workflow.

Primary implementation files:

- `run.py`: command-line entry point.
- `src/sublevel_detect/cli.py`: CLI contract and workflow dispatch.
- `src/sublevel_detect/paths.py`: repository-relative path defaults; the formal output root is `output/`.
- `src/sublevel_detect/main_pipeline.py`: main fullscan configuration, forward-prior preparation, and post-evaluation dispatch.
- `src/sublevel_detect/model.py`: data loading, physical kernel, loss function, optimizer, training loop, metrics, hyperparameter search, and K-neutral selector.
- `src/sublevel_detect/ablation_pipeline.py`: selector-only and no-forward-anchor-gap ablation workflows.
- `src/sublevel_detect/robustness_pipeline.py`: selector-weight perturbation and leave-one-retarding-voltage-out robustness workflows.
- `src/sublevel_detect/sensitivity_pipeline.py`: corrected prior-strength scan, residual-bootstrap perturbation, noise perturbation, peak-window-radius perturbation, and K=4 uncertainty summaries.
- `scripts/build_source_data_package.py`: curated package builder, hash generator, validation runner, and path sanitizer.

## 2. Input Data and Output Contract

The default input is `data/argon/FHdata.xlsx`. The loader accepts Excel and CSV inputs and resolves common column names for curve identifier, retarding voltage `Vr`, accelerating voltage `Va`, and current. If the normal Excel engine is unavailable for `.xlsx`, the code falls back to a direct workbook XML reader.

The formal output root is `output/`. The main workflow writes:

- `output/main/fullscan/decision.json`
- `output/main/fullscan/model_selection_table.csv`
- `output/main/fullscan/scan_summary.csv`
- `output/main/k_selected_full/prediction_points.csv`
- `output/main/paper_summary.json`

The selected-model prediction table has the fixed columns:

```text
curve_id,Vr,Va,observed,predicted,residual
```

This table is required by residual bootstrap. Sensitivity analysis fails if the table is absent.

## 3. Full Workflow

The formal recomputation sequence is:

```powershell
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

The workflow stages are:

1. Read and normalize the input curves.
2. Estimate forward evidence from the observed oscillatory structure.
3. Compile a soft forward prior and apply it to the production configuration.
4. Optionally run successive-halving hyperparameter optimization.
5. Train K candidates over the configured level range.
6. Retain the best K=1 run and the selected K run.
7. Export prediction points and residuals for the selected model.
8. Write model-selection, structure, physical-response, degeneracy, and paper-summary outputs.
9. Run ablation, robustness, and sensitivity workflows when requested.
10. Build the curated source-data package and validation report.

## 4. Configuration Defaults

The production fullscan uses:

| Parameter | Value |
| --- | ---: |
| `epochs` | 3500 |
| `scan_seeds` | `0,1,2` |
| `level_scan_min`, `level_scan_max` | 1, 8 |
| `optimizer` | `muon_hybrid` |
| `lr` after hyperopt | 0.0025 |
| `weight_decay` | 0.0001 |
| `grad_clip` | 5.0 |
| `early_stop_min_epochs` | 300 |
| `early_stop_warmup` | 300 |
| `early_stop_patience` | 45 |
| `peak_window_radius` | 1.5 |
| `kernel_mode` | `k_neutral` |
| `cluster_tolerance_eV` | 0.02 |
| `cpu_workers` | 4 |
| `dispatch_strategy` | `cpu_4` on CPU |

The production forward prior recorded in `config_used.json` has:

| Parameter | Value |
| --- | ---: |
| `forward_main_spacing` | 11.5 |
| `forward_spacing_std` | 0.4310839052125854 |
| `forward_confidence` | 0.9475202202349895 |
| `forward_anchor_step` | 0.25 |
| `w_prior_anchor` | 0.0005790080880939959 |
| `w_prior_gap` | 0.0003895040440469979 |

## 5. Physical Kernel

The model class is `PoissonRateFHCoreMultiLevel`. For a point with accelerating voltage `Va` and retarding voltage `Vr`, the core uses:

```text
E_collision = clamp(Va, min=0)
E_collect = Va - vr_scale * Vr
drive = softplus(E_collision - v_emit)
envelope = amp * drive^power / norm + offset + slope * E_collision
collector_transmission = collector_floor + (1 - collector_floor) * sigmoid((E_collect - collector_threshold) / collector_width)
```

High-voltage and late-response gates are:

```text
late_gate = sigmoid((E_collision - late_onset) / late_width)
high_vr_gate = sigmoid((Vr - 6.0) / 2.0)
vr_norm = clamp((Vr - 5.0) / 5.0, -1.5, 1.5)
contrast_scale = exp(clamp(vr_contrast * vr_norm + late_contrast * late_gate + vr_late_contrast * vr_norm * late_gate, -0.65, 0.65))
```

The high-energy attenuation factor is:

```text
high_energy_gate = sigmoid((E_collision - high_energy_loss_onset) / high_energy_loss_width)
high_energy_loss = 1 - high_energy_loss_strength * high_energy_gate * (0.35 + 0.65 * high_vr_gate)
```

For K effective channels with ordered energies `E_i` and softmax weights `w_i`, the oscillatory response is:

```text
phase_i = (E_collision + phase) / E_i
residual_i = phase_i - round(phase_i)
dip_i = exp(-0.5 * (residual_i * E_i / width)^2)
weighted_dip = sum_i w_i * dip_i
decay = exp(-damping * E_collision)
modulation = clamp(1 - osc_amp * contrast_scale * weighted_dip * decay, 0.03, 1.15)
prediction = clamp(envelope * collector_transmission * high_energy_loss * modulation, min=0)
```

Per-curve nuisance terms apply gain, bias, and accelerating-voltage offset after the neutral core. Energy parameters are bounded by construction: the first energy is mapped into the interval 9.0 to 14.5 V, and subsequent energies are produced by positive gaps with a minimum-gap regularizer.

## 6. Loss Function

For each curve, the training loss includes raw current error, first-derivative error, second-derivative error, peak-window weighted current error, smoothness, late-bias, late-amplitude-ratio, high-retarding-voltage valley-depth, and regularization. The total loss is:

```text
L = w_raw * L_raw
  + w_d1 * L_d1
  + w_d2 * L_d2
  + w_peak_window * L_peak
  + w_smooth * L_smooth
  + w_vr_late_bias * L_late_bias
  + w_vr_amplitude_ratio * L_late_ratio
  + w_high_vr_valley_depth * L_high_vr_valley
  + L_reg
```

The production weights are:

| Term | Weight |
| --- | ---: |
| `w_raw` | 1.0 |
| `w_d1` | 0.04 |
| `w_d2` | 0.02 |
| `w_peak_window` | 0.08 |
| `w_smooth` after hyperopt | 0.005 |
| `w_vr_late_bias` | 0.04 |
| `w_vr_amplitude_ratio` | 0.04 |
| `w_high_vr_valley_depth` | 0.06 |
| `w_reg` | 0.0001 |
| `level_weight_entropy` | 0.001 |

The regularization term includes parameter L2 penalties, a channel-weight entropy penalty, a minimum-gap penalty, and a forward-anchor penalty. The forward-anchor grid starts at the production main spacing when the forward prior is active; otherwise it falls back to 11.55 V with a 0.25 V step.

The early-stopping monitor is intentionally stricter than the train loss:

```text
L_monitor = L_total
          + 0.25 * L_d1
          + 0.10 * L_d2
          + 0.25 * L_peak
          + 0.20 * L_late_bias
          + 0.10 * L_late_ratio
          + 0.20 * L_high_vr_valley
```

## 7. Optimizer and Training

The default optimizer is `muon_hybrid`.

The local Muon branch keeps a momentum buffer for vector or matrix parameters and updates:

```text
buffer_t = momentum * buffer_{t-1} + gradient
denom = sqrt(mean(buffer_t^2))
theta_t = theta_{t-1} - lr * buffer_t / max(denom, 1e-8)
```

Weight decay is added to the gradient before the momentum update. Scalar parameters and parameters that do not fit the Muon branch use AdamW as a fallback. Setting `optimizer="adamw"` uses AdamW for all trainable parameters.

The training loop supports checkpoint resume in fullscan mode. A scorecard is considered current only if the model schema, requested epochs, early-stop floor, and configuration hash match. Full CPU scans use process-level parallelism with `cpu_workers=4`.

## 8. Hyperparameter Optimization

Fullscan enables hyperparameter optimization unless `--exclude hpopt` is passed. The method is successive halving:

| Stage | Epochs | Levels | Seeds | Retained candidates |
| --- | ---: | --- | --- | ---: |
| Stage 1 | 80 | 1,2,4,6,8 | 0 | 4 |
| Stage 2 | 240 | 2,4,6,8 | 0,1 | 2 |
| Stage 3 | 300 | 1,2,3,4,5,6,7,8 | 0,1 | 1 |

The candidate update set includes baseline, structure guard, low-smooth peak, stronger second derivative, balanced first/second derivative, low-smooth high-peak, retarding-voltage response guard, stronger retarding-voltage late-bias guard, stronger prior anchor, stronger prior gap, lower learning rate with longer patience, and higher learning rate with guard terms. The selected production candidate is `lr_higher_guarded`, which yields `lr=0.0025` and `w_smooth=0.005`.

The hyperopt score is:

```text
score = rmse_mean
      + 0.10 * structure_score
      + 0.08 * vr_physical_response_score
      + 0.03 * d1_rmse_mean
      + 0.03 * d2_rmse_mean
      + 0.35 if flatline_guard_pass is false
```

## 9. K-Neutral Model Selection

The selector scans K=1 through K=8 and uses rank aggregation rather than a single metric. Default rank weights are:

| Component | Weight |
| --- | ---: |
| fit RMSE | 1.0 |
| summary/CV RMSE | 1.0 |
| first derivative | 1.0 |
| second derivative | 1.0 |
| structure score | 1.25 |
| physical-response score | 1.0 |
| BIC | 1.0 |
| AIC | 0.5 |
| degeneracy penalty | 1.0 |

The composite score is the weighted sum of ranks, with penalties for top-heavy or low-weight channel degeneracy, clustered effective energies, and flatline failures. The selected K is the smallest composite score, with lower K used as the tie breaker.

The current production result is:

| Quantity | Value |
| --- | --- |
| selected K | 4 |
| best K by fit | 4 |
| best K by first derivative | 8 |
| best K by second derivative | 8 |
| best K by structure | 8 |
| best K by physical response | 4 |
| best K by BIC | 4 |
| best K by AIC | 4 |

This means K=4 is the selected effective response model under the production prior and selector workflow. K=8 remains an explicit boundary case for derivative and structure metrics.

## 10. Production K=4 Parameters

The retained production K=4 anchors are:

| Channel | Energy V | Weight |
| ---: | ---: | ---: |
| 1 | 11.500251770019531 | 0.3593961000442505 |
| 2 | 11.739255905151367 | 0.3961048126220703 |
| 3 | 12.59369945526123 | 0.08622404932975769 |
| 4 | 13.96471881866455 | 0.15827499330043793 |

These channels are effective response channels inferred from the teaching-laboratory curves. They are not term-resolved spectroscopic assignments.

## 11. Ablation Design

The ablation workflow has two classes.

Selector-only ablations recompute the decision table without retraining:

- remove weight-degeneracy penalty;
- remove energy-cluster degeneracy penalty;
- use a metric/BIC/AIC-only selector variant.

The no-forward-anchor-gap ablation retrains the scan with the forward anchor and gap prior disabled. In the current evidence set, selector-only ablations retain K=4, while the no-forward-anchor-gap retrain selects K=7. This defines a real boundary: K=4 is conditional on the production physical prior and audit workflow.

## 12. Robustness Design

The robustness workflow has two parts.

Selector-weight perturbation recomputes decisions over a fixed grid of rank-weight settings without retraining. The current package records K=4 in all 23 selector-weight perturbation scenarios.

Leave-one-retarding-voltage-out retraining excludes one retarding-voltage curve at a time. It reuses the main hyperparameter configuration and does not rerun hyperopt for each fold. The current summary selects K=4 in 3 of 5 folds; the remaining folds expose K=1 and K=6 boundary outcomes.

## 13. Sensitivity and Perturbation Design

The sensitivity workflow uses the existing main fullscan as its baseline.

Prior-strength scanning evaluates factors:

```text
0, 0.25, 0.5, 1, 2, 4
```

The `1x` row is defined as the production prior weights recorded in `main/fullscan/config_used.json`. It is not recompiled from an already prior-applied configuration. Current selected-K outcomes are:

| Prior factor | Selected K |
| ---: | ---: |
| 0 | 8 |
| 0.25 | 4 |
| 0.5 | 4 |
| 1 | 4 |
| 2 | 8 |
| 4 | 8 |

Uncertainty perturbations include:

- seed-dependent initialization jitter;
- residual bootstrap from selected-model residuals;
- curve-level noise perturbation from local raw-current residuals;
- peak-window-radius perturbation.

Residual bootstrap is strict: it samples from `main/k_selected_full/prediction_points.csv`, grouped by curve, and fails if that table is missing. It does not fall back to centered raw current. The package validation reports bootstrap noise standard deviation 0.07944132053812393 uA, selected-model residual standard deviation 0.08032634434767792 uA, and absolute difference 0.000885023809553992 uA.

K=4 uncertainty is reported in two separate forms:

- `conditional_k4_all_fits`: all fitted K=4 channels under stress conditions; this is a stress-test drift table.
- `production_anchor_matched_k4`: perturbed K=4 channels matched back to the four production anchors by minimum total absolute energy difference; this is the production-anchor matched uncertainty table.

## 14. Source-Data Package

The curated package contains:

- 29 manuscript-facing CSV/JSON source tables;
- 71 figure assets;
- 16 retained K=1 and selected K=4 run-record files;
- 193 selected output-result files;
- `FILE_INDEX.csv`, `SHA256SUMS.txt`, `validation_report.json`, and `source_data_package_manifest.md`.

The current validation status is `pass`. The package builder normalizes local absolute paths in text payloads before hashing and writing the manifest.

To rebuild the package from an existing formal output root:

```powershell
$env:SUBLEVEL_OUTPUT='output'
python scripts/build_source_data_package.py
```

If the manuscript visualization source-data and figure directories are outside this repository, provide them explicitly:

```powershell
$env:SUBLEVEL_SOURCE_TABLES='<path-to-source-data>'
$env:SUBLEVEL_FIGURES='<path-to-figures>'
python scripts/build_source_data_package.py
```

## 15. Verification Commands

The repository-level checks are:

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest -q
python scripts/build_source_data_package.py
```

The package-level consistency checks are stored in `source_data_package/validation_report.json` and cover:

- source-table count equals 29;
- figure-asset count equals 71;
- prior-strength `1x` selects K=4;
- production anchor energies match the retained K=4 anchors;
- residual-bootstrap noise scale is within 0.01 uA of the selected-model residual scale;
- all package files have SHA256 hashes.
