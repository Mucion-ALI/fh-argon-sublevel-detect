# SubLevel Detect Memory

## 2026-06-29 Source Data Package Commit Preparation

Plan:

- Package the manuscript source data into the GitHub-facing source repository
  without mixing ordinary rerun outputs into version control.
- Keep normal runtime outputs ignored, but allow the curated
  `source_data_package/` archive to include retained checkpoints and run
  records needed for manuscript-level file checking.
- Update the English and Chinese GitHub README files so repository users can
  distinguish normal reproducible runs from the committed manuscript source
  data package.

Execution result:

- Created `source_data_package/` with `manuscript_source_tables/`,
  `figures/main/`, `figures/supplementary/`, `run_records/k1_full/`, and
  `run_records/k_selected_full/`.
- Added package metadata files: `README.md`, `FILE_INDEX.csv`,
  `SHA256SUMS.txt`, `validation_report.json`, and
  `source_data_package_manifest.md`.
- Updated `.gitignore` so newly generated checkpoints remain ignored while
  curated checkpoint files under `source_data_package/` can be committed.
- Updated `README.md` and `README.zh-CN.md` with the source-data package scope
  and validation summary.

Verification:

- Manifest validation passed for 88 payload files: 19 manuscript source tables,
  55 generated figure files, and 14 retained K=1/K=4 run-record files.
- File sizes and SHA256 hashes matched the source manifest.
- The 19 manuscript-facing CSV/JSON files parsed successfully.
- Remote synchronization completed on the GitHub repository at commit
  `1da880e8c5e5b11b7a738a384032666ab153fac3`.
- Remote verification read back `README.md`,
  `source_data_package/validation_report.json`, and
  `source_data_package/SHA256SUMS.txt` from `main`.

## 2026-06-29 Sensitivity Pipeline Supplement

Plan:

- Add a `--sensitivity` workflow for forward-anchor prior-strength scanning and
  conditional channel-parameter uncertainty estimation.
- Keep the main production workflow unchanged by default; sensitivity runs
  should reuse existing main fullscan outputs and skip hyperparameter
  optimization when requested.

Execution result:

- Added `src/sublevel_detect/sensitivity_pipeline.py` and wired `--sensitivity`
  into the CLI.
- Added `Config.init_jitter_scale` with default `0.0`, seed-dependent model
  initialization jitter for sensitivity runs, and `peak_window_radius` plumbing
  into `curve_to_tensors`.
- Added forward-prior strength scaling for factors `0`, `0.25`, `0.5`, `1`,
  `2`, and `4`.
- Added condition-level skip/reuse logic so completed sensitivity scans can be
  summarized without retraining.
- Added manuscript-facing outputs:
  `prior_strength_selection.csv`, `prior_strength_channel_drift.csv`,
  `channel_uncertainty_samples.csv`, `channel_uncertainty_summary.csv`,
  `uncertainty_selection_summary.csv`, and `sensitivity_summary.json`.
- Fixed `summarize_channel_uncertainty` so K=4 intervals are computed from
  all `n_levels=4` fits, not only scenarios where `selected_k=4`.

Verification:

- Full sensitivity run completed with 6 prior-strength rows and 7740
  uncertainty sample rows.
- Prior-strength result: K=4 only at 0.25x and 0.5x; 0, 1x, 2x, and 4x select
  K=8.
- Uncertainty selected-K result: seed jitter selects K=8; window-radius scans
  select K=8 in 4/5 and K=7 in 1/5; residual-bootstrap and noise-perturbation
  scenarios select K=5-8 and never K=4.
- Conditional K=4 channel summary contains 215 samples per channel plus the
  E2-E1 gap row.
- Full test suite passed: `python -m pytest -q` reported 22 passing tests.
- Sensitivity smoke test passed with
  `python run.py --mode smoke --exclude hpopt --sensitivity --output C:\tmp\sublevel_detect_sensitivity_smoke --device cpu`;
  the temporary smoke directory was removed after output checks.

## 2026-06-29 Fig. 11 Bar-Chart Visual Refinement

Plan:

- Refine the manuscript-facing K=4 channel uncertainty plot after visual
  feedback that the bar chart was too heavy.
- Keep the figure as a channel-indexed vertical bar chart with uncertainty
  intervals; do not change the underlying source data or uncertainty
  definition.

Execution result:

- Narrowed the Fig. 11 bars, increased channel spacing, and slightly lightened
  the uncertainty error bars in
  `ESSAY/ajp_argon_sublevel_manuscript/visualization/scripts/make_figures.py`.
- Added a regression assertion in `tests/test_manuscript_figures.py` so the
  channel-uncertainty plot continues to use narrow bars with channel spacing.

Verification:

- `python -m pytest tests\test_manuscript_figures.py -q` passed.
- Full test suite passed after the figure refinement:
  `python -m pytest -q` reported 23 passing tests.

## 2026-06-29 Sensitivity Pipeline Second-Pass Repair and Recompute

Plan:

- Repair sensitivity workflow defects before recomputing: prior-strength
  scaling must use the recorded production prior weights directly, residual
  bootstrap must sample selected-model residuals, and K=4 uncertainty must be
  split into conditional stress-test drift and production-anchor matched
  summaries.
- Keep legacy `outputs/` intact. Write the new audit run to
  `outputs_recomputed_v2/` and keep paper directories unchanged until manual
  review.
- Update repository documentation so the public reproduction target is
  `SubLevel_Detect` and the recommended audit commands use the independent
  recomputation output root.

Execution result:

- Added targeted tests for production-prior `1x` scaling, residual-bootstrap
  residual-source enforcement, residual noise scale, anchor-matched K=4
  summaries, and `outputs_recomputed_v2` CLI output isolation.
- Added selected-model point prediction export to
  `main/k_selected_full/prediction_points.csv` with fixed columns
  `curve_id,Vr,Va,observed,predicted,residual`.
- Updated sensitivity prior-strength scans so factors `0.25`, `0.5`, `1`,
  `2`, and `4` scale the production weights recorded in
  `main/fullscan/config_used.json`; `1x` is no longer recompiled from an
  already prior-applied config.
- Removed the residual-bootstrap `current - mean(current)` fallback. Missing
  `prediction_points.csv` now raises a clear error and asks the user to run the
  main workflow first.
- Added `channel_uncertainty_conditional_k4.csv`,
  `channel_uncertainty_anchor_matched.csv`, and
  `channel_uncertainty_anchor_matched_samples.csv`; retained
  `channel_uncertainty_summary.csv` as the conditional K=4 compatibility alias.
- Updated `README.md`, `README.zh-CN.md`, `docs/reproduction.md`,
  `docs/outputs.md`, and `docs/paper_evidence.md` with the new reproduction
  commands and sensitivity interpretation rules.

Verification so far:

- `python -m pytest tests/test_sensitivity_pipeline.py -q` passed with 11
  tests.
- `python -m pytest tests/test_cli_contract.py -q` passed with 14 tests.
- `python -m compileall -q run.py src/sublevel_detect` passed.
- `python -m pytest -q` passed with 28 tests.
- Smoke validation passed with
  `python run.py --mode smoke --exclude hpopt --ablation --sensitivity --output C:\tmp\sublevel_detect_smoke_v2 --device cpu`;
  the temporary smoke directory was removed after verification.
- Full recomputation completed under `outputs_recomputed_v2/`:
  main fullscan selected K=4 with hyperparameter optimization enabled;
  ablation and robustness completed; sensitivity reused the recomputed main
  sweep and produced 6 prior-strength rows plus 7740 uncertainty sample rows.
- Main selected K=4 production channels were unchanged relative to legacy
  `outputs/` and the root core artifact: energies
  11.500252, 11.739256, 12.593699, and 13.964719 V with weights
  0.359396, 0.396105, 0.086224, and 0.158275.
- Model-selection diagnostics were unchanged: selected K=4; best-by-fit K=4;
  best-by-d1/d2/structure K=8; best-by-physical/BIC/AIC K=4.
- Ablation and robustness were unchanged: no-forward-anchor-gap selected K=7;
  selector-only ablations selected K=4; selector perturbation selected K=4 in
  23/23 scenarios; leave-one-Vr-out selected K=4 in 3/5 folds, with K=1 and
  K=6 in the remaining folds.
- Prior-strength labels changed as expected after fixing the scaling basis:
  the recomputed `1x` weights equal production
  (`w_prior_anchor=0.0005790080880939`,
  `w_prior_gap=0.0003895040440469`) and select K=4. The old `1x` row used
  inflated weights and selected K=8.
- Residual bootstrap scale changed from fallback-like strong perturbation to
  true selected-model residual perturbation: old replicate 000 noise std was
  0.723779, while recomputed replicate 000 noise std was 0.079441, matching
  the selected-model residual std 0.080326 and far below the raw current std
  0.757117.
- Recomputed production-anchor matched K=4 uncertainty means were
  11.497609, 11.734655, 12.418945, and 13.882520 V. Channels 1 and 2 are
  essentially anchored to production; channels 3 and 4 show negative stress
  drift of -0.174755 V and -0.082199 V.
- Independent review outputs were written under `recomputed_review/` with
  comparison tables and PNG figures. No paper files under `ESSAY/` were
  modified.

## 2026-06-30 Output Path Consolidation and Source Data Package Rebuild

Plan:

- Treat `output/` as the single active formal output root.
- Archive legacy runtime data rather than deleting it.
- Rebuild the manuscript-facing `source_data_package/` from current `output/`
  and the manuscript visualization workspace.
- Update code, tests, README, and manuscript-facing scripts so active
  reproduction no longer depends on `outputs_recomputed_v2/`,
  `recomputed_review/`, or legacy `Core_Data` artifacts.

Execution result:

- Renamed the completed recomputation directory from `outputs_recomputed_v2/`
  to `output/`.
- Archived legacy `outputs/`, `recomputed_review/`, and the previous
  `source_data_package/` snapshot under `FH-old/archive_data/`.
- Updated defaults in `paths.py`, `model.py`, and `monitor.py` so the default
  output root and monitor root are `output/`.
- Updated CLI contract tests to assert the single formal output directory name.
- Updated manuscript figure/source-data scripts to read from `output/` and to
  support the current `ablation_summary.csv` group naming.
- Added `scripts/build_source_data_package.py` to rebuild the curated source
  data bundle reproducibly.
- Rebuilt `source_data_package/` with 29 manuscript source tables, 71 figure
  assets, 16 minimal K=1/K=4 run-record files, and selected `output/` result
  files.

Verification snapshot:

- `python scripts/build_source_data_package.py` passed.
- Package validation status: pass.
- Package file count: 309.
- Key checks passed: 29 source tables, 71 figures,
  `prior_strength=1x selected_k=4`, four production anchor energies
  `11.500251770019531`, `11.739255905151367`, `12.59369945526123`, and
  `13.96471881866455`, and residual-bootstrap noise std within 0.01 uA of the
  selected-model residual std.

## 2026-06-30 Local GitHub Clone Synchronization and Technical Reports

Plan:

- Clone the remote GitHub source repository into the local workspace and
  compare cloud, cloned, and current `SubLevel_Detect` states.
- Treat current `SubLevel_Detect` as the implementation source of truth and
  synchronize the local GitHub clone without copying the full ignored
  `output/` runtime tree into version control.
- Add complete English and Chinese technical reports covering workflow,
  physical kernel, loss terms, training, optimizer choice, hyperparameter
  search, selector design, perturbations, ablations, robustness, and source
  data packaging.
- Update both repository README files and keep local absolute paths out of
  committed source-data payloads.

Execution result:

- Cloned the remote repository into the requested local workspace path.
- Verified the clone and remote initially matched commit
  `1da880e8c5e5b11b7a738a384032666ab153fac3`.
- Synchronized current source code, tests, docs, scripts, and the curated
  source-data package into the local clone.
- Added `Techique_Report.md` and `Techique_Report_zh.md` using the filename
  spelling requested by the user.
- Updated `README.md` and `README.zh-CN.md` with technical report links,
  effective multichannel response language, package rebuild instructions, and
  validation status.
- Updated `scripts/build_source_data_package.py` so the standalone repository
  resolves paths from the repository root, accepts environment-variable
  overrides, skips duplicate supplementary heatmap assets, normalizes text
  payload paths, and writes hashes after normalization.
- Rebuilt `source_data_package/` from the current formal output and manuscript
  visualization source data. Validation status remained `pass` with 309 files,
  29 source tables, 71 figure assets, 16 run-record files, and 193 selected
  output-result files.

Verification status:

- `python -m compileall -q run.py src/sublevel_detect` passed.
- `python -m pytest -q --basetemp _pytest_tmp` passed with the external
  manuscript figure regression skipped when the manuscript project is not part
  of the standalone source repository contract.
- `python scripts/build_source_data_package.py` passed with
  `SUBLEVEL_OUTPUT` pointing to the current formal output root.
- Source-data package validation remained `pass`.
- Searches over reports, README files, docs, source, tests, scripts, and the
  source-data package found no local absolute workspace paths or explicit
  personal GitHub repository strings.
- Old uncertainty summary values were absent from reports, README files, docs,
  source, tests, scripts, and memory. Similar numeric values may remain inside
  raw source-data sample rows, where they are legitimate perturbation samples
  rather than narrative uncertainty claims.

## 2026-06-30 GitHub Publication Commit

Plan:

- Commit the synchronized standalone repository state to the GitHub remote.
- Keep the commit scope limited to the current workflow synchronization,
  technical reports, README updates, source-data package refresh, and tests.
- Re-run the lightweight verification checks before committing and remove
  temporary pytest and Python cache directories.

Execution result:

- Confirmed the repository remote points to the intended GitHub project.
- Verified the working tree changes match the expected synchronization scope.
- Re-ran compile, pytest, validation-report, and path-anonymity checks before
  staging.
- Prepared the repository for a single publication commit.
