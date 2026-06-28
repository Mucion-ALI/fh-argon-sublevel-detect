# Output Structure

`outputs/main/fullscan/` contains the main K scan:

- `decision.json`: selected K and selector diagnostics.
- `model_selection_table.csv`: one row per scored K.
- `scan_summary.csv`: training and metric summary.
- `vr_physical_response.csv`: per-curve physical response audit rows.
- `structure_metrics.csv`: peak-valley and shape-preservation summary.
- `production_run_summary.json`: command-level run metadata.

`outputs/main/` contains paper-facing summaries:

- `paper_summary.json`
- `paper_summary.md`

`outputs/ablation/` contains ablation evidence:

- `ablation_summary.csv`
- `ablation_summary.json`
- `ablation_report.md`
- `selector_ablation_decision.json`
- `ablation_metric_table.csv`
- `energy_cluster_table.csv`

Generated checkpoint files are intentionally ignored by git.
