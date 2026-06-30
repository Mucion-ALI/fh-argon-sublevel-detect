# Source Data Package

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

Validation status: `pass`.
