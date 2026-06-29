# Source Data Package

This directory contains the manuscript source-data package for the Frank-Hertz argon sublevel-detection study. It is a curated archive of the files listed in `source_data_package_manifest.md` and is intended to support figure-level checking and manuscript-table reproduction without expanding Appendix C into a full file inventory.

## Contents

- `manuscript_source_tables/`: CSV/JSON tables used directly by manuscript figures, model-selection tables, physical-response audits, ablation discussion, and robustness checks.
- `figures/main/`: main-text figure assets in PDF/PNG/SVG formats.
- `figures/supplementary/`: supplementary figure assets in PDF/PNG formats.
- `run_records/k1_full/` and `run_records/k_selected_full/`: retained K=1 and selected-K run records, including metrics, parameters, scorecards, logs, status files, and checkpoints used to document the retained runs.
- `FILE_INDEX.csv`: package path, original local source path, file size, and SHA256 for every payload file.
- `SHA256SUMS.txt`: checksum list for payload files and package metadata.
- `validation_report.json`: machine-readable validation summary generated from the manifest.

## Validation

The package was generated from `ESSAY/ajp_argon_sublevel_manuscript/source_data_package_manifest.md`. The generation check verified 88 manifest payload files: 19 manuscript source tables, 55 generated figure files, and 14 retained K=1/K=4 run-record files. File sizes and SHA256 hashes matched the manifest, and the 19 manuscript-facing CSV/JSON source tables were parsed successfully.

The package is separate from normal runtime outputs. New smoke-test or rerun outputs should not be mixed into this directory unless the manifest and checksum files are regenerated.
