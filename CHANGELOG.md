# Changelog

## 2026-08-03

### Added
- Added support for carrying `target_mass_flow_g_per_h` and `target_displacement_um` from `step.json` into the batch metadata pipeline.
- Included both new fields in the `ALL_RUNS.csv` summary schema.
- Included both new fields in the `ALL_PARTICLE_RESULTS.csv` per-particle output schema.

### Updated
- Refreshed existing combo CSVs so older result files also receive the new metadata columns when the update path is run.
- Rebuilt the combined and all-runs exports to reflect the new schema.

### Notes
- The current workspace script folder is not a Git repository, so these changes cannot be pushed from this location until the actual repo checkout is opened.
