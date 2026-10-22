# Changelog

## 2026-10-22

### Added
- Added `FORCE_REANALYSIS_RUNS` to rerun selected experiment folders while retaining registry checks for other runs. Both global and folder-specific forced reanalysis are disabled in this version.
- Added material catalog validation, legacy glycerol ID resolution, canonical `Material_ID` values, and surface tension, density, and viscosity enrichment in particle and run-summary exports.
- Added nozzle catalog validation and dimensionless droplet diameter (`dd/di = ECD_mm / inner_diameter_mm`) in particle exports.
- Added `Frequency_index` labels within each run and `Displacement_index` labels within material/frequency/nozzle/target-flow groups, including refresh of existing CSVs when measurement levels change.
- Added `--update-material-id-only` and `--update-dimensionless-diameter-only` commands to refresh existing results without repeating image analysis.
- Added 12 regression tests for level labeling, material enrichment, catalog validation, dimensionless diameter, and repeatable CSV updates.
- Added editable workflow diagrams, Mermaid sources, a browser viewer, and methods documentation.
- Added Git ignore rules for Python caches and diagrams.net temporary/backup files.

### Updated
- Changed the input root from `DoE_V1/VERSUCHE` to `DoE_V1/SCREENING`.
- Set calibration to `0.01742 mm/px` and the crop to x=690, y=10, width=1660, height=3900 pixels; aligned workflow documentation with these settings.
- Integrated material properties and level labels into batch processing, metadata updates, and aggregate CSV rebuilds.

### Validation
- `python -B -m unittest -v test_master_batch`: all 12 tests passed.
- Image analysis and result regeneration were not run as part of this release.

## 2026-08-03

### Added
- Added support for carrying `target_mass_flow_g_per_h` and `target_displacement_um` from `step.json` into the batch metadata pipeline.
- Included both new fields in the `ALL_RUNS.csv` summary schema.
- Included both new fields in the `ALL_PARTICLE_RESULTS.csv` per-particle output schema.

### Updated
- Refreshed existing combo CSVs so older result files also receive the new metadata columns when the update path is run.
- Rebuilt the combined and all-runs exports to reflect the new schema.
