"""
Master Batch Program for Automated Particle Image Analysis
==========================================================

This file handles the infrastructure around the image analysis:
    - discover experiment/run folders in DATA_ROOT
    - discover combo folders below run_folder/steps/combo_*/
    - check analysis_registry.csv to prevent duplicate combo analysis
    - read step.json metadata
    - pass the images/ folder to image_analysis_core.py
    - add metadata columns to the particle results
    - save one CSV per combo folder
    - rebuild one combined ALL_PARTICLE_RESULTS.csv

The actual image analysis is performed in image_analysis_core.py.
Keep the segmentation and measurement logic there unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from image_analysis_core import AnalysisConfig, process_image_folder


# =============================================================================
# Main settings: edit this section
# =============================================================================

DATA_ROOT = Path(
    r"W:\PROMOTION_SHARIFSOLTANI_VERKAPSELUNG_DFG_TH1817_9_1_MT\DoE_V1\VERSUCHE"
)

RESULTS_ROOT = Path(
    r"W:\PROMOTION_SHARIFSOLTANI_VERKAPSELUNG_DFG_TH1817_9_1_MT\BILDANALYSE\TS\RESULTS"
)

# Output folders created below RESULTS_ROOT
CSV_ROOT = RESULTS_ROOT / "CSV"
IMAGE_OUTPUT_ROOT = RESULTS_ROOT / "IMAGES"
REGISTRY_PATH = RESULTS_ROOT / "analysis_registry.csv"
COMBINED_CSV_PATH = CSV_ROOT / "ALL_PARTICLE_RESULTS.csv"
ALL_RUNS_CSV_PATH = CSV_ROOT / "ALL_RUNS.csv"

# Re-run combos even if they are marked as completed in analysis_registry.csv.
FORCE_REANALYSIS = True

# Calibration
PIXEL_SIZE_MM = 0.0784

# Crop region
CROP_X = 1000
CROP_Y = 456
CROP_WIDTH = 1400
CROP_HEIGHT = 2430

# QC/output behavior
SAVE_OVERLAY = True
SAVE_QC_PANELS = True
QC_EVERY_N_IMAGES = 10

# Segmentation settings
THRESHOLD_METHOD = "fixed"
FIXED_THRESHOLD = 2
OTSU_RELAXATION_FACTOR = 0.35
PERCENTILE_THRESHOLD = 90.0
ADAPTIVE_BLOCK_SIZE = 151
ADAPTIVE_OFFSET = 15.0

# Background correction
BACKGROUND_SIGMA = 80.0
DARK_PARTICLES = True

# Contrast enhancement before thresholding
USE_CONTRAST_ENHANCEMENT = True
ENHANCEMENT_MIN_INTENSITY = 5
ENHANCEMENT_FACTOR = 2.5

# Particle filtering
MIN_AREA_PX = 20
MAX_AREA_PX = None
MIN_CIRCULARITY = 0.50
MAX_CIRCULARITY = 1.10
EXCLUDE_BORDER = True

# Mask cleanup
USE_OPENING = False
USE_CLOSING = False
OPENING_RADIUS = 1
CLOSING_RADIUS = 1
FILL_HOLES = True

# Watershed
USE_WATERSHED = False
WATERSHED_MIN_DISTANCE = 20

# Diagnostics
PRINT_DIAGNOSTICS = True


# =============================================================================
# Registry handling
# =============================================================================

REGISTRY_COLUMNS = [
    "run_folder",
    "combo_folder",
    "status",
    "completed_at",
    "csv_path",
    "n_images",
    "n_particles",
    "error",
]


def load_registry(path: Path) -> pd.DataFrame:
    """Load the combo-level analysis registry, or create an empty one."""
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    registry = pd.read_csv(path)

    for col in REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = ""

    return registry[REGISTRY_COLUMNS].copy()


def save_registry(registry: pd.DataFrame, path: Path) -> None:
    """Save the analysis registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)


def combo_is_completed(registry: pd.DataFrame, run_folder: str, combo_folder: str) -> bool:
    """Return True if this run/combo pair is already marked completed."""
    if registry.empty:
        return False

    match = (
        (registry["run_folder"] == run_folder)
        & (registry["combo_folder"] == combo_folder)
        & (registry["status"] == "completed")
    )

    return bool(match.any())


def update_registry(
    registry: pd.DataFrame,
    *,
    run_folder: str,
    combo_folder: str,
    status: str,
    csv_path: Path | None = None,
    n_images: int = 0,
    n_particles: int = 0,
    error: str = "",
) -> pd.DataFrame:
    """Insert or update one registry row for one combo folder."""
    now = datetime.now().isoformat(timespec="seconds")

    csv_path_str = str(csv_path) if csv_path is not None else ""

    row = {
        "run_folder": run_folder,
        "combo_folder": combo_folder,
        "status": status,
        "completed_at": now if status == "completed" else "",
        "csv_path": csv_path_str,
        "n_images": n_images,
        "n_particles": n_particles,
        "error": error,
    }

    match = (
        (registry["run_folder"] == run_folder)
        & (registry["combo_folder"] == combo_folder)
    )

    if match.any():
        idx = registry.index[match][0]
        for key, value in row.items():
            registry.at[idx, key] = value
    else:
        registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)

    return registry


# =============================================================================
# Folder discovery and metadata handling
# =============================================================================


def count_jpeg_files(folder: Path) -> int:
    """Count JPEG images directly inside one folder."""
    if not folder.exists():
        return 0

    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def discover_combo_folders(data_root: Path) -> list[tuple[Path, Path]]:
    """
    Discover all valid combo folders below DATA_ROOT.

    Returns a list of tuples:
        (run_folder, combo_folder)
    """
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    combo_folders: list[tuple[Path, Path]] = []

    for run_folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        steps_folder = run_folder / "steps"

        if not steps_folder.exists():
            print(f"Skipping run folder without steps/: {run_folder}")
            continue

        for combo_folder in sorted(p for p in steps_folder.iterdir() if p.is_dir()):
            images_folder = combo_folder / "images"
            step_json = combo_folder / "step.json"

            if not images_folder.exists():
                print(f"Skipping combo without images/: {combo_folder}")
                continue

            if not step_json.exists():
                print(f"Skipping combo without step.json: {combo_folder}")
                continue

            combo_folders.append((run_folder, combo_folder))

    return combo_folders


def load_step_json(combo_folder: Path) -> dict[str, Any]:
    """Read step.json from one combo folder."""
    step_json_path = combo_folder / "step.json"

    with step_json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_key(key: str) -> str:
    """Normalize JSON keys for robust metadata lookup."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def find_value_recursive(data: Any, aliases: set[str]) -> Any | None:
    """
    Search nested dictionaries/lists for a key matching one of the aliases.

    The comparison is case-insensitive and ignores underscores, hyphens, and spaces.
    """
    normalized_aliases = {normalize_key(alias) for alias in aliases}

    if isinstance(data, dict):
        for key, value in data.items():
            if normalize_key(str(key)) in normalized_aliases:
                return value

        for value in data.values():
            found = find_value_recursive(value, aliases)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = find_value_recursive(item, aliases)
            if found is not None:
                return found

    return None


def normalize_column_name(name: str) -> str:
    """Normalize DataFrame column names for key matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_scale_csv_path(combo_folder: Path, step_data: dict[str, Any]) -> Path | None:
    """Resolve the scale.csv path from step.json or the run folder."""
    scale_csv_value = find_value_recursive(
        step_data,
        {"scale_csv", "scale.csv", "scaleCsv", "scale"},
    )

    if scale_csv_value is not None:
        candidate = Path(str(scale_csv_value))
        if not candidate.is_absolute():
            candidate = combo_folder / candidate

        if candidate.exists():
            return candidate

        run_folder = combo_folder.parent.parent
        candidate = run_folder / candidate.name
        if candidate.exists():
            return candidate

    run_folder = combo_folder.parent.parent
    fallback = run_folder / "scale.csv"
    if fallback.exists():
        return fallback

    return None


def resolve_oscilloscope_csv_path(combo_folder: Path, step_data: dict[str, Any]) -> Path | None:
    """Resolve the oscilloscope.csv path from step.json or the run folder."""
    oscilloscope_csv_value = find_value_recursive(
        step_data,
        {"oscilloscope_csv", "oscilloscope.csv", "oscilloscopeCsv", "oscilloscope"},
    )

    if oscilloscope_csv_value is not None:
        candidate = Path(str(oscilloscope_csv_value))
        if not candidate.is_absolute():
            candidate = combo_folder / candidate

        if candidate.exists():
            return candidate

        run_folder = combo_folder.parent.parent
        candidate = run_folder / candidate.name
        if candidate.exists():
            return candidate

    run_folder = combo_folder.parent.parent
    fallback = run_folder / "oscilloscope.csv"
    if fallback.exists():
        return fallback

    return None


def compute_avg_pump_speed(combo_folder: Path, step_data: dict[str, Any]) -> tuple[float | None, float | None]:
    """Compute the average pump speed in g/s and g/h for one combo."""
    combo_index_value = find_value_recursive(
        step_data,
        {
            "combo_index",
            "combo index",
            "comboindex",
            "step_index",
            "step index",
            "stepindex",
        },
    )

    if combo_index_value is None:
        print(f"Warning: combo index not found in step.json for {combo_folder}")
        return None, None

    try:
        combo_index = int(float(combo_index_value))
    except (TypeError, ValueError):
        print(f"Warning: could not parse combo index '{combo_index_value}' in {combo_folder}")
        return None, None

    scale_csv_path = resolve_scale_csv_path(combo_folder, step_data)
    if scale_csv_path is None:
        print(f"Warning: scale.csv not found for {combo_folder}")
        return None, None

    try:
        scale_df = pd.read_csv(scale_csv_path, sep=None, engine="python")
    except Exception:
        try:
            scale_df = pd.read_csv(scale_csv_path, sep=";", engine="python")
        except Exception as exc:
            print(f"Warning: failed to read scale.csv at {scale_csv_path}: {exc}")
            return None, None

    normalized_columns = {normalize_column_name(c): c for c in scale_df.columns}
    combo_col = normalized_columns.get("comboindex")
    weight_col = normalized_columns.get("weightg")
    elapsed_col = normalized_columns.get("elapseds")

    if combo_col is None or weight_col is None or elapsed_col is None:
        print(
            f"Warning: scale.csv is missing required columns for {combo_folder}. "
            "Expected combo index, weight_g, and elapsed_s."
        )
        return None, None

    scale_df = scale_df[[combo_col, weight_col, elapsed_col]].copy()
    scale_df[combo_col] = pd.to_numeric(scale_df[combo_col], errors="coerce")
    scale_df[weight_col] = pd.to_numeric(scale_df[weight_col], errors="coerce")
    scale_df[elapsed_col] = pd.to_numeric(scale_df[elapsed_col], errors="coerce")

    subset = scale_df[scale_df[combo_col] == combo_index]
    if subset.empty:
        print(f"Warning: no rows found for combo index {combo_index} in {scale_csv_path}")
        return None, None

    subset = subset.sort_values(elapsed_col)
    if len(subset) < 2:
        print(f"Warning: not enough rows for combo index {combo_index} in {scale_csv_path}")
        return None, None

    first_row = subset.iloc[0]
    last_row = subset.iloc[-1]
    delta_weight = first_row[weight_col] - last_row[weight_col]
    delta_time = last_row[elapsed_col] - first_row[elapsed_col]

    if pd.isna(delta_weight) or pd.isna(delta_time):
        print(
            f"Warning: invalid weight or elapsed values for combo index {combo_index} in {scale_csv_path}"
        )
        return None, None

    if delta_time == 0:
        print(f"Warning: zero elapsed time for combo index {combo_index} in {scale_csv_path}")
        return None, None

    avg_g_s = delta_weight / delta_time
    avg_g_h = avg_g_s * 3600.0
    return avg_g_s, avg_g_h


def compute_mean_p2p_displacement(combo_folder: Path, step_data: dict[str, Any]) -> float | None:
    """Compute the mean p2p displacement from oscilloscope.csv for one combo."""
    combo_index_value = find_value_recursive(
        step_data,
        {
            "combo_index",
            "combo index",
            "comboindex",
            "step_index",
            "step index",
            "stepindex",
        },
    )

    if combo_index_value is None:
        print(f"Warning: combo index not found in step.json for {combo_folder}")
        return None

    try:
        combo_index = int(float(combo_index_value))
    except (TypeError, ValueError):
        print(f"Warning: could not parse combo index '{combo_index_value}' in {combo_folder}")
        return None

    osc_csv_path = resolve_oscilloscope_csv_path(combo_folder, step_data)
    if osc_csv_path is None:
        print(f"Warning: oscilloscope.csv not found for {combo_folder}")
        return None

    try:
        osc_df = pd.read_csv(osc_csv_path, sep=None, engine="python")
    except Exception:
        try:
            osc_df = pd.read_csv(osc_csv_path, sep=";", engine="python")
        except Exception as exc:
            print(f"Warning: failed to read oscilloscope.csv at {osc_csv_path}: {exc}")
            return None

    normalized_columns = {normalize_column_name(c): c for c in osc_df.columns}
    combo_col = normalized_columns.get("comboindex")
    p2p_col = normalized_columns.get("p2pdisplacementum")

    if combo_col is None or p2p_col is None:
        print(
            f"Warning: oscilloscope.csv is missing required columns for {combo_folder}. "
            "Expected combo index and p2p_displacement_um."
        )
        return None

    osc_df = osc_df[[combo_col, p2p_col]].copy()
    osc_df[combo_col] = pd.to_numeric(osc_df[combo_col], errors="coerce")
    osc_df[p2p_col] = pd.to_numeric(osc_df[p2p_col], errors="coerce")

    subset = osc_df[osc_df[combo_col] == combo_index]
    if subset.empty:
        print(f"Warning: no rows found for combo index {combo_index} in {osc_csv_path}")
        return None

    mean_value = subset[p2p_col].mean(skipna=True)
    if pd.isna(mean_value):
        print(f"Warning: no valid p2p_displacement_um values for combo index {combo_index} in {osc_csv_path}")
        return None

    return float(mean_value)


def run_timestamp_from_folder(run_folder: Path) -> str:
    """Extract the timestamp part from a run folder name."""
    return run_folder.name.split("__", maxsplit=1)[0]


def experiment_id_from_folder(run_folder: Path) -> str:
    """Extract the ExperimentID part from a run folder name, if present."""
    parts = run_folder.name.split("__", maxsplit=1)
    if len(parts) == 2:
        return parts[1]
    return run_folder.name


def extract_metadata(step_data: dict[str, Any], run_folder: Path, combo_folder: Path) -> dict[str, Any]:
    """Extract required metadata from step.json, with safe fallbacks."""
    metadata = {
        "run_folder": run_folder.name,
        "combo_folder": combo_folder.name,
        "run_timestamp": run_timestamp_from_folder(run_folder),
        "experiment_id": find_value_recursive(
            step_data,
            {"experiment_id", "Experiment_ID", "ExperimentID", "experimentID"},
        ),
        "nozzle_id": find_value_recursive(
            step_data,
            {"nozzle_id", "Nozzle_ID", "NozzleID", "nozzleID"},
        ),
        "set_speed_rpm": find_value_recursive(
            step_data,
            {"set_speed_rpm", "SetSpeedRPM", "setSpeedRpm", "speed_rpm", "rpm"},
        ),
        "target_mass_flow_g_per_h": find_value_recursive(
            step_data,
            {
                "target_mass_flow_g_per_h",
                "TargetMassFlowGPerH",
                "targetMassFlowGPerH",
                "mass_flow_g_per_h",
                "target_mass_flow",
            },
        ),
        "target_displacement_um": find_value_recursive(
            step_data,
            {
                "target_displacement_um",
                "TargetDisplacementUm",
                "targetDisplacementUm",
                "displacement_um",
                "target_displacement",
            },
        ),
        "set_frequency_hz": find_value_recursive(
            step_data,
            {
                "set_frequency_hz",
                "SetFrequencyHz",
                "setFrequencyHz",
                "frequency_hz",
                "frequency",
            },
        ),
        "set_amplitude_vpp": find_value_recursive(
            step_data,
            {
                "set_amplitude_vpp",
                "SetAmplitudeVpp",
                "setAmplitudeVpp",
                "amplitude_vpp",
                "amplitude",
            },
        ),
    }

    if metadata["experiment_id"] is None:
        metadata["experiment_id"] = experiment_id_from_folder(run_folder)

    for key, value in list(metadata.items()):
        if value is None:
            metadata[key] = ""

    return metadata


# =============================================================================
# Output helpers
# =============================================================================


def safe_filename(name: str) -> str:
    """Return a Windows-safe filename component."""
    name = re.sub(r"[<>:\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._ ")


def make_combo_label(run_folder: Path, combo_folder: Path) -> str:
    """Create a stable label for output files from run + combo folder names."""
    return safe_filename(f"{run_folder.name}__{combo_folder.name}")


def add_metadata_columns(result_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Add run/step metadata columns to the particle result table."""
    result_df = result_df.copy()

    columns_in_order = [
        "run_folder",
        "combo_folder",
        "run_timestamp",
        "experiment_id",
        "nozzle_id",
        "set_speed_rpm",
        "target_mass_flow_g_per_h",
        "target_displacement_um",
        "set_frequency_hz",
        "set_amplitude_vpp",
        "avg_pump_speed_g_s",
        "avg_pump_speed_g_h",
        "mean_p2p_displacement_um",
    ]

    for col in reversed(columns_in_order):
        if col not in result_df.columns:
            result_df.insert(0, col, metadata.get(col, ""))

    return result_df


def ensure_metadata_columns(result_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Insert any missing metadata columns into an existing particle result table."""
    result_df = result_df.copy()

    columns_in_order = [
        "run_folder",
        "combo_folder",
        "run_timestamp",
        "experiment_id",
        "nozzle_id",
        "set_speed_rpm",
        "target_mass_flow_g_per_h",
        "target_displacement_um",
        "set_frequency_hz",
        "set_amplitude_vpp",
        "avg_pump_speed_g_s",
        "avg_pump_speed_g_h",
        "mean_p2p_displacement_um",
    ]

    for col in reversed(columns_in_order):
        if col not in result_df.columns:
            result_df.insert(0, col, metadata.get(col, ""))

    return result_df


def save_combo_csv(result_df: pd.DataFrame, csv_path: Path) -> None:
    """Save one combo result CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)


def update_pump_speed_columns(
    result_df: pd.DataFrame,
    metadata: dict[str, Any],
    avg_g_s: float | None,
    avg_g_h: float | None,
    mean_p2p_displacement_um: float | None,
) -> pd.DataFrame:
    """Update or insert pump-speed and metadata columns in an existing combo DataFrame."""
    result_df = ensure_metadata_columns(result_df, metadata)

    columns_present = {
        "avg_pump_speed_g_s",
        "avg_pump_speed_g_h",
        "mean_p2p_displacement_um",
    }

    if columns_present.issubset(result_df.columns):
        result_df["avg_pump_speed_g_s"] = avg_g_s
        result_df["avg_pump_speed_g_h"] = avg_g_h
        result_df["mean_p2p_displacement_um"] = mean_p2p_displacement_um
        return result_df

    insert_after = "set_amplitude_vpp"
    if insert_after in result_df.columns:
        insert_at = result_df.columns.get_loc(insert_after) + 1
    else:
        insert_at = len(result_df.columns)

    for column_name, value in [
        ("avg_pump_speed_g_s", avg_g_s),
        ("avg_pump_speed_g_h", avg_g_h),
        ("mean_p2p_displacement_um", mean_p2p_displacement_um),
    ]:
        if column_name in result_df.columns:
            result_df[column_name] = value
        else:
            result_df.insert(insert_at, column_name, value)
            insert_at += 1

    return result_df


def update_pump_speeds_for_combo(run_folder: Path, combo_folder: Path) -> None:
    """Update pump speed columns in an existing combo CSV without re-running image analysis."""
    label = make_combo_label(run_folder, combo_folder)
    csv_path = CSV_ROOT / f"{label}.csv"

    if not csv_path.exists():
        print(f"Warning: CSV not found for {run_folder.name} / {combo_folder.name}: {csv_path}")
        return

    try:
        step_data = load_step_json(combo_folder)
        metadata = extract_metadata(step_data, run_folder, combo_folder)
        avg_pump_speed_g_s, avg_pump_speed_g_h = compute_avg_pump_speed(combo_folder, step_data)
        mean_p2p_displacement_um = compute_mean_p2p_displacement(combo_folder, step_data)

        combo_df = pd.read_csv(csv_path)
        combo_df = update_pump_speed_columns(
            combo_df,
            metadata,
            avg_pump_speed_g_s if avg_pump_speed_g_s is not None else pd.NA,
            avg_pump_speed_g_h if avg_pump_speed_g_h is not None else pd.NA,
            mean_p2p_displacement_um if mean_p2p_displacement_um is not None else pd.NA,
        )
        save_combo_csv(combo_df, csv_path)
        print(f"Updated pump speed and oscilloscope metadata for {csv_path}")
    except Exception as exc:
        print(f"ERROR updating pump speed for {run_folder.name} / {combo_folder.name}: {exc}")
        traceback.print_exc()


def update_all_pump_speeds() -> None:
    """Update pump speed columns for all discovered combo CSVs and rebuild the combined CSV."""
    combo_folders = discover_combo_folders(DATA_ROOT)

    print(f"Found {len(combo_folders)} combos for pump speed update.")

    for run_folder, combo_folder in combo_folders:
        update_pump_speeds_for_combo(run_folder, combo_folder)

    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def rebuild_only() -> None:
    """Rebuild the combined and run-summary CSVs from existing data only."""
    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def rebuild_all_runs_only() -> None:
    """Rebuild only the run-summary CSV from existing data."""
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def rebuild_combined_csv(csv_root: Path, combined_csv_path: Path) -> None:
    """Rebuild ALL_PARTICLE_RESULTS.csv from all individual combo CSV files."""
    excluded_names = {
        combined_csv_path.name,
        ALL_RUNS_CSV_PATH.name,
        "analysis_registry.csv",
    }

    combo_csvs = sorted(
        p
        for p in csv_root.glob("*.csv")
        if p.name not in excluded_names
    )

    if not combo_csvs:
        pd.DataFrame().to_csv(combined_csv_path, index=False)
        return

    frames: list[pd.DataFrame] = []

    for csv_path in combo_csvs:
        try:
            frames.append(pd.read_csv(csv_path))
        except pd.errors.EmptyDataError:
            continue

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv_path, index=False)


ALL_RUNS_COLUMNS = [
    "run_folder",
    "combo_folder",
    "run_timestamp",
    "experiment_id",
    "nozzle_id",
    "set_speed_rpm",
    "target_mass_flow_g_per_h",
    "target_displacement_um",
    "set_frequency_hz",
    "set_amplitude_vpp",
    "mean_p2p_displacement_um",
    "displacement_live_um",
    "pump_speed_live_gs",
    "avg_pump_speed_g_s",
    "avg_pump_speed_g_h",
    "folder",
]


def first_non_empty_value(df: pd.DataFrame, column_name: str) -> Any:
    """Return the first non-empty value from a column, or an empty string."""
    if column_name not in df.columns:
        return ""

    values = df[column_name].dropna()
    if values.empty:
        return ""

    return values.iloc[0]


def build_all_runs_row(run_folder: Path, combo_folder: Path) -> dict[str, Any]:
    """Build one run-summary row for one combo folder."""
    step_data = load_step_json(combo_folder)
    metadata = extract_metadata(step_data, run_folder, combo_folder)

    avg_pump_speed_g_s, avg_pump_speed_g_h = compute_avg_pump_speed(combo_folder, step_data)
    mean_p2p_displacement_um = compute_mean_p2p_displacement(combo_folder, step_data)

    row = {
        "run_folder": metadata.get("run_folder", ""),
        "combo_folder": metadata.get("combo_folder", ""),
        "run_timestamp": metadata.get("run_timestamp", ""),
        "experiment_id": metadata.get("experiment_id", ""),
        "nozzle_id": metadata.get("nozzle_id", ""),
        "set_speed_rpm": metadata.get("set_speed_rpm", ""),
        "target_mass_flow_g_per_h": metadata.get("target_mass_flow_g_per_h", ""),
        "target_displacement_um": metadata.get("target_displacement_um", ""),
        "set_frequency_hz": metadata.get("set_frequency_hz", ""),
        "set_amplitude_vpp": metadata.get("set_amplitude_vpp", ""),
        "mean_p2p_displacement_um": (
            mean_p2p_displacement_um if mean_p2p_displacement_um is not None else pd.NA
        ),
        "displacement_live_um": pd.NA,
        "pump_speed_live_gs": pd.NA,
        "avg_pump_speed_g_s": avg_pump_speed_g_s if avg_pump_speed_g_s is not None else pd.NA,
        "avg_pump_speed_g_h": avg_pump_speed_g_h if avg_pump_speed_g_h is not None else pd.NA,
        "folder": combo_folder.name,
    }

    label = make_combo_label(run_folder, combo_folder)
    combo_csv_path = CSV_ROOT / f"{label}.csv"
    if combo_csv_path.exists():
        try:
            combo_df = pd.read_csv(combo_csv_path)
            for optional_column in ["displacement_live_um", "pump_speed_live_gs", "folder"]:
                value = first_non_empty_value(combo_df, optional_column)
                if value != "":
                    row[optional_column] = value
        except pd.errors.EmptyDataError:
            pass

    return row


def rebuild_all_runs_csv(all_runs_csv_path: Path) -> None:
    """Rebuild ALL_RUNS.csv with one row per discovered combo/run."""
    rows: list[dict[str, Any]] = []

    for run_folder, combo_folder in discover_combo_folders(DATA_ROOT):
        try:
            rows.append(build_all_runs_row(run_folder, combo_folder))
        except Exception as exc:
            print(f"Warning: failed to build all-runs row for {combo_folder}: {exc}")

    all_runs = pd.DataFrame(rows, columns=ALL_RUNS_COLUMNS)
    all_runs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(all_runs_csv_path, index=False)


# =============================================================================
# Analysis execution
# =============================================================================


def make_analysis_config(images_folder: Path, output_folder: Path, label: str) -> AnalysisConfig:
    """Create the AnalysisConfig passed to image_analysis_core.py."""
    return AnalysisConfig(
        input_root=images_folder,
        results_root=output_folder,
        csv_root=CSV_ROOT,
        label=label,
        pixel_size_mm=PIXEL_SIZE_MM,
        crop_x=CROP_X,
        crop_y=CROP_Y,
        crop_width=CROP_WIDTH,
        crop_height=CROP_HEIGHT,
        threshold_method=THRESHOLD_METHOD,
        fixed_threshold=FIXED_THRESHOLD,
        otsu_relaxation_factor=OTSU_RELAXATION_FACTOR,
        percentile_threshold=PERCENTILE_THRESHOLD,
        adaptive_block_size=ADAPTIVE_BLOCK_SIZE,
        adaptive_offset=ADAPTIVE_OFFSET,
        background_sigma=BACKGROUND_SIGMA,
        dark_particles=DARK_PARTICLES,
        use_contrast_enhancement=USE_CONTRAST_ENHANCEMENT,
        enhancement_min_intensity=ENHANCEMENT_MIN_INTENSITY,
        enhancement_factor=ENHANCEMENT_FACTOR,
        min_area_px=MIN_AREA_PX,
        max_area_px=MAX_AREA_PX,
        min_circularity=MIN_CIRCULARITY,
        max_circularity=MAX_CIRCULARITY,
        exclude_border=EXCLUDE_BORDER,
        use_opening=USE_OPENING,
        use_closing=USE_CLOSING,
        opening_radius=OPENING_RADIUS,
        closing_radius=CLOSING_RADIUS,
        fill_holes=FILL_HOLES,
        use_watershed=USE_WATERSHED,
        watershed_min_distance=WATERSHED_MIN_DISTANCE,
        save_overlay=SAVE_OVERLAY,
        save_qc_panels=SAVE_QC_PANELS,
        qc_every_n_images=QC_EVERY_N_IMAGES,
        print_diagnostics=PRINT_DIAGNOSTICS,
    )


def analyze_one_combo(
    run_folder: Path,
    combo_folder: Path,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze one combo folder and update the registry."""
    run_name = run_folder.name
    combo_name = combo_folder.name

    if not FORCE_REANALYSIS and combo_is_completed(registry, run_name, combo_name):
        print(f"Skipping already completed combo: {run_name} / {combo_name}")
        return registry

    print("\n" + "=" * 90)
    print(f"Analyzing combo: {run_name} / {combo_name}")
    print("=" * 90)

    images_folder = combo_folder / "images"
    n_images = count_jpeg_files(images_folder)
    label = make_combo_label(run_folder, combo_folder)
    output_folder = IMAGE_OUTPUT_ROOT / safe_filename(run_name) / safe_filename(combo_name)
    csv_path = CSV_ROOT / f"{label}.csv"

    try:
        step_data = load_step_json(combo_folder)
        metadata = extract_metadata(step_data, run_folder, combo_folder)

        avg_pump_speed_g_s, avg_pump_speed_g_h = compute_avg_pump_speed(combo_folder, step_data)
        mean_p2p_displacement_um = compute_mean_p2p_displacement(combo_folder, step_data)

        metadata["avg_pump_speed_g_s"] = avg_pump_speed_g_s if avg_pump_speed_g_s is not None else pd.NA
        metadata["avg_pump_speed_g_h"] = avg_pump_speed_g_h if avg_pump_speed_g_h is not None else pd.NA
        metadata["mean_p2p_displacement_um"] = (
            mean_p2p_displacement_um if mean_p2p_displacement_um is not None else pd.NA
        )

        cfg = make_analysis_config(images_folder, output_folder, label)
        result_df = process_image_folder(images_folder, output_folder, cfg)
        result_df = add_metadata_columns(result_df, metadata)

        save_combo_csv(result_df, csv_path)

        registry = update_registry(
            registry,
            run_folder=run_name,
            combo_folder=combo_name,
            status="completed",
            csv_path=csv_path,
            n_images=n_images,
            n_particles=len(result_df),
            error="",
        )

        save_registry(registry, REGISTRY_PATH)

        print(f"Saved combo CSV: {csv_path}")
        print(f"Completed combo: {run_name} / {combo_name}")

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"ERROR while processing {run_name} / {combo_name}: {error_message}")
        traceback.print_exc()

        registry = update_registry(
            registry,
            run_folder=run_name,
            combo_folder=combo_name,
            status="failed",
            csv_path=csv_path,
            n_images=n_images,
            n_particles=0,
            error=error_message,
        )

        save_registry(registry, REGISTRY_PATH)

    return registry


def run_master_batch() -> None:
    """Run automated batch analysis for all new/incomplete combo folders."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    registry = load_registry(REGISTRY_PATH)
    combo_folders = discover_combo_folders(DATA_ROOT)

    print(f"Found {len(combo_folders)} valid combo folders below: {DATA_ROOT}")

    for run_folder, combo_folder in combo_folders:
        registry = analyze_one_combo(run_folder, combo_folder, registry)

    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)

    print("\nBatch analysis finished.")
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Combined CSV: {COMBINED_CSV_PATH}")
    print(f"All-runs CSV: {ALL_RUNS_CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Batch Program for Automated Particle Image Analysis"
    )
    parser.add_argument(
        "--update-pump-speed-only",
        action="store_true",
        help="Update pump speed and oscilloscope metadata columns in existing combo CSVs without re-running image analysis",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Rebuild the combined CSV from existing combo CSVs only",
    )
    parser.add_argument(
        "--rebuild-all-runs-only",
        action="store_true",
        help="Rebuild ALL_RUNS.csv from existing run/combo metadata only",
    )
    args = parser.parse_args()

    if args.update_pump_speed_only:
        update_all_pump_speeds()
    elif args.rebuild_all_runs_only:
        rebuild_all_runs_only()
    elif args.rebuild_only:
        rebuild_only()
    else:
        run_master_batch()
