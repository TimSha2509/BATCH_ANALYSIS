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
import math
import re
import traceback
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from image_analysis_core import AnalysisConfig, process_image_folder


# =============================================================================
# Main settings: edit this section
# =============================================================================

DATA_ROOT = Path(
    r"W:\PROMOTION_SHARIFSOLTANI_VERKAPSELUNG_DFG_TH1817_9_1_MT\DoE_V1\SCREENING"
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
MATERIALS_CSV_PATH = Path(
    r"W:\PROMOTION_SHARIFSOLTANI_VERKAPSELUNG_DFG_TH1817_9_1_MT\MATERIALS AND NOZZLES\Materials.csv"
)
NOZZLES_CSV_PATH = Path(
    r"W:\PROMOTION_SHARIFSOLTANI_VERKAPSELUNG_DFG_TH1817_9_1_MT\MATERIALS AND NOZZLES\Nozzles.csv"
)

MATERIAL_PROPERTY_COLUMNS = [
    "surface tension_mNm",
    "density_kg_l",
    "viscosity_ref_40_1s",
]

# Re-run combos even if they are marked as completed in analysis_registry.csv.
FORCE_REANALYSIS = False

# Force only these run folders; clear this set after the desired reanalysis.
# Other runs still follow the normal completed/incomplete registry checks.
FORCE_REANALYSIS_RUNS = set() # type folders as {"2026-10-19T13-13-31__Screening_Glycerol_87",} instead of set() for rerun of spec. folders

# Calibration
PIXEL_SIZE_MM = 0.01742

# Crop region
CROP_X = 690
CROP_Y = 10
CROP_WIDTH = 1660
CROP_HEIGHT = 3900

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


def material_id_from_legacy_experiment_id(experiment_id: Any) -> str:
    """Derive the current dataset's material ID from its glycerol experiment name."""
    if experiment_id is None:
        return ""

    # Existing folders contain both ``Glycerol_90`` and the historic
    # ``Glyceroll_90`` spelling. Do not guess a material for other naming
    # schemes; future datasets can provide Material_ID explicitly in step.json.
    match = re.search(r"glycerol+[_\s-]*(\d{2})(?!\d)", str(experiment_id), flags=re.I)
    if match is None:
        return ""
    return f"Glycerol_{match.group(1)}"


def apply_material_ids(result_df: pd.DataFrame) -> pd.DataFrame:
    """Fill Material_ID from Experiment_ID while preserving explicit values."""
    result_df = result_df.copy()
    experiment_ids = (
        result_df["experiment_id"]
        if "experiment_id" in result_df.columns
        else pd.Series("", index=result_df.index, dtype="object")
    )
    derived_ids = [
        material_id_from_legacy_experiment_id(experiment_id)
        for experiment_id in experiment_ids
    ]

    if "Material_ID" in result_df.columns:
        material_ids = [
            value
            if pd.notna(value) and str(value).strip() != ""
            else derived
            for value, derived in zip(result_df["Material_ID"], derived_ids)
        ]
        result_df = result_df.drop(columns=["Material_ID"])
    else:
        material_ids = derived_ids

    insert_at = (
        result_df.columns.get_loc("experiment_id") + 1
        if "experiment_id" in result_df.columns
        else len(result_df.columns)
    )
    result_df.insert(insert_at, "Material_ID", material_ids)
    return result_df


def material_id_key(value: Any) -> str:
    """Return a trimmed, case-insensitive key for matching material IDs."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def load_material_catalog(materials_csv_path: Path = MATERIALS_CSV_PATH) -> pd.DataFrame:
    """Load Materials.csv and validate the fields required for result enrichment."""
    required_columns = {
        "Material_ID",
        "plasticizer",
        "concentration_plasticizer",
        *MATERIAL_PROPERTY_COLUMNS,
    }

    if not materials_csv_path.exists():
        raise FileNotFoundError(f"Materials catalog does not exist: {materials_csv_path}")

    materials = pd.read_csv(materials_csv_path)
    missing_columns = sorted(required_columns - set(materials.columns))
    if missing_columns:
        raise ValueError(
            f"Materials catalog is missing required columns: {', '.join(missing_columns)}"
        )

    material_keys = materials["Material_ID"].map(material_id_key)
    if material_keys.eq("").any():
        raise ValueError("Materials catalog contains an empty Material_ID")

    duplicate_keys = material_keys[material_keys.duplicated(keep=False)]
    if not duplicate_keys.empty:
        duplicate_ids = sorted(materials.loc[duplicate_keys.index, "Material_ID"].astype(str).unique())
        raise ValueError(
            f"Materials catalog contains duplicate Material_ID values: {', '.join(duplicate_ids)}"
        )

    for column in MATERIAL_PROPERTY_COLUMNS:
        numeric_values = pd.to_numeric(materials[column], errors="coerce")
        non_empty_values = materials[column].notna() & materials[column].astype(str).str.strip().ne("")
        if (non_empty_values & numeric_values.isna()).any():
            raise ValueError(f"Materials catalog column {column!r} contains non-numeric values")
        materials[column] = numeric_values.astype("float64")

    return materials


def build_material_lookup(materials: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build exact and legacy aliases that resolve to authoritative catalog rows."""
    lookup: dict[str, dict[str, Any]] = {}

    def register(key: str, record: dict[str, Any]) -> None:
        if not key:
            return
        existing = lookup.get(key)
        if existing is not None and existing["Material_ID"] != record["Material_ID"]:
            raise ValueError(
                "Ambiguous material alias "
                f"{key!r}: {existing['Material_ID']!r} and {record['Material_ID']!r}"
            )
        lookup[key] = record

    records: list[dict[str, Any]] = []
    for _, row in materials.iterrows():
        record = {
            "Material_ID": str(row["Material_ID"]).strip(),
            **{column: row[column] for column in MATERIAL_PROPERTY_COLUMNS},
        }
        records.append(record)
        register(material_id_key(record["Material_ID"]), record)

    for (_, row), record in zip(materials.iterrows(), records):
        canonical_id = record["Material_ID"]
        if canonical_id.casefold().startswith("doe_"):
            register(material_id_key(canonical_id[4:]), record)

        plasticizer = str(row["plasticizer"]).strip().casefold()
        try:
            concentration = float(row["concentration_plasticizer"])
        except (TypeError, ValueError):
            concentration = math.nan

        if plasticizer == "glycerol" and math.isfinite(concentration) and concentration.is_integer():
            register(material_id_key(f"Glycerol_{int(concentration)}"), record)

    return lookup


def enrich_material_properties(
    result_df: pd.DataFrame,
    materials: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Canonicalize Material_ID and refresh material properties from Materials.csv."""
    if materials is None:
        materials = load_material_catalog()

    result_df = apply_material_ids(result_df)
    lookup = build_material_lookup(materials)
    canonical_ids: list[str] = []
    property_values = {column: [] for column in MATERIAL_PROPERTY_COLUMNS}
    unknown_ids: set[str] = set()

    for value in result_df["Material_ID"]:
        key = material_id_key(value)
        if not key:
            canonical_ids.append("")
            for column in MATERIAL_PROPERTY_COLUMNS:
                property_values[column].append(pd.NA)
            continue

        record = lookup.get(key)
        if record is None:
            unknown_ids.add(str(value).strip())
            canonical_ids.append(str(value).strip())
            for column in MATERIAL_PROPERTY_COLUMNS:
                property_values[column].append(pd.NA)
            continue

        canonical_ids.append(record["Material_ID"])
        for column in MATERIAL_PROPERTY_COLUMNS:
            property_values[column].append(record[column])

    if unknown_ids:
        raise ValueError(
            "Material_ID values not found in Materials.csv: "
            + ", ".join(sorted(unknown_ids))
        )

    columns_to_replace = ["Material_ID", *MATERIAL_PROPERTY_COLUMNS]
    result_df = result_df.drop(
        columns=[column for column in columns_to_replace if column in result_df.columns]
    )
    insert_at = (
        result_df.columns.get_loc("experiment_id") + 1
        if "experiment_id" in result_df.columns
        else len(result_df.columns)
    )
    result_df.insert(insert_at, "Material_ID", canonical_ids)
    insert_at += 1
    for column in MATERIAL_PROPERTY_COLUMNS:
        numeric_values = pd.to_numeric(
            pd.Series(property_values[column], index=result_df.index),
            errors="coerce",
        ).astype("float64")
        result_df.insert(insert_at, column, numeric_values)
        insert_at += 1

    return result_df


def dataframes_equal_for_csv(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare CSV-bound frames while treating empty strings and NA as equivalent."""
    if left.columns.tolist() != right.columns.tolist() or len(left) != len(right):
        return False

    def values_equal(left_value: Any, right_value: Any) -> bool:
        left_empty = left_value is None or pd.isna(left_value) or (
            isinstance(left_value, str) and not left_value.strip()
        )
        right_empty = right_value is None or pd.isna(right_value) or (
            isinstance(right_value, str) and not right_value.strip()
        )
        if left_empty or right_empty:
            return left_empty and right_empty
        if isinstance(left_value, Real) and isinstance(right_value, Real):
            return math.isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        return bool(left_value == right_value)

    for column in left.columns:
        if not all(
            values_equal(left_value, right_value)
            for left_value, right_value in zip(left[column], right[column])
        ):
            return False
    return True


def nozzle_id_key(value: Any) -> str:
    """Return a trimmed, case-insensitive key for matching nozzle IDs."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def load_nozzle_catalog(nozzles_csv_path: Path = NOZZLES_CSV_PATH) -> pd.DataFrame:
    """Load Nozzles.csv and validate identifiers and inner diameters."""
    required_columns = {"Nozzle_ID", "inner_diameter_mm"}
    if not nozzles_csv_path.exists():
        raise FileNotFoundError(f"Nozzle catalog does not exist: {nozzles_csv_path}")

    nozzles = pd.read_csv(nozzles_csv_path)
    missing_columns = sorted(required_columns - set(nozzles.columns))
    if missing_columns:
        raise ValueError(
            f"Nozzle catalog is missing required columns: {', '.join(missing_columns)}"
        )

    nozzle_keys = nozzles["Nozzle_ID"].map(nozzle_id_key)
    if nozzle_keys.eq("").any():
        raise ValueError("Nozzle catalog contains an empty Nozzle_ID")

    duplicate_keys = nozzle_keys[nozzle_keys.duplicated(keep=False)]
    if not duplicate_keys.empty:
        duplicate_ids = sorted(nozzles.loc[duplicate_keys.index, "Nozzle_ID"].astype(str).unique())
        raise ValueError(
            f"Nozzle catalog contains duplicate Nozzle_ID values: {', '.join(duplicate_ids)}"
        )

    raw_diameters = nozzles["inner_diameter_mm"]
    inner_diameters = pd.to_numeric(raw_diameters, errors="coerce")
    non_empty_values = raw_diameters.notna() & raw_diameters.astype(str).str.strip().ne("")
    if (non_empty_values & inner_diameters.isna()).any():
        raise ValueError("Nozzle catalog column 'inner_diameter_mm' contains non-numeric values")
    if inner_diameters.dropna().le(0).any():
        raise ValueError("Nozzle catalog inner_diameter_mm values must be greater than zero")

    nozzles["inner_diameter_mm"] = inner_diameters.astype("float64")
    return nozzles


def enrich_dimensionless_diameter(
    result_df: pd.DataFrame,
    nozzles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate dd/di as ECD_mm divided by the matched nozzle inner diameter."""
    if nozzles is None:
        nozzles = load_nozzle_catalog()

    result_df = result_df.copy()
    diameter_by_nozzle = {
        nozzle_id_key(row["Nozzle_ID"]): row["inner_diameter_mm"]
        for _, row in nozzles.iterrows()
    }
    nozzle_ids = (
        result_df["nozzle_id"]
        if "nozzle_id" in result_df.columns
        else pd.Series("", index=result_df.index, dtype="object")
    )
    droplet_diameters = (
        pd.to_numeric(result_df["ECD_mm"], errors="coerce")
        if "ECD_mm" in result_df.columns
        else pd.Series(math.nan, index=result_df.index, dtype="float64")
    )
    inner_diameters = pd.to_numeric(
        pd.Series(
            [diameter_by_nozzle.get(nozzle_id_key(value), math.nan) for value in nozzle_ids],
            index=result_df.index,
        ),
        errors="coerce",
    ).astype("float64")
    dimensionless_diameters = (droplet_diameters / inner_diameters).astype("float64")

    if "dd/di" in result_df.columns:
        result_df = result_df.drop(columns=["dd/di"])
    insert_at = (
        result_df.columns.get_loc("ECD_mm") + 1
        if "ECD_mm" in result_df.columns
        else len(result_df.columns)
    )
    result_df.insert(insert_at, "dd/di", dimensionless_diameters)
    return result_df


FREQUENCY_ALIASES = {
    'set_frequency_hz',
    'SetFrequencyHz',
    'setFrequencyHz',
    'frequency_hz',
    'frequency',
}

def finite_float(value: Any) -> float | None:
    '''Return a finite numeric value, or None when it cannot be parsed.'''
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def frequency_as_float(value: Any) -> float | None:
    '''Return a finite numeric frequency, or None when it cannot be parsed.'''
    return finite_float(value)


def frequency_index_for_rank(rank: int) -> str:
    '''Return L, M, H, H_2, H_3, ... for a zero-based frequency rank.'''
    if rank < 0:
        raise ValueError('Frequency rank cannot be negative')
    if rank == 0:
        return 'L'
    if rank == 1:
        return 'M'
    if rank == 2:
        return 'H'
    return f'H_{rank - 1}'


def displacement_index_for_rank(rank: int, level_count: int) -> str:
    '''Return the displacement label for a zero-based rank and group size.'''
    if level_count < 1:
        raise ValueError('Displacement level count must be positive')
    if rank < 0 or rank >= level_count:
        raise ValueError('Displacement rank must be within the group')
    if level_count == 2 and rank == 1:
        return 'H'
    return frequency_index_for_rank(rank)


def combo_key_from_csv_path(csv_path: Path) -> tuple[str, str] | None:
    """Recover run/combo folder names from an individual combo CSV filename."""
    marker = "__combo_"
    if marker not in csv_path.stem:
        return None

    run_name, combo_suffix = csv_path.stem.split(marker, maxsplit=1)
    return run_name, f"combo_{combo_suffix}"


def frequency_from_combo_name(combo_name: str) -> float | None:
    """Extract the numeric f...Hz component from a combo folder name."""
    match = re.search(r"(?:^|_)f(-?\d+(?:\.\d+)?)Hz(?:_|$)", combo_name, flags=re.I)
    return frequency_as_float(match.group(1)) if match else None


def read_combo_frequency_from_csv(
    csv_path: Path,
) -> tuple[tuple[str, str], float] | None:
    """Read run/combo/frequency metadata, falling back to the CSV filename."""
    combo_key = combo_key_from_csv_path(csv_path)
    frequency = frequency_from_combo_name(combo_key[1]) if combo_key else None

    try:
        combo_df = pd.read_csv(csv_path, nrows=1)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        combo_df = pd.DataFrame()

    if not combo_df.empty:
        row = combo_df.iloc[0]
        run_value = row.get("run_folder")
        combo_value = row.get("combo_folder")
        frequency_value = frequency_as_float(row.get("set_frequency_hz"))

        if pd.notna(run_value) and pd.notna(combo_value):
            combo_key = (str(run_value), str(combo_value))
        if frequency_value is not None:
            frequency = frequency_value

    if combo_key is None or frequency is None:
        return None
    return combo_key, frequency


def build_frequency_index_lookup(
    combo_folders: list[tuple[Path, Path]],
    csv_root: Path | None = None,
) -> dict[tuple[str, str], str]:
    '''Index distinct frequencies independently within each experimental run folder.'''
    frequency_by_combo: dict[tuple[str, str], float] = {}
    folder_frequency_by_combo: dict[tuple[str, str], float] = {}
    frequencies_by_run: dict[str, set[float]] = {}

    for run_folder, combo_folder in combo_folders:
        try:
            step_data = load_step_json(combo_folder)
        except (OSError, json.JSONDecodeError) as exc:
            print(f'Warning: could not read frequency metadata from {combo_folder}: {exc}')
            continue

        frequency = frequency_as_float(find_value_recursive(step_data, FREQUENCY_ALIASES))
        if frequency is None:
            continue

        combo_key = (run_folder.name, combo_folder.name)
        frequency_by_combo[combo_key] = frequency
        folder_frequency_by_combo[combo_key] = frequency
        frequencies_by_run.setdefault(run_folder.name, set()).add(frequency)

    if csv_root is not None and csv_root.exists():
        excluded_names = {
            COMBINED_CSV_PATH.name,
            ALL_RUNS_CSV_PATH.name,
            "analysis_registry.csv",
        }
        for csv_path in csv_root.glob("*.csv"):
            if csv_path.name in excluded_names:
                continue
            csv_frequency = read_combo_frequency_from_csv(csv_path)
            if csv_frequency is None:
                continue
            combo_key, frequency = csv_frequency
            frequency_by_combo[combo_key] = frequency
            frequencies_by_run.setdefault(combo_key[0], set()).add(frequency)

    frequency_by_combo.update(folder_frequency_by_combo)
    frequencies_by_run = {}
    for (run_name, _), frequency in frequency_by_combo.items():
        frequencies_by_run.setdefault(run_name, set()).add(frequency)

    labels_by_run: dict[str, dict[float, str]] = {}
    for run_name, frequencies in frequencies_by_run.items():
        labels_by_run[run_name] = {
            frequency: frequency_index_for_rank(rank)
            for rank, frequency in enumerate(sorted(frequencies))
        }

    return {
        combo_key: labels_by_run[combo_key[0]][frequency]
        for combo_key, frequency in frequency_by_combo.items()
    }


def canonical_material_key_for_index(
    material_id: Any,
    experiment_id: Any,
    material_lookup: dict[str, dict[str, Any]],
) -> str:
    '''Resolve one material value to the canonical catalog key used for grouping.'''
    material_key = material_id_key(material_id)
    if not material_key:
        material_key = material_id_key(
            material_id_from_legacy_experiment_id(experiment_id)
        )

    record = material_lookup.get(material_key)
    if record is None:
        return ''
    return material_id_key(record['Material_ID'])


def displacement_group_record(
    metadata: Any,
    material_lookup: dict[str, dict[str, Any]],
) -> tuple[tuple[str, float, str, float], float] | None:
    '''Return the normalized displacement group and value from row-like metadata.'''
    material_key = canonical_material_key_for_index(
        metadata.get('Material_ID'),
        metadata.get('experiment_id'),
        material_lookup,
    )
    frequency = finite_float(metadata.get('set_frequency_hz'))
    nozzle_key = nozzle_id_key(metadata.get('nozzle_id'))
    mass_flow = finite_float(metadata.get('target_mass_flow_g_per_h'))
    displacement = finite_float(metadata.get('target_displacement_um'))

    if (
        not material_key
        or frequency is None
        or not nozzle_key
        or mass_flow is None
        or displacement is None
    ):
        return None

    return (material_key, frequency, nozzle_key, mass_flow), displacement


def read_combo_displacement_from_csv(
    csv_path: Path,
    material_lookup: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], tuple[str, float, str, float], float] | None:
    '''Read combo identity and normalized displacement grouping from one CSV.'''
    combo_key = combo_key_from_csv_path(csv_path)

    try:
        combo_df = pd.read_csv(csv_path, nrows=1)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    if combo_df.empty:
        return None

    row = combo_df.iloc[0]
    run_value = row.get('run_folder')
    combo_value = row.get('combo_folder')
    if pd.notna(run_value) and pd.notna(combo_value):
        combo_key = (str(run_value), str(combo_value))

    group_record = displacement_group_record(row, material_lookup)
    if combo_key is None or group_record is None:
        return None

    group_key, displacement = group_record
    return combo_key, group_key, displacement


def build_displacement_index_lookup(
    combo_folders: list[tuple[Path, Path]],
    csv_root: Path | None = None,
    materials: pd.DataFrame | None = None,
) -> dict[tuple[str, str], str]:
    '''Index displacement levels within each material/frequency/nozzle/flow group.'''
    if materials is None:
        materials = load_material_catalog()
    material_lookup = build_material_lookup(materials)

    displacement_by_combo: dict[
        tuple[str, str],
        tuple[tuple[str, float, str, float], float],
    ] = {}

    if csv_root is not None and csv_root.exists():
        excluded_names = {
            COMBINED_CSV_PATH.name,
            ALL_RUNS_CSV_PATH.name,
            'analysis_registry.csv',
        }
        for csv_path in csv_root.glob('*.csv'):
            if csv_path.name in excluded_names:
                continue
            csv_displacement = read_combo_displacement_from_csv(
                csv_path,
                material_lookup,
            )
            if csv_displacement is None:
                continue
            combo_key, group_key, displacement = csv_displacement
            displacement_by_combo[combo_key] = (group_key, displacement)

    for run_folder, combo_folder in combo_folders:
        combo_key = (run_folder.name, combo_folder.name)
        try:
            step_data = load_step_json(combo_folder)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f'Warning: could not read displacement metadata from '
                f'{combo_folder}: {exc}'
            )
            continue

        metadata = extract_metadata(step_data, run_folder, combo_folder)
        group_record = displacement_group_record(metadata, material_lookup)
        if group_record is None:
            displacement_by_combo.pop(combo_key, None)
        else:
            displacement_by_combo[combo_key] = group_record

    displacements_by_group: dict[
        tuple[str, float, str, float],
        set[float],
    ] = {}
    for group_key, displacement in displacement_by_combo.values():
        displacements_by_group.setdefault(group_key, set()).add(displacement)

    labels_by_group: dict[
        tuple[str, float, str, float],
        dict[float, str],
    ] = {}
    for group_key, displacements in displacements_by_group.items():
        sorted_displacements = sorted(displacements)
        level_count = len(sorted_displacements)
        labels_by_group[group_key] = {
            displacement: displacement_index_for_rank(rank, level_count)
            for rank, displacement in enumerate(sorted_displacements)
        }

    return {
        combo_key: labels_by_group[group_key][displacement]
        for combo_key, (group_key, displacement) in displacement_by_combo.items()
    }


def apply_frequency_index_lookup(
    result_df: pd.DataFrame,
    frequency_index_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    '''Add or refresh Frequency_index from run/combo folder names.'''
    result_df = result_df.copy()

    required_columns = {'run_folder', 'combo_folder'}
    if not required_columns.issubset(result_df.columns):
        return result_df

    frequency_indices = [
        frequency_index_lookup.get((str(run_name), str(combo_name)), '')
        for run_name, combo_name in zip(
            result_df['run_folder'],
            result_df['combo_folder'],
        )
    ]

    if 'Frequency_index' in result_df.columns:
        result_df = result_df.drop(columns=['Frequency_index'])

    if 'set_frequency_hz' in result_df.columns:
        insert_at = result_df.columns.get_loc('set_frequency_hz') + 1
    else:
        insert_at = len(result_df.columns)

    result_df.insert(insert_at, 'Frequency_index', frequency_indices)
    return result_df


def apply_displacement_index_lookup(
    result_df: pd.DataFrame,
    displacement_index_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    '''Add or refresh Displacement_index from run/combo folder names.'''
    result_df = result_df.copy()

    required_columns = {'run_folder', 'combo_folder'}
    if not required_columns.issubset(result_df.columns):
        return result_df

    displacement_indices = [
        displacement_index_lookup.get((str(run_name), str(combo_name)), '')
        for run_name, combo_name in zip(
            result_df['run_folder'],
            result_df['combo_folder'],
        )
    ]

    if 'Displacement_index' in result_df.columns:
        result_df = result_df.drop(columns=['Displacement_index'])

    if 'target_displacement_um' in result_df.columns:
        insert_at = result_df.columns.get_loc('target_displacement_um') + 1
    else:
        insert_at = len(result_df.columns)

    result_df.insert(insert_at, 'Displacement_index', displacement_indices)
    return result_df


def extract_metadata(
    step_data: dict[str, Any],
    run_folder: Path,
    combo_folder: Path,
    frequency_index: str = '',
    displacement_index: str = '',
) -> dict[str, Any]:
    """Extract required metadata from step.json, with safe fallbacks."""
    metadata = {
        "run_folder": run_folder.name,
        "combo_folder": combo_folder.name,
        "run_timestamp": run_timestamp_from_folder(run_folder),
        "experiment_id": find_value_recursive(
            step_data,
            {"experiment_id", "Experiment_ID", "ExperimentID", "experimentID"},
        ),
        "Material_ID": find_value_recursive(
            step_data,
            {"material_id", "Material_ID", "MaterialID", "materialID"},
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
        "set_frequency_hz": find_value_recursive(step_data, FREQUENCY_ALIASES),
        "Frequency_index": frequency_index,
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

    if metadata["Material_ID"] is None or str(metadata["Material_ID"]).strip() == "":
        metadata["Material_ID"] = material_id_from_legacy_experiment_id(
            metadata["experiment_id"]
        )

    metadata['Displacement_index'] = displacement_index

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
        "Material_ID",
        *MATERIAL_PROPERTY_COLUMNS,
        "nozzle_id",
        "set_speed_rpm",
        "target_mass_flow_g_per_h",
        "target_displacement_um",
        "set_frequency_hz",
        "Frequency_index",
        "set_amplitude_vpp",
        "avg_pump_speed_g_s",
        "avg_pump_speed_g_h",
        "mean_p2p_displacement_um",
    ]

    columns_in_order.insert(
        columns_in_order.index('target_displacement_um') + 1,
        'Displacement_index',
    )

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
        "Material_ID",
        *MATERIAL_PROPERTY_COLUMNS,
        "nozzle_id",
        "set_speed_rpm",
        "target_mass_flow_g_per_h",
        "target_displacement_um",
        "set_frequency_hz",
        "Frequency_index",
        "set_amplitude_vpp",
        "avg_pump_speed_g_s",
        "avg_pump_speed_g_h",
        "mean_p2p_displacement_um",
    ]

    columns_in_order.insert(
        columns_in_order.index('target_displacement_um') + 1,
        'Displacement_index',
    )

    for col in reversed(columns_in_order):
        if col not in result_df.columns:
            result_df.insert(0, col, metadata.get(col, ""))

    return result_df


def save_combo_csv(
    result_df: pd.DataFrame,
    csv_path: Path,
    materials: pd.DataFrame | None = None,
    nozzles: pd.DataFrame | None = None,
) -> None:
    """Validate, enrich, and save one combo result CSV."""
    result_df = enrich_material_properties(result_df, materials)
    result_df = enrich_dimensionless_diameter(result_df, nozzles)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)


def update_material_ids_in_existing_csvs(csv_root: Path) -> int:
    """Synchronize material IDs and properties in existing combo CSVs."""
    excluded_names = {
        COMBINED_CSV_PATH.name,
        ALL_RUNS_CSV_PATH.name,
        "analysis_registry.csv",
    }
    materials = load_material_catalog()
    pending_updates: list[tuple[Path, pd.DataFrame]] = []

    for csv_path in csv_root.glob("*.csv"):
        if csv_path.name in excluded_names or combo_key_from_csv_path(csv_path) is None:
            continue

        try:
            combo_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            combo_df = pd.DataFrame()

        enriched_df = enrich_material_properties(combo_df, materials)
        if not dataframes_equal_for_csv(enriched_df, combo_df):
            pending_updates.append((csv_path, enriched_df))

    for csv_path, enriched_df in pending_updates:
        save_combo_csv(enriched_df, csv_path, materials)

    return len(pending_updates)


def update_dimensionless_diameter_in_existing_csvs(csv_root: Path) -> int:
    """Add or refresh dd/di in all existing individual combo CSVs."""
    excluded_names = {
        COMBINED_CSV_PATH.name,
        ALL_RUNS_CSV_PATH.name,
        "analysis_registry.csv",
    }
    materials = load_material_catalog()
    nozzles = load_nozzle_catalog()
    pending_updates: list[tuple[Path, pd.DataFrame]] = []

    for csv_path in csv_root.glob("*.csv"):
        if csv_path.name in excluded_names or combo_key_from_csv_path(csv_path) is None:
            continue

        try:
            combo_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            combo_df = pd.DataFrame()

        enriched_df = enrich_material_properties(combo_df, materials)
        enriched_df = enrich_dimensionless_diameter(enriched_df, nozzles)
        if not dataframes_equal_for_csv(enriched_df, combo_df):
            pending_updates.append((csv_path, enriched_df))

    for csv_path, enriched_df in pending_updates:
        save_combo_csv(enriched_df, csv_path, materials, nozzles)

    return len(pending_updates)


def update_frequency_indices_in_existing_csvs(
    csv_root: Path,
    frequency_index_lookup: dict[tuple[str, str], str],
) -> int:
    """Add or refresh Frequency_index in all existing individual combo CSVs."""
    excluded_names = {
        COMBINED_CSV_PATH.name,
        ALL_RUNS_CSV_PATH.name,
        "analysis_registry.csv",
    }
    updated_count = 0

    for csv_path in csv_root.glob("*.csv"):
        if csv_path.name in excluded_names:
            continue

        combo_key = combo_key_from_csv_path(csv_path)
        if combo_key is None:
            continue
        frequency_index = frequency_index_lookup.get(combo_key, "")

        try:
            combo_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            combo_df = pd.DataFrame()

        if "Frequency_index" in combo_df.columns:
            if combo_df.empty or combo_df["Frequency_index"].eq(frequency_index).all():
                continue
            combo_df["Frequency_index"] = frequency_index
        else:
            if "set_frequency_hz" in combo_df.columns:
                insert_at = combo_df.columns.get_loc("set_frequency_hz") + 1
            else:
                insert_at = len(combo_df.columns)
            combo_df.insert(insert_at, "Frequency_index", frequency_index)

        save_combo_csv(combo_df, csv_path)
        updated_count += 1

    return updated_count


def update_level_indices_in_existing_csvs(
    csv_root: Path,
    frequency_index_lookup: dict[tuple[str, str], str],
    displacement_index_lookup: dict[tuple[str, str], str],
    materials: pd.DataFrame | None = None,
    nozzles: pd.DataFrame | None = None,
) -> int:
    '''Add or refresh both level indexes in all individual combo CSVs.'''
    if materials is None:
        materials = load_material_catalog()
    if nozzles is None:
        nozzles = load_nozzle_catalog()

    excluded_names = {
        COMBINED_CSV_PATH.name,
        ALL_RUNS_CSV_PATH.name,
        'analysis_registry.csv',
    }
    updated_count = 0

    for csv_path in csv_root.glob('*.csv'):
        if csv_path.name in excluded_names:
            continue

        combo_key = combo_key_from_csv_path(csv_path)
        if combo_key is None:
            continue

        try:
            combo_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            combo_df = pd.DataFrame()

        indexed_df = apply_frequency_index_lookup(
            combo_df,
            frequency_index_lookup,
        )
        indexed_df = apply_displacement_index_lookup(
            indexed_df,
            displacement_index_lookup,
        )
        if dataframes_equal_for_csv(indexed_df, combo_df):
            continue

        save_combo_csv(indexed_df, csv_path, materials, nozzles)
        updated_count += 1

    return updated_count


def refresh_level_indices_in_existing_csvs(
    combo_folders: list[tuple[Path, Path]] | None = None,
    csv_root: Path = CSV_ROOT,
    materials: pd.DataFrame | None = None,
    nozzles: pd.DataFrame | None = None,
) -> int:
    '''Rebuild both lookups and refresh every existing individual combo CSV.'''
    if combo_folders is None:
        combo_folders = discover_combo_folders(DATA_ROOT)
    if materials is None:
        materials = load_material_catalog()
    if nozzles is None:
        nozzles = load_nozzle_catalog()

    frequency_index_lookup = build_frequency_index_lookup(combo_folders, csv_root)
    displacement_index_lookup = build_displacement_index_lookup(
        combo_folders,
        csv_root,
        materials,
    )
    return update_level_indices_in_existing_csvs(
        csv_root,
        frequency_index_lookup,
        displacement_index_lookup,
        materials,
        nozzles,
    )


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


def _update_pump_speeds_for_combo(
    run_folder: Path,
    combo_folder: Path,
    displacement_index: str = '',
    frequency_index: str = "",
) -> None:
    """Update pump speed columns in an existing combo CSV without re-running image analysis."""
    label = make_combo_label(run_folder, combo_folder)
    csv_path = CSV_ROOT / f"{label}.csv"

    if not csv_path.exists():
        print(f"Warning: CSV not found for {run_folder.name} / {combo_folder.name}: {csv_path}")
        return

    try:
        step_data = load_step_json(combo_folder)
        metadata = extract_metadata(
            step_data,
            run_folder,
            combo_folder,
            frequency_index,
            displacement_index,
        )
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
        combo_df["Frequency_index"] = frequency_index
        combo_df['Displacement_index'] = displacement_index
        save_combo_csv(combo_df, csv_path)
        print(f"Updated pump speed and oscilloscope metadata for {csv_path}")
    except Exception as exc:
        print(f"ERROR updating pump speed for {run_folder.name} / {combo_folder.name}: {exc}")
        traceback.print_exc()


def update_pump_speeds_for_combo(
    run_folder: Path,
    combo_folder: Path,
    frequency_index: str = '',
    displacement_index: str = '',
) -> None:
    '''Update one combo while preserving the existing frequency argument order.'''
    _update_pump_speeds_for_combo(
        run_folder,
        combo_folder,
        displacement_index=displacement_index,
        frequency_index=frequency_index,
    )


def update_all_pump_speeds() -> None:
    """Update pump speed columns for all discovered combo CSVs and rebuild the combined CSV."""
    combo_folders = discover_combo_folders(DATA_ROOT)
    frequency_index_lookup = build_frequency_index_lookup(combo_folders, CSV_ROOT)
    displacement_index_lookup = build_displacement_index_lookup(
        combo_folders,
        CSV_ROOT,
    )

    print(f"Found {len(combo_folders)} combos for pump speed update.")

    for run_folder, combo_folder in combo_folders:
        frequency_index = frequency_index_lookup.get((run_folder.name, combo_folder.name), "")
        displacement_index = displacement_index_lookup.get(
            (run_folder.name, combo_folder.name),
            '',
        )
        update_pump_speeds_for_combo(
            run_folder,
            combo_folder,
            frequency_index=frequency_index,
            displacement_index=displacement_index,
        )

    updated_legacy_csvs = update_level_indices_in_existing_csvs(
        CSV_ROOT,
        frequency_index_lookup,
        displacement_index_lookup,
    )
    print(f"Added or refreshed frequency indexes in {updated_legacy_csvs} legacy CSVs.")

    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def rebuild_only() -> None:
    """Rebuild the combined and run-summary CSVs from existing data only."""
    refresh_level_indices_in_existing_csvs()
    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def update_material_ids_only() -> None:
    """Synchronize material IDs/properties and rebuild aggregate outputs."""
    updated_count = update_material_ids_in_existing_csvs(CSV_ROOT)
    refresh_level_indices_in_existing_csvs()
    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    rebuild_all_runs_csv(ALL_RUNS_CSV_PATH)
    print(f"Synchronized material data in {updated_count} combo CSVs.")
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")
    print(f"Rebuilt all-runs CSV: {ALL_RUNS_CSV_PATH}")


def update_dimensionless_diameter_only() -> None:
    """Refresh dd/di in combo CSVs and rebuild the combined particle results."""
    updated_count = update_dimensionless_diameter_in_existing_csvs(CSV_ROOT)
    refresh_level_indices_in_existing_csvs()
    rebuild_combined_csv(CSV_ROOT, COMBINED_CSV_PATH)
    print(f"Added or refreshed dd/di in {updated_count} combo CSVs.")
    print(f"Rebuilt combined CSV: {COMBINED_CSV_PATH}")


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
        combined = enrich_material_properties(combined)
        combined = enrich_dimensionless_diameter(combined)
        combo_folders = discover_combo_folders(DATA_ROOT)
        frequency_index_lookup = build_frequency_index_lookup(combo_folders, csv_root)
        combined = apply_frequency_index_lookup(combined, frequency_index_lookup)
        displacement_index_lookup = build_displacement_index_lookup(
            combo_folders,
            csv_root,
        )
        combined = apply_displacement_index_lookup(
            combined,
            displacement_index_lookup,
        )
    else:
        combined = pd.DataFrame()

    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv_path, index=False)


ALL_RUNS_COLUMNS = [
    "run_folder",
    "combo_folder",
    "run_timestamp",
    "experiment_id",
    "Material_ID",
    *MATERIAL_PROPERTY_COLUMNS,
    "nozzle_id",
    "set_speed_rpm",
    "target_mass_flow_g_per_h",
    "target_displacement_um",
    "set_frequency_hz",
    "Frequency_index",
    "set_amplitude_vpp",
    "mean_p2p_displacement_um",
    "displacement_live_um",
    "pump_speed_live_gs",
    "avg_pump_speed_g_s",
    "avg_pump_speed_g_h",
    "folder",
]
ALL_RUNS_COLUMNS.insert(
    ALL_RUNS_COLUMNS.index('target_displacement_um') + 1,
    'Displacement_index',
)


def first_non_empty_value(df: pd.DataFrame, column_name: str) -> Any:
    """Return the first non-empty value from a column, or an empty string."""
    if column_name not in df.columns:
        return ""

    values = df[column_name].dropna()
    if values.empty:
        return ""

    return values.iloc[0]


def build_all_runs_row(
    run_folder: Path,
    combo_folder: Path,
    frequency_index: str = "",
    displacement_index: str = '',
) -> dict[str, Any]:
    """Build one run-summary row for one combo folder."""
    step_data = load_step_json(combo_folder)
    metadata = extract_metadata(
        step_data,
        run_folder,
        combo_folder,
        frequency_index,
        displacement_index,
    )

    avg_pump_speed_g_s, avg_pump_speed_g_h = compute_avg_pump_speed(combo_folder, step_data)
    mean_p2p_displacement_um = compute_mean_p2p_displacement(combo_folder, step_data)

    row = {
        "run_folder": metadata.get("run_folder", ""),
        "combo_folder": metadata.get("combo_folder", ""),
        "run_timestamp": metadata.get("run_timestamp", ""),
        "experiment_id": metadata.get("experiment_id", ""),
        "Material_ID": metadata.get("Material_ID", ""),
        "nozzle_id": metadata.get("nozzle_id", ""),
        "set_speed_rpm": metadata.get("set_speed_rpm", ""),
        "target_mass_flow_g_per_h": metadata.get("target_mass_flow_g_per_h", ""),
        "target_displacement_um": metadata.get("target_displacement_um", ""),
        "set_frequency_hz": metadata.get("set_frequency_hz", ""),
        "Frequency_index": metadata.get("Frequency_index", ""),
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

    row['Displacement_index'] = metadata.get('Displacement_index', '')

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
    combo_folders = discover_combo_folders(DATA_ROOT)
    frequency_index_lookup = build_frequency_index_lookup(combo_folders, CSV_ROOT)
    displacement_index_lookup = build_displacement_index_lookup(
        combo_folders,
        CSV_ROOT,
    )

    for run_folder, combo_folder in combo_folders:
        try:
            frequency_index = frequency_index_lookup.get(
                (run_folder.name, combo_folder.name),
                "",
            )
            displacement_index = displacement_index_lookup.get(
                (run_folder.name, combo_folder.name),
                '',
            )
            rows.append(
                build_all_runs_row(
                    run_folder,
                    combo_folder,
                    frequency_index,
                    displacement_index,
                )
            )
        except Exception as exc:
            print(f"Warning: failed to build all-runs row for {combo_folder}: {exc}")

    all_runs = pd.DataFrame(rows, columns=ALL_RUNS_COLUMNS)
    all_runs = enrich_material_properties(all_runs)
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
    frequency_index: str = "",
    displacement_index: str = '',
) -> pd.DataFrame:
    """Analyze one combo folder and update the registry."""
    run_name = run_folder.name
    combo_name = combo_folder.name

    force_reanalysis = FORCE_REANALYSIS or run_name in FORCE_REANALYSIS_RUNS
    if not force_reanalysis and combo_is_completed(registry, run_name, combo_name):
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
        metadata = extract_metadata(
            step_data,
            run_folder,
            combo_folder,
            frequency_index,
            displacement_index,
        )

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
    frequency_index_lookup = build_frequency_index_lookup(combo_folders, CSV_ROOT)
    displacement_index_lookup = build_displacement_index_lookup(
        combo_folders,
        CSV_ROOT,
    )

    print(f"Found {len(combo_folders)} valid combo folders below: {DATA_ROOT}")

    for run_folder, combo_folder in combo_folders:
        frequency_index = frequency_index_lookup.get((run_folder.name, combo_folder.name), "")
        displacement_index = displacement_index_lookup.get(
            (run_folder.name, combo_folder.name),
            '',
        )
        registry = analyze_one_combo(
            run_folder,
            combo_folder,
            registry,
            frequency_index=frequency_index,
            displacement_index=displacement_index,
        )

    refreshed_combo_folders = discover_combo_folders(DATA_ROOT)
    refresh_level_indices_in_existing_csvs(
        refreshed_combo_folders,
        CSV_ROOT,
    )
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
        "--update-material-id-only",
        action="store_true",
        help="Synchronize Material_ID and properties in combo and aggregate CSVs",
    )
    parser.add_argument(
        "--update-dimensionless-diameter-only",
        action="store_true",
        help="Add or refresh dd/di in existing combo and combined result CSVs",
    )
    parser.add_argument(
        "--rebuild-all-runs-only",
        action="store_true",
        help="Rebuild ALL_RUNS.csv from existing run/combo metadata only",
    )
    args = parser.parse_args()

    if args.update_pump_speed_only:
        update_all_pump_speeds()
    elif args.update_dimensionless_diameter_only:
        update_dimensionless_diameter_only()
    elif args.update_material_id_only:
        update_material_ids_only()
    elif args.rebuild_all_runs_only:
        rebuild_all_runs_only()
    elif args.rebuild_only:
        rebuild_only()
    else:
        run_master_batch()
