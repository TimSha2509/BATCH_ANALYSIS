"""
Particle Analysis Pipeline for Droplet/Capsule Images
=====================================================

Batch image analysis pipeline for dark droplets / capsules on a brighter background.

Workflow:
    crop -> background correction -> threshold -> mask cleanup
    -> particle labeling -> measurement -> CSV + overlay + QC export

Important logic:
    In the original grayscale image:
        dark droplets = low intensity

    After background correction:
        dark droplets become bright contrast signal

    Therefore thresholding is done as:
        corrected > threshold

Required packages:
    pip install numpy pandas opencv-python scikit-image scipy matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, threshold_otsu


ThresholdMethod = Literal[
    "fixed",
    "otsu",
    "relaxed_otsu",
    "adaptive",
    "percentile",
]


@dataclass
class AnalysisConfig:
    # Input/output paths
    input_root: Path
    results_root: Path
    csv_root: Path
    label: str

    # Calibration
    pixel_size_mm: float = 0.0260835

    # Crop region
    crop_x: int = 920
    crop_y: int = 530
    crop_width: int = 890
    crop_height: int = 2890

    # -------------------------------------------------------------------------
    # Segmentation settings
    # -------------------------------------------------------------------------

    # Best starting point for faint satellite droplets:
    # use "fixed" and tune fixed_threshold between ~3 and 20.
    threshold_method: ThresholdMethod = "fixed"

    # For fixed thresholding on the background-corrected image.
    # Faint satellite droplets often need low values like 3, 5, 8, 10, 15.
    fixed_threshold: int = 5

    # For relaxed Otsu:
    # threshold = otsu_relaxation_factor * otsu_threshold
    otsu_relaxation_factor: float = 0.35

    # For percentile thresholding.
    # Lower values detect more faint droplets but may include more noise.
    percentile_threshold: float = 90.0

    # Adaptive thresholding.
    # In skimage threshold_local, increasing offset makes detection easier
    # because the offset is subtracted from the local threshold.
    adaptive_block_size: int = 151
    adaptive_offset: float = 15.0

    # -------------------------------------------------------------------------
    # Background correction
    # -------------------------------------------------------------------------

    # Larger sigma = smoother background estimate.
    # Try 50, 80, 100, 150 depending on illumination gradients.
    background_sigma: float = 80.0

    # True for dark droplets on bright background.
    dark_particles: bool = True

    # -------------------------------------------------------------------------
    # Contrast enhancement before thresholding
    # -------------------------------------------------------------------------

    # If True, pixels in the corrected image above enhancement_min_intensity
    # are multiplied by enhancement_factor before creating the binary mask.
    use_contrast_enhancement: bool = True

    # Only pixels brighter than this value are enhanced.
    enhancement_min_intensity: int = 5

    # Multiplication factor for pixels brighter than enhancement_min_intensity.
    enhancement_factor: float = 2.5

    # -------------------------------------------------------------------------
    # Particle filtering
    # -------------------------------------------------------------------------

    # During debugging, use 20 or 50.
    # After segmentation works, increase back to 100 if desired.
    min_area_px: int = 20

    max_area_px: int | None = None

    # During debugging, keep circularity loose.
    # After detection works, try min_circularity = 0.30 or 0.50.
    min_circularity: float = 0.50
    max_circularity: float = 1.10

    exclude_border: bool = True

    # -------------------------------------------------------------------------
    # Mask cleanup
    # -------------------------------------------------------------------------

    # Opening can delete small/faint satellite droplets.
    # Keep False while optimizing small-particle detection.
    use_opening: bool = False
    use_closing: bool = False

    opening_radius: int = 1
    closing_radius: int = 1

    fill_holes: bool = True

    # -------------------------------------------------------------------------
    # Watershed
    # -------------------------------------------------------------------------

    use_watershed: bool = False
    watershed_min_distance: int = 20

    # -------------------------------------------------------------------------
    # Export options
    # -------------------------------------------------------------------------

    save_overlay: bool = True
    save_qc_panels: bool = True

    # Use 1 during debugging to generate a QC panel for every image.
    qc_every_n_images: int = 1

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print_diagnostics: bool = True


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def list_image_files(folder: Path) -> list[Path]:
    """Return all JPEG files in a folder without duplicates."""
    allowed_extensions = {".jpg", ".jpeg"}

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_extensions
    ]

    return sorted(files)


def safe_crop(image: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """Crop image safely and raise a clear error if crop is outside image bounds."""
    h, w = image.shape[:2]

    x1 = cfg.crop_x
    y1 = cfg.crop_y
    x2 = cfg.crop_x + cfg.crop_width
    y2 = cfg.crop_y + cfg.crop_height

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(
            f"Crop region is outside image bounds. "
            f"Image size: width={w}, height={h}; "
            f"crop: x={x1}:{x2}, y={y1}:{y2}"
        )

    return image[y1:y2, x1:x2].copy()


def convert_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image from OpenCV to grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def background_correct(gray: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """
    Estimate local background and return a contrast image.

    If dark_particles=True:
        dark particles in the original grayscale image become bright signal
        in the corrected image.

    Original grayscale:
        dark particle = low intensity

    Corrected image:
        dark particle = high intensity
    """
    gray_float = gray.astype(np.float32)

    background = cv2.GaussianBlur(
        gray_float,
        ksize=(0, 0),
        sigmaX=cfg.background_sigma,
        sigmaY=cfg.background_sigma,
    )

    if cfg.dark_particles:
        corrected = background - gray_float
    else:
        corrected = gray_float - background

    corrected = np.clip(corrected, 0, None)

    if corrected.max() > 0:
        corrected = corrected / corrected.max() * 255.0

    return corrected.astype(np.uint8)


def enhance_corrected_image(corrected: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """
    Brighten pixels above a defined intensity threshold.

    Pixels <= enhancement_min_intensity are kept unchanged.
    Pixels > enhancement_min_intensity are multiplied by enhancement_factor.

    This enhanced image is used for thresholding if use_contrast_enhancement=True.
    """
    if not cfg.use_contrast_enhancement:
        return corrected

    enhanced = corrected.astype(np.float32).copy()

    bright_mask = enhanced > cfg.enhancement_min_intensity
    enhanced[bright_mask] = enhanced[bright_mask] * cfg.enhancement_factor

    enhanced = np.clip(enhanced, 0, 255)

    return enhanced.astype(np.uint8)


def create_binary_mask(corrected: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """
    Create a binary mask from the background-corrected contrast image.

    Important:
    For dark_particles=True, background_correct() already converts dark particles
    into bright signal. Therefore particles are detected using:

        corrected > threshold
    """
    positive_pixels = corrected[corrected > 0]

    if cfg.threshold_method == "fixed":
        thr = cfg.fixed_threshold
        mask = corrected > thr

        if cfg.print_diagnostics:
            print(f"Fixed threshold: {thr}")

    elif cfg.threshold_method == "otsu":
        thr = threshold_otsu(corrected)
        mask = corrected > thr

        if cfg.print_diagnostics:
            print(f"Otsu threshold: {thr:.2f}")

    elif cfg.threshold_method == "relaxed_otsu":
        otsu_thr = threshold_otsu(corrected)
        thr = cfg.otsu_relaxation_factor * otsu_thr
        mask = corrected > thr

        if cfg.print_diagnostics:
            print(
                f"Relaxed Otsu threshold: {thr:.2f} "
                f"({cfg.otsu_relaxation_factor:.2f} × {otsu_thr:.2f})"
            )

    elif cfg.threshold_method == "percentile":
        if positive_pixels.size == 0:
            if cfg.print_diagnostics:
                print("Percentile threshold: no positive pixels")
            return np.zeros_like(corrected, dtype=bool)

        thr = np.percentile(positive_pixels, cfg.percentile_threshold)
        mask = corrected > thr

        if cfg.print_diagnostics:
            print(
                f"Percentile threshold: "
                f"p{cfg.percentile_threshold:.1f} = {thr:.2f}"
            )

    elif cfg.threshold_method == "adaptive":
        block_size = cfg.adaptive_block_size

        if block_size % 2 == 0:
            block_size += 1

        local_thr = threshold_local(
            corrected,
            block_size=block_size,
            offset=cfg.adaptive_offset,
        )

        mask = corrected > local_thr

        if cfg.print_diagnostics:
            print(
                f"Adaptive threshold: block_size={block_size}, "
                f"offset={cfg.adaptive_offset}"
            )

    else:
        raise ValueError(f"Unknown threshold method: {cfg.threshold_method}")

    return mask.astype(bool)


def clean_mask(mask: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """
    Clean binary mask.

    For small/faint satellite droplets, binary opening should usually be disabled
    because it can remove small objects.
    """
    mask = morphology.remove_small_objects(mask, max_size=cfg.min_area_px)

    if cfg.fill_holes:
        mask = ndi.binary_fill_holes(mask)

    if cfg.use_opening:
        mask = morphology.binary_opening(
            mask,
            morphology.disk(cfg.opening_radius),
        )

    if cfg.use_closing:
        mask = morphology.binary_closing(
            mask,
            morphology.disk(cfg.closing_radius),
        )

    if cfg.exclude_border:
        mask = segmentation.clear_border(mask)

    return mask.astype(bool)


def label_particles(mask: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """
    Label particles.

    If watershed is enabled, touching droplets/capsules can be separated using
    the distance transform.
    """
    if not cfg.use_watershed:
        return measure.label(mask)

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        min_distance=cfg.watershed_min_distance,
        labels=mask,
    )

    markers = np.zeros(mask.shape, dtype=np.int32)

    if len(coords) > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        return measure.label(mask)

    labels = segmentation.watershed(-distance, markers, mask=mask)

    return labels


def print_object_area_stats(mask_or_labels: np.ndarray, name: str) -> None:
    """
    Print object area statistics to see at which step small particles disappear.
    """
    labels_tmp = measure.label(mask_or_labels > 0)
    props = measure.regionprops(labels_tmp)
    areas = np.array([p.area for p in props])

    if len(areas) == 0:
        print(f"{name}: no objects")
        return

    print(
        f"{name}: "
        f"n={len(areas)}, "
        f"min={areas.min():.0f}, "
        f"p10={np.percentile(areas, 10):.0f}, "
        f"median={np.median(areas):.0f}, "
        f"p90={np.percentile(areas, 90):.0f}, "
        f"max={areas.max():.0f}"
    )


def measure_particles(
    labels: np.ndarray,
    gray: np.ndarray,
    image_name: str,
    folder_name: str,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Measure particle properties and convert important values to mm."""
    if labels.max() == 0:
        return pd.DataFrame()

    props = measure.regionprops_table(
        labels,
        intensity_image=gray,
        properties=[
            "label",
            "area",
            "centroid",
            "perimeter",
            "eccentricity",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
            "equivalent_diameter_area",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "bbox",
        ],
    )

    df = pd.DataFrame(props)

    if df.empty:
        return df

    # Avoid division by zero for tiny/degenerate objects
    df = df[df["perimeter"] > 0].copy()

    if df.empty:
        return df

    # ImageJ-style circularity: 4*pi*Area/Perimeter^2
    df["circularity"] = 4.0 * np.pi * df["area"] / (df["perimeter"] ** 2)

    # Apply particle filters
    df = df[df["area"] >= cfg.min_area_px].copy()

    if cfg.max_area_px is not None:
        df = df[df["area"] <= cfg.max_area_px].copy()

    df = df[
        (df["circularity"] >= cfg.min_circularity)
        & (df["circularity"] <= cfg.max_circularity)
    ].copy()

    if df.empty:
        return df

    # Convert to physical units
    px = cfg.pixel_size_mm

    df["area_mm2"] = df["area"] * px**2
    df["ECD_mm"] = df["equivalent_diameter_area"] * px
    df["perimeter_mm"] = df["perimeter"] * px
    df["major_axis_mm"] = df["major_axis_length"] * px
    df["minor_axis_mm"] = df["minor_axis_length"] * px
    df["centroid_x_mm"] = df["centroid-1"] * px
    df["centroid_y_mm"] = df["centroid-0"] * px

    # Add metadata
    df.insert(0, "folder", folder_name)
    df.insert(1, "image", image_name)

    return df.reset_index(drop=True)


def annotate_overlay_with_labels(
    overlay: np.ndarray,
    labels: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Draw thicker boundaries and numeric labels for each object."""
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    boundaries = morphology.dilation(boundaries, morphology.disk(2))
    overlay[boundaries] = color

    for prop in measure.regionprops(labels):
        row, col = prop.centroid
        text = str(prop.label)
        position = (int(col) + 2, int(row) + 2)
        cv2.putText(
            overlay,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def save_overlay_image(
    crop_bgr: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save an RGB overlay image with particle boundaries."""
    overlay = crop_bgr.copy()
    annotate_overlay_with_labels(overlay, labels, (0, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def save_qc_panel(
    crop_bgr: np.ndarray,
    gray: np.ndarray,
    corrected: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a QC panel showing the major processing steps.

    Important:
    - "Corrected" shows the background-corrected image before enhancement.
    - "Enhanced corrected" is the actual image used for thresholding.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Overlay particle boundaries on the original cropped image
    overlay = crop_rgb.copy()
    annotate_overlay_with_labels(overlay, labels, (255, 0, 0))

    fig, axes = plt.subplots(1, 6, figsize=(22, 5))

    axes[0].imshow(crop_rgb)
    axes[0].set_title("Crop")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(corrected, cmap="gray")
    axes[2].set_title("Corrected")
    axes[2].axis("off")

    axes[3].imshow(enhanced, cmap="gray")
    axes[3].set_title("Enhanced corrected")
    axes[3].axis("off")

    axes[4].imshow(mask, cmap="gray")
    axes[4].set_title("Binary mask")
    axes[4].axis("off")

    axes[5].imshow(overlay)
    axes[5].set_title("Overlay")
    axes[5].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------


def process_single_image(
    image_path: Path,
    folder_output_dir: Path,
    qc_output_dir: Path,
    image_index: int,
    folder_name: str,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Process one image and return a DataFrame with particle measurements."""
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print(f"Could not open image: {image_path}")
        return pd.DataFrame()

    try:
        crop_bgr = safe_crop(image_bgr, cfg)
    except ValueError as exc:
        print(f"Skipping {image_path.name}: {exc}")
        return pd.DataFrame()

    gray = convert_to_gray(crop_bgr)
    corrected = background_correct(gray, cfg)

    enhanced = enhance_corrected_image(corrected, cfg)

    mask_raw = create_binary_mask(enhanced, cfg)

    if cfg.print_diagnostics:
        print_object_area_stats(mask_raw, "raw mask after threshold")

    mask = clean_mask(mask_raw, cfg)

    if cfg.print_diagnostics:
        print_object_area_stats(mask, "clean mask")

    labels = label_particles(mask, cfg)

    if cfg.print_diagnostics:
        print_object_area_stats(labels, "labels before measurement")

    df = measure_particles(
        labels=labels,
        gray=gray,
        image_name=image_path.name,
        folder_name=folder_name,
        cfg=cfg,
    )

    if cfg.print_diagnostics:
        if not df.empty:
            print(
                f"after dataframe filters: "
                f"n={len(df)}, "
                f"min area={df['area'].min():.0f}, "
                f"median area={df['area'].median():.0f}, "
                f"max area={df['area'].max():.0f}"
            )
        else:
            print("after dataframe filters: no particles")

    if cfg.save_overlay:
        overlay_path = folder_output_dir / f"marked_{image_index:03d}_{image_path.stem}.png"
        save_overlay_image(crop_bgr, labels, overlay_path)

    if cfg.save_qc_panels and (image_index == 1 or image_index % cfg.qc_every_n_images == 0):
        qc_path = qc_output_dir / f"qc_{image_index:03d}_{image_path.stem}.png"
        save_qc_panel(crop_bgr, gray, corrected, enhanced, mask, labels, qc_path)

    return df


def process_image_folder(
    images_folder: Path,
    folder_output_dir: Path,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """
    Process all JPEG images in one images folder and return particle measurements.

    This function is intended for automated batch orchestration from master_batch.py.
    The actual single-image analysis pipeline is not changed; this function only
    controls which folder is processed and where QC/overlay images are written.
    """
    print(f"\nProcessing image folder: {images_folder}")

    image_files = list_image_files(images_folder)

    if not image_files:
        print(f"No JPEG images found in {images_folder}")
        return pd.DataFrame()

    qc_output_dir = folder_output_dir / "QC"
    folder_output_dir.mkdir(parents=True, exist_ok=True)
    qc_output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []

    for image_index, image_path in enumerate(image_files, start=1):
        print(f"\n  [{image_index}/{len(image_files)}] {image_path.name}")

        df = process_single_image(
            image_path=image_path,
            folder_output_dir=folder_output_dir,
            qc_output_dir=qc_output_dir,
            image_index=image_index,
            folder_name=images_folder.parent.name,
            cfg=cfg,
        )

        if not df.empty:
            all_results.append(df)

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    print(f"Detected particles in image folder: {len(result_df)}")

    return result_df


def process_folder(folder: Path, folder_index: int, cfg: AnalysisConfig) -> pd.DataFrame:
    """Process all JPEG images in one subfolder and save one CSV."""
    print(f"\nProcessing folder {folder_index:02d}: {folder}")

    image_files = list_image_files(folder)

    if not image_files:
        print(f"No JPEG images found in {folder}")
        return pd.DataFrame()

    folder_output_dir = cfg.results_root / f"{cfg.label}_{folder_index:02d}"
    qc_output_dir = folder_output_dir / "QC"

    folder_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.csv_root.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []

    for image_index, image_path in enumerate(image_files, start=1):
        print(f"\n  [{image_index}/{len(image_files)}] {image_path.name}")

        df = process_single_image(
            image_path=image_path,
            folder_output_dir=folder_output_dir,
            qc_output_dir=qc_output_dir,
            image_index=image_index,
            folder_name=folder.name,
            cfg=cfg,
        )

        if not df.empty:
            all_results.append(df)

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.csv_root / f"{cfg.label}_{folder_index:02d}_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Detected particles in folder: {len(result_df)}")

    return result_df


def run_batch_analysis(cfg: AnalysisConfig) -> pd.DataFrame:
    """Process every subfolder in the input root."""
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.csv_root.mkdir(parents=True, exist_ok=True)

    subfolders = sorted([p for p in cfg.input_root.iterdir() if p.is_dir()])

    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in {cfg.input_root}")

    all_folder_results: list[pd.DataFrame] = []

    for folder_index, folder in enumerate(subfolders, start=1):
        folder_df = process_folder(folder, folder_index, cfg)

        if not folder_df.empty:
            all_folder_results.append(folder_df)

    if all_folder_results:
        combined_df = pd.concat(all_folder_results, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv = cfg.csv_root / f"{cfg.label}_ALL_RESULTS_{timestamp}.csv"
    combined_df.to_csv(combined_csv, index=False)

    print("\nBatch analysis finished.")
    print(f"Combined CSV: {combined_csv}")
    print(f"Total detected particles: {len(combined_df)}")

    return combined_df
