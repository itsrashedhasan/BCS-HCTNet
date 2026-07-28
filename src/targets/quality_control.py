"""Step 05B: Numerical and visual quality control for derived targets.

This CPU-only stage reads the persistent Step 04 split artifacts and the
persistent Step 05A target artifacts. It performs deterministic numerical
reproduction checks for every ISIC 2018 target bundle and creates a stratified
visual-review package.

Automatic numerical checks do not replace visual review. Training remains
blocked until the generated contact sheets are reviewed and the final Step 05C
sign-off is recorded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data.analyze_overlap import normalize_image_id, read_csv_rows
from src.targets.generate_targets import find_step04_split_directory
from src.targets.target_geometry import (
    TargetGeometryConfig,
    generate_target_bundle,
    sha256_file,
    target_statistics,
    validate_target_bundle,
)
from src.utils import config


EXPECTED_SPLIT_COUNTS = {
    "train": 2594,
    "val": 100,
    "internal_test": 1000,
}

EXPECTED_TOTAL_IMAGES = 3694
EXPECTED_FULL_FOREGROUND_ID = "ISIC_0023056"
QC_PROTOCOL_VERSION = "BCS-HCTNet-target-qc-v1"

VISUAL_SAMPLES_PER_STRATUM = 2
VISUAL_SAMPLES_PER_SHEET = 4
VISUAL_TILE_SIZE = 210


def find_step05a_artifact() -> tuple[Path, Path, Path]:
    """Locate the persistent Step 05A target artifact."""

    input_root = Path("/kaggle/input")

    candidates: list[
        tuple[Path, Path, Path]
    ] = []

    for manifest_path in input_root.rglob(
        "isic2018_derived_targets_352.csv"
    ):
        artifact_root = manifest_path.parents[2]

        target_root = (
            artifact_root
            / "data"
            / "targets"
            / "isic2018_352"
        )

        config_path = (
            target_root
            / "TARGET_CONFIG.json"
        )

        report_path = (
            artifact_root
            / "outputs"
            / "reports"
            / "step05_target_generation_report.json"
        )

        if (
            target_root.is_dir()
            and config_path.is_file()
            and report_path.is_file()
        ):
            candidates.append(
                (
                    artifact_root.resolve(),
                    manifest_path.resolve(),
                    target_root.resolve(),
                )
            )

    unique_candidates = sorted(
        set(candidates),
        key=lambda item: str(
            item[0]
        ),
    )

    preferred = [
        candidate
        for candidate in unique_candidates
        if (
            "bcs-hctnet-step05a-target-artifacts"
            in str(
                candidate[0]
            ).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(unique_candidates) == 1:
        return unique_candidates[0]

    if not unique_candidates:
        raise FileNotFoundError(
            "Could not locate the persistent "
            "Step 05A target artifact."
        )

    raise RuntimeError(
        "Multiple Step 05A artifacts were found: "
        f"{[str(item[0]) for item in unique_candidates]}"
    )


def load_geometry_config(
    target_root: Path,
) -> TargetGeometryConfig:
    """Load and verify the persisted target configuration."""

    config_path = (
        target_root
        / "TARGET_CONFIG.json"
    )

    payload = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    geometry_payload = payload.get(
        "target_geometry",
        {},
    )

    geometry_config = TargetGeometryConfig(
        output_size=int(
            geometry_payload[
                "output_size"
            ]
        ),
        contour_width=int(
            geometry_payload[
                "contour_width"
            ]
        ),
        boundary_band_radius=int(
            geometry_payload[
                "boundary_band_radius"
            ]
        ),
        sdm_clip_distance=float(
            geometry_payload[
                "sdm_clip_distance"
            ]
        ),
        connectivity=int(
            geometry_payload[
                "connectivity"
            ]
        ),
    )

    geometry_config.validate()

    expected_fingerprint = str(
        payload.get(
            "target_geometry_sha256",
            "",
        )
    )

    if (
        geometry_config.fingerprint()
        != expected_fingerprint
    ):
        raise RuntimeError(
            "Persisted target configuration "
            "fingerprint mismatch."
        )

    return geometry_config


def resolve_persistent_path(
    absolute_value: object,
    relative_value: object,
    relative_root: Path | None = None,
) -> Path:
    """Resolve a persisted path without filename guessing."""

    absolute_text = str(
        absolute_value or ""
    ).strip()

    if (
        absolute_text
        and "::" not in absolute_text
    ):
        direct_path = Path(
            absolute_text
        )

        if direct_path.is_file():
            return direct_path

        marker = "/kaggle/input/"

        if marker in absolute_text:
            rebased = (
                Path("/kaggle/input")
                / absolute_text.split(
                    marker,
                    1,
                )[1]
            )

            if rebased.is_file():
                return rebased

    relative_text = str(
        relative_value or ""
    ).strip()

    if (
        relative_text
        and "::" not in relative_text
    ):
        candidate_roots: list[
            Path
        ] = []

        if relative_root is not None:
            candidate_roots.append(
                relative_root
            )

        candidate_roots.append(
            Path("/kaggle/input")
        )

        for root in candidate_roots:
            candidate = (
                root
                / relative_text
            )

            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Could not resolve persisted path. "
        f"absolute={absolute_text!r}, "
        f"relative={relative_text!r}, "
        f"relative_root={str(relative_root)!r}."
    )


def target_bundle_paths(
    row: dict[str, str],
    target_root: Path,
) -> dict[str, Path]:
    """Resolve all four persisted target files."""

    mapping = {
        "mask": (
            "target_mask_path",
            "target_mask_relative_path",
        ),
        "contour": (
            "target_contour_path",
            "target_contour_relative_path",
        ),
        "boundary_band": (
            "target_boundary_band_path",
            "target_boundary_band_relative_path",
        ),
        "sdm": (
            "target_sdm_path",
            "target_sdm_relative_path",
        ),
    }

    return {
        name: resolve_persistent_path(
            row.get(
                absolute_field,
                "",
            ),
            row.get(
                relative_field,
                "",
            ),
            target_root,
        )
        for (
            name,
            (
                absolute_field,
                relative_field,
            ),
        ) in mapping.items()
    }


def load_saved_bundle(
    paths: dict[str, Path],
    geometry_config: TargetGeometryConfig,
) -> dict[str, np.ndarray]:
    """Load and structurally validate a target bundle."""

    binary_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for name in [
        "mask",
        "contour",
        "boundary_band",
    ]:
        with Image.open(
            paths[name]
        ) as opened_image:
            image = (
                opened_image
                .convert("L")
            )

            image.load()

            binary_arrays[
                name
            ] = np.asarray(
                image,
                dtype=np.uint8,
            )

    sdm = np.load(
        paths["sdm"],
        allow_pickle=False,
    )

    bundle = {
        **binary_arrays,
        "sdm": sdm,
    }

    validate_target_bundle(
        bundle,
        geometry_config,
    )

    return bundle


def exact_binary_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> int:
    """Count unequal binary pixels."""

    return int(
        np.count_nonzero(
            np.asarray(first)
            != np.asarray(second)
        )
    )


def inspect_sample(
    row: dict[str, str],
    target_root: Path,
    geometry_config: TargetGeometryConfig,
    lesion_size_group: str,
) -> dict[str, Any]:
    """Reproduce and validate one saved target bundle."""

    image_id = normalize_image_id(
        row["image_id"]
    )

    split = str(
        row["split"]
    )

    source_mask_path = (
        resolve_persistent_path(
            row.get(
                "source_mask_path",
                "",
            ),
            row.get(
                "source_mask_relative_path",
                "",
            ),
        )
    )

    source_image_path = (
        resolve_persistent_path(
            row.get(
                "source_image_path",
                "",
            ),
            row.get(
                "source_image_relative_path",
                "",
            ),
        )
    )

    paths = target_bundle_paths(
        row,
        target_root,
    )

    saved_bundle = load_saved_bundle(
        paths,
        geometry_config,
    )

    with Image.open(
        source_mask_path
    ) as opened_mask:
        source_mask = (
            opened_mask
            .convert("L")
            .copy()
        )

        source_mask_size = (
            source_mask.size
        )

    reproduced_bundle = (
        generate_target_bundle(
            source_mask,
            geometry_config,
        )
    )

    mask_difference_pixels = (
        exact_binary_difference(
            saved_bundle["mask"],
            reproduced_bundle["mask"],
        )
    )

    contour_difference_pixels = (
        exact_binary_difference(
            saved_bundle["contour"],
            reproduced_bundle["contour"],
        )
    )

    boundary_band_difference_pixels = (
        exact_binary_difference(
            saved_bundle[
                "boundary_band"
            ],
            reproduced_bundle[
                "boundary_band"
            ],
        )
    )

    sdm_absolute_difference = np.abs(
        np.asarray(
            saved_bundle["sdm"],
            dtype=np.float32,
        )
        - np.asarray(
            reproduced_bundle["sdm"],
            dtype=np.float32,
        )
    )

    sdm_max_absolute_difference = float(
        np.max(
            sdm_absolute_difference
        )
    )

    sdm_mean_absolute_difference = float(
        np.mean(
            sdm_absolute_difference
        )
    )

    statistics = target_statistics(
        saved_bundle,
        geometry_config,
    )

    mask = (
        np.asarray(
            saved_bundle["mask"]
        )
        > 0
    )

    contour = (
        np.asarray(
            saved_bundle["contour"]
        )
        > 0
    )

    boundary_band = (
        np.asarray(
            saved_bundle[
                "boundary_band"
            ]
        )
        > 0
    )

    sdm = np.asarray(
        saved_bundle["sdm"],
        dtype=np.float32,
    )

    border_touching = bool(
        np.any(
            mask[0, :]
        )
        or np.any(
            mask[-1, :]
        )
        or np.any(
            mask[:, 0]
        )
        or np.any(
            mask[:, -1]
        )
    )

    contour_inside_boundary_band = bool(
        np.all(
            np.logical_or(
                np.logical_not(
                    contour
                ),
                boundary_band,
            )
        )
    )

    background = np.logical_not(
        mask
    )

    sdm_inside_positive = bool(
        np.all(
            sdm[mask] > 0
        )
    )

    sdm_outside_negative = bool(
        not np.any(
            background
        )
        or np.all(
            sdm[
                background
            ]
            < 0
        )
    )

    saved_hashes_match_manifest = {
        "mask": (
            not row.get(
                "target_mask_sha256",
                "",
            ).strip()
            or sha256_file(
                paths["mask"]
            )
            == row[
                "target_mask_sha256"
            ]
        ),
        "contour": (
            not row.get(
                "target_contour_sha256",
                "",
            ).strip()
            or sha256_file(
                paths["contour"]
            )
            == row[
                "target_contour_sha256"
            ]
        ),
        "boundary_band": (
            not row.get(
                "target_boundary_band_sha256",
                "",
            ).strip()
            or sha256_file(
                paths[
                    "boundary_band"
                ]
            )
            == row[
                "target_boundary_band_sha256"
            ]
        ),
        "sdm": (
            not row.get(
                "target_sdm_sha256",
                "",
            ).strip()
            or sha256_file(
                paths["sdm"]
            )
            == row[
                "target_sdm_sha256"
            ]
        ),
    }

    sample_checks = {
        "mask_reproduces_exactly": (
            mask_difference_pixels
            == 0
        ),
        "contour_reproduces_exactly": (
            contour_difference_pixels
            == 0
        ),
        "boundary_band_reproduces_exactly": (
            boundary_band_difference_pixels
            == 0
        ),
        "sdm_reproduces_within_tolerance": (
            sdm_max_absolute_difference
            <= 1e-7
        ),
        "contour_inside_boundary_band": (
            contour_inside_boundary_band
        ),
        "sdm_inside_positive": (
            sdm_inside_positive
        ),
        "sdm_outside_negative": (
            sdm_outside_negative
        ),
        "saved_hashes_match_manifest": (
            all(
                saved_hashes_match_manifest
                .values()
            )
        ),
    }

    return {
        "image_id": image_id,
        "split": split,
        "lesion_size_group": (
            lesion_size_group
        ),
        "source_image_path": str(
            source_image_path
        ),
        "source_mask_path": str(
            source_mask_path
        ),
        "source_mask_width": int(
            source_mask_size[0]
        ),
        "source_mask_height": int(
            source_mask_size[1]
        ),
        "target_mask_path": str(
            paths["mask"]
        ),
        "target_contour_path": str(
            paths["contour"]
        ),
        "target_boundary_band_path": str(
            paths[
                "boundary_band"
            ]
        ),
        "target_sdm_path": str(
            paths["sdm"]
        ),
        "mask_difference_pixels": (
            mask_difference_pixels
        ),
        "contour_difference_pixels": (
            contour_difference_pixels
        ),
        "boundary_band_difference_pixels": (
            boundary_band_difference_pixels
        ),
        "sdm_max_absolute_difference": (
            sdm_max_absolute_difference
        ),
        "sdm_mean_absolute_difference": (
            sdm_mean_absolute_difference
        ),
        "border_touching": (
            border_touching
        ),
        "contour_inside_boundary_band": (
            contour_inside_boundary_band
        ),
        "sdm_inside_positive": (
            sdm_inside_positive
        ),
        "sdm_outside_negative": (
            sdm_outside_negative
        ),
        "target_hashes_match_manifest": (
            all(
                saved_hashes_match_manifest
                .values()
            )
        ),
        "sample_passed": (
            all(
                sample_checks.values()
            )
        ),
        "failed_checks": ";".join(
            name
            for (
                name,
                passed,
            ) in sample_checks.items()
            if not passed
        ),
        **statistics,
    }


def stable_sample_key(
    image_id: str,
) -> str:
    """Return a deterministic pseudo-random ordering key."""

    return hashlib.sha256(
        (
            f"{QC_PROTOCOL_VERSION}:"
            f"{image_id}"
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def add_selection_reason(
    selected: dict[
        str,
        set[str],
    ],
    image_id: str,
    reason: str,
) -> None:
    """Add a visual-selection reason."""

    selected.setdefault(
        image_id,
        set(),
    ).add(
        reason
    )


def select_visual_samples(
    results: Sequence[
        dict[str, Any]
    ],
) -> dict[str, set[str]]:
    """Select stratified and edge-case visual samples."""

    selected: dict[
        str,
        set[str],
    ] = {}

    for split in [
        "train",
        "val",
        "internal_test",
    ]:
        for size_group in [
            "small",
            "medium",
            "large",
        ]:
            candidates = sorted(
                (
                    row
                    for row in results
                    if (
                        row["split"]
                        == split
                        and row[
                            "lesion_size_group"
                        ]
                        == size_group
                    )
                ),
                key=lambda row: (
                    stable_sample_key(
                        str(
                            row[
                                "image_id"
                            ]
                        )
                    )
                ),
            )

            if not candidates:
                raise RuntimeError(
                    "No sample available for "
                    "visual stratum "
                    f"split={split}, "
                    f"size={size_group}."
                )

            for row in candidates[
                :VISUAL_SAMPLES_PER_STRATUM
            ]:
                add_selection_reason(
                    selected,
                    str(
                        row[
                            "image_id"
                        ]
                    ),
                    (
                        f"stratified_{split}_"
                        f"{size_group}"
                    ),
                )

    by_ratio = sorted(
        results,
        key=lambda row: (
            float(
                row[
                    "mask_foreground_ratio"
                ]
            ),
            str(
                row[
                    "image_id"
                ]
            ),
        ),
    )

    for row in by_ratio[:3]:
        add_selection_reason(
            selected,
            str(
                row["image_id"]
            ),
            "global_smallest_mask_ratio",
        )

    for row in by_ratio[-3:]:
        add_selection_reason(
            selected,
            str(
                row["image_id"]
            ),
            "global_largest_mask_ratio",
        )

    by_components = sorted(
        results,
        key=lambda row: (
            -int(
                row[
                    "connected_component_count"
                ]
            ),
            str(
                row[
                    "image_id"
                ]
            ),
        ),
    )

    for row in by_components[:4]:
        add_selection_reason(
            selected,
            str(
                row["image_id"]
            ),
            "highest_component_count",
        )

    border_rows = sorted(
        (
            row
            for row in results
            if bool(
                row[
                    "border_touching"
                ]
            )
        ),
        key=lambda row: (
            -float(
                row[
                    "mask_foreground_ratio"
                ]
            ),
            str(
                row[
                    "image_id"
                ]
            ),
        ),
    )

    for row in border_rows[:4]:
        add_selection_reason(
            selected,
            str(
                row["image_id"]
            ),
            "border_touching_mask",
        )

    full_foreground_rows = [
        row
        for row in results
        if bool(
            row[
                "mask_is_full_foreground"
            ]
        )
    ]

    for row in full_foreground_rows:
        add_selection_reason(
            selected,
            str(
                row["image_id"]
            ),
            "full_foreground_special_case",
        )

    return selected


def load_font(
    size: int,
) -> ImageFont.ImageFont:
    """Load a readable font when available."""

    candidates = [
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/"
            "liberation2/"
            "LiberationSans-Regular.ttf"
        ),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(
                str(candidate),
                size=size,
            )

    return ImageFont.load_default()


def resized_training_image(
    image_path: Path,
    size: int,
) -> Image.Image:
    """Resize the source image to the target square."""

    with Image.open(
        image_path
    ) as opened_image:
        image = ImageOps.exif_transpose(
            opened_image
        ).convert(
            "RGB"
        )

        resampling = getattr(
            Image,
            "Resampling",
            Image,
        )

        return image.resize(
            (
                size,
                size,
            ),
            resample=(
                resampling.BILINEAR
            ),
        )


def overlay_binary(
    base_image: Image.Image,
    binary_array: np.ndarray,
    color: tuple[
        int,
        int,
        int,
    ],
    alpha: int,
) -> Image.Image:
    """Overlay a binary target on an RGB image."""

    base = base_image.convert(
        "RGBA"
    )

    binary = (
        np.asarray(
            binary_array
        )
        > 0
    )

    overlay = np.zeros(
        (
            binary.shape[0],
            binary.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    overlay[
        binary,
        0,
    ] = color[0]

    overlay[
        binary,
        1,
    ] = color[1]

    overlay[
        binary,
        2,
    ] = color[2]

    overlay[
        binary,
        3,
    ] = alpha

    overlay_image = Image.fromarray(
        overlay,
        mode="RGBA",
    )

    return Image.alpha_composite(
        base,
        overlay_image,
    ).convert(
        "RGB"
    )


def sdm_visualization(
    sdm: np.ndarray,
) -> Image.Image:
    """Create an RGB signed-distance visualization."""

    values = np.asarray(
        sdm,
        dtype=np.float32,
    )

    normalized = np.clip(
        (
            values + 1.0
        )
        * 127.5,
        0,
        255,
    ).astype(
        np.uint8
    )

    colored_bgr = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO,
    )

    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        colored_rgb,
        mode="RGB",
    )


def add_tile_title(
    tile: Image.Image,
    title: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Add a title strip above a visual tile."""

    canvas = Image.new(
        "RGB",
        (
            tile.width,
            tile.height + 28,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            6,
            5,
        ),
        title,
        fill="black",
        font=font,
    )

    canvas.paste(
        tile,
        (
            0,
            28,
        ),
    )

    return canvas


def create_visual_panel(
    row: dict[str, Any],
    reasons: Sequence[str],
) -> Image.Image:
    """Create one five-view target-QC panel."""

    target_size = int(
        row["height"]
    )

    image = resized_training_image(
        Path(
            str(
                row[
                    "source_image_path"
                ]
            )
        ),
        target_size,
    )

    with Image.open(
        Path(
            str(
                row[
                    "target_mask_path"
                ]
            )
        )
    ) as opened_mask:
        mask = np.asarray(
            opened_mask.convert(
                "L"
            ),
            dtype=np.uint8,
        )

    with Image.open(
        Path(
            str(
                row[
                    "target_contour_path"
                ]
            )
        )
    ) as opened_contour:
        contour = np.asarray(
            opened_contour.convert(
                "L"
            ),
            dtype=np.uint8,
        )

    with Image.open(
        Path(
            str(
                row[
                    "target_boundary_band_path"
                ]
            )
        )
    ) as opened_band:
        boundary_band = np.asarray(
            opened_band.convert(
                "L"
            ),
            dtype=np.uint8,
        )

    sdm = np.load(
        Path(
            str(
                row[
                    "target_sdm_path"
                ]
            )
        ),
        allow_pickle=False,
    )

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    )

    views = [
        (
            "Resized image",
            image,
        ),
        (
            "Mask overlay",
            overlay_binary(
                image,
                mask,
                (
                    255,
                    0,
                    0,
                ),
                105,
            ),
        ),
        (
            "Contour overlay",
            overlay_binary(
                image,
                contour,
                (
                    0,
                    255,
                    0,
                ),
                220,
            ),
        ),
        (
            "Boundary band",
            overlay_binary(
                image,
                boundary_band,
                (
                    255,
                    215,
                    0,
                ),
                145,
            ),
        ),
        (
            "Signed distance map",
            sdm_visualization(
                sdm
            ),
        ),
    ]

    tile_font = load_font(
        14
    )

    title_font = load_font(
        17
    )

    body_font = load_font(
        13
    )

    titled_tiles: list[
        Image.Image
    ] = []

    for (
        title,
        view,
    ) in views:
        resized_view = view.resize(
            (
                VISUAL_TILE_SIZE,
                VISUAL_TILE_SIZE,
            ),
            resample=(
                resampling.BILINEAR
            ),
        )

        titled_tiles.append(
            add_tile_title(
                resized_view,
                title,
                tile_font,
            )
        )

    panel_width = (
        VISUAL_TILE_SIZE
        * len(
            titled_tiles
        )
    )

    panel_height = (
        VISUAL_TILE_SIZE
        + 110
    )

    panel = Image.new(
        "RGB",
        (
            panel_width,
            panel_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        panel
    )

    headline = (
        f"{row['image_id']} | "
        f"{row['split']} | "
        f"size={row['lesion_size_group']} | "
        f"mask="
        f"{float(row['mask_foreground_ratio']):.4f} | "
        f"components="
        f"{int(row['connected_component_count'])} | "
        f"border="
        f"{bool(row['border_touching'])}"
    )

    draw.text(
        (
            8,
            5,
        ),
        headline,
        fill="black",
        font=title_font,
    )

    reason_text = (
        "Selection: "
        + "; ".join(
            reasons
        )
    )

    draw.text(
        (
            8,
            31,
        ),
        reason_text,
        fill="black",
        font=body_font,
    )

    metric_text = (
        "Reproduction differences: "
        f"mask="
        f"{int(row['mask_difference_pixels'])}, "
        f"contour="
        f"{int(row['contour_difference_pixels'])}, "
        f"band="
        f"{int(row['boundary_band_difference_pixels'])}, "
        f"SDM max="
        f"{float(row['sdm_max_absolute_difference']):.2e}"
    )

    draw.text(
        (
            8,
            52,
        ),
        metric_text,
        fill="black",
        font=body_font,
    )

    y_offset = 78

    for (
        index,
        tile,
    ) in enumerate(
        titled_tiles
    ):
        panel.paste(
            tile,
            (
                index
                * VISUAL_TILE_SIZE,
                y_offset,
            ),
        )

    return panel


def create_contact_sheets(
    selected_rows: Sequence[
        tuple[
            dict[str, Any],
            Sequence[str],
        ]
    ],
    output_directory: Path,
) -> list[dict[str, Any]]:
    """Create deterministic visual-review sheets."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_rows: list[
        dict[str, Any]
    ] = []

    panel_height = (
        VISUAL_TILE_SIZE
        + 110
    )

    panel_width = (
        VISUAL_TILE_SIZE
        * 5
    )

    for sheet_start in range(
        0,
        len(
            selected_rows
        ),
        VISUAL_SAMPLES_PER_SHEET,
    ):
        sheet_rows = selected_rows[
            sheet_start:
            sheet_start
            + VISUAL_SAMPLES_PER_SHEET
        ]

        sheet_number = (
            sheet_start
            // VISUAL_SAMPLES_PER_SHEET
        ) + 1

        sheet = Image.new(
            "RGB",
            (
                panel_width,
                panel_height
                * len(
                    sheet_rows
                ),
            ),
            "white",
        )

        for (
            panel_index,
            (
                row,
                reasons,
            ),
        ) in enumerate(
            sheet_rows
        ):
            panel = create_visual_panel(
                row,
                reasons,
            )

            sheet.paste(
                panel,
                (
                    0,
                    panel_index
                    * panel_height,
                ),
            )

            index_rows.append(
                {
                    "visual_rank": (
                        sheet_start
                        + panel_index
                        + 1
                    ),
                    "contact_sheet": (
                        f"target_qc_sheet_"
                        f"{sheet_number:03d}.jpg"
                    ),
                    "panel_position": (
                        panel_index
                        + 1
                    ),
                    "image_id": (
                        row[
                            "image_id"
                        ]
                    ),
                    "split": (
                        row[
                            "split"
                        ]
                    ),
                    "lesion_size_group": (
                        row[
                            "lesion_size_group"
                        ]
                    ),
                    "selection_reasons": (
                        ";".join(
                            reasons
                        )
                    ),
                }
            )

        output_path = (
            output_directory
            / (
                f"target_qc_sheet_"
                f"{sheet_number:03d}.jpg"
            )
        )

        sheet.save(
            output_path,
            format="JPEG",
            quality=92,
            optimize=True,
        )

    return index_rows


def write_csv(
    output_path: Path,
    rows: Sequence[
        dict[str, Any]
    ],
    fieldnames: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV output."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                fieldnames
            ),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def numeric_summary(
    values: Sequence[
        float | int
    ],
) -> dict[str, float | int]:
    """Return reproducible descriptive statistics."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        return {}

    return {
        "count": int(
            array.size
        ),
        "minimum": float(
            np.min(
                array
            )
        ),
        "p01": float(
            np.quantile(
                array,
                0.01,
            )
        ),
        "p05": float(
            np.quantile(
                array,
                0.05,
            )
        ),
        "p25": float(
            np.quantile(
                array,
                0.25,
            )
        ),
        "median": float(
            np.quantile(
                array,
                0.50,
            )
        ),
        "p75": float(
            np.quantile(
                array,
                0.75,
            )
        ),
        "p95": float(
            np.quantile(
                array,
                0.95,
            )
        ),
        "p99": float(
            np.quantile(
                array,
                0.99,
            )
        ),
        "maximum": float(
            np.max(
                array
            )
        ),
        "mean": float(
            np.mean(
                array
            )
        ),
    }


def main() -> int:
    """Run exhaustive numerical and visual target QC."""

    config.ensure_all_dirs()

    (
        artifact_root,
        target_manifest_path,
        target_root,
    ) = find_step05a_artifact()

    split_directory = (
        find_step04_split_directory()
    )

    locked_manifest_path = (
        split_directory
        / "isic2018_all_locked.csv"
    )

    geometry_config = (
        load_geometry_config(
            target_root
        )
    )

    target_rows = read_csv_rows(
        target_manifest_path
    )

    locked_rows = read_csv_rows(
        locked_manifest_path
    )

    if (
        len(
            target_rows
        )
        != EXPECTED_TOTAL_IMAGES
    ):
        raise RuntimeError(
            "Unexpected target-manifest "
            "count: expected "
            f"{EXPECTED_TOTAL_IMAGES}, "
            f"found {len(target_rows)}."
        )

    lesion_size_by_id = {
        normalize_image_id(
            row[
                "image_id"
            ]
        ): str(
            row[
                "lesion_size_group"
            ]
        )
        for row in locked_rows
    }

    if (
        len(
            lesion_size_by_id
        )
        != EXPECTED_TOTAL_IMAGES
    ):
        raise RuntimeError(
            "Unexpected locked split "
            "unique-image count."
        )

    configured_workers = int(
        os.environ.get(
            "BCS_TARGET_QC_WORKERS",
            "0",
        )
    )

    max_workers = (
        configured_workers
        if configured_workers > 0
        else min(
            8,
            max(
                1,
                os.cpu_count() or 1,
            ),
        )
    )

    print(
        "=== Step 05B: Derived Target "
        "Quality Control ==="
    )

    print(
        "Step 05A artifact : "
        f"{artifact_root}"
    )

    print(
        "Step 04 splits    : "
        f"{split_directory}"
    )

    print(
        "Target root       : "
        f"{target_root}"
    )

    print(
        "Samples           : "
        f"{len(target_rows)}"
    )

    print(
        "CPU workers       : "
        f"{max_workers}"
    )

    results: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    completed = 0

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_identity = {
            executor.submit(
                inspect_sample,
                row,
                target_root,
                geometry_config,
                lesion_size_by_id[
                    normalize_image_id(
                        row[
                            "image_id"
                        ]
                    )
                ],
            ): (
                str(
                    row[
                        "split"
                    ]
                ),
                normalize_image_id(
                    row[
                        "image_id"
                    ]
                ),
            )
            for row in target_rows
        }

        for future in as_completed(
            future_to_identity
        ):
            (
                split,
                image_id,
            ) = future_to_identity[
                future
            ]

            try:
                results.append(
                    future.result()
                )

            except Exception as error:
                failures.append(
                    {
                        "split": split,
                        "image_id": (
                            image_id
                        ),
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

            completed += 1

            if (
                completed % 250 == 0
                or completed
                == len(
                    target_rows
                )
            ):
                print(
                    "  numerical QC: "
                    f"{completed}/"
                    f"{len(target_rows)}"
                )

    report_directory = Path(
        config.REPORTS_DIR
    )

    manifest_directory = Path(
        config.MANIFEST_DIR
    )

    figure_directory = (
        report_directory.parent
        / "figures"
        / "step05b_target_qc"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if failures:
        failure_path = (
            report_directory
            / (
                "step05b_target_"
                "qc_failures.json"
            )
        )

        failure_path.write_text(
            json.dumps(
                failures,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Target QC failed to inspect "
            "one or more samples. "
            f"See {failure_path}."
        )

    split_order = {
        "train": 0,
        "val": 1,
        "internal_test": 2,
    }

    results.sort(
        key=lambda row: (
            split_order[
                str(
                    row[
                        "split"
                    ]
                )
            ],
            str(
                row[
                    "image_id"
                ]
            ),
        )
    )

    failed_result_rows = [
        row
        for row in results
        if not bool(
            row[
                "sample_passed"
            ]
        )
    ]

    numerical_qc_path = (
        manifest_directory
        / (
            "step05b_target_"
            "numerical_qc.csv"
        )
    )

    numerical_fields = [
        "image_id",
        "split",
        "lesion_size_group",
        "source_image_path",
        "source_mask_path",
        "source_mask_width",
        "source_mask_height",
        "target_mask_path",
        "target_contour_path",
        "target_boundary_band_path",
        "target_sdm_path",
        "mask_difference_pixels",
        "contour_difference_pixels",
        "boundary_band_difference_pixels",
        "sdm_max_absolute_difference",
        "sdm_mean_absolute_difference",
        "border_touching",
        "contour_inside_boundary_band",
        "sdm_inside_positive",
        "sdm_outside_negative",
        "target_hashes_match_manifest",
        "sample_passed",
        "failed_checks",
        "target_protocol_version",
        "target_config_sha256",
        "height",
        "width",
        "mask_foreground_pixels",
        "mask_foreground_ratio",
        "mask_is_full_foreground",
        "connected_component_count",
        "contour_pixels",
        "contour_ratio",
        "boundary_band_pixels",
        "boundary_band_ratio",
        "sdm_min",
        "sdm_max",
        "sdm_mean",
        "sdm_inside_mean",
        "sdm_outside_mean",
    ]

    write_csv(
        numerical_qc_path,
        results,
        numerical_fields,
    )

    selected_reasons = (
        select_visual_samples(
            results
        )
    )

    result_by_id = {
        str(
            row[
                "image_id"
            ]
        ): row
        for row in results
    }

    selected_rows = sorted(
        (
            (
                result_by_id[
                    image_id
                ],
                sorted(
                    reasons
                ),
            )
            for (
                image_id,
                reasons,
            ) in selected_reasons.items()
        ),
        key=lambda item: (
            split_order[
                str(
                    item[0][
                        "split"
                    ]
                )
            ],
            str(
                item[0][
                    "lesion_size_group"
                ]
            ),
            str(
                item[0][
                    "image_id"
                ]
            ),
        ),
    )

    contact_sheet_index = (
        create_contact_sheets(
            selected_rows,
            figure_directory,
        )
    )

    visual_sample_path = (
        manifest_directory
        / (
            "step05b_visual_"
            "qc_samples.csv"
        )
    )

    write_csv(
        visual_sample_path,
        contact_sheet_index,
        [
            "visual_rank",
            "contact_sheet",
            "panel_position",
            "image_id",
            "split",
            "lesion_size_group",
            "selection_reasons",
        ],
    )

    observed_split_counts = dict(
        Counter(
            str(
                row[
                    "split"
                ]
            )
            for row in results
        )
    )

    full_foreground_ids = sorted(
        str(
            row[
                "image_id"
            ]
        )
        for row in results
        if bool(
            row[
                "mask_is_full_foreground"
            ]
        )
    )

    border_touching_count = sum(
        bool(
            row[
                "border_touching"
            ]
        )
        for row in results
    )

    multi_component_count = sum(
        int(
            row[
                "connected_component_count"
            ]
        )
        > 1
        for row in results
    )

    visual_strata = {
        (
            str(
                row[
                    "split"
                ]
            ),
            str(
                row[
                    "lesion_size_group"
                ]
            ),
        )
        for (
            row,
            _,
        ) in selected_rows
    }

    expected_visual_strata = {
        (
            split,
            size_group,
        )
        for split in [
            "train",
            "val",
            "internal_test",
        ]
        for size_group in [
            "small",
            "medium",
            "large",
        ]
    }

    contact_sheet_files = sorted(
        figure_directory.glob(
            "target_qc_sheet_*.jpg"
        )
    )

    checks = {
        "all_samples_inspected": (
            len(
                results
            )
            == EXPECTED_TOTAL_IMAGES
        ),
        "split_counts_correct": (
            observed_split_counts
            == EXPECTED_SPLIT_COUNTS
        ),
        "all_samples_passed_numerical_qc": (
            not failed_result_rows
        ),
        "all_saved_hashes_match_manifest": all(
            bool(
                row[
                    "target_hashes_match_manifest"
                ]
            )
            for row in results
        ),
        "all_masks_reproduce_exactly": all(
            int(
                row[
                    "mask_difference_pixels"
                ]
            )
            == 0
            for row in results
        ),
        "all_contours_reproduce_exactly": all(
            int(
                row[
                    "contour_difference_pixels"
                ]
            )
            == 0
            for row in results
        ),
        "all_boundary_bands_reproduce_exactly": all(
            int(
                row[
                    "boundary_band_difference_pixels"
                ]
            )
            == 0
            for row in results
        ),
        "all_sdms_reproduce_within_tolerance": all(
            float(
                row[
                    "sdm_max_absolute_difference"
                ]
            )
            <= 1e-7
            for row in results
        ),
        "full_foreground_case_preserved": (
            full_foreground_ids
            == [
                EXPECTED_FULL_FOREGROUND_ID
            ]
        ),
        "all_visual_strata_covered": (
            expected_visual_strata
            <= visual_strata
        ),
        "visual_samples_generated": (
            len(
                contact_sheet_index
            )
            == len(
                selected_rows
            )
            and bool(
                contact_sheet_files
            )
        ),
    }

    automatic_qc_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "05B_derived_target_"
            "quality_control"
        ),
        "qc_protocol_version": (
            QC_PROTOCOL_VERSION
        ),
        "step05a_artifact_root": str(
            artifact_root
        ),
        "step04_split_directory": str(
            split_directory
        ),
        "target_manifest": {
            "path": str(
                target_manifest_path
            ),
            "sha256": sha256_file(
                target_manifest_path
            ),
        },
        "target_configuration": (
            geometry_config.to_dict()
        ),
        "target_configuration_sha256": (
            geometry_config.fingerprint()
        ),
        "counts": {
            "samples_inspected": len(
                results
            ),
            "split_counts": (
                observed_split_counts
            ),
            "numerical_failures": len(
                failed_result_rows
            ),
            "full_foreground_mask_count": len(
                full_foreground_ids
            ),
            "full_foreground_mask_ids": (
                full_foreground_ids
            ),
            "border_touching_mask_count": (
                border_touching_count
            ),
            "multi_component_mask_count": (
                multi_component_count
            ),
            "visual_samples": len(
                selected_rows
            ),
            "contact_sheet_files": len(
                contact_sheet_files
            ),
        },
        "reproduction_tolerances": {
            "binary_target_difference_pixels": 0,
            "sdm_max_absolute_difference": 1e-7,
        },
        "statistics": {
            "mask_foreground_ratio": (
                numeric_summary(
                    [
                        float(
                            row[
                                "mask_foreground_ratio"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "contour_ratio": (
                numeric_summary(
                    [
                        float(
                            row[
                                "contour_ratio"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "boundary_band_ratio": (
                numeric_summary(
                    [
                        float(
                            row[
                                "boundary_band_ratio"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "connected_component_count": (
                numeric_summary(
                    [
                        int(
                            row[
                                "connected_component_count"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "sdm_min": (
                numeric_summary(
                    [
                        float(
                            row[
                                "sdm_min"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "sdm_max": (
                numeric_summary(
                    [
                        float(
                            row[
                                "sdm_max"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            (
                "sdm_reproduction_"
                "max_absolute_difference"
            ): numeric_summary(
                [
                    float(
                        row[
                            "sdm_max_absolute_difference"
                        ]
                    )
                    for row in results
                ]
            ),
        },
        "visual_sampling": {
            "method": (
                "Deterministic split-by-lesion-size "
                "stratification plus extreme and "
                "special-case enrichment."
            ),
            "samples_per_split_size_stratum": (
                VISUAL_SAMPLES_PER_STRATUM
            ),
            "additional_cases": [
                "three globally smallest masks",
                "three globally largest masks",
                "four highest component counts",
                "four largest border-touching masks",
                "all full-foreground masks",
            ],
            "visual_review_status": (
                "pending_manual_review"
            ),
        },
        "checks": checks,
        "automatic_qc_passed": (
            automatic_qc_passed
        ),
        "outputs": {
            "numerical_qc_csv": str(
                numerical_qc_path
            ),
            "visual_sample_index": str(
                visual_sample_path
            ),
            "contact_sheet_directory": str(
                figure_directory
            ),
        },
        "training_allowed": False,
        "training_block_reason": (
            "Automated numerical QC passed, "
            "but the generated target contact "
            "sheets still require visual review "
            "and Step 05C sign-off."
        ),
    }

    report_path = (
        report_directory
        / (
            "step05b_target_"
            "qc_report.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Step 05B Results ==="
    )

    print(
        "Numerically inspected images      : "
        f"{len(results)}"
    )

    print(
        "Numerical QC failures             : "
        f"{len(failed_result_rows)}"
    )

    print(
        "Full-foreground masks             : "
        f"{len(full_foreground_ids)}"
    )

    print(
        "Border-touching masks             : "
        f"{border_touching_count}"
    )

    print(
        "Multi-component masks             : "
        f"{multi_component_count}"
    )

    print(
        "Visual-review samples             : "
        f"{len(selected_rows)}"
    )

    print(
        "Contact sheets generated          : "
        f"{len(contact_sheet_files)}"
    )

    print(
        "Automatic validation checks passed: "
        f"{automatic_qc_passed}"
    )

    print("\nOutputs:")
    print(
        f" - {numerical_qc_path}"
    )
    print(
        f" - {visual_sample_path}"
    )
    print(
        f" - {figure_directory}"
    )
    print(
        f" - {report_path}"
    )

    print(
        "\nTraining remains blocked until "
        "the visual QC sheets receive "
        "final sign-off."
    )

    return (
        0
        if automatic_qc_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())