"""Step 05A: Generate fixed-resolution ISIC 2018 supervision targets.

This CPU-only stage reads the persistent Step 04 locked split manifests and
generates four targets for every ISIC 2018 image:

1. resized binary segmentation mask;
2. controlled-width contour target;
3. dilated boundary-band target;
4. normalized signed-distance map.

The stage is resumable. Existing targets are loaded and validated before they
are reused. A configuration mismatch causes a hard failure rather than mixing
targets generated with different geometry settings.

Persistent inputs are read-only. Outputs are written under /kaggle/working.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.data.analyze_overlap import (
    normalize_image_id,
    read_csv_rows,
)
from src.data.split_dataset import resolve_mask_path
from src.targets.target_geometry import (
    TARGET_PROTOCOL_VERSION,
    TargetGeometryConfig,
    generate_target_bundle,
    save_binary_png,
    save_float32_npy,
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

GENERATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-target-generation-v1"
)


def find_step04_split_directory() -> Path:
    """Locate the persistent Step 04 split directory."""

    input_root = Path("/kaggle/input")
    candidates: list[Path] = []

    for train_path in input_root.rglob(
        "isic2018_train.csv"
    ):
        directory = train_path.parent

        required_files = [
            directory / "isic2018_val.csv",
            directory / "isic2018_internal_test.csv",
            directory / "isic2018_all_locked.csv",
        ]

        if all(
            path.is_file()
            for path in required_files
        ):
            candidates.append(
                directory.resolve()
            )

    candidates = sorted(
        set(candidates)
    )

    preferred = [
        candidate
        for candidate in candidates
        if (
            "bcs-hctnet-step04-split-artifacts"
            in str(candidate).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not locate the persistent "
            "Step 04 split directory."
        )

    raise RuntimeError(
        "Multiple Step 04 split directories "
        "were found: "
        f"{[str(path) for path in candidates]}"
    )


def environment_flag(
    name: str,
    default: bool = False,
) -> bool:
    """Read a Boolean environment flag."""

    raw_value = os.environ.get(
        name,
        "",
    ).strip().lower()

    if not raw_value:
        return default

    if raw_value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if raw_value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"{name} must be a Boolean value, "
        f"received {raw_value!r}."
    )


def load_target_config() -> TargetGeometryConfig:
    """Load configurable target geometry values."""

    active_config = TargetGeometryConfig(
        output_size=int(
            os.environ.get(
                "BCS_TARGET_SIZE",
                "352",
            )
        ),
        contour_width=int(
            os.environ.get(
                "BCS_CONTOUR_WIDTH",
                "2",
            )
        ),
        boundary_band_radius=int(
            os.environ.get(
                "BCS_BOUNDARY_BAND_RADIUS",
                "3",
            )
        ),
        sdm_clip_distance=float(
            os.environ.get(
                "BCS_SDM_CLIP_DISTANCE",
                "20",
            )
        ),
        connectivity=int(
            os.environ.get(
                "BCS_TARGET_CONNECTIVITY",
                "8",
            )
        ),
    )

    active_config.validate()

    return active_config


def read_locked_splits(
    split_directory: Path,
) -> dict[str, list[dict[str, str]]]:
    """Read and validate the three official splits."""

    file_mapping = {
        "train": (
            split_directory
            / "isic2018_train.csv"
        ),
        "val": (
            split_directory
            / "isic2018_val.csv"
        ),
        "internal_test": (
            split_directory
            / "isic2018_internal_test.csv"
        ),
    }

    split_rows: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for split, file_path in file_mapping.items():
        rows = read_csv_rows(
            file_path
        )

        if len(rows) != EXPECTED_SPLIT_COUNTS[
            split
        ]:
            raise RuntimeError(
                f"Unexpected {split} count: "
                f"expected "
                f"{EXPECTED_SPLIT_COUNTS[split]}, "
                f"found {len(rows)}."
            )

        normalized_rows: list[
            dict[str, str]
        ] = []

        for source_row in rows:
            row = dict(
                source_row
            )

            image_id = normalize_image_id(
                row["image_id"]
            )

            row["image_id"] = image_id
            row["split"] = split

            normalized_rows.append(
                row
            )

        normalized_rows.sort(
            key=lambda row: row["image_id"]
        )

        split_rows[split] = (
            normalized_rows
        )

    all_ids = [
        row["image_id"]
        for split in [
            "train",
            "val",
            "internal_test",
        ]
        for row in split_rows[split]
    ]

    if len(all_ids) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Unexpected total ISIC 2018 "
            f"image count: {len(all_ids)}."
        )

    duplicate_ids = sorted(
        image_id
        for image_id, count
        in Counter(all_ids).items()
        if count > 1
    )

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate image IDs occur across "
            f"locked splits: {duplicate_ids[:20]}"
        )

    return split_rows


def target_paths(
    target_root: Path,
    split: str,
    image_id: str,
) -> dict[str, Path]:
    """Return deterministic output paths."""

    return {
        "mask": (
            target_root
            / "mask"
            / split
            / f"{image_id}.png"
        ),
        "contour": (
            target_root
            / "contour"
            / split
            / f"{image_id}.png"
        ),
        "boundary_band": (
            target_root
            / "boundary_band"
            / split
            / f"{image_id}.png"
        ),
        "sdm": (
            target_root
            / "sdm"
            / split
            / f"{image_id}.npy"
        ),
    }


def load_existing_bundle(
    paths: dict[str, Path],
    geometry_config: TargetGeometryConfig,
) -> dict[str, np.ndarray]:
    """Load and validate an existing target bundle."""

    missing = [
        name
        for name, path
        in paths.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Incomplete existing target bundle. "
            f"Missing: {missing}"
        )

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
            image = opened_image.convert(
                "L"
            )

            image.load()

            binary_arrays[name] = (
                np.asarray(
                    image,
                    dtype=np.uint8,
                )
            )

    sdm = np.load(
        paths["sdm"],
        allow_pickle=False,
    )

    if sdm.dtype != np.float32:
        raise ValueError(
            "Existing SDM must be float32, "
            f"received {sdm.dtype}."
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


def save_target_bundle(
    bundle: dict[str, np.ndarray],
    paths: dict[str, Path],
) -> None:
    """Save a generated target bundle."""

    save_binary_png(
        bundle["mask"],
        paths["mask"],
    )

    save_binary_png(
        bundle["contour"],
        paths["contour"],
    )

    save_binary_png(
        bundle["boundary_band"],
        paths["boundary_band"],
    )

    save_float32_npy(
        bundle["sdm"],
        paths["sdm"],
    )


def prepare_target_root(
    target_root: Path,
    geometry_config: TargetGeometryConfig,
    force: bool,
) -> Path:
    """Create and lock the target configuration."""

    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        target_root
        / "TARGET_CONFIG.json"
    )

    expected_payload = {
        "generation_protocol_version": (
            GENERATION_PROTOCOL_VERSION
        ),
        "target_protocol_version": (
            TARGET_PROTOCOL_VERSION
        ),
        "target_geometry": (
            geometry_config.to_dict()
        ),
        "target_geometry_sha256": (
            geometry_config.fingerprint()
        ),
    }

    if config_path.is_file():
        existing_payload = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            existing_payload
            != expected_payload
        ):
            if not force:
                raise RuntimeError(
                    "Existing target directory uses "
                    "a different configuration. "
                    "Set BCS_TARGET_FORCE=true only "
                    "after intentionally removing or "
                    "replacing the old target outputs."
                )

            config_path.unlink()

    config_path.write_text(
        json.dumps(
            expected_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return config_path


def process_sample(
    row: dict[str, str],
    target_root: Path,
    geometry_config: TargetGeometryConfig,
    force: bool,
) -> dict[str, Any]:
    """Generate or validate one target bundle."""

    image_id = normalize_image_id(
        row["image_id"]
    )

    split = row["split"]

    source_mask_path = (
        resolve_mask_path(
            row
        )
    )

    paths = target_paths(
        target_root,
        split,
        image_id,
    )

    all_outputs_exist = all(
        path.is_file()
        for path in paths.values()
    )

    if (
        all_outputs_exist
        and not force
    ):
        bundle = load_existing_bundle(
            paths,
            geometry_config,
        )

        generation_action = (
            "reused_validated_existing"
        )

    else:
        with Image.open(
            source_mask_path
        ) as opened_mask:
            source_mask = (
                opened_mask
                .convert("L")
                .copy()
            )

        bundle = generate_target_bundle(
            source_mask,
            geometry_config,
        )

        save_target_bundle(
            bundle,
            paths,
        )

        bundle = load_existing_bundle(
            paths,
            geometry_config,
        )

        generation_action = (
            "generated"
        )

    statistics = target_statistics(
        bundle,
        geometry_config,
    )

    return {
        "image_id": image_id,
        "split": split,
        "source_image_path": row.get(
            "image_path",
            "",
        ),
        "source_image_relative_path": row.get(
            "image_relative_path",
            "",
        ),
        "source_mask_path": str(
            source_mask_path
        ),
        "source_mask_relative_path": row.get(
            "mask_relative_path",
            "",
        ),
        "source_decoded_binary_mask_sha256": (
            row.get(
                "decoded_binary_mask_sha256",
                "",
            )
        ),
        "source_mask_foreground_ratio": (
            row.get(
                "mask_foreground_ratio",
                "",
            )
        ),
        "source_mask_is_full_foreground": (
            row.get(
                "mask_is_full_foreground",
                "",
            )
        ),
        "target_mask_path": str(
            paths["mask"]
        ),
        "target_contour_path": str(
            paths["contour"]
        ),
        "target_boundary_band_path": str(
            paths["boundary_band"]
        ),
        "target_sdm_path": str(
            paths["sdm"]
        ),
        "target_mask_relative_path": (
            paths["mask"]
            .relative_to(
                target_root
            )
            .as_posix()
        ),
        "target_contour_relative_path": (
            paths["contour"]
            .relative_to(
                target_root
            )
            .as_posix()
        ),
        "target_boundary_band_relative_path": (
            paths["boundary_band"]
            .relative_to(
                target_root
            )
            .as_posix()
        ),
        "target_sdm_relative_path": (
            paths["sdm"]
            .relative_to(
                target_root
            )
            .as_posix()
        ),
        "target_mask_sha256": (
            sha256_file(
                paths["mask"]
            )
        ),
        "target_contour_sha256": (
            sha256_file(
                paths["contour"]
            )
        ),
        "target_boundary_band_sha256": (
            sha256_file(
                paths[
                    "boundary_band"
                ]
            )
        ),
        "target_sdm_sha256": (
            sha256_file(
                paths["sdm"]
            )
        ),
        "generation_action": (
            generation_action
        ),
        **statistics,
    }


def write_csv(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV data."""

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
            np.min(array)
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
            np.max(array)
        ),
        "mean": float(
            np.mean(array)
        ),
    }


def directory_size_bytes(
    directory: Path,
) -> int:
    """Calculate the size of generated files."""

    return sum(
        file_path.stat().st_size
        for file_path in directory.rglob("*")
        if file_path.is_file()
    )


def main() -> int:
    """Generate all ISIC 2018 supervision targets."""

    config.ensure_all_dirs()

    geometry_config = (
        load_target_config()
    )

    force = environment_flag(
        "BCS_TARGET_FORCE",
        default=False,
    )

    configured_workers = int(
        os.environ.get(
            "BCS_TARGET_WORKERS",
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

    split_directory = (
        find_step04_split_directory()
    )

    split_rows = read_locked_splits(
        split_directory
    )

    data_root = (
        Path(
            config.MANIFEST_DIR
        ).parent
    )

    target_root = (
        data_root
        / "targets"
        / (
            f"isic2018_"
            f"{geometry_config.output_size}"
        )
    )

    manifest_directory = Path(
        config.MANIFEST_DIR
    )

    report_directory = Path(
        config.REPORTS_DIR
    )

    manifest_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_config_path = (
        prepare_target_root(
            target_root,
            geometry_config,
            force,
        )
    )

    all_rows = [
        row
        for split in [
            "train",
            "val",
            "internal_test",
        ]
        for row in split_rows[split]
    ]

    print(
        "=== Step 05A: Generate "
        "ISIC 2018 Derived Targets ==="
    )

    print(
        "Persistent split directory: "
        f"{split_directory}"
    )

    print(
        "Target root              : "
        f"{target_root}"
    )

    print(
        "Target resolution        : "
        f"{geometry_config.output_size}"
        " x "
        f"{geometry_config.output_size}"
    )

    print(
        "Contour width            : "
        f"{geometry_config.contour_width}"
    )

    print(
        "Boundary-band radius     : "
        f"{geometry_config.boundary_band_radius}"
    )

    print(
        "SDM clipping distance    : "
        f"{geometry_config.sdm_clip_distance}"
    )

    print(
        "CPU workers              : "
        f"{max_workers}"
    )

    print(
        "Force regeneration       : "
        f"{force}"
    )

    results: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    completed = 0
    total = len(
        all_rows
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_sample = {
            executor.submit(
                process_sample,
                row,
                target_root,
                geometry_config,
                force,
            ): (
                row["split"],
                row["image_id"],
            )
            for row in all_rows
        }

        for future in as_completed(
            future_to_sample
        ):
            split, image_id = (
                future_to_sample[
                    future
                ]
            )

            try:
                result = future.result()

                results.append(
                    result
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
                or completed == total
            ):
                print(
                    "  target generation: "
                    f"{completed}/{total}"
                )

    if failures:
        failure_path = (
            report_directory
            / (
                "step05_target_"
                "generation_failures.json"
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
            "Target generation failed for "
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
                    row["split"]
                )
            ],
            str(
                row["image_id"]
            ),
        )
    )

    manifest_fieldnames = [
        "image_id",
        "split",
        "source_image_path",
        "source_image_relative_path",
        "source_mask_path",
        "source_mask_relative_path",
        "source_decoded_binary_mask_sha256",
        "source_mask_foreground_ratio",
        "source_mask_is_full_foreground",
        "target_mask_path",
        "target_contour_path",
        "target_boundary_band_path",
        "target_sdm_path",
        "target_mask_relative_path",
        "target_contour_relative_path",
        "target_boundary_band_relative_path",
        "target_sdm_relative_path",
        "target_mask_sha256",
        "target_contour_sha256",
        "target_boundary_band_sha256",
        "target_sdm_sha256",
        "generation_action",
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

    manifest_path = (
        manifest_directory
        / (
            "isic2018_derived_"
            "targets_352.csv"
        )
    )

    write_csv(
        manifest_path,
        results,
        manifest_fieldnames,
    )

    observed_split_counts = dict(
        Counter(
            str(row["split"])
            for row in results
        )
    )

    action_counts = dict(
        Counter(
            str(
                row[
                    "generation_action"
                ]
            )
            for row in results
        )
    )

    full_foreground_ids = sorted(
        str(row["image_id"])
        for row in results
        if bool(
            row[
                "mask_is_full_foreground"
            ]
        )
    )

    checks = {
        "all_samples_processed": (
            len(results)
            == EXPECTED_TOTAL_IMAGES
        ),
        "split_counts_correct": (
            observed_split_counts
            == EXPECTED_SPLIT_COUNTS
        ),
        "all_target_files_exist": all(
            Path(
                str(
                    row[path_field]
                )
            ).is_file()
            for row in results
            for path_field in [
                "target_mask_path",
                "target_contour_path",
                "target_boundary_band_path",
                "target_sdm_path",
            ]
        ),
        "all_target_shapes_correct": all(
            int(row["height"])
            == geometry_config.output_size
            and int(row["width"])
            == geometry_config.output_size
            for row in results
        ),
        "all_contours_nonempty": all(
            int(
                row[
                    "contour_pixels"
                ]
            )
            > 0
            for row in results
        ),
        "all_boundary_bands_nonempty": all(
            int(
                row[
                    "boundary_band_pixels"
                ]
            )
            > 0
            for row in results
        ),
        "all_sdm_values_bounded": all(
            float(row["sdm_min"])
            >= -1.0
            and float(
                row["sdm_max"]
            )
            <= 1.0
            for row in results
        ),
        "full_foreground_case_preserved": (
            full_foreground_ids
            == [
                "ISIC_0023056"
            ]
        ),
        "configuration_file_exists": (
            target_config_path.is_file()
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "05A_isic2018_"
            "target_generation"
        ),
        "generation_protocol_version": (
            GENERATION_PROTOCOL_VERSION
        ),
        "target_protocol_version": (
            TARGET_PROTOCOL_VERSION
        ),
        "persistent_split_directory": str(
            split_directory
        ),
        "target_root": str(
            target_root
        ),
        "target_configuration": (
            geometry_config.to_dict()
        ),
        "target_configuration_sha256": (
            geometry_config.fingerprint()
        ),
        "counts": {
            "total_images": len(
                results
            ),
            "split_counts": (
                observed_split_counts
            ),
            "generation_actions": (
                action_counts
            ),
            "full_foreground_mask_count": (
                len(
                    full_foreground_ids
                )
            ),
            "full_foreground_mask_ids": (
                full_foreground_ids
            ),
            "target_files": (
                len(results) * 4
            ),
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
            "contour_pixels": (
                numeric_summary(
                    [
                        int(
                            row[
                                "contour_pixels"
                            ]
                        )
                        for row in results
                    ]
                )
            ),
            "boundary_band_pixels": (
                numeric_summary(
                    [
                        int(
                            row[
                                "boundary_band_pixels"
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
        },
        "storage": {
            "target_directory_bytes": (
                directory_size_bytes(
                    target_root
                )
            ),
            "target_file_count": (
                sum(
                    1
                    for path
                    in target_root.rglob("*")
                    if path.is_file()
                )
            ),
            "binary_targets_format": (
                "8-bit PNG with values 0 and 255"
            ),
            "sdm_format": (
                "NumPy NPY float32"
            ),
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            "target_root": str(
                target_root
            ),
            "target_config": str(
                target_config_path
            ),
            "target_manifest": str(
                manifest_path
            ),
        },
        "training_allowed": False,
        "training_block_reason": (
            "Numerical and visual derived-target "
            "quality control remains incomplete."
        ),
    }

    report_path = (
        report_directory
        / (
            "step05_target_"
            "generation_report.json"
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
        "\n=== Step 05A Results ==="
    )

    print(
        "Generated/validated images     : "
        f"{len(results)}"
    )

    print(
        "Train targets                  : "
        f"{observed_split_counts.get('train', 0)}"
    )

    print(
        "Validation targets             : "
        f"{observed_split_counts.get('val', 0)}"
    )

    print(
        "Internal-test targets          : "
        f"{observed_split_counts.get('internal_test', 0)}"
    )

    print(
        "Full-foreground masks retained : "
        f"{len(full_foreground_ids)}"
    )

    print(
        "Generated target files         : "
        f"{len(results) * 4}"
    )

    print(
        "Target directory size GB       : "
        f"{directory_size_bytes(target_root) / 1024**3:.3f}"
    )

    print(
        "All validation checks passed   : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {target_root}")
    print(f" - {target_config_path}")
    print(f" - {manifest_path}")
    print(f" - {report_path}")

    print(
        "\nNo persistent input artifact "
        "was modified."
    )

    print(
        "Training remains blocked until "
        "target quality control passes."
    )

    return (
        0
        if all_checks_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())