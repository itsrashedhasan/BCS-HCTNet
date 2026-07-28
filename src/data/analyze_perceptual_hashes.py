"""Step 03C: Calibrate perceptual hashes and screen IMA++ candidates.

This stage is non-destructive and CPU-only.

It uses the 3,568 known same-ID ISIC 2018–IMA++ pairs as positive controls.
It then finds the closest ISIC 2018 perceptual-hash neighbors for every
remaining IMA++ candidate.

No image is excluded in this stage. The output is used to select defensible
review thresholds instead of guessing them.

Persistent inputs:
    Step 02 manifests under /kaggle/input

Temporary outputs:
    /kaggle/working/data/manifests/
    /kaggle/working/outputs/reports/
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.data.analyze_exact_pixels import (
    resolve_image_path,
    select_unique_image_rows,
)
from src.data.analyze_overlap import (
    EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES,
    EXPECTED_ISIC2018_UNIQUE_IMAGES,
    find_step02_manifest_directory,
    normalize_image_id,
    read_csv_rows,
    sha256_file,
)
from src.utils import config


EXPECTED_EXACT_ID_OVERLAP = 3568
EXPECTED_NONOVERLAP_CANDIDATES = 11399

POPCOUNT_LOOKUP = np.array(
    [bin(value).count("1") for value in range(256)],
    dtype=np.uint8,
)


def bits_to_uint64(bits: np.ndarray) -> int:
    """Pack exactly 64 Boolean values into an unsigned integer."""

    flattened = np.asarray(
        bits,
        dtype=np.uint8,
    ).reshape(-1)

    if flattened.size != 64:
        raise ValueError(
            f"Expected 64 bits, received {flattened.size}."
        )

    result = 0

    for bit in flattened:
        result = (result << 1) | int(bit)

    return result


def phash64(grayscale: np.ndarray) -> int:
    """Calculate a 64-bit DCT perceptual hash."""

    resized = cv2.resize(
        grayscale,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    coefficients = cv2.dct(resized)
    low_frequencies = coefficients[:8, :8]

    flattened = low_frequencies.reshape(-1)

    median = float(
        np.median(flattened[1:])
    )

    bits = low_frequencies > median

    return bits_to_uint64(bits)


def dhash64(grayscale: np.ndarray) -> int:
    """Calculate a 64-bit horizontal difference hash."""

    resized = cv2.resize(
        grayscale,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )

    bits = resized[:, 1:] > resized[:, :-1]

    return bits_to_uint64(bits)


def ahash64(grayscale: np.ndarray) -> int:
    """Calculate a 64-bit average hash."""

    resized = cv2.resize(
        grayscale,
        (8, 8),
        interpolation=cv2.INTER_AREA,
    )

    bits = resized > float(resized.mean())

    return bits_to_uint64(bits)


def hash_image(
    image_id: str,
    row: dict[str, str],
    dataset: str,
) -> tuple[str, dict[str, object]]:
    """Decode one image and calculate three perceptual hashes."""

    image_path = resolve_image_path(row)

    with Image.open(image_path) as opened_image:
        oriented = ImageOps.exif_transpose(
            opened_image
        )

        rgb_image = oriented.convert("RGB")
        rgb_image.load()

        width, height = rgb_image.size

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.uint8,
        )

    grayscale = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2GRAY,
    )

    return image_id, {
        "dataset": dataset,
        "image_id": image_id,
        "image_path": str(image_path),
        "image_relative_path": row.get(
            "image_relative_path",
            "",
        ),
        "width": width,
        "height": height,
        "phash": phash64(grayscale),
        "dhash": dhash64(grayscale),
        "ahash": ahash64(grayscale),
    }


def hash_dataset(
    dataset: str,
    rows_by_id: dict[str, dict[str, str]],
    max_workers: int,
) -> dict[str, dict[str, object]]:
    """Calculate perceptual hashes with bounded CPU threading."""

    results: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []

    total = len(rows_by_id)
    completed = 0

    print(
        f"Hashing {dataset}: {total} images "
        f"with {max_workers} workers"
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_id = {
            executor.submit(
                hash_image,
                image_id,
                row,
                dataset,
            ): image_id
            for image_id, row in rows_by_id.items()
        }

        for future in as_completed(future_to_id):
            image_id = future_to_id[future]

            try:
                result_id, result = future.result()
                results[result_id] = result

            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "image_id": image_id,
                        "error": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )

            completed += 1

            if (
                completed % 500 == 0
                or completed == total
            ):
                print(
                    f"  {dataset}: "
                    f"{completed}/{total}"
                )

    if failures:
        failure_path = (
            Path(config.REPORTS_DIR)
            / "perceptual_hash_failures.json"
        )

        failure_path.write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Perceptual hashing failed. "
            f"See {failure_path}."
        )

    return results


def hamming_distance(
    first: int,
    second: int,
) -> int:
    """Return the Hamming distance between two 64-bit hashes."""

    return int(
        (int(first) ^ int(second)).bit_count()
    )


def hamming_matrix(
    query_values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    """Calculate a chunked query-by-reference Hamming matrix."""

    xor_values = np.bitwise_xor(
        query_values[:, None],
        reference_values[None, :],
    )

    byte_view = xor_values.view(np.uint8).reshape(
        query_values.shape[0],
        reference_values.shape[0],
        8,
    )

    return POPCOUNT_LOOKUP[
        byte_view
    ].sum(axis=2)


def nearest_neighbors(
    candidate_signatures: dict[str, dict[str, object]],
    isic_signatures: dict[str, dict[str, object]],
    chunk_size: int = 128,
) -> list[dict[str, object]]:
    """Find nearest ISIC neighbors under pHash, dHash, and combined score."""

    candidate_ids = sorted(candidate_signatures)
    isic_ids = sorted(isic_signatures)

    isic_phashes = np.asarray(
        [
            isic_signatures[image_id]["phash"]
            for image_id in isic_ids
        ],
        dtype=np.uint64,
    )

    isic_dhashes = np.asarray(
        [
            isic_signatures[image_id]["dhash"]
            for image_id in isic_ids
        ],
        dtype=np.uint64,
    )

    isic_ahashes = np.asarray(
        [
            isic_signatures[image_id]["ahash"]
            for image_id in isic_ids
        ],
        dtype=np.uint64,
    )

    results: list[dict[str, object]] = []

    for start in range(
        0,
        len(candidate_ids),
        chunk_size,
    ):
        chunk_ids = candidate_ids[
            start : start + chunk_size
        ]

        candidate_phashes = np.asarray(
            [
                candidate_signatures[
                    image_id
                ]["phash"]
                for image_id in chunk_ids
            ],
            dtype=np.uint64,
        )

        candidate_dhashes = np.asarray(
            [
                candidate_signatures[
                    image_id
                ]["dhash"]
                for image_id in chunk_ids
            ],
            dtype=np.uint64,
        )

        candidate_ahashes = np.asarray(
            [
                candidate_signatures[
                    image_id
                ]["ahash"]
                for image_id in chunk_ids
            ],
            dtype=np.uint64,
        )

        phash_distances = hamming_matrix(
            candidate_phashes,
            isic_phashes,
        )

        dhash_distances = hamming_matrix(
            candidate_dhashes,
            isic_dhashes,
        )

        ahash_distances = hamming_matrix(
            candidate_ahashes,
            isic_ahashes,
        )

        combined_scores = (
            phash_distances.astype(np.int16) * 2
            + dhash_distances.astype(np.int16)
            + ahash_distances.astype(np.int16)
        )

        for row_index, image_id in enumerate(
            chunk_ids
        ):
            phash_index = int(
                np.argmin(
                    phash_distances[row_index]
                )
            )

            dhash_index = int(
                np.argmin(
                    dhash_distances[row_index]
                )
            )

            combined_index = int(
                np.argmin(
                    combined_scores[row_index]
                )
            )

            results.append(
                {
                    "imaplusplus_image_id": image_id,
                    "nearest_phash_isic2018_id": (
                        isic_ids[phash_index]
                    ),
                    "nearest_phash_distance": int(
                        phash_distances[
                            row_index,
                            phash_index,
                        ]
                    ),
                    "nearest_dhash_isic2018_id": (
                        isic_ids[dhash_index]
                    ),
                    "nearest_dhash_distance": int(
                        dhash_distances[
                            row_index,
                            dhash_index,
                        ]
                    ),
                    "nearest_combined_isic2018_id": (
                        isic_ids[combined_index]
                    ),
                    "combined_pair_phash_distance": int(
                        phash_distances[
                            row_index,
                            combined_index,
                        ]
                    ),
                    "combined_pair_dhash_distance": int(
                        dhash_distances[
                            row_index,
                            combined_index,
                        ]
                    ),
                    "combined_pair_ahash_distance": int(
                        ahash_distances[
                            row_index,
                            combined_index,
                        ]
                    ),
                    "combined_score": int(
                        combined_scores[
                            row_index,
                            combined_index,
                        ]
                    ),
                    "imaplusplus_image_path": (
                        candidate_signatures[
                            image_id
                        ]["image_path"]
                    ),
                    "matching_isic2018_image_path": (
                        isic_signatures[
                            isic_ids[combined_index]
                        ]["image_path"]
                    ),
                }
            )

        print(
            "  nearest-neighbor search: "
            f"{min(start + chunk_size, len(candidate_ids))}"
            f"/{len(candidate_ids)}"
        )

    return results


def distribution_summary(
    values: Sequence[int],
) -> dict[str, float | int]:
    """Return reproducible descriptive statistics."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        return {}

    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def threshold_counts(
    values: Sequence[int],
) -> dict[str, int]:
    """Count observations at several non-decision thresholds."""

    thresholds = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        16,
    ]

    return {
        f"distance_le_{threshold}": sum(
            value <= threshold
            for value in values
        )
        for threshold in thresholds
    }


def write_csv(
    output_path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Write rows as UTF-8 CSV."""

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
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run perceptual-hash calibration and candidate screening."""

    config.ensure_all_dirs()

    manifest_directory = (
        find_step02_manifest_directory()
    )

    isic_manifest_path = (
        manifest_directory
        / "isic2018_all.csv"
    )

    ima_manifest_path = (
        manifest_directory
        / "imaplusplus_all.csv"
    )

    print(
        "=== Step 03C: Perceptual Hash "
        "Calibration and Screening ==="
    )

    print(
        "Persistent manifest directory: "
        f"{manifest_directory}"
    )

    isic_rows = read_csv_rows(
        isic_manifest_path
    )

    ima_rows = read_csv_rows(
        ima_manifest_path
    )

    isic_unique_rows = (
        select_unique_image_rows(isic_rows)
    )

    ima_unique_rows = (
        select_unique_image_rows(ima_rows)
    )

    if len(isic_unique_rows) != (
        EXPECTED_ISIC2018_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "Unexpected ISIC 2018 unique-image count."
        )

    if len(ima_unique_rows) != (
        EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "Unexpected IMA++ unique-image count."
        )

    isic_ids = set(isic_unique_rows)
    ima_ids = set(ima_unique_rows)

    known_overlap_ids = sorted(
        isic_ids & ima_ids
    )

    candidate_ids = sorted(
        ima_ids - isic_ids
    )

    if len(known_overlap_ids) != (
        EXPECTED_EXACT_ID_OVERLAP
    ):
        raise RuntimeError(
            "Known overlap count changed: "
            f"expected {EXPECTED_EXACT_ID_OVERLAP}, "
            f"found {len(known_overlap_ids)}."
        )

    if len(candidate_ids) != (
        EXPECTED_NONOVERLAP_CANDIDATES
    ):
        raise RuntimeError(
            "Candidate count changed: "
            f"expected {EXPECTED_NONOVERLAP_CANDIDATES}, "
            f"found {len(candidate_ids)}."
        )

    configured_workers = int(
        os.environ.get(
            "BCS_HASH_WORKERS",
            "0",
        )
    )

    max_workers = (
        configured_workers
        if configured_workers > 0
        else min(
            8,
            max(1, os.cpu_count() or 1),
        )
    )

    isic_signatures = hash_dataset(
        "isic2018",
        isic_unique_rows,
        max_workers,
    )

    ima_signatures = hash_dataset(
        "imaplusplus",
        ima_unique_rows,
        max_workers,
    )

    calibration_rows: list[
        dict[str, object]
    ] = []

    for image_id in known_overlap_ids:
        isic_signature = (
            isic_signatures[image_id]
        )

        ima_signature = (
            ima_signatures[image_id]
        )

        calibration_rows.append(
            {
                "image_id": image_id,
                "phash_distance": hamming_distance(
                    int(isic_signature["phash"]),
                    int(ima_signature["phash"]),
                ),
                "dhash_distance": hamming_distance(
                    int(isic_signature["dhash"]),
                    int(ima_signature["dhash"]),
                ),
                "ahash_distance": hamming_distance(
                    int(isic_signature["ahash"]),
                    int(ima_signature["ahash"]),
                ),
                "isic2018_image_path": (
                    isic_signature["image_path"]
                ),
                "imaplusplus_image_path": (
                    ima_signature["image_path"]
                ),
            }
        )

    candidate_signatures = {
        image_id: ima_signatures[image_id]
        for image_id in candidate_ids
    }

    neighbor_rows = nearest_neighbors(
        candidate_signatures,
        isic_signatures,
    )

    calibration_path = (
        Path(config.MANIFEST_DIR)
        / "step03c_known_overlap_hash_calibration.csv"
    )

    neighbors_path = (
        Path(config.MANIFEST_DIR)
        / "step03c_candidate_nearest_hash_neighbors.csv"
    )

    write_csv(
        calibration_path,
        calibration_rows,
        [
            "image_id",
            "phash_distance",
            "dhash_distance",
            "ahash_distance",
            "isic2018_image_path",
            "imaplusplus_image_path",
        ],
    )

    write_csv(
        neighbors_path,
        neighbor_rows,
        [
            "imaplusplus_image_id",
            "nearest_phash_isic2018_id",
            "nearest_phash_distance",
            "nearest_dhash_isic2018_id",
            "nearest_dhash_distance",
            "nearest_combined_isic2018_id",
            "combined_pair_phash_distance",
            "combined_pair_dhash_distance",
            "combined_pair_ahash_distance",
            "combined_score",
            "imaplusplus_image_path",
            "matching_isic2018_image_path",
        ],
    )

    calibration_phash = [
        int(row["phash_distance"])
        for row in calibration_rows
    ]

    calibration_dhash = [
        int(row["dhash_distance"])
        for row in calibration_rows
    ]

    calibration_ahash = [
        int(row["ahash_distance"])
        for row in calibration_rows
    ]

    candidate_phash = [
        int(row["nearest_phash_distance"])
        for row in neighbor_rows
    ]

    candidate_dhash = [
        int(row["nearest_dhash_distance"])
        for row in neighbor_rows
    ]

    candidate_combined = [
        int(row["combined_score"])
        for row in neighbor_rows
    ]

    checks = {
        "known_overlap_count_correct": (
            len(calibration_rows)
            == EXPECTED_EXACT_ID_OVERLAP
        ),
        "candidate_count_correct": (
            len(neighbor_rows)
            == EXPECTED_NONOVERLAP_CANDIDATES
        ),
        "all_isic_images_hashed": (
            len(isic_signatures)
            == EXPECTED_ISIC2018_UNIQUE_IMAGES
        ),
        "all_ima_images_hashed": (
            len(ima_signatures)
            == EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
        ),
        "no_candidate_has_same_identifier": (
            not any(
                row["imaplusplus_image_id"]
                in isic_ids
                for row in neighbor_rows
            )
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "03C_perceptual_hash_"
            "calibration_and_screening"
        ),
        "method": {
            "phash": (
                "64-bit 32x32 DCT, "
                "8x8 low-frequency block"
            ),
            "dhash": (
                "64-bit horizontal "
                "difference hash"
            ),
            "ahash": (
                "64-bit average hash"
            ),
            "combined_score": (
                "2 * pHash distance + "
                "dHash distance + "
                "aHash distance"
            ),
            "automatic_exclusion_performed": False,
        },
        "inputs": {
            "isic2018_manifest": {
                "path": str(
                    isic_manifest_path
                ),
                "sha256": sha256_file(
                    isic_manifest_path
                ),
            },
            "imaplusplus_manifest": {
                "path": str(
                    ima_manifest_path
                ),
                "sha256": sha256_file(
                    ima_manifest_path
                ),
            },
        },
        "counts": {
            "known_same_id_control_pairs": (
                len(calibration_rows)
            ),
            "nonoverlap_candidates_screened": (
                len(neighbor_rows)
            ),
        },
        "known_overlap_calibration": {
            "phash_distribution": (
                distribution_summary(
                    calibration_phash
                )
            ),
            "dhash_distribution": (
                distribution_summary(
                    calibration_dhash
                )
            ),
            "ahash_distribution": (
                distribution_summary(
                    calibration_ahash
                )
            ),
            "phash_threshold_counts": (
                threshold_counts(
                    calibration_phash
                )
            ),
            "dhash_threshold_counts": (
                threshold_counts(
                    calibration_dhash
                )
            ),
        },
        "candidate_nearest_neighbors": {
            "nearest_phash_distribution": (
                distribution_summary(
                    candidate_phash
                )
            ),
            "nearest_dhash_distribution": (
                distribution_summary(
                    candidate_dhash
                )
            ),
            "combined_score_distribution": (
                distribution_summary(
                    candidate_combined
                )
            ),
            "nearest_phash_threshold_counts": (
                threshold_counts(
                    candidate_phash
                )
            ),
            "nearest_dhash_threshold_counts": (
                threshold_counts(
                    candidate_dhash
                )
            ),
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            "known_overlap_calibration": str(
                calibration_path
            ),
            "candidate_nearest_neighbors": str(
                neighbors_path
            ),
        },
        "final_threshold_selected": False,
        "final_clean_manifest_created": False,
        "training_allowed": False,
        "training_block_reason": (
            "Perceptual thresholds and suspicious "
            "pairs must be reviewed before creating "
            "the final IMA++ non-overlap manifest."
        ),
    }

    report_path = (
        Path(config.REPORTS_DIR)
        / "imaplusplus_perceptual_hash_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== Perceptual Hash Results ==="
    )

    print(
        "Known same-ID calibration pairs : "
        f"{len(calibration_rows)}"
    )

    print(
        "Non-overlap candidates screened : "
        f"{len(neighbor_rows)}"
    )

    print(
        "Known-overlap pHash median       : "
        f"{report['known_overlap_calibration']}"
        "['phash_distribution']['median']}"
    )

    print(
        "Candidate nearest pHash minimum  : "
        f"{report['candidate_nearest_neighbors']}"
        "['nearest_phash_distribution']['minimum']}"
    )

    print(
        "Candidate nearest pHash median   : "
        f"{report['candidate_nearest_neighbors']}"
        "['nearest_phash_distribution']['median']}"
    )

    print(
        "All validation checks passed     : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {calibration_path}")
    print(f" - {neighbors_path}")
    print(f" - {report_path}")

    print(
        "\nNo image was excluded automatically."
    )

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())