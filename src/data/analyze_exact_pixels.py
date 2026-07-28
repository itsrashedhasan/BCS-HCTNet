"""Step 03B: Detect exact decoded-pixel overlap between IMA++ and ISIC 2018.

This stage is session-reset safe. It reads the persistent Step 02 manifests
from /kaggle/input and does not depend on Step 03A files in /kaggle/working.

It hashes normalized decoded RGB pixels, not encoded file bytes. Therefore,
files with different container metadata but exactly identical decoded pixels
receive the same signature.

Outputs are written to /kaggle/working and are non-destructive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps

from src.data.analyze_overlap import (
    EXPECTED_IMAPLUSPLUS_ROWS,
    EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES,
    EXPECTED_ISIC2018_UNIQUE_IMAGES,
    find_step02_manifest_directory,
    normalize_image_id,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
)
from src.utils import config


EXPECTED_EXACT_ID_OVERLAP_UNIQUE_IMAGES = 3568


def resolve_image_path(row: dict[str, str]) -> Path:
    """Resolve an image path without making ambiguous guesses."""

    absolute_value = row.get("image_path", "").strip()

    if absolute_value and "::" not in absolute_value:
        absolute_path = Path(absolute_value)

        if absolute_path.is_file():
            return absolute_path

    relative_value = row.get(
        "image_relative_path",
        "",
    ).strip()

    if relative_value and "::" not in relative_value:
        relative_path = (
            Path("/kaggle/input") / relative_value
        )

        if relative_path.is_file():
            return relative_path

    raise FileNotFoundError(
        "Could not resolve manifest image path without guessing. "
        f"image_id={row.get('image_id', '')!r}, "
        f"image_path={absolute_value!r}, "
        f"image_relative_path={relative_value!r}"
    )


def decoded_rgb_signature(
    image_path: Path,
) -> dict[str, object]:
    """Return a hash of normalized decoded RGB pixels."""

    with Image.open(image_path) as opened_image:
        oriented_image = ImageOps.exif_transpose(
            opened_image
        )

        rgb_image = oriented_image.convert("RGB")
        rgb_image.load()

        width, height = rgb_image.size

        digest = hashlib.sha256()

        digest.update(
            b"BCS-HCTNet-decoded-RGB-v1\0"
        )

        digest.update(
            struct.pack(
                ">II",
                width,
                height,
            )
        )

        digest.update(rgb_image.tobytes())

    return {
        "decoded_width": width,
        "decoded_height": height,
        "decoded_mode": "RGB",
        "decoded_pixel_sha256": digest.hexdigest(),
    }


def select_unique_image_rows(
    rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Choose one deterministic image row for every image ID."""

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        image_id = normalize_image_id(
            row["image_id"]
        )

        grouped[image_id].append(row)

    selected: dict[
        str,
        dict[str, str],
    ] = {}

    for image_id, image_rows in grouped.items():
        selected[image_id] = sorted(
            image_rows,
            key=lambda row: (
                row.get(
                    "image_relative_path",
                    "",
                ),
                row.get(
                    "image_path",
                    "",
                ),
            ),
        )[0]

    return selected


def hash_dataset_images(
    dataset: str,
    rows_by_id: dict[str, dict[str, str]],
    max_workers: int,
) -> tuple[
    dict[str, dict[str, object]],
    list[dict[str, str]],
]:
    """Hash one image per ID with bounded parallelism."""

    results: dict[
        str,
        dict[str, object],
    ] = {}

    failures: list[dict[str, str]] = []

    total = len(rows_by_id)

    def process_one(
        item: tuple[
            str,
            dict[str, str],
        ],
    ) -> tuple[
        str,
        dict[str, object],
    ]:
        image_id, row = item

        image_path = resolve_image_path(row)

        signature = decoded_rgb_signature(
            image_path
        )

        signature.update(
            {
                "dataset": dataset,
                "image_id": image_id,
                "image_path": str(image_path),
                "image_relative_path": row.get(
                    "image_relative_path",
                    "",
                ),
            }
        )

        return image_id, signature

    print(
        f"Hashing {dataset}: {total} unique images "
        f"with {max_workers} workers"
    )

    completed = 0

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_id = {
            executor.submit(
                process_one,
                item,
            ): item[0]
            for item in rows_by_id.items()
        }

        for future in as_completed(
            future_to_id
        ):
            image_id = future_to_id[future]

            try:
                result_id, signature = (
                    future.result()
                )

                results[result_id] = signature

            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "image_id": image_id,
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
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

    return results, failures


def group_ids_by_hash(
    signatures_by_id: dict[
        str,
        dict[str, object],
    ],
) -> dict[str, list[str]]:
    """Group image IDs by decoded-pixel hash."""

    grouped: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for image_id, signature in (
        signatures_by_id.items()
    ):
        pixel_hash = str(
            signature[
                "decoded_pixel_sha256"
            ]
        )

        grouped[pixel_hash].append(
            image_id
        )

    return {
        pixel_hash: sorted(image_ids)
        for pixel_hash, image_ids
        in grouped.items()
    }


def write_hash_ledger(
    output_path: Path,
    signatures: Iterable[
        dict[str, object]
    ],
) -> None:
    """Write one reproducibility row for each image hash."""

    fieldnames = [
        "dataset",
        "image_id",
        "decoded_pixel_sha256",
        "decoded_width",
        "decoded_height",
        "decoded_mode",
        "image_path",
        "image_relative_path",
    ]

    rows = sorted(
        signatures,
        key=lambda row: (
            str(row["dataset"]),
            str(row["image_id"]),
        ),
    )

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
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run exact decoded-pixel overlap detection."""

    config.ensure_all_dirs()

    manifest_directory = (
        find_step02_manifest_directory()
    )

    isic2018_path = (
        manifest_directory
        / "isic2018_all.csv"
    )

    imaplusplus_path = (
        manifest_directory
        / "imaplusplus_all.csv"
    )

    print(
        "=== Step 03B: Exact Decoded-Pixel "
        "Overlap Analysis ==="
    )

    print(
        "Persistent manifest directory: "
        f"{manifest_directory}"
    )

    isic2018_rows = read_csv_rows(
        isic2018_path
    )

    imaplusplus_rows = read_csv_rows(
        imaplusplus_path
    )

    if len(imaplusplus_rows) != (
        EXPECTED_IMAPLUSPLUS_ROWS
    ):
        raise RuntimeError(
            "IMA++ row count changed unexpectedly: "
            f"expected {EXPECTED_IMAPLUSPLUS_ROWS}, "
            f"found {len(imaplusplus_rows)}."
        )

    isic_unique_rows = (
        select_unique_image_rows(
            isic2018_rows
        )
    )

    ima_unique_rows = (
        select_unique_image_rows(
            imaplusplus_rows
        )
    )

    if len(isic_unique_rows) != (
        EXPECTED_ISIC2018_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "ISIC 2018 unique-image count changed: "
            f"expected "
            f"{EXPECTED_ISIC2018_UNIQUE_IMAGES}, "
            f"found {len(isic_unique_rows)}."
        )

    if len(ima_unique_rows) != (
        EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "IMA++ unique-image count changed: "
            f"expected "
            f"{EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES}, "
            f"found {len(ima_unique_rows)}."
        )

    isic_ids = set(isic_unique_rows)
    ima_ids = set(ima_unique_rows)

    exact_id_overlap_ids = (
        isic_ids & ima_ids
    )

    if len(exact_id_overlap_ids) != (
        EXPECTED_EXACT_ID_OVERLAP_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "Exact-ID overlap count changed from "
            "the validated Step 03A result: "
            f"expected "
            f"{EXPECTED_EXACT_ID_OVERLAP_UNIQUE_IMAGES}, "
            f"found {len(exact_id_overlap_ids)}."
        )

    candidate_ids = sorted(
        ima_ids - isic_ids
    )

    candidate_unique_rows = {
        image_id: ima_unique_rows[image_id]
        for image_id in candidate_ids
    }

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
            max(
                1,
                os.cpu_count() or 1,
            ),
        )
    )

    isic_signatures, isic_failures = (
        hash_dataset_images(
            "isic2018",
            isic_unique_rows,
            max_workers,
        )
    )

    (
        candidate_signatures,
        candidate_failures,
    ) = hash_dataset_images(
        "imaplusplus_candidate",
        candidate_unique_rows,
        max_workers,
    )

    failures = (
        isic_failures
        + candidate_failures
    )

    if failures:
        failure_path = (
            Path(config.REPORTS_DIR)
            / "exact_pixel_hash_failures.json"
        )

        failure_path.write_text(
            json.dumps(
                failures,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Image hashing failed for one or "
            "more files. "
            f"See {failure_path}."
        )

    isic_ids_by_hash = (
        group_ids_by_hash(
            isic_signatures
        )
    )

    candidate_ids_by_hash = (
        group_ids_by_hash(
            candidate_signatures
        )
    )

    shared_hashes = sorted(
        set(isic_ids_by_hash)
        & set(candidate_ids_by_hash)
    )

    exact_pixel_overlap_ids = sorted(
        {
            image_id
            for pixel_hash in shared_hashes
            for image_id in (
                candidate_ids_by_hash[
                    pixel_hash
                ]
            )
        }
    )

    exact_pixel_overlap_id_set = set(
        exact_pixel_overlap_ids
    )

    candidate_after_pixel_ids = sorted(
        set(candidate_ids)
        - exact_pixel_overlap_id_set
    )

    original_fieldnames = list(
        imaplusplus_rows[0].keys()
    )

    extra_fieldnames = [
        "step03_status",
        "overlap_basis",
        "matching_isic2018_ids",
        "exact_pixel_sha256",
        "decoded_width",
        "decoded_height",
    ]

    output_fieldnames = (
        original_fieldnames
        + [
            field
            for field in extra_fieldnames
            if field
            not in original_fieldnames
        ]
    )

    exact_pixel_overlap_rows: list[
        dict[str, str]
    ] = []

    candidate_after_pixel_rows: list[
        dict[str, str]
    ] = []

    for original_row in imaplusplus_rows:
        image_id = normalize_image_id(
            original_row["image_id"]
        )

        if image_id in exact_id_overlap_ids:
            continue

        row = dict(original_row)
        row["image_id"] = image_id

        signature = candidate_signatures[
            image_id
        ]

        pixel_hash = str(
            signature[
                "decoded_pixel_sha256"
            ]
        )

        row["exact_pixel_sha256"] = (
            pixel_hash
        )

        row["decoded_width"] = str(
            signature["decoded_width"]
        )

        row["decoded_height"] = str(
            signature["decoded_height"]
        )

        if (
            image_id
            in exact_pixel_overlap_id_set
        ):
            matching_ids = (
                isic_ids_by_hash[
                    pixel_hash
                ]
            )

            row[
                "is_overlap_with_isic2018"
            ] = "yes_exact_decoded_pixels"

            row["step03_status"] = (
                "excluded_exact_decoded_"
                "pixel_overlap"
            )

            row["overlap_basis"] = (
                "same_dimensions_and_"
                "decoded_rgb_pixels"
            )

            row[
                "matching_isic2018_ids"
            ] = ";".join(matching_ids)

            exact_pixel_overlap_rows.append(
                row
            )

        else:
            row[
                "is_overlap_with_isic2018"
            ] = (
                "no_exact_id_or_pixel_match_"
                "perceptual_check_pending"
            )

            row["step03_status"] = (
                "candidate_after_exact_"
                "pixel_filter"
            )

            row["overlap_basis"] = (
                "no_exact_identifier_or_"
                "decoded_pixel_match"
            )

            row[
                "matching_isic2018_ids"
            ] = ""

            candidate_after_pixel_rows.append(
                row
            )

    annotation_count_by_id: dict[
        str,
        int,
    ] = defaultdict(int)

    for row in imaplusplus_rows:
        image_id = normalize_image_id(
            row["image_id"]
        )

        annotation_count_by_id[
            image_id
        ] += 1

    overlap_id_rows: list[
        dict[str, str]
    ] = []

    for image_id in exact_pixel_overlap_ids:
        signature = candidate_signatures[
            image_id
        ]

        pixel_hash = str(
            signature[
                "decoded_pixel_sha256"
            ]
        )

        overlap_id_rows.append(
            {
                "imaplusplus_image_id": (
                    image_id
                ),
                "matching_isic2018_ids": (
                    ";".join(
                        isic_ids_by_hash[
                            pixel_hash
                        ]
                    )
                ),
                "decoded_pixel_sha256": (
                    pixel_hash
                ),
                "decoded_width": str(
                    signature[
                        "decoded_width"
                    ]
                ),
                "decoded_height": str(
                    signature[
                        "decoded_height"
                    ]
                ),
                "imaplusplus_annotation_rows": (
                    str(
                        annotation_count_by_id[
                            image_id
                        ]
                    )
                ),
                "imaplusplus_image_path": str(
                    signature["image_path"]
                ),
            }
        )

    manifest_output_directory = Path(
        config.MANIFEST_DIR
    )

    report_output_directory = Path(
        config.REPORTS_DIR
    )

    manifest_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlap_rows_path = (
        manifest_output_directory
        / (
            "imaplusplus_exact_pixel_"
            "overlap_rows.csv"
        )
    )

    overlap_ids_path = (
        manifest_output_directory
        / (
            "imaplusplus_exact_pixel_"
            "overlap_ids.csv"
        )
    )

    candidate_rows_path = (
        manifest_output_directory
        / (
            "imaplusplus_nonoverlap_after_"
            "exact_pixel_filter_candidate.csv"
        )
    )

    hash_ledger_path = (
        manifest_output_directory
        / (
            "step03b_decoded_pixel_"
            "hash_ledger.csv"
        )
    )

    write_csv_rows(
        overlap_rows_path,
        exact_pixel_overlap_rows,
        output_fieldnames,
    )

    write_csv_rows(
        candidate_rows_path,
        candidate_after_pixel_rows,
        output_fieldnames,
    )

    write_csv_rows(
        overlap_ids_path,
        overlap_id_rows,
        [
            "imaplusplus_image_id",
            "matching_isic2018_ids",
            "decoded_pixel_sha256",
            "decoded_width",
            "decoded_height",
            "imaplusplus_annotation_rows",
            "imaplusplus_image_path",
        ],
    )

    write_hash_ledger(
        hash_ledger_path,
        [
            *isic_signatures.values(),
            *candidate_signatures.values(),
        ],
    )

    exact_pixel_overlap_annotation_rows = (
        len(exact_pixel_overlap_rows)
    )

    candidate_after_pixel_annotation_rows = (
        len(candidate_after_pixel_rows)
    )

    internal_isic_duplicate_groups = {
        pixel_hash: image_ids
        for pixel_hash, image_ids
        in isic_ids_by_hash.items()
        if len(image_ids) > 1
    }

    internal_candidate_duplicate_groups = {
        pixel_hash: image_ids
        for pixel_hash, image_ids
        in candidate_ids_by_hash.items()
        if len(image_ids) > 1
    }

    expected_candidate_annotation_rows = sum(
        normalize_image_id(
            row["image_id"]
        )
        not in exact_id_overlap_ids
        for row in imaplusplus_rows
    )

    checks = {
        "all_images_hashed_successfully": (
            len(isic_signatures)
            == len(isic_unique_rows)
            and len(candidate_signatures)
            == len(candidate_unique_rows)
        ),
        "candidate_unique_partition_preserved": (
            len(exact_pixel_overlap_ids)
            + len(candidate_after_pixel_ids)
            == len(candidate_ids)
        ),
        "candidate_annotation_partition_preserved": (
            exact_pixel_overlap_annotation_rows
            + candidate_after_pixel_annotation_rows
            == expected_candidate_annotation_rows
        ),
        "remaining_candidate_has_no_shared_exact_hash": (
            not any(
                str(
                    candidate_signatures[
                        image_id
                    ][
                        "decoded_pixel_sha256"
                    ]
                )
                in isic_ids_by_hash
                for image_id
                in candidate_after_pixel_ids
            )
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "03B_exact_decoded_pixel_"
            "overlap_analysis"
        ),
        "persistent_manifest_directory": str(
            manifest_directory
        ),
        "hash_definition": {
            "decoder": "Pillow",
            "orientation": (
                "ImageOps.exif_transpose"
            ),
            "mode": "RGB",
            "signature_version": (
                "BCS-HCTNet-decoded-RGB-v1"
            ),
            "includes_dimensions": True,
            "cryptographic_hash": "SHA-256",
        },
        "inputs": {
            "isic2018_manifest": {
                "path": str(isic2018_path),
                "sha256": sha256_file(
                    isic2018_path
                ),
                "rows": len(isic2018_rows),
                "unique_images": len(
                    isic_unique_rows
                ),
            },
            "imaplusplus_manifest": {
                "path": str(
                    imaplusplus_path
                ),
                "sha256": sha256_file(
                    imaplusplus_path
                ),
                "rows": len(
                    imaplusplus_rows
                ),
                "unique_images": len(
                    ima_unique_rows
                ),
            },
        },
        "exact_identifier_filter": {
            "excluded_unique_images": len(
                exact_id_overlap_ids
            ),
            "candidate_unique_images": len(
                candidate_ids
            ),
        },
        "exact_decoded_pixel_overlap": {
            "shared_hash_groups": len(
                shared_hashes
            ),
            "excluded_unique_images": len(
                exact_pixel_overlap_ids
            ),
            "excluded_annotation_rows": (
                exact_pixel_overlap_annotation_rows
            ),
            "remaining_candidate_unique_images": (
                len(candidate_after_pixel_ids)
            ),
            "remaining_candidate_annotation_rows": (
                candidate_after_pixel_annotation_rows
            ),
        },
        "internal_exact_duplicates_report_only": {
            "isic2018_hash_groups_with_multiple_ids": (
                internal_isic_duplicate_groups
            ),
            (
                "imaplusplus_candidate_hash_"
                "groups_with_multiple_ids"
            ): internal_candidate_duplicate_groups,
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            "exact_pixel_overlap_rows": str(
                overlap_rows_path
            ),
            "exact_pixel_overlap_ids": str(
                overlap_ids_path
            ),
            (
                "candidate_after_exact_"
                "pixel_filter"
            ): str(candidate_rows_path),
            "decoded_pixel_hash_ledger": str(
                hash_ledger_path
            ),
        },
        "final_clean_manifest_created": False,
        "perceptual_duplicate_check_pending": (
            True
        ),
        "training_allowed": False,
        "training_block_reason": (
            "Perceptual near-duplicate analysis, "
            "final exclusion ledger, fixed split "
            "creation, target generation, and QC "
            "remain incomplete."
        ),
    }

    report_path = (
        report_output_directory
        / (
            "imaplusplus_exact_pixel_"
            "overlap_report.json"
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
        "\n=== Exact Decoded-Pixel Results ==="
    )

    print(
        "Exact-ID-filtered candidate images     : "
        f"{len(candidate_ids)}"
    )

    print(
        "Exact pixel-overlap unique images      : "
        f"{len(exact_pixel_overlap_ids)}"
    )

    print(
        "Exact pixel-overlap annotation rows    : "
        f"{exact_pixel_overlap_annotation_rows}"
    )

    print(
        "Remaining candidate unique images      : "
        f"{len(candidate_after_pixel_ids)}"
    )

    print(
        "Remaining candidate annotation rows    : "
        f"{candidate_after_pixel_annotation_rows}"
    )

    print(
        "All validation checks passed           : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {overlap_rows_path}")
    print(f" - {overlap_ids_path}")
    print(f" - {candidate_rows_path}")
    print(f" - {hash_ledger_path}")
    print(f" - {report_path}")

    print(
        "\nNo persistent Step 02 file was modified."
    )

    print(
        "The remaining candidate is not final "
        "until perceptual near-duplicate "
        "analysis passes."
    )

    return (
        0
        if all_checks_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())