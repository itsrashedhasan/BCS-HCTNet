"""Step 03A: Analyze exact ISIC-ID overlap between IMA++ and ISIC 2018.

This stage is deliberately non-destructive.

It reads the persistent Step 02 manifests from /kaggle/input and writes:

    /kaggle/working/data/manifests/
        imaplusplus_exact_id_overlap_rows.csv
        imaplusplus_nonoverlap_after_id_filter_candidate.csv
        imaplusplus_exact_id_overlap_ids.csv

    /kaggle/working/outputs/reports/
        imaplusplus_id_overlap_report.json

Important:
    The non-overlap candidate is not yet the final clean IMA++ manifest.
    Content-hash and perceptual-duplicate checks must still be completed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.utils import config


REQUIRED_MANIFEST_FILES = {
    "isic2018": "isic2018_all.csv",
    "imaplusplus": "imaplusplus_all.csv",
}

EXPECTED_ISIC2018_UNIQUE_IMAGES = 3694
EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES = 14967
EXPECTED_IMAPLUSPLUS_ROWS = 22472


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_step02_manifest_directory() -> Path:
    """Locate the persistent Step 02 manifest directory.

    The function does not rely on a fixed Kaggle username or dataset slug.
    """

    input_root = Path("/kaggle/input")

    if not input_root.exists():
        raise FileNotFoundError(
            "Kaggle input root does not exist: /kaggle/input"
        )

    candidates: list[Path] = []

    for manifest_path in input_root.rglob(
        REQUIRED_MANIFEST_FILES["isic2018"]
    ):
        manifest_directory = manifest_path.parent

        all_required_exist = all(
            (
                manifest_directory
                / required_filename
            ).exists()
            for required_filename
            in REQUIRED_MANIFEST_FILES.values()
        )

        if all_required_exist:
            candidates.append(
                manifest_directory.resolve()
            )

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            "Could not locate the persistent Step 02 "
            "manifest directory under /kaggle/input."
        )

    preferred_candidates = [
        candidate
        for candidate in candidates
        if "bcs-hctnet-step02-artifacts"
        in str(candidate).lower()
    ]

    if len(preferred_candidates) == 1:
        return preferred_candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Multiple possible Step 02 manifest directories "
        f"were found: {[str(path) for path in candidates]}"
    )


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file as dictionaries."""

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {file_path}"
            )

        return [
            {
                key: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            }
            for row in reader
        ]


def validate_required_columns(
    rows: Sequence[dict[str, str]],
    required_columns: Iterable[str],
    manifest_name: str,
) -> None:
    """Check that a manifest includes all required columns."""

    if not rows:
        raise RuntimeError(
            f"{manifest_name} manifest is empty."
        )

    available_columns = set(rows[0])

    missing_columns = sorted(
        set(required_columns) - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"{manifest_name} is missing columns: "
            f"{missing_columns}"
        )


def normalize_image_id(value: str) -> str:
    """Normalize an ISIC identifier."""

    return value.strip().upper().replace("-", "_")


def write_csv_rows(
    output_path: Path,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    """Write dictionary rows to a UTF-8 CSV file."""

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


def count_imaplusplus_mask_types(
    rows: Sequence[dict[str, str]],
) -> dict[str, int]:
    """Count IMA++ annotation categories."""

    counts = Counter(
        row.get("mask_type", "") or "unspecified"
        for row in rows
    )

    return dict(sorted(counts.items()))


def main() -> int:
    """Run exact identifier-overlap analysis."""

    config.ensure_all_dirs()

    manifest_directory = (
        find_step02_manifest_directory()
    )

    isic2018_path = (
        manifest_directory
        / REQUIRED_MANIFEST_FILES["isic2018"]
    )

    imaplusplus_path = (
        manifest_directory
        / REQUIRED_MANIFEST_FILES["imaplusplus"]
    )

    print("=== Step 03A: Exact ISIC-ID Overlap Analysis ===")
    print(
        "Persistent manifest directory: "
        f"{manifest_directory}"
    )
    print(f"ISIC 2018 manifest: {isic2018_path}")
    print(f"IMA++ manifest    : {imaplusplus_path}")

    isic2018_rows = read_csv_rows(
        isic2018_path
    )

    imaplusplus_rows = read_csv_rows(
        imaplusplus_path
    )

    required_columns = {
        "image_id",
        "dataset",
        "image_path",
        "image_relative_path",
        "mask_path",
        "mask_relative_path",
        "mask_type",
        "source_split",
        "split",
        "is_overlap_with_isic2018",
    }

    validate_required_columns(
        isic2018_rows,
        required_columns,
        "ISIC 2018",
    )

    validate_required_columns(
        imaplusplus_rows,
        required_columns,
        "IMA++",
    )

    if len(imaplusplus_rows) != (
        EXPECTED_IMAPLUSPLUS_ROWS
    ):
        raise RuntimeError(
            "IMA++ row count changed unexpectedly: "
            f"expected {EXPECTED_IMAPLUSPLUS_ROWS}, "
            f"found {len(imaplusplus_rows)}."
        )

    isic_rows_by_id: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in isic2018_rows:
        image_id = normalize_image_id(
            row["image_id"]
        )

        isic_rows_by_id[image_id].append(row)

    duplicated_isic_ids = {
        image_id: len(rows)
        for image_id, rows
        in isic_rows_by_id.items()
        if len(rows) > 1
    }

    if duplicated_isic_ids:
        raise RuntimeError(
            "ISIC 2018 contains duplicate image IDs "
            "across manifest rows: "
            f"{duplicated_isic_ids}"
        )

    if len(isic_rows_by_id) != (
        EXPECTED_ISIC2018_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "ISIC 2018 unique-image count changed: "
            f"expected "
            f"{EXPECTED_ISIC2018_UNIQUE_IMAGES}, "
            f"found {len(isic_rows_by_id)}."
        )

    ima_rows_by_id: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in imaplusplus_rows:
        image_id = normalize_image_id(
            row["image_id"]
        )

        ima_rows_by_id[image_id].append(row)

    if len(ima_rows_by_id) != (
        EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "IMA++ unique-image count changed: "
            f"expected "
            f"{EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES}, "
            f"found {len(ima_rows_by_id)}."
        )

    isic_ids = set(isic_rows_by_id)
    ima_ids = set(ima_rows_by_id)

    overlap_ids = sorted(
        isic_ids & ima_ids
    )

    nonoverlap_ids = sorted(
        ima_ids - isic_ids
    )

    overlap_id_set = set(overlap_ids)

    overlap_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []

    original_fieldnames = list(
        imaplusplus_rows[0].keys()
    )

    additional_fieldnames = [
        "step03_status",
        "overlap_basis",
        "overlap_isic2018_source_split",
        "overlap_isic2018_split",
    ]

    output_fieldnames = (
        original_fieldnames
        + [
            field
            for field in additional_fieldnames
            if field not in original_fieldnames
        ]
    )

    for original_row in imaplusplus_rows:
        row = dict(original_row)

        image_id = normalize_image_id(
            row["image_id"]
        )

        row["image_id"] = image_id

        if image_id in overlap_id_set:
            matching_isic_row = (
                isic_rows_by_id[image_id][0]
            )

            row[
                "is_overlap_with_isic2018"
            ] = "yes_exact_identifier"

            row["step03_status"] = (
                "excluded_exact_identifier_overlap"
            )

            row["overlap_basis"] = (
                "same_normalized_isic_identifier"
            )

            row[
                "overlap_isic2018_source_split"
            ] = matching_isic_row.get(
                "source_split",
                "",
            )

            row[
                "overlap_isic2018_split"
            ] = matching_isic_row.get(
                "split",
                "",
            )

            overlap_rows.append(row)

        else:
            row[
                "is_overlap_with_isic2018"
            ] = (
                "no_exact_identifier_match_"
                "content_check_pending"
            )

            row["step03_status"] = (
                "candidate_after_exact_id_filter"
            )

            row["overlap_basis"] = (
                "no_normalized_identifier_match"
            )

            row[
                "overlap_isic2018_source_split"
            ] = ""

            row[
                "overlap_isic2018_split"
            ] = ""

            candidate_rows.append(row)

    overlap_id_rows: list[dict[str, str]] = []

    overlap_id_fieldnames = [
        "image_id",
        "isic2018_source_split",
        "isic2018_split",
        "imaplusplus_annotation_rows",
        "imaplusplus_individual_annotations",
        "imaplusplus_consensus_annotations",
        "imaplusplus_other_annotations",
    ]

    for image_id in overlap_ids:
        ima_rows = ima_rows_by_id[image_id]
        isic_row = isic_rows_by_id[image_id][0]

        mask_type_counts = Counter(
            row.get("mask_type", "")
            for row in ima_rows
        )

        overlap_id_rows.append(
            {
                "image_id": image_id,
                "isic2018_source_split": (
                    isic_row.get(
                        "source_split",
                        "",
                    )
                ),
                "isic2018_split": (
                    isic_row.get(
                        "split",
                        "",
                    )
                ),
                "imaplusplus_annotation_rows": str(
                    len(ima_rows)
                ),
                "imaplusplus_individual_annotations": str(
                    mask_type_counts.get(
                        "individual_annotation",
                        0,
                    )
                ),
                "imaplusplus_consensus_annotations": str(
                    mask_type_counts.get(
                        "consensus",
                        0,
                    )
                ),
                "imaplusplus_other_annotations": str(
                    len(ima_rows)
                    - mask_type_counts.get(
                        "individual_annotation",
                        0,
                    )
                    - mask_type_counts.get(
                        "consensus",
                        0,
                    )
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
        / "imaplusplus_exact_id_overlap_rows.csv"
    )

    candidate_rows_path = (
        manifest_output_directory
        / (
            "imaplusplus_nonoverlap_after_"
            "id_filter_candidate.csv"
        )
    )

    overlap_ids_path = (
        manifest_output_directory
        / "imaplusplus_exact_id_overlap_ids.csv"
    )

    write_csv_rows(
        overlap_rows_path,
        overlap_rows,
        output_fieldnames,
    )

    write_csv_rows(
        candidate_rows_path,
        candidate_rows,
        output_fieldnames,
    )

    write_csv_rows(
        overlap_ids_path,
        overlap_id_rows,
        overlap_id_fieldnames,
    )

    overlap_rows_by_isic_split = Counter(
        row[
            "overlap_isic2018_source_split"
        ]
        for row in overlap_rows
    )

    overlap_unique_ids_by_isic_split = Counter(
        isic_rows_by_id[image_id][0].get(
            "source_split",
            "",
        )
        for image_id in overlap_ids
    )

    checks = {
        "input_isic2018_unique_images_correct": (
            len(isic_rows_by_id)
            == EXPECTED_ISIC2018_UNIQUE_IMAGES
        ),
        "input_imaplusplus_rows_correct": (
            len(imaplusplus_rows)
            == EXPECTED_IMAPLUSPLUS_ROWS
        ),
        "input_imaplusplus_unique_images_correct": (
            len(ima_rows_by_id)
            == EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
        ),
        "unique_image_partition_preserved": (
            len(overlap_ids)
            + len(nonoverlap_ids)
            == len(ima_rows_by_id)
        ),
        "annotation_row_partition_preserved": (
            len(overlap_rows)
            + len(candidate_rows)
            == len(imaplusplus_rows)
        ),
        "candidate_contains_no_exact_id_overlap": (
            not any(
                normalize_image_id(
                    row["image_id"]
                )
                in isic_ids
                for row in candidate_rows
            )
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": "03A_exact_identifier_overlap_analysis",
        "persistent_manifest_directory": str(
            manifest_directory
        ),
        "inputs": {
            "isic2018_manifest": {
                "path": str(isic2018_path),
                "sha256": sha256_file(
                    isic2018_path
                ),
                "rows": len(isic2018_rows),
                "unique_images": len(
                    isic_rows_by_id
                ),
            },
            "imaplusplus_manifest": {
                "path": str(imaplusplus_path),
                "sha256": sha256_file(
                    imaplusplus_path
                ),
                "rows": len(
                    imaplusplus_rows
                ),
                "unique_images": len(
                    ima_rows_by_id
                ),
            },
        },
        "exact_identifier_overlap": {
            "n_overlap_unique_images": len(
                overlap_ids
            ),
            "n_overlap_annotation_rows": len(
                overlap_rows
            ),
            "n_nonoverlap_candidate_unique_images": len(
                nonoverlap_ids
            ),
            "n_nonoverlap_candidate_annotation_rows": len(
                candidate_rows
            ),
            "overlap_unique_images_by_isic2018_source_split": dict(
                sorted(
                    overlap_unique_ids_by_isic_split.items()
                )
            ),
            "overlap_annotation_rows_by_isic2018_source_split": dict(
                sorted(
                    overlap_rows_by_isic_split.items()
                )
            ),
            "overlap_mask_type_counts": (
                count_imaplusplus_mask_types(
                    overlap_rows
                )
            ),
            "candidate_mask_type_counts": (
                count_imaplusplus_mask_types(
                    candidate_rows
                )
            ),
        },
        "checks": checks,
        "all_checks_passed": all_checks_passed,
        "outputs": {
            "overlap_rows": str(
                overlap_rows_path
            ),
            "candidate_rows": str(
                candidate_rows_path
            ),
            "overlap_ids": str(
                overlap_ids_path
            ),
        },
        "final_clean_manifest_created": False,
        "content_duplicate_check_pending": True,
        "perceptual_duplicate_check_pending": True,
        "training_allowed": False,
        "training_block_reason": (
            "Step 03A only removes exact identifier "
            "overlap from the candidate set. Exact "
            "decoded-pixel and perceptual near-duplicate "
            "checks are still required."
        ),
    }

    report_path = (
        report_output_directory
        / "imaplusplus_id_overlap_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Exact Identifier Results ===")
    print(
        "ISIC 2018 unique images              : "
        f"{len(isic_rows_by_id)}"
    )
    print(
        "IMA++ unique images                  : "
        f"{len(ima_rows_by_id)}"
    )
    print(
        "IMA++ annotation rows                : "
        f"{len(imaplusplus_rows)}"
    )
    print(
        "Exact overlapping unique images      : "
        f"{len(overlap_ids)}"
    )
    print(
        "Exact overlapping annotation rows    : "
        f"{len(overlap_rows)}"
    )
    print(
        "Non-overlap candidate unique images  : "
        f"{len(nonoverlap_ids)}"
    )
    print(
        "Non-overlap candidate annotation rows: "
        f"{len(candidate_rows)}"
    )
    print(
        "All validation checks passed         : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {overlap_rows_path}")
    print(f" - {candidate_rows_path}")
    print(f" - {overlap_ids_path}")
    print(f" - {report_path}")

    print(
        "\nNo persistent Step 02 file was modified."
    )
    print(
        "The candidate manifest is not final until "
        "content-duplicate checks pass."
    )

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())