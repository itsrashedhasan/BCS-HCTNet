"""Step 01: Read-only audit of all four research datasets.

The audit does not assume exact Kaggle dataset slugs or nesting depths.
It searches all files under /kaggle/input and also supports ZIP archives.

This module never modifies, extracts, renames, or deletes dataset files.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.data.source_index import (
    SourceFile,
    build_source_index,
    default_scan_roots,
    is_image_openable,
)
from src.utils import config


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def compact_text(value: str) -> str:
    """Remove separators and normalize text for path matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalized_path(value: str) -> str:
    """Normalize path separators and capitalization."""

    return value.lower().replace("\\", "/")


def unique_by_name(
    entries: Iterable[SourceFile],
) -> dict[str, SourceFile]:
    """Deduplicate files by filename.

    Directly mounted files are preferred over equivalent ZIP members.
    """

    selected: dict[str, SourceFile] = {}

    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.name.lower(),
            entry.kind == "zip",
            entry.virtual_path,
        ),
    )

    for entry in sorted_entries:
        selected.setdefault(entry.name.lower(), entry)

    return selected


def source_summary(
    entries: Iterable[SourceFile],
) -> dict:
    """Summarize where selected dataset files were discovered."""

    entry_list = list(entries)

    containers = sorted(
        {
            entry.container
            for entry in entry_list
        }
    )

    storage_modes = sorted(
        {
            entry.kind
            for entry in entry_list
        }
    )

    return {
        "storage_modes": storage_modes,
        "containers": containers[:20],
        "n_containers": len(containers),
        "only_inside_zip": (
            bool(entry_list)
            and storage_modes == ["zip"]
        ),
    }


def verify_sample(
    entries: Iterable[SourceFile],
    limit: int = 50,
) -> list[str]:
    """Verify that a sample of images and masks can be opened."""

    corrupt_files: list[str] = []

    unique_entries = list(
        unique_by_name(entries).values()
    )

    for entry in unique_entries[:limit]:
        if not is_image_openable(entry):
            corrupt_files.append(entry.virtual_path)

    return corrupt_files


def build_pair_report(
    images: Sequence[SourceFile],
    masks: Sequence[SourceFile],
    expected_mask_name: Callable[[str], str],
    expected_counts: tuple[int, int] | None = None,
) -> dict:
    """Generate image-to-mask pairing statistics."""

    image_map = unique_by_name(images)
    mask_map = unique_by_name(masks)

    image_name_counts = Counter(
        entry.name.lower()
        for entry in images
    )

    mask_name_counts = Counter(
        entry.name.lower()
        for entry in masks
    )

    missing_masks: list[str] = []

    for _, image_reference in sorted(image_map.items()):
        expected_name = expected_mask_name(
            image_reference.name
        ).lower()

        if expected_name not in mask_map:
            missing_masks.append(image_reference.name)

    report = {
        "n_images": len(image_map),
        "n_masks": len(mask_map),
        "missing_masks": missing_masks,
        "n_missing_masks": len(missing_masks),
        "duplicate_image_copies": {
            name: count
            for name, count in image_name_counts.items()
            if count > 1
        },
        "duplicate_mask_copies": {
            name: count
            for name, count in mask_name_counts.items()
            if count > 1
        },
        "corrupt_image_sample": verify_sample(images),
        "corrupt_mask_sample": verify_sample(masks),
        "image_sources": source_summary(images),
        "mask_sources": source_summary(masks),
    }

    if expected_counts is not None:
        expected_images, expected_masks = expected_counts

        report["expected_images"] = expected_images
        report["expected_masks"] = expected_masks

        report["image_count_matches_expected"] = (
            len(image_map) == expected_images
        )

        report["mask_count_matches_expected"] = (
            len(mask_map) == expected_masks
        )

    return report


def select_entries(
    entries: Iterable[SourceFile],
    required_key: str,
    suffixes: set[str],
) -> list[SourceFile]:
    """Select files by normalized path text and extension."""

    selected: list[SourceFile] = []

    for entry in entries:
        if entry.suffix not in suffixes:
            continue

        compact_path = compact_text(
            entry.virtual_path
        )

        if required_key in compact_path:
            selected.append(entry)

    return selected


def audit_isic2018(
    entries: Sequence[SourceFile],
) -> dict:
    """Audit official ISIC 2018 Task 1 files."""

    specifications = {
        "train": (
            "isic2018task12traininginput",
            "isic2018task1traininggroundtruth",
            (2594, 2594),
        ),
        "val": (
            "isic2018task12validationinput",
            "isic2018task1validationgroundtruth",
            (100, 100),
        ),
        "test": (
            "isic2018task12testinput",
            "isic2018task1testgroundtruth",
            (1000, 1000),
        ),
    }

    report = {
        "dataset": "isic2018",
        "splits": {},
    }

    all_found: list[SourceFile] = []

    for split_name, specification in specifications.items():
        image_key, mask_key, expected_counts = specification

        images = select_entries(
            entries,
            image_key,
            {".jpg", ".jpeg"},
        )

        masks = select_entries(
            entries,
            mask_key,
            {".png"},
        )

        all_found.extend(images)
        all_found.extend(masks)

        split_report = build_pair_report(
            images=images,
            masks=masks,
            expected_mask_name=lambda image_name: (
                f"{Path(image_name).stem}_segmentation.png"
            ),
            expected_counts=expected_counts,
        )

        if not images:
            split_report["note"] = (
                "No matching official ISIC 2018 files were found. "
                "The dataset may not be attached, or its folder "
                "names may differ from the official structure."
            )

        report["splits"][split_name] = split_report

    report["source_summary"] = source_summary(all_found)

    report["ready_for_manifest"] = all(
        split_report["n_images"] > 0
        and split_report["n_missing_masks"] == 0
        for split_report in report["splits"].values()
    )

    return report


def audit_isic2017(
    entries: Sequence[SourceFile],
) -> dict:
    """Audit official ISIC 2017 Task 1 files."""

    specifications = {
        "train": (
            "isic2017trainingdata",
            "isic2017trainingpart1groundtruth",
            (2000, 2000),
        ),
        "val": (
            "isic2017validationdata",
            "isic2017validationpart1groundtruth",
            (150, 150),
        ),
        "test": (
            "isic2017testv2data",
            "isic2017testv2part1groundtruth",
            (600, 600),
        ),
    }

    report = {
        "dataset": "isic2017",
        "splits": {},
    }

    all_found: list[SourceFile] = []

    for split_name, specification in specifications.items():
        image_key, mask_key, expected_counts = specification

        images = [
            entry
            for entry in select_entries(
                entries,
                image_key,
                {".jpg", ".jpeg"},
            )
            if "superpixel" not in entry.name.lower()
        ]

        masks = select_entries(
            entries,
            mask_key,
            {".png"},
        )

        all_found.extend(images)
        all_found.extend(masks)

        split_report = build_pair_report(
            images=images,
            masks=masks,
            expected_mask_name=lambda image_name: (
                f"{Path(image_name).stem}_segmentation.png"
            ),
            expected_counts=expected_counts,
        )

        if not images:
            split_report["note"] = (
                "No matching official ISIC 2017 files were found."
            )

        report["splits"][split_name] = split_report

    report["source_summary"] = source_summary(all_found)

    report["ready_for_external_evaluation"] = all(
        split_report["n_images"] > 0
        and split_report["n_missing_masks"] == 0
        for split_report in report["splits"].values()
    )

    return report


def extract_ph2_id(
    filename: str,
) -> str | None:
    """Extract an identifier such as IMD003 from a PH2 filename."""

    match = re.search(
        r"(IMD\d+)",
        filename,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).upper()


def audit_ph2(
    entries: Sequence[SourceFile],
) -> dict:
    """Audit PH2 images and lesion masks."""

    images: list[SourceFile] = []
    masks: list[SourceFile] = []

    for entry in entries:
        if entry.suffix not in IMAGE_EXTENSIONS:
            continue

        path = normalized_path(entry.virtual_path)
        compact_path = compact_text(path)
        filename = entry.name.lower()

        image_id = extract_ph2_id(entry.name)

        if image_id is None:
            continue

        is_mask = (
            "_lesion" in filename
            or "/trainy/" in path
            or "lesion_mask" in path
            or "lesionmask" in compact_path
        )

        is_image = (
            "/trainx/" in path
            or "trainxzip" in compact_path
            or "dermoscopic_image" in path
            or "dermoscopicimage" in compact_path
        ) and not is_mask

        if is_mask:
            masks.append(entry)

        elif is_image:
            images.append(entry)

    image_by_id: dict[str, SourceFile] = {}
    mask_by_id: dict[str, SourceFile] = {}

    image_counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()

    for entry in images:
        image_id = extract_ph2_id(entry.name)

        if image_id is None:
            continue

        image_counts[image_id] += 1

        existing = image_by_id.get(image_id)

        if (
            existing is None
            or (
                existing.kind == "zip"
                and entry.kind == "file"
            )
        ):
            image_by_id[image_id] = entry

    for entry in masks:
        image_id = extract_ph2_id(entry.name)

        if image_id is None:
            continue

        mask_counts[image_id] += 1

        existing = mask_by_id.get(image_id)

        if (
            existing is None
            or (
                existing.kind == "zip"
                and entry.kind == "file"
            )
        ):
            mask_by_id[image_id] = entry

    missing_mask_ids = sorted(
        set(image_by_id) - set(mask_by_id)
    )

    report = {
        "dataset": "ph2",
        "n_images": len(image_by_id),
        "n_masks": len(mask_by_id),
        "expected_images": 200,
        "expected_masks": 200,
        "image_count_matches_expected": (
            len(image_by_id) == 200
        ),
        "mask_count_matches_expected": (
            len(mask_by_id) == 200
        ),
        "missing_mask_ids": missing_mask_ids,
        "n_missing_masks": len(missing_mask_ids),
        "duplicate_image_copies": {
            key: value
            for key, value in image_counts.items()
            if value > 1
        },
        "duplicate_mask_copies": {
            key: value
            for key, value in mask_counts.items()
            if value > 1
        },
        "corrupt_image_sample": verify_sample(
            image_by_id.values()
        ),
        "corrupt_mask_sample": verify_sample(
            mask_by_id.values()
        ),
        "image_sources": source_summary(images),
        "mask_sources": source_summary(masks),
        "ready_for_external_evaluation": (
            bool(image_by_id)
            and not missing_mask_ids
        ),
    }

    if not images:
        report["note"] = (
            "No PH2 files were found in either supported layout: "
            "trainx/trainy or the official PH2 lesion folders."
        )

    return report


def extract_isic_id(
    value: str,
) -> str | None:
    """Extract and normalize an ISIC identifier."""

    match = re.search(
        r"ISIC[_-]?\d+",
        value,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return (
        match.group(0)
        .upper()
        .replace("-", "_")
    )


def audit_imaplusplus(
    entries: Sequence[SourceFile],
) -> dict:
    """Audit IMA++ images, segmentation masks, and metadata."""

    images: list[SourceFile] = []
    masks: list[SourceFile] = []
    metadata: list[SourceFile] = []

    for entry in entries:
        path = normalized_path(entry.virtual_path)
        compact_path = compact_text(path)

        if (
            entry.suffix == ".csv"
            and any(
                token in entry.name.lower()
                for token in (
                    "metadata",
                    "train",
                    "val",
                    "test",
                    "seg",
                )
            )
            and (
                "imaplus" in compact_path
                or "14201693" in compact_path
            )
        ):
            metadata.append(entry)

        if (
            entry.suffix in {".jpg", ".jpeg"}
            and (
                "isic-images" in path
                or "isicimages" in compact_path
            )
        ):
            images.append(entry)

        elif (
            entry.suffix == ".png"
            and (
                "/segs/" in path
                or "segszip" in compact_path
                or "14201693segs" in compact_path
            )
        ):
            masks.append(entry)

    image_by_id: dict[str, SourceFile] = {}
    mask_ids: set[str] = set()

    mask_count_by_id: Counter[str] = Counter()

    for entry in images:
        image_id = extract_isic_id(entry.name)

        if image_id is None:
            continue

        existing = image_by_id.get(image_id)

        if (
            existing is None
            or (
                existing.kind == "zip"
                and entry.kind == "file"
            )
        ):
            image_by_id[image_id] = entry

    for entry in masks:
        image_id = extract_isic_id(entry.name)

        if image_id is None:
            continue

        mask_ids.add(image_id)
        mask_count_by_id[image_id] += 1

    images_without_masks = sorted(
        set(image_by_id) - mask_ids
    )

    annotation_distribution = Counter(
        mask_count_by_id.values()
    )

    report = {
        "dataset": "imaplusplus",
        "n_images": len(image_by_id),
        "n_mask_files": len(masks),
        "n_unique_ids_in_masks": len(mask_ids),
        "images_without_any_mask": images_without_masks,
        "n_images_without_any_mask": len(images_without_masks),
        "annotator_mask_count_distribution": {
            str(mask_count): image_count
            for mask_count, image_count
            in sorted(annotation_distribution.items())
        },
        "metadata_files_found": sorted(
            {
                entry.name
                for entry in metadata
            }
        ),
        "corrupt_image_sample": verify_sample(
            image_by_id.values()
        ),
        "corrupt_mask_sample": verify_sample(masks),
        "image_sources": source_summary(images),
        "mask_sources": source_summary(masks),
        "ready_for_overlap_removal": (
            bool(image_by_id)
            and not images_without_masks
        ),
    }

    if not images:
        report["note"] = (
            "No IMA++ ISIC image files were found."
        )

    return report


def build_overall_status(
    reports: Mapping[str, dict],
) -> dict:
    """Summarize whether the project can continue to later steps."""

    isic2018_ready = reports["isic2018"].get(
        "ready_for_manifest",
        False,
    )

    external_status = {
        "isic2017": reports["isic2017"].get(
            "ready_for_external_evaluation",
            False,
        ),
        "ph2": reports["ph2"].get(
            "ready_for_external_evaluation",
            False,
        ),
        "imaplusplus": reports["imaplusplus"].get(
            "ready_for_overlap_removal",
            False,
        ),
    }

    return {
        "isic2018_ready_for_step_02": isic2018_ready,
        "external_datasets_discovered": external_status,
        "can_continue_to_step_02": isic2018_ready,
        "training_allowed": False,
        "training_block_reason": (
            "Training remains blocked until manifests, overlap "
            "removal, split creation, target generation, visual "
            "quality control, and tests are completed."
        ),
    }


def run_all_audits() -> dict:
    """Discover all files, run all audits, and save a JSON report."""

    config.ensure_all_dirs()

    roots = default_scan_roots()

    print("=== Building read-only source index ===")
    print(f"Scan roots: {roots}")

    entries = build_source_index(roots)

    print(
        "Indexed files and ZIP members: "
        f"{len(entries)}"
    )

    reports = {
        "isic2018": audit_isic2018(entries),
        "isic2017": audit_isic2017(entries),
        "ph2": audit_ph2(entries),
        "imaplusplus": audit_imaplusplus(entries),
    }

    reports["overall_status"] = build_overall_status(
        reports
    )

    reports["scan"] = {
        "roots": roots,
        "n_indexed_entries": len(entries),
        "n_direct_files": sum(
            entry.kind == "file"
            for entry in entries
        ),
        "n_zip_members": sum(
            entry.kind == "zip"
            for entry in entries
        ),
        "zip_archives": sorted(
            {
                entry.container
                for entry in entries
                if entry.kind == "zip"
            }
        ),
    }

    for dataset_name in (
        "isic2018",
        "isic2017",
        "ph2",
        "imaplusplus",
    ):
        print(
            f"\n--- Audit result: {dataset_name} ---"
        )

        print(
            json.dumps(
                reports[dataset_name],
                indent=2,
            )[:8000]
        )

    print("\n--- Overall status ---")

    print(
        json.dumps(
            reports["overall_status"],
            indent=2,
        )
    )

    output_path = os.path.join(
        config.REPORTS_DIR,
        "audit_report.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            reports,
            output_file,
            indent=2,
        )

    print(
        f"\nFull report saved to: {output_path}"
    )

    return reports


if __name__ == "__main__":
    run_all_audits()