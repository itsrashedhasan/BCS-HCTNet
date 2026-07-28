"""Step 02: Build reproducible CSV manifests for all four datasets.

Generated files:

    data/manifests/isic2018_all.csv
    data/manifests/ph2_all.csv
    data/manifests/imaplusplus_all.csv
    data/manifests/isic2017_all.csv
    outputs/reports/manifest_summary.json

Important research rules:

1. ISIC 2018 is the only development dataset.
2. PH2 is strict external validation only.
3. IMA++ remains pending until ISIC 2018 overlap removal.
4. ISIC 2017 is cross-year robustness evaluation only.
5. This script does not train, split, modify, or copy images.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageStat

from src.data.source_index import (
    SourceFile,
    build_source_index,
    default_scan_roots,
    open_source,
)
from src.utils import config


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "image_path",
    "mask_path",
    "image_relative_path",
    "mask_relative_path",
    "image_filename",
    "mask_filename",
    "image_source_kind",
    "mask_source_kind",
    "image_container",
    "mask_container",
    "image_member",
    "mask_member",
    "mask_type",
    "annotator_id",
    "consensus_type",
    "source_split",
    "split",
    "height",
    "width",
    "mask_foreground_ratio",
    "diagnosis",
    "lesion_size_group",
    "contrast_score",
    "artifact_score",
    "is_overlap_with_isic2018",
    "external_role",
    "patient_id",
    "notes",
]


def compact_text(value: str) -> str:
    """Normalize text for tolerant folder matching."""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def normalized_path(value: str) -> str:
    """Normalize path separators and capitalization."""

    return value.lower().replace("\\", "/")


def extract_isic_id(value: str) -> str | None:
    """Extract an identifier such as ISIC_0000123."""

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


def extract_ph2_id(value: str) -> str | None:
    """Extract an identifier such as IMD003."""

    match = re.search(
        r"IMD\d+",
        value,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(0).upper()


def relative_source_path(
    source: SourceFile,
) -> str:
    """Return a portable path relative to Kaggle input when possible."""

    if source.kind == "zip":
        container = source.container

        if container.startswith("/kaggle/input/"):
            container = os.path.relpath(
                container,
                "/kaggle/input",
            )

        return f"{container}::{source.member}"

    path = source.container

    if path.startswith("/kaggle/input/"):
        return os.path.relpath(
            path,
            "/kaggle/input",
        )

    repository_root = str(
        Path(__file__).resolve().parents[2]
    )

    try:
        return os.path.relpath(
            path,
            repository_root,
        )

    except ValueError:
        return path


def source_fields(
    source: SourceFile,
    prefix: str,
) -> dict:
    """Create manifest fields for an image or mask source."""

    return {
        f"{prefix}_path": source.virtual_path,
        f"{prefix}_relative_path": (
            relative_source_path(source)
        ),
        f"{prefix}_filename": source.name,
        f"{prefix}_source_kind": source.kind,
        f"{prefix}_container": source.container,
        f"{prefix}_member": source.member,
    }


def choose_preferred_source(
    entries: Iterable[SourceFile],
) -> SourceFile:
    """Choose one deterministic copy of the same file.

    Directly mounted files are preferred over ZIP members.
    """

    candidates = list(entries)

    if not candidates:
        raise ValueError(
            "Cannot choose a source from an empty list."
        )

    return sorted(
        candidates,
        key=lambda entry: (
            entry.kind == "zip",
            entry.virtual_path,
        ),
    )[0]


def choose_canonical_ph2_source(
    entries: Iterable[SourceFile],
    source_type: str,
) -> SourceFile:
    """Choose the simplified PH2 trainx/trainy copy.

    Pixel equality was verified separately in Step 01B.
    """

    candidates = list(entries)

    if not candidates:
        raise ValueError(
            "Cannot select a PH2 source from an empty list."
        )

    preferred_folder = (
        "/trainx/"
        if source_type == "image"
        else "/trainy/"
    )

    return sorted(
        candidates,
        key=lambda entry: (
            preferred_folder
            not in normalized_path(
                entry.virtual_path
            ),
            entry.kind == "zip",
            entry.virtual_path,
        ),
    )[0]


def unique_by_filename(
    entries: Iterable[SourceFile],
) -> dict[str, SourceFile]:
    """Deduplicate files using lowercase filenames."""

    grouped: dict[
        str,
        list[SourceFile],
    ] = defaultdict(list)

    for entry in entries:
        grouped[entry.name.lower()].append(entry)

    return {
        filename: choose_preferred_source(copies)
        for filename, copies in grouped.items()
    }


def select_entries(
    entries: Iterable[SourceFile],
    required_key: str,
    suffixes: set[str],
) -> list[SourceFile]:
    """Select files by normalized path key and extension."""

    return [
        entry
        for entry in entries
        if (
            entry.suffix in suffixes
            and required_key
            in compact_text(entry.virtual_path)
        )
    ]


def inspect_image(
    source: SourceFile,
) -> dict:
    """Read image dimensions and a normalized contrast score."""

    with open_source(source) as stream:
        with Image.open(stream) as image:
            image.load()

            width, height = image.size

            grayscale = image.convert("L")
            contrast_score = (
                float(
                    ImageStat.Stat(
                        grayscale
                    ).stddev[0]
                )
                / 255.0
            )

    return {
        "width": width,
        "height": height,
        "contrast_score": round(
            contrast_score,
            8,
        ),
    }


def inspect_mask(
    source: SourceFile,
) -> dict:
    """Calculate the proportion of nonzero mask pixels."""

    with open_source(source) as stream:
        with Image.open(stream) as mask:
            normalized = mask.convert("L")
            normalized.load()

            width, height = normalized.size
            histogram = normalized.histogram()

    total_pixels = width * height

    if total_pixels == 0:
        foreground_ratio = 0.0

    else:
        foreground_pixels = sum(
            histogram[1:]
        )

        foreground_ratio = (
            foreground_pixels / total_pixels
        )

    return {
        "mask_width": width,
        "mask_height": height,
        "mask_foreground_ratio": round(
            foreground_ratio,
            8,
        ),
    }


def blank_manifest_row() -> dict:
    """Return an empty row with every required column."""

    return {
        column: ""
        for column in MANIFEST_COLUMNS
    }


def build_paired_row(
    *,
    dataset: str,
    image_id: str,
    image_source: SourceFile,
    mask_source: SourceFile,
    source_split: str,
    split: str,
    mask_type: str,
    annotator_id: str = "",
    consensus_type: str = "",
    external_role: str = "",
    overlap_status: str = "",
    diagnosis: str = "",
    patient_id: str = "",
    notes: str = "",
) -> dict:
    """Build one complete manifest row."""

    image_information = inspect_image(
        image_source
    )

    mask_information = inspect_mask(
        mask_source
    )

    if (
        image_information["width"]
        != mask_information["mask_width"]
        or image_information["height"]
        != mask_information["mask_height"]
    ):
        raise ValueError(
            "Image-mask dimension mismatch for "
            f"{dataset}/{image_id}: "
            f"image="
            f"{image_information['width']}x"
            f"{image_information['height']}, "
            f"mask="
            f"{mask_information['mask_width']}x"
            f"{mask_information['mask_height']}"
        )

    row = blank_manifest_row()

    row.update(
        {
            "image_id": image_id,
            "dataset": dataset,
            "mask_type": mask_type,
            "annotator_id": annotator_id,
            "consensus_type": consensus_type,
            "source_split": source_split,
            "split": split,
            "height": image_information["height"],
            "width": image_information["width"],
            "mask_foreground_ratio": (
                mask_information[
                    "mask_foreground_ratio"
                ]
            ),
            "diagnosis": diagnosis,
            "lesion_size_group": "",
            "contrast_score": (
                image_information[
                    "contrast_score"
                ]
            ),
            "artifact_score": "",
            "is_overlap_with_isic2018": (
                overlap_status
            ),
            "external_role": external_role,
            "patient_id": patient_id,
            "notes": notes,
        }
    )

    row.update(
        source_fields(
            image_source,
            "image",
        )
    )

    row.update(
        source_fields(
            mask_source,
            "mask",
        )
    )

    return row


def build_isic_manifest(
    entries: Sequence[SourceFile],
    *,
    dataset: str,
    specifications: dict,
    external_role: str,
) -> list[dict]:
    """Build an ISIC 2018 or ISIC 2017 manifest."""

    rows: list[dict] = []

    for source_split, specification in (
        specifications.items()
    ):
        image_key = specification["image_key"]
        mask_key = specification["mask_key"]
        split = specification["split"]
        expected_count = (
            specification["expected_count"]
        )

        images = unique_by_filename(
            select_entries(
                entries,
                image_key,
                {".jpg", ".jpeg"},
            )
        )

        masks = unique_by_filename(
            select_entries(
                entries,
                mask_key,
                {".png"},
            )
        )

        if len(images) != expected_count:
            raise RuntimeError(
                f"{dataset}/{source_split}: "
                f"expected {expected_count} images, "
                f"found {len(images)}."
            )

        if len(masks) != expected_count:
            raise RuntimeError(
                f"{dataset}/{source_split}: "
                f"expected {expected_count} masks, "
                f"found {len(masks)}."
            )

        for image_source in sorted(
            images.values(),
            key=lambda source: source.name,
        ):
            image_id = extract_isic_id(
                image_source.name
            )

            if image_id is None:
                raise RuntimeError(
                    "Could not extract ISIC ID from "
                    f"{image_source.virtual_path}"
                )

            expected_mask_name = (
                f"{image_id}_segmentation.png"
            ).lower()

            mask_source = masks.get(
                expected_mask_name
            )

            if mask_source is None:
                raise RuntimeError(
                    f"Missing mask for {image_id} "
                    f"in {dataset}/{source_split}."
                )

            overlap_status = (
                "reference_dataset"
                if dataset == "isic2018"
                else ""
            )

            rows.append(
                build_paired_row(
                    dataset=dataset,
                    image_id=image_id,
                    image_source=image_source,
                    mask_source=mask_source,
                    source_split=source_split,
                    split=split,
                    mask_type=(
                        "official_binary_ground_truth"
                    ),
                    external_role=external_role,
                    overlap_status=overlap_status,
                    notes=(
                        "Official dataset image-mask pair."
                    ),
                )
            )

    return rows


def require_ph2_duplicate_verification() -> None:
    """Require successful PH2 duplicate verification."""

    report_path = (
        Path(config.REPORTS_DIR)
        / "ph2_duplicate_verification.json"
    )

    if not report_path.exists():
        raise RuntimeError(
            "PH2 duplicate verification report is missing. "
            "Run scripts/01b_verify_ph2_duplicates.py "
            "before building manifests."
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8",
        )
    )

    safe = report.get(
        "summary",
        {},
    ).get(
        "safe_to_choose_one_canonical_copy",
        False,
    )

    if not safe:
        raise RuntimeError(
            "PH2 duplicate verification did not pass. "
            "Do not build a canonical PH2 manifest."
        )


def classify_ph2_entry(
    entry: SourceFile,
) -> str | None:
    """Classify a PH2 file as image, mask, or unrelated."""

    if entry.suffix not in IMAGE_SUFFIXES:
        return None

    if extract_ph2_id(entry.name) is None:
        return None

    path = normalized_path(
        entry.virtual_path
    )

    compact_path = compact_text(path)
    filename = entry.name.lower()

    is_mask = (
        "_lesion" in filename
        or "/trainy/" in path
        or "lesion_mask" in path
        or "lesionmask" in compact_path
    )

    if is_mask:
        return "mask"

    is_image = (
        "/trainx/" in path
        or "trainxzip" in compact_path
        or "dermoscopic_image" in path
        or "dermoscopicimage" in compact_path
    )

    if is_image:
        return "image"

    return None


def build_ph2_manifest(
    entries: Sequence[SourceFile],
) -> list[dict]:
    """Build the canonical 200-image PH2 manifest."""

    require_ph2_duplicate_verification()

    grouped_images: dict[
        str,
        list[SourceFile],
    ] = defaultdict(list)

    grouped_masks: dict[
        str,
        list[SourceFile],
    ] = defaultdict(list)

    for entry in entries:
        entry_type = classify_ph2_entry(
            entry
        )

        if entry_type is None:
            continue

        image_id = extract_ph2_id(
            entry.name
        )

        if image_id is None:
            continue

        if entry_type == "image":
            grouped_images[image_id].append(
                entry
            )

        else:
            grouped_masks[image_id].append(
                entry
            )

    if len(grouped_images) != 200:
        raise RuntimeError(
            "PH2 expected 200 unique images, "
            f"found {len(grouped_images)}."
        )

    if len(grouped_masks) != 200:
        raise RuntimeError(
            "PH2 expected 200 unique masks, "
            f"found {len(grouped_masks)}."
        )

    rows: list[dict] = []

    for image_id in sorted(grouped_images):
        if image_id not in grouped_masks:
            raise RuntimeError(
                f"PH2 mask missing for {image_id}."
            )

        image_source = (
            choose_canonical_ph2_source(
                grouped_images[image_id],
                "image",
            )
        )

        mask_source = (
            choose_canonical_ph2_source(
                grouped_masks[image_id],
                "mask",
            )
        )

        rows.append(
            build_paired_row(
                dataset="ph2",
                image_id=image_id,
                image_source=image_source,
                mask_source=mask_source,
                source_split="all",
                split="external",
                mask_type=(
                    "official_binary_ground_truth"
                ),
                external_role="strict_external",
                overlap_status="not_applicable",
                notes=(
                    "Canonical trainx/trainy copy selected "
                    "after pixel-identical duplicate "
                    "verification."
                ),
            )
        )

    return rows


def classify_imaplusplus_mask(
    filename: str,
) -> dict:
    """Extract annotation metadata from an IMA++ mask filename."""

    stem = Path(filename).stem

    image_id = extract_isic_id(stem)

    if image_id is None:
        return {
            "mask_type": "unknown",
            "annotator_id": "",
            "consensus_type": "",
        }

    remaining = stem[len(image_id):].lstrip(
        "_-"
    )

    tokens = remaining.split("_")

    first_token = (
        tokens[0].upper()
        if tokens
        else ""
    )

    if first_token == "MV":
        return {
            "mask_type": "consensus",
            "annotator_id": "",
            "consensus_type": "MV",
        }

    if first_token == "ST":
        return {
            "mask_type": "consensus",
            "annotator_id": "",
            "consensus_type": "ST",
        }

    annotator_id = (
        first_token
        if re.fullmatch(
            r"A\d+",
            first_token,
        )
        else ""
    )

    return {
        "mask_type": "individual_annotation",
        "annotator_id": annotator_id,
        "consensus_type": "",
    }


def build_imaplusplus_manifest(
    entries: Sequence[SourceFile],
) -> list[dict]:
    """Build one row per IMA++ image-mask annotation pair."""

    images_by_id: dict[
        str,
        list[SourceFile],
    ] = defaultdict(list)

    masks_by_id: dict[
        str,
        list[SourceFile],
    ] = defaultdict(list)

    for entry in entries:
        path = normalized_path(
            entry.virtual_path
        )

        compact_path = compact_text(path)

        if (
            entry.suffix in {".jpg", ".jpeg"}
            and (
                "isic-images" in path
                or "isicimages" in compact_path
            )
        ):
            image_id = extract_isic_id(
                entry.name
            )

            if image_id is not None:
                images_by_id[image_id].append(
                    entry
                )

        elif (
            entry.suffix == ".png"
            and (
                "/segs/" in path
                or "segszip" in compact_path
                or "14201693segs"
                in compact_path
            )
        ):
            image_id = extract_isic_id(
                entry.name
            )

            if image_id is not None:
                masks_by_id[image_id].append(
                    entry
                )

    if len(images_by_id) != 14967:
        raise RuntimeError(
            "IMA++ expected 14,967 unique images, "
            f"found {len(images_by_id)}."
        )

    missing_mask_ids = sorted(
        set(images_by_id)
        - set(masks_by_id)
    )

    if missing_mask_ids:
        raise RuntimeError(
            "IMA++ contains images without masks: "
            f"{missing_mask_ids[:20]}"
        )

    rows: list[dict] = []

    for image_id in sorted(images_by_id):
        image_source = choose_preferred_source(
            images_by_id[image_id]
        )

        masks = sorted(
            masks_by_id[image_id],
            key=lambda source: source.name,
        )

        for mask_source in masks:
            mask_metadata = (
                classify_imaplusplus_mask(
                    mask_source.name
                )
            )

            rows.append(
                build_paired_row(
                    dataset="imaplusplus",
                    image_id=image_id,
                    image_source=image_source,
                    mask_source=mask_source,
                    source_split="all",
                    split=(
                        "supplementary_external_"
                        "pending_overlap_removal"
                    ),
                    mask_type=(
                        mask_metadata[
                            "mask_type"
                        ]
                    ),
                    annotator_id=(
                        mask_metadata[
                            "annotator_id"
                        ]
                    ),
                    consensus_type=(
                        mask_metadata[
                            "consensus_type"
                        ]
                    ),
                    external_role=(
                        "supplementary_external_style_"
                        "pending_overlap_removal"
                    ),
                    overlap_status="pending",
                    notes=(
                        "Do not use for evaluation until "
                        "Step 03 removes all ISIC 2018 "
                        "identifier overlap."
                    ),
                )
            )

    return rows


def validate_manifest(
    name: str,
    rows: Sequence[dict],
) -> dict:
    """Validate one completed manifest."""

    if not rows:
        raise RuntimeError(
            f"{name} manifest is empty."
        )

    missing_paths = [
        row["image_id"]
        for row in rows
        if (
            not row["image_path"]
            or not row["mask_path"]
        )
    ]

    invalid_dimensions = [
        row["image_id"]
        for row in rows
        if (
            int(row["width"]) <= 0
            or int(row["height"]) <= 0
        )
    ]

    invalid_ratios = [
        row["image_id"]
        for row in rows
        if not (
            0.0
            <= float(
                row["mask_foreground_ratio"]
            )
            <= 1.0
        )
    ]

    if missing_paths:
        raise RuntimeError(
            f"{name}: rows with missing paths: "
            f"{missing_paths[:20]}"
        )

    if invalid_dimensions:
        raise RuntimeError(
            f"{name}: invalid image dimensions: "
            f"{invalid_dimensions[:20]}"
        )

    if invalid_ratios:
        raise RuntimeError(
            f"{name}: invalid mask ratios: "
            f"{invalid_ratios[:20]}"
        )

    unique_image_ids = {
        row["image_id"]
        for row in rows
    }

    split_counts: dict[str, int] = defaultdict(
        int
    )

    for row in rows:
        split_counts[row["split"]] += 1

    return {
        "n_rows": len(rows),
        "n_unique_images": len(
            unique_image_ids
        ),
        "split_row_counts": dict(
            sorted(split_counts.items())
        ),
        "validation_passed": True,
    }


def write_manifest(
    output_path: Path,
    rows: Sequence[dict],
) -> None:
    """Write one UTF-8 CSV manifest."""

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
            fieldnames=MANIFEST_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Build all four manifests."""

    config.ensure_all_dirs()

    roots = default_scan_roots()

    print("=== Step 02: Build Dataset Manifests ===")
    print(f"Scan roots: {roots}")

    entries = build_source_index(roots)

    print(
        "Indexed files and ZIP members: "
        f"{len(entries)}"
    )

    isic2018_rows = build_isic_manifest(
        entries,
        dataset="isic2018",
        specifications={
            "official_train": {
                "image_key": (
                    "isic2018task12traininginput"
                ),
                "mask_key": (
                    "isic2018task1traininggroundtruth"
                ),
                "split": "train",
                "expected_count": 2594,
            },
            "official_validation": {
                "image_key": (
                    "isic2018task12validationinput"
                ),
                "mask_key": (
                    "isic2018task1validationgroundtruth"
                ),
                "split": "val",
                "expected_count": 100,
            },
            "official_test": {
                "image_key": (
                    "isic2018task12testinput"
                ),
                "mask_key": (
                    "isic2018task1testgroundtruth"
                ),
                "split": "internal_test",
                "expected_count": 1000,
            },
        },
        external_role="development_internal",
    )

    ph2_rows = build_ph2_manifest(entries)

    imaplusplus_rows = (
        build_imaplusplus_manifest(entries)
    )

    isic2017_rows = build_isic_manifest(
        entries,
        dataset="isic2017",
        specifications={
            "official_train": {
                "image_key": (
                    "isic2017trainingdata"
                ),
                "mask_key": (
                    "isic2017trainingpart1groundtruth"
                ),
                "split": "robustness",
                "expected_count": 2000,
            },
            "official_validation": {
                "image_key": (
                    "isic2017validationdata"
                ),
                "mask_key": (
                    "isic2017validationpart1groundtruth"
                ),
                "split": "robustness",
                "expected_count": 150,
            },
            "official_test": {
                "image_key": (
                    "isic2017testv2data"
                ),
                "mask_key": (
                    "isic2017testv2part1groundtruth"
                ),
                "split": "robustness",
                "expected_count": 600,
            },
        },
        external_role="cross_year_robustness",
    )

    manifests = {
        "isic2018_all": isic2018_rows,
        "ph2_all": ph2_rows,
        "imaplusplus_all": imaplusplus_rows,
        "isic2017_all": isic2017_rows,
    }

    expected_unique_images = {
        "isic2018_all": 3694,
        "ph2_all": 200,
        "imaplusplus_all": 14967,
        "isic2017_all": 2750,
    }

    summary = {
        "scan_roots": roots,
        "n_indexed_entries": len(entries),
        "manifests": {},
        "training_allowed": False,
        "training_block_reason": (
            "IMA++ overlap removal, fixed split export, "
            "target generation, visual QC, and tests "
            "are not complete."
        ),
    }

    for manifest_name, rows in (
        manifests.items()
    ):
        manifest_summary = (
            validate_manifest(
                manifest_name,
                rows,
            )
        )

        expected_unique = (
            expected_unique_images[
                manifest_name
            ]
        )

        actual_unique = (
            manifest_summary[
                "n_unique_images"
            ]
        )

        if actual_unique != expected_unique:
            raise RuntimeError(
                f"{manifest_name}: expected "
                f"{expected_unique} unique images, "
                f"found {actual_unique}."
            )

        output_path = (
            Path(config.MANIFEST_DIR)
            / f"{manifest_name}.csv"
        )

        write_manifest(
            output_path,
            rows,
        )

        manifest_summary["path"] = str(
            output_path
        )

        manifest_summary[
            "expected_unique_images"
        ] = expected_unique

        summary["manifests"][
            manifest_name
        ] = manifest_summary

        print(
            f"\n{manifest_name}:"
            f"\n  rows          = "
            f"{manifest_summary['n_rows']}"
            f"\n  unique images = "
            f"{actual_unique}"
            f"\n  saved to      = "
            f"{output_path}"
        )

    report_path = (
        Path(config.REPORTS_DIR)
        / "manifest_summary.json"
    )

    report_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nManifest summary saved to: "
        f"{report_path}"
    )

    print(
        "\nStep 02 core manifest generation passed."
    )

    print(
        "Training remains blocked until later "
        "data-preparation steps pass."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())