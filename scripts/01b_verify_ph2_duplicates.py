"""Step 01B: Verify that duplicate PH2 files are pixel-identical.

Run from the repository root:

    python3 scripts/01b_verify_ph2_duplicates.py

The Kaggle PH2 dataset may contain both a simplified trainx/trainy layout and
an official PH2 folder layout. This script groups files by PH2 image ID and
checks whether repeated image and mask copies decode to exactly the same
pixels.

It does not modify any dataset files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


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


def normalized_path(value: str) -> str:
    """Normalize path separators and capitalization."""

    return value.lower().replace("\\", "/")


def compact_text(value: str) -> str:
    """Remove separators for tolerant path matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_ph2_id(filename: str) -> str | None:
    """Extract an identifier such as IMD003."""

    match = re.search(
        r"(IMD\d+)",
        filename,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).upper()


def classify_ph2_file(
    entry: SourceFile,
) -> str | None:
    """Classify a PH2 entry as an image, mask, or unrelated file."""

    if entry.suffix not in IMAGE_SUFFIXES:
        return None

    if extract_ph2_id(entry.name) is None:
        return None

    path = normalized_path(entry.virtual_path)
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


def decoded_pixel_signature(
    entry: SourceFile,
    mode: str,
) -> dict:
    """Return dimensions and a hash of normalized decoded pixels."""

    with open_source(entry) as stream:
        with Image.open(stream) as image:
            normalized = image.convert(mode)
            normalized.load()

            digest = hashlib.sha256()
            digest.update(
                str(normalized.size).encode("utf-8")
            )
            digest.update(mode.encode("utf-8"))
            digest.update(normalized.tobytes())

            return {
                "size": list(normalized.size),
                "mode": mode,
                "pixel_sha256": digest.hexdigest(),
            }


def compare_group(
    entries: list[SourceFile],
    mode: str,
) -> dict:
    """Compare every copy belonging to one PH2 identifier."""

    copies: list[dict] = []

    for entry in sorted(
        entries,
        key=lambda item: item.virtual_path,
    ):
        copy_report = {
            "path": entry.virtual_path,
            "kind": entry.kind,
        }

        try:
            copy_report.update(
                decoded_pixel_signature(
                    entry,
                    mode,
                )
            )

        except Exception as error:
            copy_report["error"] = (
                f"{type(error).__name__}: {error}"
            )

        copies.append(copy_report)

    successful = [
        copy
        for copy in copies
        if "error" not in copy
    ]

    signatures = {
        (
            tuple(copy["size"]),
            copy["mode"],
            copy["pixel_sha256"],
        )
        for copy in successful
    }

    return {
        "n_copies": len(copies),
        "all_readable": (
            len(successful) == len(copies)
        ),
        "pixel_identical": (
            bool(successful)
            and len(signatures) == 1
        ),
        "copies": copies,
    }


def main() -> int:
    """Verify all repeated PH2 image and mask copies."""

    config.ensure_all_dirs()

    roots = default_scan_roots()
    entries = build_source_index(roots)

    grouped: dict[
        str,
        dict[str, list[SourceFile]],
    ] = {
        "image": defaultdict(list),
        "mask": defaultdict(list),
    }

    for entry in entries:
        file_type = classify_ph2_file(entry)

        if file_type is None:
            continue

        image_id = extract_ph2_id(entry.name)

        if image_id is not None:
            grouped[file_type][image_id].append(entry)

    report: dict = {
        "scan_roots": roots,
        "images": {},
        "masks": {},
    }

    for image_id, copies in sorted(
        grouped["image"].items()
    ):
        if len(copies) > 1:
            report["images"][image_id] = (
                compare_group(
                    copies,
                    mode="RGB",
                )
            )

    for image_id, copies in sorted(
        grouped["mask"].items()
    ):
        if len(copies) > 1:
            report["masks"][image_id] = (
                compare_group(
                    copies,
                    mode="L",
                )
            )

    image_mismatches = [
        image_id
        for image_id, result
        in report["images"].items()
        if (
            not result["all_readable"]
            or not result["pixel_identical"]
        )
    ]

    mask_mismatches = [
        image_id
        for image_id, result
        in report["masks"].items()
        if (
            not result["all_readable"]
            or not result["pixel_identical"]
        )
    ]

    report["summary"] = {
        "n_duplicate_image_ids": len(
            report["images"]
        ),
        "n_duplicate_mask_ids": len(
            report["masks"]
        ),
        "image_mismatch_ids": image_mismatches,
        "mask_mismatch_ids": mask_mismatches,
        "all_duplicate_images_pixel_identical": (
            not image_mismatches
        ),
        "all_duplicate_masks_pixel_identical": (
            not mask_mismatches
        ),
        "safe_to_choose_one_canonical_copy": (
            bool(report["images"])
            and bool(report["masks"])
            and not image_mismatches
            and not mask_mismatches
        ),
    }

    output_path = (
        Path(config.REPORTS_DIR)
        / "ph2_duplicate_verification.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=== PH2 Duplicate Verification ===")
    print(
        "Duplicate image IDs        : "
        f"{len(report['images'])}"
    )
    print(
        "Duplicate mask IDs         : "
        f"{len(report['masks'])}"
    )
    print(
        "Image mismatches           : "
        f"{image_mismatches}"
    )
    print(
        "Mask mismatches            : "
        f"{mask_mismatches}"
    )
    print(
        "Safe to choose one copy    : "
        f"{report['summary']['safe_to_choose_one_canonical_copy']}"
    )
    print(
        "Full report saved to       : "
        f"{output_path}"
    )

    if report["summary"][
        "safe_to_choose_one_canonical_copy"
    ]:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())