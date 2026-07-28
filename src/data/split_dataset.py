"""Step 04: Lock official ISIC 2018 splits and fixed external cohorts.

The audited Kaggle dataset contains the official ISIC 2018 training,
validation, and test releases. Therefore, this stage preserves those official
partitions instead of creating a new random 70/15/15 split. The fallback
70/15/15 rule in the research plan applies only when official partitions are
unavailable.

This CPU-only stage also:

- calculates lesion foreground ratios for ISIC 2018;
- defines lesion-size groups from training-only tertiles;
- verifies image-ID, decoded-pixel, and available patient/group leakage;
- creates fixed evaluation-only manifests for PH2, cleaned IMA++, and ISIC 2017;
- writes reproducibility reports and split hashes.

All persistent inputs are read-only. Outputs are written under
/kaggle/working/data/splits and /kaggle/working/outputs/reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from src.data.analyze_exact_pixels import (
    decoded_rgb_signature,
    resolve_image_path,
    select_unique_image_rows,
)
from src.data.analyze_overlap import (
    EXPECTED_ISIC2018_UNIQUE_IMAGES,
    find_step02_manifest_directory,
    normalize_image_id,
    read_csv_rows,
    sha256_file,
)
from src.utils import config


EXPECTED_ISIC2018_SPLIT_COUNTS = {
    "train": 2594,
    "val": 100,
    "internal_test": 1000,
}

EXPECTED_PH2_UNIQUE_IMAGES = 200
EXPECTED_ISIC2017_UNIQUE_IMAGES = 2750

EXPECTED_IMA_PRIMARY_UNIQUE_IMAGES = 11294
EXPECTED_IMA_PRIMARY_ROWS = 14204

EXPECTED_IMA_SENSITIVITY_UNIQUE_IMAGES = 11319
EXPECTED_IMA_SENSITIVITY_ROWS = 14243

SPLIT_PROTOCOL_VERSION = (
    "BCS-HCTNet-split-protocol-v1"
)


def find_step03_final_manifest_directory() -> Path:
    """Locate the verified Step 03 final manifest directory."""

    input_root = Path("/kaggle/input")
    candidates: list[Path] = []

    for primary_path in input_root.rglob(
        "imaplusplus_final_clean_primary.csv"
    ):
        directory = primary_path.parent

        required_files = [
            directory
            / (
                "imaplusplus_clean_"
                "sensitivity_tier1_only.csv"
            ),
            directory
            / "imaplusplus_overlap_exclusion_ledger.csv",
            directory
            / (
                "imaplusplus_perceptual_"
                "candidate_decisions.csv"
            ),
        ]

        if all(
            path.is_file()
            for path in required_files
        ):
            candidates.append(
                directory.resolve()
            )

    candidates = sorted(set(candidates))

    preferred = [
        candidate
        for candidate in candidates
        if (
            "bcs-hctnet-step03-final-artifacts"
            in str(candidate).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not find the persistent "
            "Step 03 final manifest directory."
        )

    raise RuntimeError(
        "Multiple Step 03 final manifest "
        "directories were found: "
        f"{[str(path) for path in candidates]}"
    )


def resolve_mask_path(
    row: dict[str, str],
) -> Path:
    """Resolve a mask path without filename guessing."""

    absolute_value = str(
        row.get(
            "mask_path",
            "",
        )
    ).strip()

    if (
        absolute_value
        and "::" not in absolute_value
    ):
        absolute_path = Path(
            absolute_value
        )

        if absolute_path.is_file():
            return absolute_path

    relative_value = str(
        row.get(
            "mask_relative_path",
            "",
        )
    ).strip()

    if (
        relative_value
        and "::" not in relative_value
    ):
        relative_path = (
            Path("/kaggle/input")
            / relative_value
        )

        if relative_path.is_file():
            return relative_path

    raise FileNotFoundError(
        "Could not resolve manifest mask path "
        "without guessing. "
        f"image_id={row.get('image_id', '')!r}, "
        f"mask_path={absolute_value!r}, "
        f"mask_relative_path={relative_value!r}"
    )


def normalize_official_split(
    row: dict[str, str],
) -> str:
    """Map the release name to the project split."""

    raw_value = str(
        row.get(
            "split",
            "",
        )
    ).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "internal_test",
        "testing": "internal_test",
        "internal_test": "internal_test",
    }

    if raw_value in aliases:
        return aliases[raw_value]

    searchable_path = " ".join(
        [
            str(
                row.get(
                    "image_path",
                    "",
                )
            ),
            str(
                row.get(
                    "image_relative_path",
                    "",
                )
            ),
            str(
                row.get(
                    "mask_path",
                    "",
                )
            ),
            str(
                row.get(
                    "mask_relative_path",
                    "",
                )
            ),
        ]
    ).lower()

    inferred_matches: list[str] = []

    if "training" in searchable_path:
        inferred_matches.append(
            "train"
        )

    if "validation" in searchable_path:
        inferred_matches.append(
            "val"
        )

    if "test" in searchable_path:
        inferred_matches.append(
            "internal_test"
        )

    inferred_matches = sorted(
        set(inferred_matches)
    )

    if len(inferred_matches) == 1:
        return inferred_matches[0]

    raise RuntimeError(
        "Could not determine the official "
        "ISIC 2018 split for "
        f"image_id={row.get('image_id', '')!r}, "
        f"split={raw_value!r}."
    )


def mask_signature(
    mask_array: np.ndarray,
) -> str:
    """Hash decoded binary mask pixels."""

    binary_mask = np.asarray(
        mask_array > 0,
        dtype=np.uint8,
    )

    height, width = binary_mask.shape

    digest = hashlib.sha256()

    digest.update(
        b"BCS-HCTNet-decoded-binary-mask-v1\0"
    )

    digest.update(
        struct.pack(
            ">II",
            width,
            height,
        )
    )

    digest.update(
        binary_mask.tobytes()
    )

    return digest.hexdigest()


def inspect_isic_sample(
    image_id: str,
    row: dict[str, str],
) -> tuple[str, dict[str, object]]:
    """Inspect one ISIC image-mask pair."""

    image_path = resolve_image_path(
        row
    )

    mask_path = resolve_mask_path(
        row
    )

    image_result = decoded_rgb_signature(
        image_path
    )

    with Image.open(
        mask_path
    ) as opened_mask:
        mask_image = opened_mask.convert(
            "L"
        )

        mask_image.load()

        mask_width, mask_height = (
            mask_image.size
        )

        mask_array = np.asarray(
            mask_image,
            dtype=np.uint8,
        )

    image_width = int(
        image_result[
            "decoded_width"
        ]
    )

    image_height = int(
        image_result[
            "decoded_height"
        ]
    )

    if (
        image_width,
        image_height,
    ) != (
        mask_width,
        mask_height,
    ):
        raise RuntimeError(
            "Image-mask dimensions do not "
            f"match for {image_id}: "
            f"image={(image_width, image_height)}, "
            f"mask={(mask_width, mask_height)}."
        )

    foreground_ratio = float(
        np.mean(
            mask_array > 0
        )
    )

    if not (
        0.0
        < foreground_ratio
        < 1.0
    ):
        raise RuntimeError(
            "ISIC 2018 mask must contain both "
            "foreground and background: "
            f"image_id={image_id}, "
            f"ratio={foreground_ratio}."
        )

    return image_id, {
        "image_id": image_id,
        "official_split": (
            normalize_official_split(
                row
            )
        ),
        "image_width": image_width,
        "image_height": image_height,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "mask_foreground_ratio": (
            foreground_ratio
        ),
        "decoded_pixel_sha256": str(
            image_result[
                "decoded_pixel_sha256"
            ]
        ),
        "decoded_binary_mask_sha256": (
            mask_signature(
                mask_array
            )
        ),
        "image_path": str(
            image_path
        ),
        "mask_path": str(
            mask_path
        ),
    }


def inspect_isic_dataset(
    rows_by_id: dict[
        str,
        dict[str, str],
    ],
    max_workers: int,
) -> dict[str, dict[str, object]]:
    """Inspect all ISIC samples using CPU threads."""

    total = len(
        rows_by_id
    )

    completed = 0

    results: dict[
        str,
        dict[str, object],
    ] = {}

    failures: list[
        dict[str, str]
    ] = []

    print(
        "Inspecting ISIC 2018 image-mask "
        f"pairs: {total} with "
        f"{max_workers} workers"
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_id = {
            executor.submit(
                inspect_isic_sample,
                image_id,
                row,
            ): image_id
            for image_id, row
            in rows_by_id.items()
        }

        for future in as_completed(
            future_to_id
        ):
            image_id = (
                future_to_id[
                    future
                ]
            )

            try:
                (
                    result_id,
                    result,
                ) = future.result()

                results[
                    result_id
                ] = result

            except Exception as error:
                failures.append(
                    {
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
                completed % 500 == 0
                or completed == total
            ):
                print(
                    "  ISIC 2018: "
                    f"{completed}/{total}"
                )

    if failures:
        failure_path = (
            Path(
                config.REPORTS_DIR
            )
            / (
                "step04_isic_"
                "inspection_failures.json"
            )
        )

        failure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        failure_path.write_text(
            json.dumps(
                failures,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "ISIC 2018 image-mask "
            "inspection failed. "
            f"See {failure_path}."
        )

    return results


def higher_quantile(
    values: Sequence[float],
    quantile: float,
) -> float:
    """Return a deterministic empirical quantile."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "Cannot calculate a quantile "
            "from an empty sequence."
        )

    try:
        return float(
            np.quantile(
                array,
                quantile,
                method="higher",
            )
        )

    except TypeError:
        return float(
            np.quantile(
                array,
                quantile,
                interpolation="higher",
            )
        )


def lesion_size_group(
    foreground_ratio: float,
    lower_threshold: float,
    upper_threshold: float,
) -> str:
    """Assign a training-calibrated size group."""

    if foreground_ratio <= lower_threshold:
        return "small"

    if foreground_ratio <= upper_threshold:
        return "medium"

    return "large"


def ensure_fields(
    rows: Sequence[
        dict[str, object]
    ],
    preferred_fields: Sequence[str],
) -> list[str]:
    """Build a stable CSV field order."""

    fields: list[str] = []

    for field in preferred_fields:
        if field not in fields:
            fields.append(
                field
            )

    for row in rows:
        for field in row.keys():
            if field not in fields:
                fields.append(
                    field
                )

    return fields


def write_csv(
    output_path: Path,
    rows: Sequence[
        dict[str, object]
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


def row_set_sha256(
    rows: Sequence[
        dict[str, object]
    ],
) -> str:
    """Hash canonical rows independently of CSV."""

    digest = hashlib.sha256()

    for row in rows:
        canonical = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

        digest.update(
            canonical.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\n"
        )

    return digest.hexdigest()


def cross_split_collisions(
    records: Sequence[
        dict[str, object]
    ],
    key: str,
) -> list[dict[str, object]]:
    """Find values occurring in multiple splits."""

    grouped: dict[
        str,
        dict[
            str,
            list[str],
        ],
    ] = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    for record in records:
        value = str(
            record.get(
                key,
                "",
            )
        ).strip()

        if not value:
            continue

        split = str(
            record[
                "split"
            ]
        )

        grouped[
            value
        ][split].append(
            str(
                record[
                    "image_id"
                ]
            )
        )

    collisions: list[
        dict[str, object]
    ] = []

    for (
        value,
        split_members,
    ) in grouped.items():
        if len(
            split_members
        ) > 1:
            collisions.append(
                {
                    "value": value,
                    "splits": sorted(
                        split_members
                    ),
                    "image_ids_by_split": {
                        split: sorted(
                            image_ids
                        )
                        for (
                            split,
                            image_ids,
                        )
                        in sorted(
                            split_members.items()
                        )
                    },
                }
            )

    return collisions


def prepare_external_rows(
    rows: Sequence[
        dict[str, str]
    ],
    split_name: str,
    external_role: str,
    status: str,
) -> list[dict[str, object]]:
    """Lock a persistent evaluation manifest."""

    prepared: list[
        dict[str, object]
    ] = []

    for source_row in rows:
        row: dict[
            str,
            object,
        ] = dict(
            source_row
        )

        row["image_id"] = (
            normalize_image_id(
                source_row[
                    "image_id"
                ]
            )
        )

        row["split"] = (
            split_name
        )

        row[
            "external_role"
        ] = external_role

        row[
            "evaluation_only"
        ] = "true"

        row[
            "training_allowed"
        ] = "false"

        row[
            "split_protocol_version"
        ] = SPLIT_PROTOCOL_VERSION

        row[
            "split_status"
        ] = status

        prepared.append(
            row
        )

    prepared.sort(
        key=lambda row: (
            str(
                row[
                    "image_id"
                ]
            ),
            str(
                row.get(
                    "annotator_id",
                    "",
                )
            ),
            str(
                row.get(
                    "mask_path",
                    "",
                )
            ),
        )
    )

    return prepared


def unique_image_count(
    rows: Sequence[
        dict[str, object]
    ],
) -> int:
    """Count normalized unique image IDs."""

    return len(
        {
            normalize_image_id(
                str(
                    row[
                        "image_id"
                    ]
                )
            )
            for row in rows
        }
    )


def split_distribution_rows(
    locked_rows: Sequence[
        dict[str, object]
    ],
) -> list[dict[str, object]]:
    """Create lesion-size distribution rows."""

    counts: dict[
        tuple[str, str],
        int,
    ] = Counter(
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
        for row in locked_rows
    )

    output: list[
        dict[str, object]
    ] = []

    for split in [
        "train",
        "val",
        "internal_test",
    ]:
        split_total = sum(
            count
            for (
                candidate_split,
                _,
            ), count
            in counts.items()
            if candidate_split
            == split
        )

        for size_group in [
            "small",
            "medium",
            "large",
        ]:
            count = counts.get(
                (
                    split,
                    size_group,
                ),
                0,
            )

            output.append(
                {
                    "split": split,
                    "lesion_size_group": (
                        size_group
                    ),
                    "image_count": count,
                    "split_fraction": (
                        round(
                            count
                            / split_total,
                            8,
                        )
                        if split_total
                        else 0.0
                    ),
                }
            )

    return output


def main() -> int:
    """Lock development and evaluation splits."""

    config.ensure_all_dirs()

    step02_directory = (
        find_step02_manifest_directory()
    )

    step03_final_directory = (
        find_step03_final_manifest_directory()
    )

    isic_manifest_path = (
        step02_directory
        / "isic2018_all.csv"
    )

    ph2_manifest_path = (
        step02_directory
        / "ph2_all.csv"
    )

    isic2017_manifest_path = (
        step02_directory
        / "isic2017_all.csv"
    )

    ima_primary_path = (
        step03_final_directory
        / "imaplusplus_final_clean_primary.csv"
    )

    ima_sensitivity_path = (
        step03_final_directory
        / (
            "imaplusplus_clean_"
            "sensitivity_tier1_only.csv"
        )
    )

    print(
        "=== Step 04: Lock Dataset Splits ==="
    )

    print(
        "Step 02 manifests   : "
        f"{step02_directory}"
    )

    print(
        "Step 03 final inputs: "
        f"{step03_final_directory}"
    )

    print(
        "Split policy        : "
        "preserve official ISIC 2018 releases"
    )

    print(
        "Random split seed   : "
        "not applicable"
    )

    isic_rows = read_csv_rows(
        isic_manifest_path
    )

    ph2_rows = read_csv_rows(
        ph2_manifest_path
    )

    isic2017_rows = read_csv_rows(
        isic2017_manifest_path
    )

    ima_primary_rows = read_csv_rows(
        ima_primary_path
    )

    ima_sensitivity_rows = read_csv_rows(
        ima_sensitivity_path
    )

    isic_unique_rows = (
        select_unique_image_rows(
            isic_rows
        )
    )

    if len(
        isic_unique_rows
    ) != EXPECTED_ISIC2018_UNIQUE_IMAGES:
        raise RuntimeError(
            "Unexpected ISIC 2018 "
            "unique-image count: "
            f"expected "
            f"{EXPECTED_ISIC2018_UNIQUE_IMAGES}, "
            f"found {len(isic_unique_rows)}."
        )

    configured_workers = int(
        os.environ.get(
            "BCS_SPLIT_WORKERS",
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

    inspection = inspect_isic_dataset(
        isic_unique_rows,
        max_workers,
    )

    training_ratios = sorted(
        float(
            record[
                "mask_foreground_ratio"
            ]
        )
        for record in inspection.values()
        if (
            record[
                "official_split"
            ]
            == "train"
        )
    )

    lower_threshold = higher_quantile(
        training_ratios,
        1.0 / 3.0,
    )

    upper_threshold = higher_quantile(
        training_ratios,
        2.0 / 3.0,
    )

    if not (
        0.0
        < lower_threshold
        <= upper_threshold
        < 1.0
    ):
        raise RuntimeError(
            "Invalid training-derived "
            "lesion-size thresholds: "
            f"lower={lower_threshold}, "
            f"upper={upper_threshold}."
        )

    locked_isic_rows: list[
        dict[str, object]
    ] = []

    for image_id in sorted(
        isic_unique_rows
    ):
        source_row = (
            isic_unique_rows[
                image_id
            ]
        )

        record = inspection[
            image_id
        ]

        split = str(
            record[
                "official_split"
            ]
        )

        ratio = float(
            record[
                "mask_foreground_ratio"
            ]
        )

        row: dict[
            str,
            object,
        ] = dict(
            source_row
        )

        row["image_id"] = (
            image_id
        )

        row[
            "official_source_split"
        ] = (
            "test"
            if split
            == "internal_test"
            else split
        )

        row["split"] = (
            split
        )

        row[
            "external_role"
        ] = "primary_internal"

        row[
            "evaluation_only"
        ] = (
            "true"
            if split
            == "internal_test"
            else "false"
        )

        row[
            "training_allowed"
        ] = (
            "true"
            if split == "train"
            else "false"
        )

        row[
            "split_protocol"
        ] = (
            "official_isic2018_release"
        )

        row[
            "split_protocol_version"
        ] = SPLIT_PROTOCOL_VERSION

        row[
            "split_seed"
        ] = (
            "not_applicable_official_release"
        )

        row[
            "mask_foreground_ratio"
        ] = f"{ratio:.10f}"

        row[
            "lesion_size_group"
        ] = lesion_size_group(
            ratio,
            lower_threshold,
            upper_threshold,
        )

        row["width"] = str(
            record[
                "image_width"
            ]
        )

        row["height"] = str(
            record[
                "image_height"
            ]
        )

        row[
            "decoded_pixel_sha256"
        ] = record[
            "decoded_pixel_sha256"
        ]

        row[
            "decoded_binary_mask_sha256"
        ] = record[
            "decoded_binary_mask_sha256"
        ]

        locked_isic_rows.append(
            row
        )

    split_rows: dict[
        str,
        list[
            dict[str, object]
        ],
    ] = {
        split: [
            row
            for row
            in locked_isic_rows
            if row[
                "split"
            ] == split
        ]
        for split in [
            "train",
            "val",
            "internal_test",
        ]
    }

    observed_split_counts = {
        split: len(rows)
        for split, rows
        in split_rows.items()
    }

    if (
        observed_split_counts
        != EXPECTED_ISIC2018_SPLIT_COUNTS
    ):
        raise RuntimeError(
            "Official ISIC 2018 split counts "
            "changed: "
            f"expected "
            f"{EXPECTED_ISIC2018_SPLIT_COUNTS}, "
            f"found {observed_split_counts}."
        )

    image_id_collisions = (
        cross_split_collisions(
            locked_isic_rows,
            "image_id",
        )
    )

    decoded_pixel_collisions = (
        cross_split_collisions(
            locked_isic_rows,
            "decoded_pixel_sha256",
        )
    )

    patient_id_values = [
        str(
            row.get(
                "patient_id",
                "",
            )
        ).strip()
        for row in locked_isic_rows
        if str(
            row.get(
                "patient_id",
                "",
            )
        ).strip()
    ]

    patient_id_available = bool(
        patient_id_values
    )

    patient_id_collisions = (
        cross_split_collisions(
            locked_isic_rows,
            "patient_id",
        )
        if patient_id_available
        else []
    )

    ph2_external_rows = (
        prepare_external_rows(
            ph2_rows,
            "external",
            "strict_external",
            "locked_evaluation_only",
        )
    )

    ima_primary_external_rows = (
        prepare_external_rows(
            ima_primary_rows,
            "supplementary_external",
            (
                "supplementary_"
                "external_style"
            ),
            (
                "locked_primary_"
                "overlap_screened_cohort"
            ),
        )
    )

    ima_sensitivity_external_rows = (
        prepare_external_rows(
            ima_sensitivity_rows,
            (
                "supplementary_"
                "external_sensitivity"
            ),
            (
                "supplementary_external_"
                "style_sensitivity"
            ),
            (
                "locked_tier1_only_"
                "sensitivity_cohort"
            ),
        )
    )

    isic2017_robustness_rows = (
        prepare_external_rows(
            isic2017_rows,
            "robustness",
            "cross_year_robustness",
            "locked_evaluation_only",
        )
    )

    external_count_checks = {
        "ph2_rows": len(
            ph2_external_rows
        ),
        "ph2_unique_images": (
            unique_image_count(
                ph2_external_rows
            )
        ),
        "ima_primary_rows": len(
            ima_primary_external_rows
        ),
        "ima_primary_unique_images": (
            unique_image_count(
                ima_primary_external_rows
            )
        ),
        "ima_sensitivity_rows": len(
            ima_sensitivity_external_rows
        ),
        "ima_sensitivity_unique_images": (
            unique_image_count(
                ima_sensitivity_external_rows
            )
        ),
        "isic2017_rows": len(
            isic2017_robustness_rows
        ),
        "isic2017_unique_images": (
            unique_image_count(
                isic2017_robustness_rows
            )
        ),
    }

    expected_external_counts = {
        "ph2_rows": 200,
        "ph2_unique_images": (
            EXPECTED_PH2_UNIQUE_IMAGES
        ),
        "ima_primary_rows": (
            EXPECTED_IMA_PRIMARY_ROWS
        ),
        "ima_primary_unique_images": (
            EXPECTED_IMA_PRIMARY_UNIQUE_IMAGES
        ),
        "ima_sensitivity_rows": (
            EXPECTED_IMA_SENSITIVITY_ROWS
        ),
        "ima_sensitivity_unique_images": (
            EXPECTED_IMA_SENSITIVITY_UNIQUE_IMAGES
        ),
        "isic2017_rows": 2750,
        "isic2017_unique_images": (
            EXPECTED_ISIC2017_UNIQUE_IMAGES
        ),
    }

    if (
        external_count_checks
        != expected_external_counts
    ):
        raise RuntimeError(
            "External cohort counts changed: "
            f"expected {expected_external_counts}, "
            f"found {external_count_checks}."
        )

    split_directory = (
        Path(
            config.MANIFEST_DIR
        ).parent
        / "splits"
    )

    report_directory = Path(
        config.REPORTS_DIR
    )

    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    isic_fieldnames = ensure_fields(
        locked_isic_rows,
        [
            "image_id",
            "dataset",
            "split",
            "official_source_split",
            "image_path",
            "mask_path",
            "mask_foreground_ratio",
            "lesion_size_group",
            "width",
            "height",
            "decoded_pixel_sha256",
            "decoded_binary_mask_sha256",
            "external_role",
            "evaluation_only",
            "training_allowed",
            "split_protocol",
            "split_protocol_version",
            "split_seed",
        ],
    )

    external_fieldnames = (
        ensure_fields(
            [
                *ph2_external_rows,
                *ima_primary_external_rows,
                *ima_sensitivity_external_rows,
                *isic2017_robustness_rows,
            ],
            [
                "image_id",
                "dataset",
                "split",
                "image_path",
                "mask_path",
                "mask_type",
                "annotator_id",
                "consensus_type",
                "external_role",
                "evaluation_only",
                "training_allowed",
                "split_status",
                "split_protocol_version",
            ],
        )
    )

    output_paths = {
        "isic2018_train": (
            split_directory
            / "isic2018_train.csv"
        ),
        "isic2018_val": (
            split_directory
            / "isic2018_val.csv"
        ),
        "isic2018_internal_test": (
            split_directory
            / "isic2018_internal_test.csv"
        ),
        "isic2018_all_locked": (
            split_directory
            / "isic2018_all_locked.csv"
        ),
        "ph2_external": (
            split_directory
            / "ph2_external.csv"
        ),
        "imaplusplus_primary_external": (
            split_directory
            / (
                "imaplusplus_"
                "nonoverlap_external.csv"
            )
        ),
        "imaplusplus_sensitivity_external": (
            split_directory
            / (
                "imaplusplus_sensitivity_"
                "tier1_only_external.csv"
            )
        ),
        "isic2017_robustness": (
            split_directory
            / "isic2017_robustness.csv"
        ),
        "split_distribution": (
            split_directory
            / (
                "isic2018_split_"
                "distribution.csv"
            )
        ),
    }

    write_csv(
        output_paths[
            "isic2018_train"
        ],
        split_rows[
            "train"
        ],
        isic_fieldnames,
    )

    write_csv(
        output_paths[
            "isic2018_val"
        ],
        split_rows[
            "val"
        ],
        isic_fieldnames,
    )

    write_csv(
        output_paths[
            "isic2018_internal_test"
        ],
        split_rows[
            "internal_test"
        ],
        isic_fieldnames,
    )

    write_csv(
        output_paths[
            "isic2018_all_locked"
        ],
        locked_isic_rows,
        isic_fieldnames,
    )

    write_csv(
        output_paths[
            "ph2_external"
        ],
        ph2_external_rows,
        external_fieldnames,
    )

    write_csv(
        output_paths[
            "imaplusplus_primary_external"
        ],
        ima_primary_external_rows,
        external_fieldnames,
    )

    write_csv(
        output_paths[
            "imaplusplus_sensitivity_external"
        ],
        ima_sensitivity_external_rows,
        external_fieldnames,
    )

    write_csv(
        output_paths[
            "isic2017_robustness"
        ],
        isic2017_robustness_rows,
        external_fieldnames,
    )

    distribution_rows = (
        split_distribution_rows(
            locked_isic_rows
        )
    )

    write_csv(
        output_paths[
            "split_distribution"
        ],
        distribution_rows,
        [
            "split",
            "lesion_size_group",
            "image_count",
            "split_fraction",
        ],
    )

    output_hashes = {
        name: sha256_file(
            path
        )
        for name, path
        in output_paths.items()
    }

    checks = {
        "isic2018_unique_count_correct": (
            len(
                locked_isic_rows
            )
            == EXPECTED_ISIC2018_UNIQUE_IMAGES
        ),
        "official_split_counts_correct": (
            observed_split_counts
            == EXPECTED_ISIC2018_SPLIT_COUNTS
        ),
        "all_isic_image_mask_pairs_inspected": (
            len(
                inspection
            )
            == EXPECTED_ISIC2018_UNIQUE_IMAGES
        ),
        "no_cross_split_image_id_collision": (
            not image_id_collisions
        ),
        "no_cross_split_decoded_pixel_collision": (
            not decoded_pixel_collisions
        ),
        (
            "no_cross_split_patient_id_"
            "collision_when_available"
        ): (
            not patient_id_collisions
        ),
        "lesion_size_thresholds_training_only": (
            True
        ),
        "all_external_counts_correct": (
            external_count_checks
            == expected_external_counts
        ),
        "ph2_is_evaluation_only": all(
            row[
                "training_allowed"
            ] == "false"
            for row
            in ph2_external_rows
        ),
        "ima_primary_is_evaluation_only": all(
            row[
                "training_allowed"
            ] == "false"
            for row
            in ima_primary_external_rows
        ),
        "isic2017_is_evaluation_only": all(
            row[
                "training_allowed"
            ] == "false"
            for row
            in isic2017_robustness_rows
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    leakage_report = {
        "stage": (
            "04_cross_split_"
            "leakage_verification"
        ),
        "split_protocol": (
            "official_isic2018_release"
        ),
        "image_id_collisions": (
            image_id_collisions
        ),
        "decoded_pixel_collisions": (
            decoded_pixel_collisions
        ),
        "patient_id_available": (
            patient_id_available
        ),
        "nonempty_patient_id_rows": len(
            patient_id_values
        ),
        "patient_id_collisions": (
            patient_id_collisions
        ),
        "patient_level_limitation": (
            (
                "Patient/group identifiers were "
                "unavailable in the audited manifest; "
                "leakage verification therefore "
                "covers image IDs and decoded image "
                "content while preserving the "
                "official release partitions."
            )
            if not patient_id_available
            else (
                "Patient/group identifiers were "
                "available and checked for "
                "cross-split overlap."
            )
        ),
        "all_checks_passed": (
            not image_id_collisions
            and not decoded_pixel_collisions
            and not patient_id_collisions
        ),
    }

    leakage_report_path = (
        report_directory
        / (
            "step04_cross_split_"
            "leakage_report.json"
        )
    )

    leakage_report_path.write_text(
        json.dumps(
            leakage_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "stage": (
            "04_fixed_split_creation"
        ),
        "protocol_version": (
            SPLIT_PROTOCOL_VERSION
        ),
        "decision": {
            "isic2018_split_source": (
                "official release partitions"
            ),
            "random_70_15_15_used": False,
            "reason": (
                "The audited dataset contains the "
                "official ISIC 2018 training, "
                "validation, and test releases. "
                "The random 70/15/15 rule is only "
                "a fallback when official "
                "partitions are unavailable."
            ),
            "official_test_role": (
                "internal_test"
            ),
            "model_selection_uses": (
                "official validation only"
            ),
            "internal_test_is_untouched": (
                True
            ),
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
            "ph2_manifest": {
                "path": str(
                    ph2_manifest_path
                ),
                "sha256": sha256_file(
                    ph2_manifest_path
                ),
            },
            "isic2017_manifest": {
                "path": str(
                    isic2017_manifest_path
                ),
                "sha256": sha256_file(
                    isic2017_manifest_path
                ),
            },
            "imaplusplus_primary_clean": {
                "path": str(
                    ima_primary_path
                ),
                "sha256": sha256_file(
                    ima_primary_path
                ),
            },
            (
                "imaplusplus_"
                "tier1_only_sensitivity"
            ): {
                "path": str(
                    ima_sensitivity_path
                ),
                "sha256": sha256_file(
                    ima_sensitivity_path
                ),
            },
        },
        "isic2018": {
            "total_unique_images": len(
                locked_isic_rows
            ),
            "split_counts": (
                observed_split_counts
            ),
            "fractions": {
                split: round(
                    count
                    / len(
                        locked_isic_rows
                    ),
                    8,
                )
                for split, count
                in observed_split_counts.items()
            },
            (
                "lesion_size_thresholds_"
                "from_training_only"
            ): {
                (
                    "small_max_"
                    "foreground_ratio"
                ): lower_threshold,
                (
                    "medium_max_"
                    "foreground_ratio"
                ): upper_threshold,
                "definition": {
                    "small": (
                        "ratio <= training "
                        "1/3 quantile"
                    ),
                    "medium": (
                        "training 1/3 quantile "
                        "< ratio <= training "
                        "2/3 quantile"
                    ),
                    "large": (
                        "ratio > training "
                        "2/3 quantile"
                    ),
                },
            },
            "split_row_set_sha256": {
                split: row_set_sha256(
                    rows
                )
                for split, rows
                in split_rows.items()
            },
            "patient_id_available": (
                patient_id_available
            ),
        },
        "fixed_evaluation_cohorts": {
            "ph2_strict_external": {
                "rows": len(
                    ph2_external_rows
                ),
                "unique_images": (
                    unique_image_count(
                        ph2_external_rows
                    )
                ),
            },
            (
                "imaplusplus_"
                "primary_supplementary"
            ): {
                "rows": len(
                    ima_primary_external_rows
                ),
                "unique_images": (
                    unique_image_count(
                        ima_primary_external_rows
                    )
                ),
            },
            (
                "imaplusplus_"
                "tier1_only_sensitivity"
            ): {
                "rows": len(
                    ima_sensitivity_external_rows
                ),
                "unique_images": (
                    unique_image_count(
                        ima_sensitivity_external_rows
                    )
                ),
            },
            (
                "isic2017_cross_year_"
                "robustness"
            ): {
                "rows": len(
                    isic2017_robustness_rows
                ),
                "unique_images": (
                    unique_image_count(
                        isic2017_robustness_rows
                    )
                ),
            },
        },
        "leakage_verification": {
            "report_path": str(
                leakage_report_path
            ),
            "image_id_collision_count": (
                len(
                    image_id_collisions
                )
            ),
            (
                "decoded_pixel_"
                "collision_count"
            ): len(
                decoded_pixel_collisions
            ),
            "patient_id_collision_count": (
                len(
                    patient_id_collisions
                )
            ),
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            name: {
                "path": str(
                    path
                ),
                "sha256": (
                    output_hashes[
                        name
                    ]
                ),
            }
            for name, path
            in output_paths.items()
        },
        "training_allowed": False,
        "training_block_reason": (
            "Derived contour, signed-distance, "
            "and boundary-band targets and their "
            "visual/numerical quality control "
            "remain incomplete."
        ),
    }

    summary_path = (
        report_directory
        / "step04_split_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Step 04 Results ==="
    )

    print(
        "ISIC 2018 train images          : "
        f"{len(split_rows['train'])}"
    )

    print(
        "ISIC 2018 validation images     : "
        f"{len(split_rows['val'])}"
    )

    print(
        "ISIC 2018 internal-test images  : "
        f"{len(split_rows['internal_test'])}"
    )

    print(
        "Cross-split ID collisions       : "
        f"{len(image_id_collisions)}"
    )

    print(
        "Cross-split pixel collisions    : "
        f"{len(decoded_pixel_collisions)}"
    )

    print(
        "Patient identifiers available   : "
        f"{patient_id_available}"
    )

    print(
        "PH2 strict-external images      : "
        f"{unique_image_count(ph2_external_rows)}"
    )

    print(
        "IMA++ primary images / rows     : "
        f"{unique_image_count(ima_primary_external_rows)}"
        " / "
        f"{len(ima_primary_external_rows)}"
    )

    print(
        "IMA++ sensitivity images / rows : "
        f"{unique_image_count(ima_sensitivity_external_rows)}"
        " / "
        f"{len(ima_sensitivity_external_rows)}"
    )

    print(
        "ISIC 2017 robustness images     : "
        f"{unique_image_count(isic2017_robustness_rows)}"
    )

    print(
        "All validation checks passed    : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")

    for path in output_paths.values():
        print(
            f" - {path}"
        )

    print(
        f" - {leakage_report_path}"
    )

    print(
        f" - {summary_path}"
    )

    print(
        "\nNo persistent input artifact "
        "was modified."
    )

    print(
        "Training remains blocked until "
        "target generation and QC pass."
    )

    return (
        0
        if all_checks_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())