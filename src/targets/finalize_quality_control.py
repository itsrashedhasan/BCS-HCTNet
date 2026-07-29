"""Step 05C: Finalize derived-target QC and unlock model development.

This CPU-only stage consolidates:

- the locked Step 04 split protocol;
- the persistent Step 05A generated targets;
- the persistent Step 05B numerical QC results;
- the completed visual review decision;
- the high-foreground annotation audit.

The official ISIC 2018 internal-test set remains unchanged for primary
evaluation. Two additional sensitivity cohorts are created before training:

1. exclude only the exactly full-foreground mask;
2. exclude every mask with foreground ratio greater than or equal to 0.98.

These sensitivity cohorts supplement rather than replace the official primary
evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.analyze_overlap import normalize_image_id, read_csv_rows
from src.targets.generate_targets import find_step04_split_directory
from src.targets.quality_control import find_step05a_artifact
from src.targets.target_geometry import sha256_file
from src.utils import config


FINALIZATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-target-qc-finalization-v1"
)

EXPECTED_TOTAL_IMAGES = 3694

EXPECTED_SPLIT_COUNTS = {
    "train": 2594,
    "val": 100,
    "internal_test": 1000,
}

EXPECTED_FULL_FOREGROUND_ID = "ISIC_0023056"

HIGH_FOREGROUND_THRESHOLD = 0.98

EXPECTED_COUNTS = {
    "foreground_ratio_ge_0_95": 24,
    "foreground_ratio_ge_0_98": 11,
    "foreground_ratio_ge_0_99": 7,
    "exact_full_foreground": 1,
    "train_foreground_ratio_ge_0_98": 3,
    "val_foreground_ratio_ge_0_98": 0,
    "internal_test_foreground_ratio_ge_0_98": 8,
}


def find_step05b_artifact() -> tuple[Path, Path, Path]:
    """Locate the persistent Step 05B QC artifact."""

    input_root = Path("/kaggle/input")

    candidates: list[
        tuple[Path, Path, Path]
    ] = []

    for report_path in input_root.rglob(
        "step05b_target_qc_report.json"
    ):
        artifact_root = (
            report_path.parents[2]
        )

        numerical_qc_path = (
            artifact_root
            / "data"
            / "manifests"
            / "step05b_target_numerical_qc.csv"
        )

        visual_index_path = (
            artifact_root
            / "data"
            / "manifests"
            / "step05b_visual_qc_samples.csv"
        )

        figure_directory = (
            artifact_root
            / "outputs"
            / "figures"
            / "step05b_target_qc"
        )

        if (
            numerical_qc_path.is_file()
            and visual_index_path.is_file()
            and figure_directory.is_dir()
        ):
            candidates.append(
                (
                    artifact_root.resolve(),
                    numerical_qc_path.resolve(),
                    visual_index_path.resolve(),
                )
            )

    unique_candidates = sorted(
        set(candidates),
        key=lambda item: str(item[0]),
    )

    preferred = [
        candidate
        for candidate in unique_candidates
        if (
            "bcs-hctnet-step05b-qc-artifacts"
            in str(candidate[0]).lower()
        )
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(unique_candidates) == 1:
        return unique_candidates[0]

    if not unique_candidates:
        raise FileNotFoundError(
            "Could not locate the persistent "
            "Step 05B QC artifact."
        )

    raise RuntimeError(
        "Multiple Step 05B QC artifacts were found: "
        f"{[str(item[0]) for item in unique_candidates]}"
    )


def parse_boolean(value: object) -> bool:
    """Parse persisted Boolean values safely."""

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
        "",
        "nan",
    }:
        return False

    raise ValueError(
        "Could not parse Boolean value "
        f"{value!r}."
    )


def write_csv(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
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
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def stable_row_hash(
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    """Hash selected manifest values deterministically."""

    digest = hashlib.sha256()

    for row in rows:
        line = "\t".join(
            str(row.get(field, ""))
            for field in fieldnames
        )

        digest.update(
            line.encode("utf-8")
        )

        digest.update(b"\n")

    return digest.hexdigest()


def main() -> int:
    """Finalize target QC and create evaluation sensitivity cohorts."""

    config.ensure_all_dirs()

    split_directory = (
        find_step04_split_directory()
    )

    (
        step05a_artifact_root,
        target_manifest_path,
        target_root,
    ) = find_step05a_artifact()

    (
        step05b_artifact_root,
        numerical_qc_path,
        visual_index_path,
    ) = find_step05b_artifact()

    step05b_report_path = (
        step05b_artifact_root
        / "outputs"
        / "reports"
        / "step05b_target_qc_report.json"
    )

    target_rows = read_csv_rows(
        target_manifest_path
    )

    numerical_rows = read_csv_rows(
        numerical_qc_path
    )

    visual_rows = read_csv_rows(
        visual_index_path
    )

    step05b_report = json.loads(
        step05b_report_path.read_text(
            encoding="utf-8"
        )
    )

    if len(target_rows) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Unexpected Step 05A target count: "
            f"{len(target_rows)}."
        )

    if len(numerical_rows) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Unexpected Step 05B numerical-QC count: "
            f"{len(numerical_rows)}."
        )

    if len(visual_rows) != 29:
        raise RuntimeError(
            "Expected 29 visual-review samples, "
            f"found {len(visual_rows)}."
        )

    contact_sheet_directory = (
        step05b_artifact_root
        / "outputs"
        / "figures"
        / "step05b_target_qc"
    )

    contact_sheets = sorted(
        contact_sheet_directory.glob(
            "target_qc_sheet_*.jpg"
        )
    )

    if len(contact_sheets) != 8:
        raise RuntimeError(
            "Expected 8 contact sheets, "
            f"found {len(contact_sheets)}."
        )

    if not bool(
        step05b_report.get(
            "automatic_qc_passed",
            False,
        )
    ):
        raise RuntimeError(
            "Step 05B automatic QC did not pass."
        )

    numerical_by_id = {
        normalize_image_id(
            row["image_id"]
        ): row
        for row in numerical_rows
    }

    if (
        len(numerical_by_id)
        != EXPECTED_TOTAL_IMAGES
    ):
        raise RuntimeError(
            "Step 05B numerical QC contains "
            "duplicate image IDs."
        )

    merged_rows: list[
        dict[str, Any]
    ] = []

    for target_row in target_rows:
        image_id = normalize_image_id(
            target_row["image_id"]
        )

        if image_id not in numerical_by_id:
            raise RuntimeError(
                "Missing numerical-QC result for "
                f"{image_id}."
            )

        qc_row = numerical_by_id[
            image_id
        ]

        if not parse_boolean(
            qc_row.get(
                "sample_passed",
                False,
            )
        ):
            raise RuntimeError(
                "Numerical target QC failed for "
                f"{image_id}."
            )

        split = str(
            target_row["split"]
        )

        foreground_ratio = float(
            target_row[
                "mask_foreground_ratio"
            ]
        )

        source_foreground_ratio = float(
            target_row[
                "source_mask_foreground_ratio"
            ]
        )

        full_foreground = parse_boolean(
            target_row[
                "mask_is_full_foreground"
            ]
        )

        background_pixels = int(
            round(
                (
                    1.0
                    - foreground_ratio
                )
                * 352
                * 352
            )
        )

        merged_rows.append(
            {
                **target_row,
                "image_id": image_id,
                "split": split,
                "source_mask_foreground_ratio": (
                    source_foreground_ratio
                ),
                "mask_foreground_ratio": (
                    foreground_ratio
                ),
                "foreground_ratio_change_after_resize": (
                    foreground_ratio
                    - source_foreground_ratio
                ),
                "background_pixels_at_352": (
                    background_pixels
                ),
                "mask_is_full_foreground": (
                    full_foreground
                ),
                "border_touching": parse_boolean(
                    qc_row.get(
                        "border_touching",
                        False,
                    )
                ),
                "connected_component_count": int(
                    qc_row[
                        "connected_component_count"
                    ]
                ),
                "sample_passed": True,
                "foreground_ratio_ge_0_95": (
                    foreground_ratio
                    >= 0.95
                ),
                "foreground_ratio_ge_0_98": (
                    foreground_ratio
                    >= HIGH_FOREGROUND_THRESHOLD
                ),
                "foreground_ratio_ge_0_99": (
                    foreground_ratio
                    >= 0.99
                ),
            }
        )

    split_counts = dict(
        Counter(
            str(row["split"])
            for row in merged_rows
        )
    )

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Unexpected merged split counts: "
            f"{split_counts}."
        )

    exact_full_rows = [
        row
        for row in merged_rows
        if bool(
            row[
                "mask_is_full_foreground"
            ]
        )
    ]

    exact_full_ids = sorted(
        str(row["image_id"])
        for row in exact_full_rows
    )

    if exact_full_ids != [
        EXPECTED_FULL_FOREGROUND_ID
    ]:
        raise RuntimeError(
            "Unexpected full-foreground IDs: "
            f"{exact_full_ids}."
        )

    threshold_counts = {
        "foreground_ratio_ge_0_95": sum(
            bool(
                row[
                    "foreground_ratio_ge_0_95"
                ]
            )
            for row in merged_rows
        ),
        "foreground_ratio_ge_0_98": sum(
            bool(
                row[
                    "foreground_ratio_ge_0_98"
                ]
            )
            for row in merged_rows
        ),
        "foreground_ratio_ge_0_99": sum(
            bool(
                row[
                    "foreground_ratio_ge_0_99"
                ]
            )
            for row in merged_rows
        ),
        "exact_full_foreground": len(
            exact_full_rows
        ),
        "train_foreground_ratio_ge_0_98": sum(
            (
                row["split"] == "train"
                and bool(
                    row[
                        "foreground_ratio_ge_0_98"
                    ]
                )
            )
            for row in merged_rows
        ),
        "val_foreground_ratio_ge_0_98": sum(
            (
                row["split"] == "val"
                and bool(
                    row[
                        "foreground_ratio_ge_0_98"
                    ]
                )
            )
            for row in merged_rows
        ),
        "internal_test_foreground_ratio_ge_0_98": sum(
            (
                row["split"]
                == "internal_test"
                and bool(
                    row[
                        "foreground_ratio_ge_0_98"
                    ]
                )
            )
            for row in merged_rows
        ),
    }

    if threshold_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "High-foreground audit counts changed: "
            f"{threshold_counts}."
        )

    internal_test_rows = sorted(
        (
            row
            for row in merged_rows
            if (
                row["split"]
                == "internal_test"
            )
        ),
        key=lambda row: str(
            row["image_id"]
        ),
    )

    internal_test_excluding_full = [
        row
        for row in internal_test_rows
        if not bool(
            row[
                "mask_is_full_foreground"
            ]
        )
    ]

    internal_test_excluding_ge_098 = [
        row
        for row in internal_test_rows
        if not bool(
            row[
                "foreground_ratio_ge_0_98"
            ]
        )
    ]

    if len(internal_test_rows) != 1000:
        raise RuntimeError(
            "Primary internal-test cohort "
            "must contain 1,000 images."
        )

    if (
        len(
            internal_test_excluding_full
        )
        != 999
    ):
        raise RuntimeError(
            "Full-foreground sensitivity "
            "cohort must contain 999 images."
        )

    if (
        len(
            internal_test_excluding_ge_098
        )
        != 992
    ):
        raise RuntimeError(
            "High-foreground sensitivity "
            "cohort must contain 992 images."
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

    audit_fieldnames = [
        "image_id",
        "split",
        "source_mask_foreground_ratio",
        "mask_foreground_ratio",
        "foreground_ratio_change_after_resize",
        "background_pixels_at_352",
        "mask_is_full_foreground",
        "foreground_ratio_ge_0_95",
        "foreground_ratio_ge_0_98",
        "foreground_ratio_ge_0_99",
        "border_touching",
        "connected_component_count",
        "sample_passed",
    ]

    high_foreground_audit_rows = sorted(
        (
            row
            for row in merged_rows
            if bool(
                row[
                    "foreground_ratio_ge_0_95"
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

    high_foreground_audit_path = (
        manifest_directory
        / (
            "step05c_high_foreground_"
            "mask_audit.csv"
        )
    )

    write_csv(
        high_foreground_audit_path,
        high_foreground_audit_rows,
        audit_fieldnames,
    )

    sensitivity_fieldnames = list(
        target_rows[0].keys()
    )

    primary_manifest_path = (
        manifest_directory
        / (
            "isic2018_internal_test_"
            "primary_1000.csv"
        )
    )

    excluding_full_path = (
        manifest_directory
        / (
            "isic2018_internal_test_"
            "sensitivity_excluding_"
            "full_foreground_999.csv"
        )
    )

    excluding_ge_098_path = (
        manifest_directory
        / (
            "isic2018_internal_test_"
            "sensitivity_excluding_"
            "foreground_ge_0_98_992.csv"
        )
    )

    write_csv(
        primary_manifest_path,
        internal_test_rows,
        sensitivity_fieldnames,
    )

    write_csv(
        excluding_full_path,
        internal_test_excluding_full,
        sensitivity_fieldnames,
    )

    write_csv(
        excluding_ge_098_path,
        internal_test_excluding_ge_098,
        sensitivity_fieldnames,
    )

    checks = {
        "step04_locked_splits_available": (
            split_directory.is_dir()
        ),
        "step05a_targets_available": (
            target_root.is_dir()
        ),
        "step05b_automatic_qc_passed": True,
        "all_3694_samples_passed_numerical_qc": (
            len(merged_rows)
            == EXPECTED_TOTAL_IMAGES
        ),
        "visual_review_sample_count_correct": (
            len(visual_rows) == 29
        ),
        "visual_review_contact_sheet_count_correct": (
            len(contact_sheets) == 8
        ),
        "visual_review_passed": True,
        "full_foreground_case_documented": (
            exact_full_ids
            == [
                EXPECTED_FULL_FOREGROUND_ID
            ]
        ),
        "official_internal_test_preserved": (
            len(internal_test_rows)
            == 1000
        ),
        "full_foreground_sensitivity_created": (
            len(
                internal_test_excluding_full
            )
            == 999
        ),
        "high_foreground_sensitivity_created": (
            len(
                internal_test_excluding_ge_098
            )
            == 992
        ),
        "validation_has_no_ge_0_98_masks": (
            threshold_counts[
                "val_foreground_ratio_ge_0_98"
            ]
            == 0
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    training_readiness = {
        "status": (
            "training_unblocked"
            if all_checks_passed
            else "training_blocked"
        ),
        "training_allowed": bool(
            all_checks_passed
        ),
        "approved_target_resolution": [
            352,
            352,
        ],
        "approved_primary_development_split": (
            "official ISIC 2018 train"
        ),
        "approved_model_selection_split": (
            "official ISIC 2018 validation"
        ),
        "approved_primary_internal_test": {
            "manifest": str(
                primary_manifest_path
            ),
            "images": 1000,
        },
        "required_internal_test_sensitivity_analyses": [
            {
                "name": (
                    "excluding_exact_full_foreground"
                ),
                "manifest": str(
                    excluding_full_path
                ),
                "images": 999,
            },
            {
                "name": (
                    "excluding_foreground_ratio_ge_0_98"
                ),
                "manifest": str(
                    excluding_ge_098_path
                ),
                "images": 992,
            },
        ],
        "prohibited_actions": [
            (
                "Do not modify the official "
                "ground-truth masks."
            ),
            (
                "Do not replace the 1,000-image "
                "official primary internal test."
            ),
            (
                "Do not use internal-test results "
                "for model selection or tuning."
            ),
            (
                "Do not omit the prespecified "
                "sensitivity analyses from final "
                "evaluation reporting."
            ),
        ],
    }

    training_readiness_path = (
        report_directory
        / "TRAINING_READINESS.json"
    )

    training_readiness_path.write_text(
        json.dumps(
            training_readiness,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "stage": (
            "05C_final_target_qc_signoff"
        ),
        "finalization_protocol_version": (
            FINALIZATION_PROTOCOL_VERSION
        ),
        "visual_review": {
            "status": "passed",
            "sheets_reviewed": 8,
            "samples_reviewed": 29,
            "criteria": [
                "image-mask spatial alignment",
                "contour placement",
                "boundary-band geometry",
                "signed-distance polarity",
                "multi-component masks",
                "border-touching masks",
                "full-foreground special case",
            ],
            "finding": (
                "No target-generation or "
                "image-mask alignment defect "
                "was identified."
            ),
        },
        "annotation_audit": {
            "high_foreground_threshold": (
                HIGH_FOREGROUND_THRESHOLD
            ),
            "threshold_counts": (
                threshold_counts
            ),
            "decision": (
                "Preserve every official sample "
                "and supplement the primary test "
                "with prespecified sensitivity "
                "analyses."
            ),
        },
        "evaluation_protocol": {
            "primary_internal_test_images": 1000,
            (
                "sensitivity_excluding_"
                "full_foreground_images"
            ): 999,
            (
                "sensitivity_excluding_"
                "foreground_ge_0_98_images"
            ): 992,
            "model_selection_uses_internal_test": (
                False
            ),
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "training_allowed": bool(
            all_checks_passed
        ),
        "inputs": {
            "step04_split_directory": str(
                split_directory
            ),
            "step05a_artifact_root": str(
                step05a_artifact_root
            ),
            "step05b_artifact_root": str(
                step05b_artifact_root
            ),
            "target_manifest_sha256": (
                sha256_file(
                    target_manifest_path
                )
            ),
            "numerical_qc_sha256": (
                sha256_file(
                    numerical_qc_path
                )
            ),
            "visual_index_sha256": (
                sha256_file(
                    visual_index_path
                )
            ),
        },
        "outputs": {
            "high_foreground_audit": str(
                high_foreground_audit_path
            ),
            "primary_internal_test_manifest": str(
                primary_manifest_path
            ),
            (
                "sensitivity_excluding_"
                "full_foreground_manifest"
            ): str(
                excluding_full_path
            ),
            (
                "sensitivity_excluding_"
                "foreground_ge_0_98_manifest"
            ): str(
                excluding_ge_098_path
            ),
            "training_readiness": str(
                training_readiness_path
            ),
        },
        "manifest_row_hashes": {
            "primary_internal_test": (
                stable_row_hash(
                    internal_test_rows,
                    [
                        "image_id",
                        "split",
                        "mask_foreground_ratio",
                    ],
                )
            ),
            (
                "sensitivity_excluding_"
                "full_foreground"
            ): stable_row_hash(
                internal_test_excluding_full,
                [
                    "image_id",
                    "split",
                    "mask_foreground_ratio",
                ],
            ),
            (
                "sensitivity_excluding_"
                "foreground_ge_0_98"
            ): stable_row_hash(
                internal_test_excluding_ge_098,
                [
                    "image_id",
                    "split",
                    "mask_foreground_ratio",
                ],
            ),
        },
    }

    report_path = (
        report_directory
        / (
            "step05c_target_qc_"
            "signoff_report.json"
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
        "=== Step 05C: Final Target-QC "
        "Sign-Off ==="
    )

    print(
        "Target samples numerically passed : "
        f"{len(merged_rows)}"
    )

    print(
        "Visual-review samples passed      : "
        f"{len(visual_rows)}"
    )

    print(
        "Visual contact sheets reviewed    : "
        f"{len(contact_sheets)}"
    )

    print(
        "Masks with foreground >= 0.98     : "
        f"{threshold_counts['foreground_ratio_ge_0_98']}"
    )

    print(
        "Internal-test masks >= 0.98       : "
        f"{threshold_counts['internal_test_foreground_ratio_ge_0_98']}"
    )

    print(
        "Exactly full masks                : "
        f"{threshold_counts['exact_full_foreground']}"
    )

    print(
        "Primary internal-test cohort      : "
        f"{len(internal_test_rows)}"
    )

    print(
        "Exclude-full sensitivity cohort   : "
        f"{len(internal_test_excluding_full)}"
    )

    print(
        "Exclude->=0.98 sensitivity cohort  : "
        f"{len(internal_test_excluding_ge_098)}"
    )

    print(
        "All final validation checks passed: "
        f"{all_checks_passed}"
    )

    print(
        "Training allowed                  : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")

    for path in [
        high_foreground_audit_path,
        primary_manifest_path,
        excluding_full_path,
        excluding_ge_098_path,
        training_readiness_path,
        report_path,
    ]:
        print(f" - {path}")

    return (
        0
        if all_checks_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())