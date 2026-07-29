"""Fully validate the five controlled baseline experiment configurations.

This script performs artifact-aware validation for:

- E01: U-Net;
- E02: UNet++;
- E03: DeepLabV3+;
- E04: TransUNet;
- E05: Swin-UNet.

It verifies that every configuration resolves the same approved manifests,
artifact roots, data protocol, augmentation protocol, optimizer/scheduler
policy, evaluation safeguards, reproducibility settings, and smoke-test
limits. Only the model identity, model parameters, experiment identity, and
output root are allowed to differ.

The script is intentionally CPU-only. It validates configuration and mounted
artifacts; it does not construct a model or start training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.train.baseline_experiment_config import (
    BASELINE_CONFIG_VALIDATION_PROTOCOL_VERSION,
    BASELINE_EXPERIMENT_MODEL_MAP,
    ValidatedBaselineExperimentConfig,
    save_baseline_validation_bundle,
    summarize_validated_baseline_configuration,
    validate_baseline_experiment_configuration,
)


BASELINE_BATCH_VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-baseline-batch-validation-v1"
)

DEFAULT_CONFIG_FILENAMES = {
    "E01": "E01_unet.yaml",
    "E02": "E02_unetpp.yaml",
    "E03": "E03_deeplabv3plus.yaml",
    "E04": "E04_transunet.yaml",
    "E05": "E05_swin_unet.yaml",
}

DEFAULT_CONFIG_DIRECTORY = Path(
    "configs/experiments"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "/kaggle/working/outputs/step07a_baseline_config_validation"
)

COMMON_VALIDATION_SECTIONS = (
    "training_readiness",
    "data",
    "augmentation",
    "training",
    "inference",
    "evaluation",
    "reproducibility",
    "smoke_test",
)

SUMMARY_CSV_FIELDS = (
    "experiment_id",
    "experiment_name",
    "model_name",
    "source_path",
    "source_file_sha256",
    "canonical_sha256",
    "train_rows",
    "validation_rows",
    "derived_target_rows",
    "training_readiness_status",
    "output_root",
)


def _atomic_write_text(
    path: Path,
    text: str,
) -> Path:
    """Atomically write UTF-8 text."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        text,
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return path


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically save a formatted JSON mapping."""

    return _atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )


def _canonical_sha256(
    value: object,
) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        serialized
    ).hexdigest()


def _require_complete_config_map(
    config_directory: str | Path,
) -> dict[str, Path]:
    """Resolve and validate the five expected configuration paths."""

    resolved_directory = Path(
        config_directory
    ).expanduser().resolve()

    if not resolved_directory.is_dir():
        raise FileNotFoundError(
            "Baseline configuration directory not found: "
            f"{resolved_directory}"
        )

    resolved_paths = {
        experiment_id: (
            resolved_directory
            / filename
        )
        for experiment_id, filename
        in DEFAULT_CONFIG_FILENAMES.items()
    }

    missing = [
        str(path)
        for path in resolved_paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing baseline configuration files: "
            f"{missing}"
        )

    return resolved_paths


def _normalized_artifact_roots(
    validated: ValidatedBaselineExperimentConfig,
) -> dict[str, str]:
    """Return artifact roots as a comparable mapping."""

    return {
        name: str(path)
        for name, path
        in sorted(
            validated.artifact_roots.items()
        )
    }


def _normalized_manifests(
    validated: ValidatedBaselineExperimentConfig,
) -> dict[str, dict[str, Any]]:
    """Return manifest provenance needed for cross-config comparison."""

    return {
        name: {
            "artifact_name": manifest.artifact_name,
            "relative_path": manifest.relative_path,
            "absolute_path": str(
                manifest.absolute_path
            ),
            "expected_rows": manifest.expected_rows,
            "observed_rows": manifest.observed_rows,
            "sha256": manifest.sha256,
        }
        for name, manifest
        in sorted(
            validated.manifests.items()
        )
    }


def _common_protocol_payload(
    validated: ValidatedBaselineExperimentConfig,
) -> dict[str, Any]:
    """Extract protocol sections that must be identical across baselines."""

    summary = validated.validation_summary

    return {
        section: summary[section]
        for section in COMMON_VALIDATION_SECTIONS
    }


def _assert_common_protocols(
    validated_by_id: Mapping[
        str,
        ValidatedBaselineExperimentConfig,
    ],
) -> dict[str, Any]:
    """Require all non-model experimental protocols to match."""

    reference_id = "E01"
    reference = validated_by_id[
        reference_id
    ]

    reference_artifacts = (
        _normalized_artifact_roots(
            reference
        )
    )

    reference_manifests = (
        _normalized_manifests(
            reference
        )
    )

    reference_common = (
        _common_protocol_payload(
            reference
        )
    )

    for experiment_id, validated in (
        validated_by_id.items()
    ):
        if experiment_id == reference_id:
            continue

        observed_artifacts = (
            _normalized_artifact_roots(
                validated
            )
        )

        if observed_artifacts != reference_artifacts:
            raise RuntimeError(
                "Artifact roots differ between "
                f"{reference_id} and {experiment_id}."
            )

        observed_manifests = (
            _normalized_manifests(
                validated
            )
        )

        if observed_manifests != reference_manifests:
            raise RuntimeError(
                "Resolved manifest provenance differs between "
                f"{reference_id} and {experiment_id}."
            )

        observed_common = (
            _common_protocol_payload(
                validated
            )
        )

        if observed_common != reference_common:
            mismatched_sections = [
                section
                for section
                in COMMON_VALIDATION_SECTIONS
                if observed_common[section]
                != reference_common[section]
            ]

            raise RuntimeError(
                "Controlled baseline protocol mismatch between "
                f"{reference_id} and {experiment_id}: "
                f"{mismatched_sections}."
            )

    return {
        "reference_experiment": reference_id,
        "artifact_roots_sha256": _canonical_sha256(
            reference_artifacts
        ),
        "manifest_provenance_sha256": _canonical_sha256(
            reference_manifests
        ),
        "common_protocol_sha256": _canonical_sha256(
            reference_common
        ),
        "common_sections": list(
            COMMON_VALIDATION_SECTIONS
        ),
        "all_common_protocols_identical": True,
    }


def _validate_experiment_assignments(
    validated_by_id: Mapping[
        str,
        ValidatedBaselineExperimentConfig,
    ],
) -> dict[str, str]:
    """Verify E01-E05 model assignments and unique output roots."""

    observed_ids = set(
        validated_by_id
    )

    expected_ids = set(
        BASELINE_EXPERIMENT_MODEL_MAP
    )

    if observed_ids != expected_ids:
        raise RuntimeError(
            "Validated baseline experiment IDs differ from the "
            f"approved map. Expected {sorted(expected_ids)}, "
            f"observed {sorted(observed_ids)}."
        )

    assignments: dict[str, str] = {}
    output_roots: list[str] = []

    for experiment_id in sorted(
        validated_by_id
    ):
        validated = validated_by_id[
            experiment_id
        ]

        experiment = validated.validation_summary[
            "experiment"
        ]

        model_and_loss = validated.validation_summary[
            "model_and_loss"
        ]

        outputs = validated.validation_summary[
            "outputs"
        ]

        if experiment[
            "id"
        ] != experiment_id:
            raise RuntimeError(
                "Configuration identity mismatch for "
                f"{experiment_id}."
            )

        expected_model = (
            BASELINE_EXPERIMENT_MODEL_MAP[
                experiment_id
            ]
        )

        observed_model = model_and_loss[
            "model_name"
        ]

        if observed_model != expected_model:
            raise RuntimeError(
                f"{experiment_id} must use {expected_model!r}, "
                f"observed {observed_model!r}."
            )

        if model_and_loss[
            "output_keys"
        ] != [
            "mask_logits",
        ]:
            raise RuntimeError(
                f"{experiment_id} violates the mask-only output contract."
            )

        if model_and_loss[
            "uses_boundary_conditioning"
        ] is not False:
            raise RuntimeError(
                f"{experiment_id} unexpectedly enables boundary conditioning."
            )

        if model_and_loss[
            "uses_auxiliary_targets"
        ] is not False:
            raise RuntimeError(
                f"{experiment_id} unexpectedly enables auxiliary targets."
            )

        assignments[
            experiment_id
        ] = observed_model

        output_roots.append(
            outputs[
                "root"
            ]
        )

    if len(
        output_roots
    ) != len(
        set(
            output_roots
        )
    ):
        raise RuntimeError(
            "Baseline output roots must be unique."
        )

    return assignments


def _experiment_summary_row(
    validated: ValidatedBaselineExperimentConfig,
) -> dict[str, Any]:
    """Build one compact CSV summary row."""

    experiment = validated.validation_summary[
        "experiment"
    ]

    model_and_loss = validated.validation_summary[
        "model_and_loss"
    ]

    outputs = validated.validation_summary[
        "outputs"
    ]

    return {
        "experiment_id": experiment[
            "id"
        ],
        "experiment_name": experiment[
            "name"
        ],
        "model_name": model_and_loss[
            "model_name"
        ],
        "source_path": str(
            validated.source_path
        ),
        "source_file_sha256": (
            validated.source_file_sha256
        ),
        "canonical_sha256": (
            validated.canonical_sha256
        ),
        "train_rows": validated.manifests[
            "train"
        ].observed_rows,
        "validation_rows": validated.manifests[
            "validation"
        ].observed_rows,
        "derived_target_rows": validated.manifests[
            "derived_targets"
        ].observed_rows,
        "training_readiness_status": (
            validated.training_readiness[
                "status"
            ]
        ),
        "output_root": outputs[
            "root"
        ],
    }


def _write_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically write the compact experiment summary CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                SUMMARY_CSV_FIELDS
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row[field]
                    for field
                    in SUMMARY_CSV_FIELDS
                }
            )

    temporary_path.replace(
        path
    )

    return path


def validate_all_baseline_configurations(
    *,
    config_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Fully validate all five configurations and save reports."""

    config_paths = _require_complete_config_map(
        config_directory
    )

    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    resolved_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_by_id: dict[
        str,
        ValidatedBaselineExperimentConfig,
    ] = {}

    saved_bundles: dict[
        str,
        dict[str, str],
    ] = {}

    for expected_id in sorted(
        config_paths
    ):
        validated = (
            validate_baseline_experiment_configuration(
                config_paths[
                    expected_id
                ]
            )
        )

        observed_id = validated.validation_summary[
            "experiment"
        ][
            "id"
        ]

        if observed_id != expected_id:
            raise RuntimeError(
                "Filename-to-experiment mismatch: expected "
                f"{expected_id}, observed {observed_id}."
            )

        validated_by_id[
            expected_id
        ] = validated

        bundle_paths = (
            save_baseline_validation_bundle(
                validated,
                resolved_output_directory
                / expected_id,
            )
        )

        saved_bundles[
            expected_id
        ] = {
            name: str(path)
            for name, path
            in bundle_paths.items()
        }

        print(
            summarize_validated_baseline_configuration(
                validated
            )
        )

        print()

    assignments = (
        _validate_experiment_assignments(
            validated_by_id
        )
    )

    common_protocol = (
        _assert_common_protocols(
            validated_by_id
        )
    )

    summary_rows = [
        _experiment_summary_row(
            validated_by_id[
                experiment_id
            ]
        )
        for experiment_id
        in sorted(
            validated_by_id
        )
    ]

    summary_csv_path = _write_summary_csv(
        resolved_output_directory
        / "BASELINE_CONFIGURATIONS.csv",
        summary_rows,
    )

    report: dict[str, Any] = {
        "status": "passed",
        "protocol_version": (
            BASELINE_BATCH_VALIDATION_PROTOCOL_VERSION
        ),
        "single_config_validation_protocol_version": (
            BASELINE_CONFIG_VALIDATION_PROTOCOL_VERSION
        ),
        "config_directory": str(
            Path(
                config_directory
            ).expanduser().resolve()
        ),
        "output_directory": str(
            resolved_output_directory
        ),
        "validated_experiment_count": len(
            validated_by_id
        ),
        "experiment_model_assignments": (
            assignments
        ),
        "common_protocol": (
            common_protocol
        ),
        "experiments": {
            experiment_id: {
                "validation": validated.to_dict(),
                "saved_bundle": saved_bundles[
                    experiment_id
                ],
            }
            for experiment_id, validated
            in sorted(
                validated_by_id.items()
            )
        },
        "summary_csv": str(
            summary_csv_path
        ),
        "all_checks_passed": True,
    }

    report_path = _atomic_write_json(
        resolved_output_directory
        / "BASELINE_CONFIGURATION_VALIDATION.json",
        report,
    )

    report[
        "report_path"
    ] = str(
        report_path
    )

    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Fully validate all E01-E05 baseline configurations "
            "against mounted Step04, Step05A, and Step05C artifacts."
        )
    )

    parser.add_argument(
        "--config-directory",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
        help=(
            "Directory containing E01-E05 YAML files. "
            f"Default: {DEFAULT_CONFIG_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for validation reports. "
            f"Default: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run full E01-E05 baseline configuration validation."""

    arguments = build_argument_parser().parse_args(
        argv
    )

    report = validate_all_baseline_configurations(
        config_directory=(
            arguments.config_directory
        ),
        output_directory=(
            arguments.output_directory
        ),
    )

    print(
        json.dumps(
            {
                "status": report[
                    "status"
                ],
                "validated_experiment_count": report[
                    "validated_experiment_count"
                ],
                "experiment_model_assignments": report[
                    "experiment_model_assignments"
                ],
                "common_protocol": report[
                    "common_protocol"
                ],
                "summary_csv": report[
                    "summary_csv"
                ],
                "report_path": report[
                    "report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if report[
        "status"
    ] != "passed":
        return 1

    if report[
        "validated_experiment_count"
    ] != len(
        BASELINE_EXPERIMENT_MODEL_MAP
    ):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
