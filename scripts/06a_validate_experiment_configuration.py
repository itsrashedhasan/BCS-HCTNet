"""Step 06A: Validate the E00 experiment configuration.

Run from the repository root:

    python3 scripts/06a_validate_experiment_configuration.py

This CPU-only stage verifies:

- persistent Step 04, Step 05A, and Step 05C artifacts;
- Step 05C training authorization;
- all configured manifest row counts;
- target geometry and preprocessing settings;
- augmentation restrictions;
- model, loss, optimizer, and evaluation safeguards;
- reproducibility and smoke-test settings.

The validated configuration and report are written under
/kaggle/working/outputs/experiments/E00/configurations.

No persistent Kaggle input artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPOSITORY_ROOT),
)


from src.train.experiment_config import (
    save_validation_bundle,
    summarize_validated_configuration,
    validate_experiment_configuration,
)


DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "e00_bcs_hctnet_foundation.yaml"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "/kaggle/working/outputs/"
    "experiments/E00/configurations"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the BCS-HCTNet E00 "
            "foundation configuration."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the experiment YAML file. "
            "Defaults to the E00 foundation config."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for the validated config "
            "copy and validation report."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Validate and persist the E00 configuration."""

    arguments = parse_arguments()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    output_directory = (
        arguments.output_directory
        .expanduser()
        .resolve()
    )

    print(
        "=== Step 06A: Validate "
        "Experiment Configuration ==="
    )

    print(
        "Repository root :",
        REPOSITORY_ROOT,
    )

    print(
        "Configuration   :",
        config_path,
    )

    print(
        "Output directory:",
        output_directory,
    )

    print(
        "Execution mode  : CPU-only validation"
    )

    validated = (
        validate_experiment_configuration(
            config_path
        )
    )

    saved_paths = save_validation_bundle(
        validated,
        output_directory,
    )

    print(
        "\n=== Validation Results ==="
    )

    print(
        summarize_validated_configuration(
            validated
        )
    )

    print("\nResolved artifact roots:")

    for name, path in (
        validated.artifact_roots.items()
    ):
        print(
            f" - {name}: {path}"
        )

    print("\nSaved outputs:")

    for name, path in (
        saved_paths.items()
    ):
        print(
            f" - {name}: {path}"
        )

    validation_report_path = (
        saved_paths[
            "validation_report"
        ]
    )

    report = json.loads(
        validation_report_path.read_text(
            encoding="utf-8"
        )
    )

    summary = report.get(
        "validation_summary",
        {},
    )

    if (
        summary.get(
            "all_checks_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Configuration validation report "
            "does not show all checks passed."
        )

    if (
        report.get(
            "training_readiness",
            {},
        ).get(
            "training_allowed"
        )
        is not True
    ):
        raise RuntimeError(
            "Validated configuration does not "
            "authorize training."
        )

    manifest_count = len(
        report.get(
            "manifests",
            {},
        )
    )

    if manifest_count != 6:
        raise RuntimeError(
            "Expected six validated manifests, "
            f"found {manifest_count}."
        )

    print(
        "\nStep 06A configuration validation: "
        "PASSED"
    )

    print(
        "GPU required: False"
    )

    print(
        "Training executed: False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())