"""Step 06D: Package persistent data-foundation artifacts.

This CPU-only script packages the completed Step 06 reports:

- validated E00 configuration;
- configuration-validation report;
- manifest-schema inspection;
- real data-pipeline validation;
- package provenance;
- SHA-256 checksum inventory.

The resulting directory can be uploaded as a private Kaggle Dataset so the
completed data foundation remains available across Kaggle sessions.

Run from the repository root:

    python3 scripts/06d_package_data_foundation_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

PACKAGE_PROTOCOL_VERSION = (
    "BCS-HCTNet-step06-data-foundation-package-v1"
)

DEFAULT_CONFIGURATION_DIRECTORY = Path(
    "/kaggle/working/outputs/"
    "experiments/E00/configurations"
)

DEFAULT_DATA_FOUNDATION_DIRECTORY = Path(
    "/kaggle/working/outputs/"
    "experiments/E00/data_foundation"
)

DEFAULT_PACKAGE_DIRECTORY = Path(
    "/kaggle/working/"
    "bcs_hctnet_step06_data_foundation_artifacts"
)

EXPECTED_SOURCE_FILES = {
    "configuration": (
        DEFAULT_CONFIGURATION_DIRECTORY
        / "e00_bcs_hctnet_foundation.yaml"
    ),
    "configuration_validation": (
        DEFAULT_CONFIGURATION_DIRECTORY
        / "CONFIG_VALIDATION.json"
    ),
    "manifest_schema_inspection": (
        DEFAULT_DATA_FOUNDATION_DIRECTORY
        / "MANIFEST_SCHEMA_INSPECTION.json"
    ),
    "data_pipeline_validation": (
        DEFAULT_DATA_FOUNDATION_DIRECTORY
        / "DATA_PIPELINE_VALIDATION.json"
    ),
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Package completed Step 06 "
            "data-foundation artifacts."
        )
    )

    parser.add_argument(
        "--configuration-directory",
        type=Path,
        default=DEFAULT_CONFIGURATION_DIRECTORY,
        help=(
            "Directory containing the validated "
            "configuration and CONFIG_VALIDATION.json."
        ),
    )

    parser.add_argument(
        "--data-foundation-directory",
        type=Path,
        default=DEFAULT_DATA_FOUNDATION_DIRECTORY,
        help=(
            "Directory containing Step 06B and "
            "Step 06C reports."
        ),
    )

    parser.add_argument(
        "--package-directory",
        type=Path,
        default=DEFAULT_PACKAGE_DIRECTORY,
        help=(
            "Output directory to package for "
            "persistent Kaggle storage."
        ),
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json_mapping(
    path: Path,
    context: str,
) -> dict[str, Any]:
    """Load and require a JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{context} not found: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{context} is invalid JSON: {path}"
        ) from error

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            f"{context} must contain a JSON object."
        )

    return dict(payload)


def require_true(
    value: object,
    context: str,
) -> None:
    """Require a value to be exactly True."""

    if value is not True:
        raise RuntimeError(
            f"{context} must be true, "
            f"received {value!r}."
        )


def require_equal(
    observed: object,
    expected: object,
    context: str,
) -> None:
    """Require exact equality."""

    if observed != expected:
        raise RuntimeError(
            f"{context} mismatch: expected "
            f"{expected!r}, found {observed!r}."
        )


def git_commit_hash(
    repository_root: Path,
) -> str | None:
    """Return the current Git commit hash."""

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    commit_hash = result.stdout.strip()

    return commit_hash or None


def validate_reports(
    *,
    configuration_validation: Mapping[str, Any],
    manifest_inspection: Mapping[str, Any],
    pipeline_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all Step 06 reports before packaging."""

    configuration_summary = configuration_validation.get(
        "validation_summary",
        {},
    )

    if not isinstance(
        configuration_summary,
        Mapping,
    ):
        raise TypeError(
            "Configuration validation summary "
            "must be a mapping."
        )

    require_true(
        configuration_summary.get(
            "all_checks_passed"
        ),
        (
            "Configuration validation "
            "all_checks_passed"
        ),
    )

    training_readiness = configuration_validation.get(
        "training_readiness",
        {},
    )

    if not isinstance(
        training_readiness,
        Mapping,
    ):
        raise TypeError(
            "Training readiness must be a mapping."
        )

    require_true(
        training_readiness.get(
            "training_allowed"
        ),
        "Training readiness authorization",
    )

    require_equal(
        training_readiness.get(
            "status"
        ),
        "training_unblocked",
        "Training readiness status",
    )

    require_true(
        manifest_inspection.get(
            "all_checks_passed"
        ),
        (
            "Manifest inspection "
            "all_checks_passed"
        ),
    )

    require_true(
        pipeline_validation.get(
            "all_checks_passed"
        ),
        (
            "Data-pipeline validation "
            "all_checks_passed"
        ),
    )

    configuration_sha256 = configuration_validation.get(
        "canonical_sha256"
    )

    require_equal(
        manifest_inspection.get(
            "configuration_sha256"
        ),
        configuration_sha256,
        (
            "Manifest-inspection "
            "configuration SHA-256"
        ),
    )

    require_equal(
        pipeline_validation.get(
            "configuration_sha256"
        ),
        configuration_sha256,
        (
            "Pipeline-validation "
            "configuration SHA-256"
        ),
    )

    manifest_reports = manifest_inspection.get(
        "manifests",
        {},
    )

    if not isinstance(
        manifest_reports,
        Mapping,
    ):
        raise TypeError(
            "Manifest inspection reports "
            "must be a mapping."
        )

    expected_manifest_rows = {
        "train": 2594,
        "validation": 100,
        "internal_test_primary": 1000,
        (
            "internal_test_"
            "excluding_full_foreground"
        ): 999,
        (
            "internal_test_"
            "excluding_foreground_ge_0_98"
        ): 992,
        "derived_targets": 3694,
    }

    for name, expected_rows in (
        expected_manifest_rows.items()
    ):
        report = manifest_reports.get(
            name,
            {},
        )

        if not isinstance(
            report,
            Mapping,
        ):
            raise TypeError(
                f"Manifest report {name!r} "
                "must be a mapping."
            )

        require_equal(
            report.get(
                "observed_rows"
            ),
            expected_rows,
            f"{name} manifest row count",
        )

    pipeline_checks = pipeline_validation.get(
        "checks",
        {},
    )

    if not isinstance(
        pipeline_checks,
        Mapping,
    ):
        raise TypeError(
            "Pipeline checks must be a mapping."
        )

    for name, passed in (
        pipeline_checks.items()
    ):
        require_true(
            passed,
            f"Pipeline check {name}",
        )

    bundle_summary = pipeline_validation.get(
        "bundle_summary",
        {},
    )

    if not isinstance(
        bundle_summary,
        Mapping,
    ):
        raise TypeError(
            "Pipeline bundle summary must "
            "be a mapping."
        )

    expected_bundle_counts = {
        "full_train_rows": 2594,
        "full_validation_rows": 100,
        "active_train_rows": 8,
        "active_validation_rows": 4,
        "train_batches": 4,
        "validation_batches": 2,
    }

    for key, expected_value in (
        expected_bundle_counts.items()
    ):
        require_equal(
            bundle_summary.get(
                key
            ),
            expected_value,
            f"Pipeline bundle {key}",
        )

    collisions = pipeline_validation.get(
        "train_validation_collisions",
        [],
    )

    require_equal(
        collisions,
        [],
        "Train-validation collisions",
    )

    execution = pipeline_validation.get(
        "execution",
        {},
    )

    if not isinstance(
        execution,
        Mapping,
    ):
        raise TypeError(
            "Pipeline execution metadata "
            "must be a mapping."
        )

    require_equal(
        execution.get(
            "device"
        ),
        "cpu",
        "Pipeline execution device",
    )

    require_equal(
        execution.get(
            "training_executed"
        ),
        False,
        "Pipeline training-executed flag",
    )

    return {
        "configuration_sha256": (
            configuration_sha256
        ),
        "training_status": (
            training_readiness.get(
                "status"
            )
        ),
        "training_allowed": True,
        "manifest_rows": (
            expected_manifest_rows
        ),
        "pipeline_counts": (
            expected_bundle_counts
        ),
        "train_validation_collisions": 0,
        "all_source_reports_passed": True,
    }


def copy_source_file(
    *,
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    """Copy one file and verify its checksum."""

    if not source.is_file():
        raise FileNotFoundError(
            f"Source artifact not found: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_sha256 = sha256_file(
        source
    )

    shutil.copy2(
        source,
        destination,
    )

    destination_sha256 = sha256_file(
        destination
    )

    if destination_sha256 != source_sha256:
        raise RuntimeError(
            "Copied artifact checksum mismatch: "
            f"{destination}"
        )

    return {
        "source_path": str(source),
        "destination_path": str(
            destination
        ),
        "bytes": destination.stat().st_size,
        "sha256": destination_sha256,
    }


def write_checksum_inventory(
    package_directory: Path,
    checksum_path: Path,
) -> list[dict[str, Any]]:
    """Write checksums for every packaged file except the inventory."""

    files = sorted(
        path
        for path in package_directory.rglob(
            "*"
        )
        if (
            path.is_file()
            and path != checksum_path
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for path in files:
        relative_path = path.relative_to(
            package_directory
        ).as_posix()

        rows.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(
                    path
                ),
            }
        )

    with checksum_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "relative_path",
                "bytes",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return rows


def verify_checksum_inventory(
    package_directory: Path,
    checksum_path: Path,
) -> dict[str, Any]:
    """Verify every checksum inventory entry."""

    with checksum_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        rows = list(
            reader
        )

    failures: list[
        dict[str, Any]
    ] = []

    for row in rows:
        relative_path = str(
            row["relative_path"]
        )

        expected_sha256 = str(
            row["sha256"]
        )

        expected_bytes = int(
            row["bytes"]
        )

        path = (
            package_directory
            / relative_path
        ).resolve()

        try:
            path.relative_to(
                package_directory.resolve()
            )

        except ValueError:
            failures.append(
                {
                    "relative_path": (
                        relative_path
                    ),
                    "reason": (
                        "path_escapes_package"
                    ),
                }
            )

            continue

        if not path.is_file():
            failures.append(
                {
                    "relative_path": (
                        relative_path
                    ),
                    "reason": "missing_file",
                }
            )

            continue

        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(
            path
        )

        if (
            observed_bytes != expected_bytes
            or observed_sha256
            != expected_sha256
        ):
            failures.append(
                {
                    "relative_path": (
                        relative_path
                    ),
                    "reason": (
                        "checksum_or_size_mismatch"
                    ),
                    "expected_bytes": (
                        expected_bytes
                    ),
                    "observed_bytes": (
                        observed_bytes
                    ),
                    "expected_sha256": (
                        expected_sha256
                    ),
                    "observed_sha256": (
                        observed_sha256
                    ),
                }
            )

    return {
        "checksummed_files": len(
            rows
        ),
        "failed_files": len(
            failures
        ),
        "failures": failures,
        "all_checks_passed": (
            len(
                failures
            )
            == 0
        ),
    }


def main() -> int:
    """Run Step 06D packaging."""

    arguments = parse_arguments()

    configuration_directory = (
        arguments
        .configuration_directory
        .expanduser()
        .resolve()
    )

    data_foundation_directory = (
        arguments
        .data_foundation_directory
        .expanduser()
        .resolve()
    )

    package_directory = (
        arguments
        .package_directory
        .expanduser()
        .resolve()
    )

    source_files = {
        "configuration": (
            configuration_directory
            / "e00_bcs_hctnet_foundation.yaml"
        ),
        "configuration_validation": (
            configuration_directory
            / "CONFIG_VALIDATION.json"
        ),
        "manifest_schema_inspection": (
            data_foundation_directory
            / "MANIFEST_SCHEMA_INSPECTION.json"
        ),
        "data_pipeline_validation": (
            data_foundation_directory
            / "DATA_PIPELINE_VALIDATION.json"
        ),
    }

    print(
        "=== Step 06D: Package Data "
        "Foundation Artifacts ==="
    )

    print(
        "Repository root :",
        REPOSITORY_ROOT,
    )

    print(
        "Package directory:",
        package_directory,
    )

    print(
        "Execution mode  : CPU-only packaging"
    )

    configuration_validation = (
        load_json_mapping(
            source_files[
                "configuration_validation"
            ],
            "Configuration validation report",
        )
    )

    manifest_inspection = (
        load_json_mapping(
            source_files[
                "manifest_schema_inspection"
            ],
            "Manifest inspection report",
        )
    )

    pipeline_validation = (
        load_json_mapping(
            source_files[
                "data_pipeline_validation"
            ],
            "Data-pipeline validation report",
        )
    )

    validation_summary = validate_reports(
        configuration_validation=(
            configuration_validation
        ),
        manifest_inspection=(
            manifest_inspection
        ),
        pipeline_validation=(
            pipeline_validation
        ),
    )

    if package_directory.exists():
        shutil.rmtree(
            package_directory
        )

    (
        package_directory
        / "configs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        package_directory
        / "reports"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        package_directory
        / "provenance"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    destinations = {
        "configuration": (
            package_directory
            / "configs"
            / "e00_bcs_hctnet_foundation.yaml"
        ),
        "configuration_validation": (
            package_directory
            / "reports"
            / "CONFIG_VALIDATION.json"
        ),
        "manifest_schema_inspection": (
            package_directory
            / "reports"
            / "MANIFEST_SCHEMA_INSPECTION.json"
        ),
        "data_pipeline_validation": (
            package_directory
            / "reports"
            / "DATA_PIPELINE_VALIDATION.json"
        ),
    }

    copied_files: dict[
        str,
        dict[str, Any],
    ] = {}

    for name, source_path in (
        source_files.items()
    ):
        copied_files[name] = (
            copy_source_file(
                source=source_path,
                destination=(
                    destinations[name]
                ),
            )
        )

    package_report_path = (
        package_directory
        / "provenance"
        / "STEP06_DATA_FOUNDATION_PACKAGE.json"
    )

    package_report = {
        "protocol_version": (
            PACKAGE_PROTOCOL_VERSION
        ),
        "stage": (
            "step06_data_foundation_complete"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "repository_root": str(
            REPOSITORY_ROOT
        ),
        "git_commit": git_commit_hash(
            REPOSITORY_ROOT
        ),
        "environment": {
            "python_version": (
                sys.version
            ),
            "platform": (
                platform.platform()
            ),
        },
        "validation_summary": (
            validation_summary
        ),
        "copied_files": (
            copied_files
        ),
        "package_policy": {
            "persistent_input_artifacts_modified": (
                False
            ),
            "gpu_required": False,
            "training_executed": False,
            "join_key": "image_id",
            "source_reports_required_to_pass": (
                True
            ),
        },
    }

    package_report_path.write_text(
        json.dumps(
            package_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checksum_path = (
        package_directory
        / "SHA256SUMS.csv"
    )

    checksum_rows = (
        write_checksum_inventory(
            package_directory=(
                package_directory
            ),
            checksum_path=checksum_path,
        )
    )

    verification = (
        verify_checksum_inventory(
            package_directory=(
                package_directory
            ),
            checksum_path=checksum_path,
        )
    )

    if not verification[
        "all_checks_passed"
    ]:
        raise RuntimeError(
            "Packaged artifact checksum "
            "verification failed."
        )

    total_files = sum(
        1
        for path in package_directory.rglob(
            "*"
        )
        if path.is_file()
    )

    total_bytes = sum(
        path.stat().st_size
        for path in package_directory.rglob(
            "*"
        )
        if path.is_file()
    )

    print(
        "\n=== Package Validation ==="
    )

    print(
        "Configuration SHA-256:",
        validation_summary[
            "configuration_sha256"
        ],
    )

    print(
        "Training status         :",
        validation_summary[
            "training_status"
        ],
    )

    print(
        "Training allowed        :",
        validation_summary[
            "training_allowed"
        ],
    )

    print(
        "Full train rows         :",
        validation_summary[
            "pipeline_counts"
        ][
            "full_train_rows"
        ],
    )

    print(
        "Full validation rows    :",
        validation_summary[
            "pipeline_counts"
        ][
            "full_validation_rows"
        ],
    )

    print(
        "Smoke train rows        :",
        validation_summary[
            "pipeline_counts"
        ][
            "active_train_rows"
        ],
    )

    print(
        "Smoke validation rows   :",
        validation_summary[
            "pipeline_counts"
        ][
            "active_validation_rows"
        ],
    )

    print(
        "Train/validation clashes:",
        validation_summary[
            "train_validation_collisions"
        ],
    )

    print(
        "\n=== Package Contents ==="
    )

    for row in checksum_rows:
        print(
            " -",
            row["relative_path"],
            "|",
            row["bytes"],
            "bytes",
        )

    print(
        "\nChecksummed files:",
        verification[
            "checksummed_files"
        ],
    )

    print(
        "Failed files     :",
        verification[
            "failed_files"
        ],
    )

    print(
        "Total files      :",
        total_files,
    )

    print(
        "Total bytes      :",
        total_bytes,
    )

    print(
        "Package directory:",
        package_directory,
    )

    print(
        "\nStep 06D data-foundation "
        "packaging: PASSED"
    )

    print(
        "GPU required: False"
    )

    print(
        "Training executed: False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )