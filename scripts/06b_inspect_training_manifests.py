"""Step 06B: Inspect training-manifest schemas and join compatibility.

Run from the repository root:

    python3 scripts/06b_inspect_training_manifests.py

This CPU-only inspection reads the manifests already validated by Step 06A
and reports:

- exact CSV column names;
- row counts;
- sample rows;
- empty-value counts;
- candidate identifier columns;
- duplicate identifier counts;
- possible join columns between split and target manifests;
- value overlap for each possible join column.

The report is used to implement the dataset loader without guessing any
manifest field names.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPOSITORY_ROOT),
)


from src.train.experiment_config import (
    ValidatedExperimentConfig,
    validate_experiment_configuration,
)


DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "e00_bcs_hctnet_foundation.yaml"
)

DEFAULT_OUTPUT_PATH = Path(
    "/kaggle/working/outputs/"
    "experiments/E00/data_foundation/"
    "MANIFEST_SCHEMA_INSPECTION.json"
)

INSPECTION_PROTOCOL_VERSION = (
    "BCS-HCTNet-manifest-schema-inspection-v1"
)

IDENTIFIER_HINTS = (
    "id",
    "image",
    "sample",
    "case",
    "lesion",
    "file",
    "name",
    "stem",
)

PATH_HINTS = (
    "path",
    "file",
    "image",
    "mask",
    "target",
    "contour",
    "boundary",
    "sdm",
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect BCS-HCTNet split and target "
            "manifest schemas."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the validated E00 "
            "experiment YAML."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for the JSON inspection report."
        ),
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help=(
            "Number of rows to include from "
            "each manifest."
        ),
    )

    return parser.parse_args()


def read_csv_manifest(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    """Read one CSV manifest."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"Manifest has no header: {path}"
            )

        fieldnames = [
            str(name)
            for name in reader.fieldnames
        ]

        rows = [
            {
                str(key): (
                    ""
                    if value is None
                    else str(value)
                )
                for key, value in row.items()
            }
            for row in reader
        ]

    return fieldnames, rows


def normalized_value(
    value: object,
) -> str:
    """Normalize one CSV value for comparisons."""

    return str(
        value
        if value is not None
        else ""
    ).strip()


def unique_nonempty_values(
    rows: list[Mapping[str, str]],
    column: str,
) -> set[str]:
    """Return unique non-empty values for a column."""

    return {
        normalized
        for row in rows
        if (
            normalized := normalized_value(
                row.get(
                    column,
                    "",
                )
            )
        )
    }


def duplicate_value_summary(
    rows: list[Mapping[str, str]],
    column: str,
) -> dict[str, Any]:
    """Summarize duplicate non-empty values."""

    values = [
        normalized
        for row in rows
        if (
            normalized := normalized_value(
                row.get(
                    column,
                    "",
                )
            )
        )
    ]

    counts = Counter(
        values
    )

    duplicate_items = {
        value: count
        for value, count in counts.items()
        if count > 1
    }

    top_duplicates = sorted(
        duplicate_items.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:10]

    return {
        "nonempty_values": len(values),
        "unique_nonempty_values": len(counts),
        "duplicate_unique_values": len(
            duplicate_items
        ),
        "duplicate_extra_rows": sum(
            count - 1
            for count in duplicate_items.values()
        ),
        "maximum_multiplicity": (
            max(
                counts.values()
            )
            if counts
            else 0
        ),
        "top_duplicates": [
            {
                "value": value,
                "count": count,
            }
            for value, count in top_duplicates
        ],
    }


def likely_identifier_columns(
    fieldnames: list[str],
) -> list[str]:
    """Find columns that may identify a sample."""

    candidates: list[str] = []

    for column in fieldnames:
        lowered = column.lower()

        if any(
            hint in lowered
            for hint in IDENTIFIER_HINTS
        ):
            candidates.append(
                column
            )

    return candidates


def likely_path_columns(
    fieldnames: list[str],
    rows: list[Mapping[str, str]],
) -> list[str]:
    """Find columns that may contain filesystem paths."""

    candidates: list[str] = []

    for column in fieldnames:
        lowered = column.lower()

        name_suggests_path = any(
            hint in lowered
            for hint in PATH_HINTS
        )

        observed_values = [
            normalized_value(
                row.get(
                    column,
                    "",
                )
            )
            for row in rows[:100]
        ]

        content_suggests_path = any(
            (
                "/" in value
                or "\\" in value
                or Path(value).suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                    ".npy",
                    ".npz",
                }
            )
            for value in observed_values
            if value
        )

        if (
            name_suggests_path
            or content_suggests_path
        ):
            candidates.append(
                column
            )

    return candidates


def column_statistics(
    fieldnames: list[str],
    rows: list[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Calculate basic statistics for every column."""

    statistics: dict[
        str,
        dict[str, Any],
    ] = {}

    for column in fieldnames:
        values = [
            normalized_value(
                row.get(
                    column,
                    "",
                )
            )
            for row in rows
        ]

        nonempty = [
            value
            for value in values
            if value
        ]

        unique_values = set(
            nonempty
        )

        sample_values: list[str] = []

        for value in nonempty:
            if value not in sample_values:
                sample_values.append(
                    value
                )

            if len(
                sample_values
            ) >= 5:
                break

        statistics[column] = {
            "rows": len(rows),
            "nonempty": len(
                nonempty
            ),
            "empty": (
                len(rows)
                - len(nonempty)
            ),
            "unique_nonempty": len(
                unique_values
            ),
            "sample_values": (
                sample_values
            ),
        }

    return statistics


def inspect_manifest(
    name: str,
    path: Path,
    expected_rows: int,
    sample_rows: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
]:
    """Inspect one manifest."""

    fieldnames, rows = read_csv_manifest(
        path
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{name}: expected {expected_rows} "
            f"rows, found {len(rows)}."
        )

    identifier_columns = (
        likely_identifier_columns(
            fieldnames
        )
    )

    duplicate_summaries = {
        column: duplicate_value_summary(
            rows,
            column,
        )
        for column in identifier_columns
    }

    report = {
        "name": name,
        "path": str(path),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "column_count": len(
            fieldnames
        ),
        "columns": fieldnames,
        "likely_identifier_columns": (
            identifier_columns
        ),
        "likely_path_columns": (
            likely_path_columns(
                fieldnames,
                rows,
            )
        ),
        "column_statistics": (
            column_statistics(
                fieldnames,
                rows,
            )
        ),
        "identifier_duplicate_summaries": (
            duplicate_summaries
        ),
        "sample_rows": rows[
            :sample_rows
        ],
    }

    return report, rows


def compare_join_columns(
    left_name: str,
    left_columns: list[str],
    left_rows: list[Mapping[str, str]],
    right_name: str,
    right_columns: list[str],
    right_rows: list[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Compare same-named columns as possible join keys."""

    shared_columns = sorted(
        set(left_columns)
        & set(right_columns)
    )

    comparisons: list[
        dict[str, Any]
    ] = []

    for column in shared_columns:
        left_values = (
            unique_nonempty_values(
                left_rows,
                column,
            )
        )

        right_values = (
            unique_nonempty_values(
                right_rows,
                column,
            )
        )

        intersection = (
            left_values
            & right_values
        )

        left_coverage = (
            len(intersection)
            / len(left_values)
            if left_values
            else 0.0
        )

        right_coverage = (
            len(intersection)
            / len(right_values)
            if right_values
            else 0.0
        )

        comparisons.append(
            {
                "left_manifest": (
                    left_name
                ),
                "right_manifest": (
                    right_name
                ),
                "column": column,
                "left_unique_nonempty": len(
                    left_values
                ),
                "right_unique_nonempty": len(
                    right_values
                ),
                "intersection_unique": len(
                    intersection
                ),
                "left_coverage": round(
                    left_coverage,
                    6,
                ),
                "right_coverage": round(
                    right_coverage,
                    6,
                ),
                "left_duplicates": (
                    duplicate_value_summary(
                        left_rows,
                        column,
                    )
                ),
                "right_duplicates": (
                    duplicate_value_summary(
                        right_rows,
                        column,
                    )
                ),
            }
        )

    comparisons.sort(
        key=lambda item: (
            -item[
                "intersection_unique"
            ],
            -item[
                "left_coverage"
            ],
            item["column"],
        )
    )

    return comparisons


def get_manifest(
    validated: ValidatedExperimentConfig,
    name: str,
):
    """Return one resolved validated manifest."""

    if name not in validated.manifests:
        raise KeyError(
            f"Validated configuration has no "
            f"manifest named {name!r}."
        )

    return validated.manifests[
        name
    ]


def choose_probable_join_columns(
    comparisons: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return high-coverage probable join columns."""

    probable: list[
        dict[str, Any]
    ] = []

    for comparison in comparisons:
        intersection = int(
            comparison[
                "intersection_unique"
            ]
        )

        coverage = float(
            comparison[
                "left_coverage"
            ]
        )

        if (
            intersection > 0
            and coverage >= 0.95
        ):
            probable.append(
                dict(
                    comparison
                )
            )

    return probable


def main() -> int:
    """Run Step 06B."""

    arguments = parse_arguments()

    if arguments.sample_rows <= 0:
        raise ValueError(
            "--sample-rows must be positive."
        )

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    output_path = (
        arguments.output
        .expanduser()
        .resolve()
    )

    print(
        "=== Step 06B: Inspect Training "
        "Manifest Schemas ==="
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
        "Output report   :",
        output_path,
    )

    print(
        "Execution mode  : CPU-only inspection"
    )

    validated = (
        validate_experiment_configuration(
            config_path
        )
    )

    manifest_reports: dict[
        str,
        dict[str, Any],
    ] = {}

    manifest_rows: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for name, resolved in (
        validated.manifests.items()
    ):
        report, rows = inspect_manifest(
            name=name,
            path=resolved.absolute_path,
            expected_rows=(
                resolved.expected_rows
            ),
            sample_rows=(
                arguments.sample_rows
            ),
        )

        manifest_reports[name] = report
        manifest_rows[name] = rows

        print(
            f"\n{name}"
        )

        print(
            "  Rows       :",
            report[
                "observed_rows"
            ],
        )

        print(
            "  Columns    :",
            report[
                "columns"
            ],
        )

        print(
            "  ID columns :",
            report[
                "likely_identifier_columns"
            ],
        )

        print(
            "  Path fields:",
            report[
                "likely_path_columns"
            ],
        )

    target_name = "derived_targets"

    target_report = manifest_reports[
        target_name
    ]

    target_rows = manifest_rows[
        target_name
    ]

    split_names = [
        "train",
        "validation",
        "internal_test_primary",
        (
            "internal_test_"
            "excluding_full_foreground"
        ),
        (
            "internal_test_"
            "excluding_foreground_ge_0_98"
        ),
    ]

    join_comparisons: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    probable_join_columns: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for split_name in split_names:
        comparisons = compare_join_columns(
            left_name=split_name,
            left_columns=(
                manifest_reports[
                    split_name
                ][
                    "columns"
                ]
            ),
            left_rows=(
                manifest_rows[
                    split_name
                ]
            ),
            right_name=target_name,
            right_columns=(
                target_report[
                    "columns"
                ]
            ),
            right_rows=target_rows,
        )

        join_comparisons[
            split_name
        ] = comparisons

        probable = (
            choose_probable_join_columns(
                comparisons
            )
        )

        probable_join_columns[
            split_name
        ] = probable

        print(
            f"\nJoin candidates: "
            f"{split_name} -> {target_name}"
        )

        if probable:
            for item in probable:
                print(
                    "  ",
                    item["column"],
                    "| intersection:",
                    item[
                        "intersection_unique"
                    ],
                    "| split coverage:",
                    item[
                        "left_coverage"
                    ],
                    "| target coverage:",
                    item[
                        "right_coverage"
                    ],
                )
        else:
            print(
                "  No same-named column reached "
                "95% split coverage."
            )

    train_probable = (
        probable_join_columns[
            "train"
        ]
    )

    validation_probable = (
        probable_join_columns[
            "validation"
        ]
    )

    checks = {
        "all_six_manifests_inspected": (
            len(
                manifest_reports
            )
            == 6
        ),
        "train_rows_correct": (
            manifest_reports[
                "train"
            ][
                "observed_rows"
            ]
            == 2594
        ),
        "validation_rows_correct": (
            manifest_reports[
                "validation"
            ][
                "observed_rows"
            ]
            == 100
        ),
        "derived_target_rows_correct": (
            target_report[
                "observed_rows"
            ]
            == 3694
        ),
        "train_has_probable_join_column": (
            len(
                train_probable
            )
            > 0
        ),
        "validation_has_probable_join_column": (
            len(
                validation_probable
            )
            > 0
        ),
        "target_has_path_like_columns": (
            len(
                target_report[
                    "likely_path_columns"
                ]
            )
            > 0
        ),
    }

    report = {
        "protocol_version": (
            INSPECTION_PROTOCOL_VERSION
        ),
        "configuration_path": str(
            validated.source_path
        ),
        "configuration_sha256": (
            validated.canonical_sha256
        ),
        "manifests": (
            manifest_reports
        ),
        "join_comparisons": (
            join_comparisons
        ),
        "probable_join_columns": (
            probable_join_columns
        ),
        "checks": checks,
        "all_checks_passed": all(
            checks.values()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Step 06B Checks ==="
    )

    for name, passed in (
        checks.items()
    ):
        print(
            f"{name}: {passed}"
        )

    print(
        "\nSaved report:",
        output_path,
    )

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "Step 06B manifest-schema "
            "inspection failed."
        )

    print(
        "\nStep 06B manifest-schema "
        "inspection: PASSED"
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