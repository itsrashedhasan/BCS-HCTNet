"""Per-image segmentation metric tracking and export.

This module accumulates evaluation metrics for every image and produces:

- one row per evaluated image;
- dataset-level descriptive statistics;
- thresholded-Jaccard pass rate;
- CSV export for later statistical analysis;
- JSON export containing protocol and summary information.

All baseline models and BCS-HCTNet must use this same tracker.

Metrics
-------
Overlap:
- Dice
- IoU / Jaccard
- ISIC thresholded Jaccard
- Precision
- Recall / sensitivity
- Specificity
- Pixel accuracy

Boundary:
- Boundary precision
- Boundary recall
- Boundary F1
- HD95
- ASSD

Dataset-level metrics are calculated from per-image values rather than from
pooled pixels.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.metrics.boundary import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    compute_boundary_metrics,
)
from src.metrics.overlap import (
    compute_overlap_metrics,
)
from src.metrics.thresholded_jaccard import (
    DEFAULT_JACCARD_QUALITY_THRESHOLD,
    compute_thresholded_jaccard_details,
)


METRIC_TRACKER_PROTOCOL_VERSION = (
    "BCS-HCTNet-segmentation-metric-tracker-v1"
)

METRIC_NAMES = (
    "dice",
    "iou",
    "thresholded_jaccard",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "hd95",
    "assd",
)

BOUNDED_METRIC_NAMES = (
    "dice",
    "iou",
    "thresholded_jaccard",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
)

DISTANCE_METRIC_NAMES = (
    "hd95",
    "assd",
)


def _validate_probability(
    value: float,
    context: str,
) -> float:
    """Validate a finite probability."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric."
        ) from error

    if (
        not math.isfinite(number)
        or not 0.0 <= number <= 1.0
    ):
        raise ValueError(
            f"{context} must be finite and "
            "within [0, 1]."
        )

    return number


def _validate_nonnegative_number(
    value: float,
    context: str,
) -> float:
    """Validate a finite non-negative number."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric."
        ) from error

    if (
        not math.isfinite(number)
        or number < 0.0
    ):
        raise ValueError(
            f"{context} must be finite and "
            "non-negative."
        )

    return number


def _normalize_sample_ids(
    sample_ids: Sequence[str],
    expected_count: int,
) -> list[str]:
    """Validate sample identifiers for one batch."""

    if isinstance(
        sample_ids,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "sample_ids must be a sequence "
            "of individual identifiers."
        )

    normalized = [
        str(sample_id).strip()
        for sample_id in sample_ids
    ]

    if len(normalized) != expected_count:
        raise ValueError(
            "sample_ids length does not match "
            f"batch size: {len(normalized)} "
            f"versus {expected_count}."
        )

    if any(
        not sample_id
        for sample_id in normalized
    ):
        raise ValueError(
            "sample_ids cannot contain empty values."
        )

    if len(
        set(normalized)
    ) != len(
        normalized
    ):
        raise ValueError(
            "sample_ids contain duplicates "
            "within the current batch."
        )

    return normalized


def _tensor_values(
    tensor: Tensor,
    expected_count: int,
    context: str,
) -> list[float]:
    """Convert a finite one-dimensional tensor to floats."""

    if not isinstance(
        tensor,
        Tensor,
    ):
        raise TypeError(
            f"{context} must be a tensor."
        )

    if tensor.ndim != 1:
        raise ValueError(
            f"{context} must have shape [B], "
            f"received {tuple(tensor.shape)}."
        )

    if tensor.numel() != expected_count:
        raise ValueError(
            f"{context} contains "
            f"{tensor.numel()} values; expected "
            f"{expected_count}."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{context} contains non-finite values."
        )

    return [
        float(value)
        for value in (
            tensor.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .tolist()
        )
    ]


def summarize_values(
    values: Sequence[float],
) -> dict[str, float | int]:
    """Calculate descriptive statistics for one metric."""

    if not values:
        raise ValueError(
            "Cannot summarize an empty sequence."
        )

    normalized = [
        float(value)
        for value in values
    ]

    if not all(
        math.isfinite(value)
        for value in normalized
    ):
        raise ValueError(
            "Metric values contain non-finite values."
        )

    count = len(
        normalized
    )

    return {
        "count": count,
        "mean": float(
            statistics.fmean(
                normalized
            )
        ),
        "standard_deviation": float(
            statistics.pstdev(
                normalized
            )
            if count > 1
            else 0.0
        ),
        "median": float(
            statistics.median(
                normalized
            )
        ),
        "minimum": float(
            min(
                normalized
            )
        ),
        "maximum": float(
            max(
                normalized
            )
        ),
    }


class SegmentationMetricTracker:
    """Accumulate per-image segmentation metrics."""

    def __init__(
        self,
        *,
        prediction_threshold: float = 0.5,
        target_threshold: float = 0.5,
        jaccard_quality_threshold: float = (
            DEFAULT_JACCARD_QUALITY_THRESHOLD
        ),
        boundary_tolerance_pixels: float = (
            DEFAULT_BOUNDARY_TOLERANCE_PIXELS
        ),
        from_logits: bool = True,
        split_name: str = "validation",
        dataset_name: str = "ISIC2018",
    ) -> None:
        """Initialize the metric tracker."""

        self.prediction_threshold = (
            _validate_probability(
                prediction_threshold,
                "prediction_threshold",
            )
        )

        self.target_threshold = (
            _validate_probability(
                target_threshold,
                "target_threshold",
            )
        )

        self.jaccard_quality_threshold = (
            _validate_probability(
                jaccard_quality_threshold,
                "jaccard_quality_threshold",
            )
        )

        self.boundary_tolerance_pixels = (
            _validate_nonnegative_number(
                boundary_tolerance_pixels,
                "boundary_tolerance_pixels",
            )
        )

        if not isinstance(
            from_logits,
            bool,
        ):
            raise TypeError(
                "from_logits must be Boolean."
            )

        self.from_logits = from_logits

        self.split_name = str(
            split_name
        ).strip()

        self.dataset_name = str(
            dataset_name
        ).strip()

        if not self.split_name:
            raise ValueError(
                "split_name cannot be empty."
            )

        if not self.dataset_name:
            raise ValueError(
                "dataset_name cannot be empty."
            )

        self._rows: list[
            dict[str, Any]
        ] = []

        self._sample_ids: set[
            str
        ] = set()

    def __len__(
        self,
    ) -> int:
        """Return the number of evaluated images."""

        return len(
            self._rows
        )

    @property
    def sample_ids(
        self,
    ) -> tuple[str, ...]:
        """Return sample identifiers in evaluation order."""

        return tuple(
            str(row["sample_id"])
            for row in self._rows
        )

    def reset(
        self,
    ) -> None:
        """Remove all accumulated observations."""

        self._rows.clear()
        self._sample_ids.clear()

    def update(
        self,
        *,
        prediction: Tensor,
        target: Tensor,
        sample_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Calculate and append metrics for one batch."""

        if not isinstance(
            prediction,
            Tensor,
        ):
            raise TypeError(
                "prediction must be a torch.Tensor."
            )

        if not isinstance(
            target,
            Tensor,
        ):
            raise TypeError(
                "target must be a torch.Tensor."
            )

        if prediction.ndim == 2:
            batch_size = 1

        else:
            batch_size = int(
                prediction.shape[0]
            )

        normalized_ids = (
            _normalize_sample_ids(
                sample_ids,
                expected_count=batch_size,
            )
        )

        duplicate_ids = sorted(
            set(
                normalized_ids
            )
            & self._sample_ids
        )

        if duplicate_ids:
            raise RuntimeError(
                "Metric tracker received sample IDs "
                "that were already evaluated: "
                f"{duplicate_ids}."
            )

        detached_prediction = (
            prediction.detach()
        )

        detached_target = (
            target.detach()
        )

        overlap_metrics = (
            compute_overlap_metrics(
                detached_prediction,
                detached_target,
                threshold=(
                    self.prediction_threshold
                ),
                target_threshold=(
                    self.target_threshold
                ),
                from_logits=(
                    self.from_logits
                ),
                reduction="none",
            )
        )

        thresholded_details = (
            compute_thresholded_jaccard_details(
                detached_prediction,
                detached_target,
                prediction_threshold=(
                    self.prediction_threshold
                ),
                target_threshold=(
                    self.target_threshold
                ),
                quality_threshold=(
                    self.jaccard_quality_threshold
                ),
                from_logits=(
                    self.from_logits
                ),
            )
        )

        boundary_metrics = (
            compute_boundary_metrics(
                detached_prediction,
                detached_target,
                prediction_threshold=(
                    self.prediction_threshold
                ),
                target_threshold=(
                    self.target_threshold
                ),
                from_logits=(
                    self.from_logits
                ),
                boundary_tolerance=(
                    self.boundary_tolerance_pixels
                ),
                reduction="none",
            )
        )

        metric_values: dict[
            str,
            list[float],
        ] = {}

        for name in (
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
            "accuracy",
        ):
            metric_values[name] = (
                _tensor_values(
                    overlap_metrics[name],
                    batch_size,
                    name,
                )
            )

        thresholded_tensor = (
            thresholded_details[
                "thresholded_jaccard"
            ]
        )

        pass_tensor = (
            thresholded_details[
                "passed_quality_threshold"
            ]
        )

        if not isinstance(
            thresholded_tensor,
            Tensor,
        ):
            raise TypeError(
                "thresholded_jaccard output "
                "must be a tensor."
            )

        if not isinstance(
            pass_tensor,
            Tensor,
        ):
            raise TypeError(
                "passed_quality_threshold output "
                "must be a tensor."
            )

        metric_values[
            "thresholded_jaccard"
        ] = _tensor_values(
            thresholded_tensor,
            batch_size,
            "thresholded_jaccard",
        )

        passed_values = [
            bool(value)
            for value in (
                pass_tensor.detach()
                .to(
                    device="cpu",
                    dtype=torch.bool,
                )
                .tolist()
            )
        ]

        if len(
            passed_values
        ) != batch_size:
            raise RuntimeError(
                "Threshold pass flags do not "
                "match batch size."
            )

        for name in (
            "boundary_precision",
            "boundary_recall",
            "boundary_f1",
            "hd95",
            "assd",
        ):
            metric_values[name] = (
                _tensor_values(
                    boundary_metrics[name],
                    batch_size,
                    name,
                )
            )

        new_rows: list[
            dict[str, Any]
        ] = []

        for sample_index, sample_id in enumerate(
            normalized_ids
        ):
            row: dict[
                str,
                Any,
            ] = {
                "evaluation_index": (
                    len(
                        self._rows
                    )
                    + sample_index
                ),
                "sample_id": sample_id,
                "dataset": (
                    self.dataset_name
                ),
                "split": (
                    self.split_name
                ),
                "passed_jaccard_quality_threshold": (
                    passed_values[
                        sample_index
                    ]
                ),
            }

            for metric_name in (
                METRIC_NAMES
            ):
                metric_value = (
                    metric_values[
                        metric_name
                    ][
                        sample_index
                    ]
                )

                if not math.isfinite(
                    metric_value
                ):
                    raise RuntimeError(
                        f"Metric {metric_name!r} "
                        "is non-finite for sample "
                        f"{sample_id!r}."
                    )

                if (
                    metric_name
                    in BOUNDED_METRIC_NAMES
                    and not (
                        0.0
                        <= metric_value
                        <= 1.0
                    )
                ):
                    raise RuntimeError(
                        f"Metric {metric_name!r} "
                        "is outside [0, 1] for "
                        f"sample {sample_id!r}: "
                        f"{metric_value}."
                    )

                if (
                    metric_name
                    in DISTANCE_METRIC_NAMES
                    and metric_value < 0.0
                ):
                    raise RuntimeError(
                        f"Distance metric "
                        f"{metric_name!r} is negative "
                        f"for sample {sample_id!r}."
                    )

                row[
                    metric_name
                ] = metric_value

            new_rows.append(
                row
            )

        self._rows.extend(
            new_rows
        )

        self._sample_ids.update(
            normalized_ids
        )

        return [
            dict(row)
            for row in new_rows
        ]

    def rows(
        self,
    ) -> list[dict[str, Any]]:
        """Return defensive copies of all per-image rows."""

        return [
            dict(row)
            for row in self._rows
        ]

    def summary(
        self,
    ) -> dict[str, Any]:
        """Calculate dataset-level descriptive statistics."""

        if not self._rows:
            raise RuntimeError(
                "Cannot summarize an empty "
                "metric tracker."
            )

        metric_statistics = {
            metric_name: summarize_values(
                [
                    float(
                        row[
                            metric_name
                        ]
                    )
                    for row in self._rows
                ]
            )
            for metric_name in (
                METRIC_NAMES
            )
        }

        passed_count = sum(
            1
            for row in self._rows
            if row[
                "passed_jaccard_quality_threshold"
            ]
        )

        image_count = len(
            self._rows
        )

        return {
            "protocol_version": (
                METRIC_TRACKER_PROTOCOL_VERSION
            ),
            "dataset": self.dataset_name,
            "split": self.split_name,
            "number_of_images": (
                image_count
            ),
            "number_passing_jaccard_quality_threshold": (
                passed_count
            ),
            "jaccard_quality_threshold_pass_rate": (
                float(
                    passed_count
                    / image_count
                )
            ),
            "evaluation_settings": {
                "prediction_threshold": (
                    self.prediction_threshold
                ),
                "target_threshold": (
                    self.target_threshold
                ),
                "jaccard_quality_threshold": (
                    self.jaccard_quality_threshold
                ),
                "boundary_tolerance_pixels": (
                    self.boundary_tolerance_pixels
                ),
                "from_logits": (
                    self.from_logits
                ),
                "distance_unit": "pixels",
                "aggregation_policy": (
                    "per_image_then_dataset_summary"
                ),
            },
            "metrics": metric_statistics,
        }

    def export_csv(
        self,
        output_path: str | Path,
    ) -> Path:
        """Export one metric row per image."""

        if not self._rows:
            raise RuntimeError(
                "Cannot export an empty "
                "metric tracker."
            )

        resolved_path = (
            Path(
                output_path
            )
            .expanduser()
            .resolve()
        )

        resolved_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "evaluation_index",
            "sample_id",
            "dataset",
            "split",
            (
                "passed_jaccard_"
                "quality_threshold"
            ),
            *METRIC_NAMES,
        ]

        with resolved_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                self._rows
            )

        return resolved_path

    def export_summary_json(
        self,
        output_path: str | Path,
        *,
        additional_metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Path:
        """Export dataset-level metrics and provenance."""

        resolved_path = (
            Path(
                output_path
            )
            .expanduser()
            .resolve()
        )

        resolved_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = (
            dict(
                additional_metadata
            )
            if additional_metadata
            is not None
            else {}
        )

        report = {
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "summary": self.summary(),
            "additional_metadata": (
                metadata
            ),
        }

        resolved_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return resolved_path

    def export(
        self,
        output_directory: str | Path,
        *,
        file_prefix: str = "evaluation",
        additional_metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> dict[str, Path]:
        """Export per-image CSV and summary JSON together."""

        normalized_prefix = str(
            file_prefix
        ).strip()

        if not normalized_prefix:
            raise ValueError(
                "file_prefix cannot be empty."
            )

        output_root = (
            Path(
                output_directory
            )
            .expanduser()
            .resolve()
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = self.export_csv(
            output_root
            / (
                f"{normalized_prefix}_"
                "per_image_metrics.csv"
            )
        )

        summary_path = (
            self.export_summary_json(
                output_root
                / (
                    f"{normalized_prefix}_"
                    "metric_summary.json"
                ),
                additional_metadata=(
                    additional_metadata
                ),
            )
        )

        return {
            "per_image_csv": csv_path,
            "summary_json": summary_path,
        }


MetricTracker = SegmentationMetricTracker


def run_metric_tracker_self_test() -> dict[str, Any]:
    """Run an offline CPU tracker and export self-test."""

    import tempfile

    target = torch.zeros(
        3,
        1,
        8,
        8,
        dtype=torch.float32,
    )

    probability = torch.full(
        (
            3,
            1,
            8,
            8,
        ),
        fill_value=0.1,
        dtype=torch.float32,
    )

    target[
        0,
        0,
        2:6,
        2:6,
    ] = 1.0

    probability[
        0,
        0,
        2:6,
        2:6,
    ] = 0.9

    target[
        1,
        0,
        2:6,
        2:6,
    ] = 1.0

    probability[
        1,
        0,
        2:6,
        3:7,
    ] = 0.9

    logits = torch.logit(
        probability.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )

    tracker = (
        SegmentationMetricTracker(
            prediction_threshold=0.5,
            target_threshold=0.5,
            jaccard_quality_threshold=0.65,
            boundary_tolerance_pixels=1.0,
            from_logits=True,
            split_name="validation",
            dataset_name="synthetic",
        )
    )

    first_rows = tracker.update(
        prediction=logits[:2],
        target=target[:2],
        sample_ids=[
            "sample_001",
            "sample_002",
        ],
    )

    second_rows = tracker.update(
        prediction=logits[2:],
        target=target[2:],
        sample_ids=[
            "sample_003",
        ],
    )

    duplicate_rejected = False

    try:
        tracker.update(
            prediction=logits[:1],
            target=target[:1],
            sample_ids=[
                "sample_001",
            ],
        )

    except RuntimeError:
        duplicate_rejected = True

    summary = tracker.summary()

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        export_paths = tracker.export(
            temporary_directory,
            file_prefix="self_test",
            additional_metadata={
                "purpose": (
                    "metric_tracker_self_test"
                )
            },
        )

        csv_path = export_paths[
            "per_image_csv"
        ]

        json_path = export_paths[
            "summary_json"
        ]

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            csv_rows = list(
                csv.DictReader(
                    input_file
                )
            )

        json_report = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        exported_csv_exists = (
            csv_path.is_file()
        )

        exported_json_exists = (
            json_path.is_file()
        )

        exported_csv_rows = len(
            csv_rows
        )

        exported_json_count = (
            json_report[
                "summary"
            ][
                "number_of_images"
            ]
        )

    rows = tracker.rows()

    metric_values_finite = all(
        math.isfinite(
            float(
                row[
                    metric_name
                ]
            )
        )
        for row in rows
        for metric_name in METRIC_NAMES
    )

    bounded_values_valid = all(
        0.0
        <= float(
            row[
                metric_name
            ]
        )
        <= 1.0
        for row in rows
        for metric_name
        in BOUNDED_METRIC_NAMES
    )

    distance_values_valid = all(
        float(
            row[
                metric_name
            ]
        )
        >= 0.0
        for row in rows
        for metric_name
        in DISTANCE_METRIC_NAMES
    )

    checks = {
        "tracker_length": (
            len(
                tracker
            )
            == 3
        ),
        "first_update_rows": (
            len(
                first_rows
            )
            == 2
        ),
        "second_update_rows": (
            len(
                second_rows
            )
            == 1
        ),
        "sample_order_preserved": (
            tracker.sample_ids
            == (
                "sample_001",
                "sample_002",
                "sample_003",
            )
        ),
        "duplicate_ids_rejected": (
            duplicate_rejected
        ),
        "all_metric_names_present": all(
            all(
                metric_name in row
                for metric_name
                in METRIC_NAMES
            )
            for row in rows
        ),
        "metric_values_finite": (
            metric_values_finite
        ),
        "bounded_values_valid": (
            bounded_values_valid
        ),
        "distance_values_valid": (
            distance_values_valid
        ),
        "perfect_sample_dice": (
            float(
                rows[0][
                    "dice"
                ]
            )
            == 1.0
        ),
        "perfect_sample_boundary_f1": (
            float(
                rows[0][
                    "boundary_f1"
                ]
            )
            == 1.0
        ),
        "empty_empty_perfect": (
            float(
                rows[2][
                    "dice"
                ]
            )
            == 1.0
            and float(
                rows[2][
                    "iou"
                ]
            )
            == 1.0
            and float(
                rows[2][
                    "boundary_f1"
                ]
            )
            == 1.0
            and float(
                rows[2][
                    "hd95"
                ]
            )
            == 0.0
        ),
        "summary_image_count": (
            summary[
                "number_of_images"
            ]
            == 3
        ),
        "summary_metric_names": (
            tuple(
                summary[
                    "metrics"
                ]
            )
            == METRIC_NAMES
        ),
        "summary_counts_correct": all(
            summary[
                "metrics"
            ][
                metric_name
            ][
                "count"
            ]
            == 3
            for metric_name
            in METRIC_NAMES
        ),
        "csv_export_exists": (
            exported_csv_exists
        ),
        "json_export_exists": (
            exported_json_exists
        ),
        "csv_export_rows": (
            exported_csv_rows
            == 3
        ),
        "json_export_count": (
            exported_json_count
            == 3
        ),
        "protocol_version_correct": (
            summary[
                "protocol_version"
            ]
            == (
                METRIC_TRACKER_PROTOCOL_VERSION
            )
        ),
    }

    return {
        "status": (
            "passed"
            if all(
                checks.values()
            )
            else "failed"
        ),
        "protocol_version": (
            METRIC_TRACKER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "sample_ids": list(
            tracker.sample_ids
        ),
        "number_of_images": (
            summary[
                "number_of_images"
            ]
        ),
        "jaccard_quality_threshold_pass_rate": (
            summary[
                "jaccard_quality_threshold_pass_rate"
            ]
        ),
        "metric_means": {
            metric_name: (
                summary[
                    "metrics"
                ][
                    metric_name
                ][
                    "mean"
                ]
            )
            for metric_name
            in METRIC_NAMES
        },
    }