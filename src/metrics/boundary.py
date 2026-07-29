"""Boundary-based metrics for binary lesion segmentation.

Implemented metrics
-------------------
Boundary F1:
    Measures precision and recall between predicted and target boundaries
    within a configurable pixel tolerance.

HD95:
    Symmetric 95th-percentile Hausdorff distance between boundary pixels.

ASSD:
    Average symmetric surface distance between predicted and target
    boundary pixels.

Evaluation conventions
----------------------
- Metrics are calculated independently for every image.
- Distances are reported in pixels by default.
- Identical empty masks receive:
    Boundary F1 = 1
    HD95 = 0
    ASSD = 0
- When exactly one mask is empty:
    Boundary F1 = 0
    HD95 = image-diagonal penalty
    ASSD = image-diagonal penalty

The explicit finite penalty prevents undefined values from silently being
excluded from experiment averages.

This module uses SciPy's exact Euclidean distance transform on CPU. Boundary
metrics are evaluation metrics and do not participate in gradient computation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor

from src.metrics.overlap import (
    prepare_binary_pair,
    reduce_metric,
)


BOUNDARY_METRICS_PROTOCOL_VERSION = (
    "BCS-HCTNet-boundary-metrics-v1"
)

DEFAULT_BOUNDARY_TOLERANCE_PIXELS = 2.0

Reduction = Literal[
    "none",
    "mean",
    "sum",
]


def validate_pixel_spacing(
    pixel_spacing: Sequence[float],
) -> tuple[float, float]:
    """Validate two-dimensional pixel spacing."""

    if isinstance(
        pixel_spacing,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "pixel_spacing must be a sequence "
            "of two positive numbers."
        )

    try:
        values = tuple(
            float(value)
            for value in pixel_spacing
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "pixel_spacing must contain "
            "numeric values."
        ) from error

    if len(values) != 2:
        raise ValueError(
            "pixel_spacing must contain exactly "
            "two values: height and width spacing."
        )

    for index, value in enumerate(
        values
    ):
        if (
            not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                "pixel_spacing values must be "
                "positive and finite. Invalid "
                f"value at index {index}: {value}."
            )

    return (
        values[0],
        values[1],
    )


def validate_boundary_tolerance(
    tolerance: float,
) -> float:
    """Validate a non-negative boundary tolerance."""

    if isinstance(
        tolerance,
        bool,
    ):
        raise TypeError(
            "tolerance must be numeric."
        )

    try:
        value = float(
            tolerance
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "tolerance must be numeric."
        ) from error

    if (
        not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            "tolerance must be finite and "
            "non-negative."
        )

    return value


def extract_binary_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    """Extract a one-pixel internal boundary from a binary mask."""

    if not isinstance(
        mask,
        np.ndarray,
    ):
        raise TypeError(
            "mask must be a NumPy array."
        )

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional, "
            f"received shape {mask.shape}."
        )

    if mask.size == 0:
        raise ValueError(
            "mask cannot be empty."
        )

    binary_mask = mask.astype(
        np.bool_,
        copy=False,
    )

    if not binary_mask.any():
        return np.zeros_like(
            binary_mask,
            dtype=np.bool_,
        )

    connectivity = (
        ndimage.generate_binary_structure(
            rank=2,
            connectivity=1,
        )
    )

    eroded = ndimage.binary_erosion(
        binary_mask,
        structure=connectivity,
        iterations=1,
        border_value=0,
    )

    boundary = (
        binary_mask
        & ~eroded
    )

    return boundary.astype(
        np.bool_,
        copy=False,
    )


def image_diagonal_penalty(
    shape: tuple[int, int],
    *,
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
) -> float:
    """Calculate the maximum corner-to-corner image distance."""

    if len(shape) != 2:
        raise ValueError(
            "shape must contain height and width."
        )

    height = int(
        shape[0]
    )

    width = int(
        shape[1]
    )

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image height and width must "
            "be positive."
        )

    spacing_height, spacing_width = (
        validate_pixel_spacing(
            pixel_spacing
        )
    )

    height_distance = (
        max(
            height - 1,
            0,
        )
        * spacing_height
    )

    width_distance = (
        max(
            width - 1,
            0,
        )
        * spacing_width
    )

    return float(
        np.hypot(
            height_distance,
            width_distance,
        )
    )


def directed_boundary_distances(
    source_boundary: np.ndarray,
    destination_boundary: np.ndarray,
    *,
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
) -> np.ndarray:
    """Calculate distances from source boundary pixels to the destination."""

    if not isinstance(
        source_boundary,
        np.ndarray,
    ) or not isinstance(
        destination_boundary,
        np.ndarray,
    ):
        raise TypeError(
            "Boundary inputs must be NumPy arrays."
        )

    if (
        source_boundary.ndim != 2
        or destination_boundary.ndim != 2
    ):
        raise ValueError(
            "Boundary arrays must be "
            "two-dimensional."
        )

    if (
        source_boundary.shape
        != destination_boundary.shape
    ):
        raise ValueError(
            "Boundary arrays must have "
            "matching shapes."
        )

    source = source_boundary.astype(
        np.bool_,
        copy=False,
    )

    destination = (
        destination_boundary.astype(
            np.bool_,
            copy=False,
        )
    )

    if not source.any():
        return np.empty(
            0,
            dtype=np.float64,
        )

    if not destination.any():
        raise ValueError(
            "destination_boundary cannot be "
            "empty when source_boundary is non-empty."
        )

    spacing = validate_pixel_spacing(
        pixel_spacing
    )

    destination_distance_map = (
        ndimage.distance_transform_edt(
            ~destination,
            sampling=spacing,
        )
    )

    distances = (
        destination_distance_map[
            source
        ]
    )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def boundary_f1_from_boundaries(
    predicted_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    tolerance: float = (
        DEFAULT_BOUNDARY_TOLERANCE_PIXELS
    ),
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
) -> dict[str, float]:
    """Calculate Boundary F1 from two boundary arrays."""

    if (
        predicted_boundary.shape
        != target_boundary.shape
    ):
        raise ValueError(
            "Predicted and target boundaries "
            "must have matching shapes."
        )

    validated_tolerance = (
        validate_boundary_tolerance(
            tolerance
        )
    )

    validated_spacing = (
        validate_pixel_spacing(
            pixel_spacing
        )
    )

    predicted = (
        predicted_boundary.astype(
            np.bool_,
            copy=False,
        )
    )

    target = target_boundary.astype(
        np.bool_,
        copy=False,
    )

    predicted_count = int(
        predicted.sum()
    )

    target_count = int(
        target.sum()
    )

    if (
        predicted_count == 0
        and target_count == 0
    ):
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
        }

    if predicted_count == 0:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
        }

    if target_count == 0:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
        }

    predicted_to_target = (
        directed_boundary_distances(
            predicted,
            target,
            pixel_spacing=(
                validated_spacing
            ),
        )
    )

    target_to_predicted = (
        directed_boundary_distances(
            target,
            predicted,
            pixel_spacing=(
                validated_spacing
            ),
        )
    )

    precision = float(
        np.mean(
            predicted_to_target
            <= validated_tolerance
        )
    )

    recall = float(
        np.mean(
            target_to_predicted
            <= validated_tolerance
        )
    )

    denominator = (
        precision
        + recall
    )

    if denominator == 0.0:
        f1 = 0.0

    else:
        f1 = float(
            2.0
            * precision
            * recall
            / denominator
        )

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def surface_distances_from_boundaries(
    predicted_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
) -> dict[str, float]:
    """Calculate symmetric HD95 and ASSD values."""

    if (
        predicted_boundary.shape
        != target_boundary.shape
    ):
        raise ValueError(
            "Predicted and target boundaries "
            "must have matching shapes."
        )

    predicted = (
        predicted_boundary.astype(
            np.bool_,
            copy=False,
        )
    )

    target = target_boundary.astype(
        np.bool_,
        copy=False,
    )

    validated_spacing = (
        validate_pixel_spacing(
            pixel_spacing
        )
    )

    predicted_count = int(
        predicted.sum()
    )

    target_count = int(
        target.sum()
    )

    if (
        predicted_count == 0
        and target_count == 0
    ):
        return {
            "hd95": 0.0,
            "assd": 0.0,
        }

    if (
        predicted_count == 0
        or target_count == 0
    ):
        penalty = image_diagonal_penalty(
            predicted.shape,
            pixel_spacing=(
                validated_spacing
            ),
        )

        return {
            "hd95": penalty,
            "assd": penalty,
        }

    predicted_to_target = (
        directed_boundary_distances(
            predicted,
            target,
            pixel_spacing=(
                validated_spacing
            ),
        )
    )

    target_to_predicted = (
        directed_boundary_distances(
            target,
            predicted,
            pixel_spacing=(
                validated_spacing
            ),
        )
    )

    symmetric_distances = np.concatenate(
        [
            predicted_to_target,
            target_to_predicted,
        ]
    )

    hd95 = float(
        np.percentile(
            symmetric_distances,
            95.0,
        )
    )

    assd = float(
        np.mean(
            symmetric_distances
        )
    )

    return {
        "hd95": hd95,
        "assd": assd,
    }


def compute_boundary_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    boundary_tolerance: float = (
        DEFAULT_BOUNDARY_TOLERANCE_PIXELS
    ),
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
    reduction: Reduction = "mean",
) -> dict[str, Tensor]:
    """Calculate per-image boundary and surface-distance metrics."""

    (
        binary_prediction,
        binary_target,
    ) = prepare_binary_pair(
        prediction,
        target,
        threshold=(
            prediction_threshold
        ),
        target_threshold=(
            target_threshold
        ),
        from_logits=from_logits,
    )

    validated_tolerance = (
        validate_boundary_tolerance(
            boundary_tolerance
        )
    )

    validated_spacing = (
        validate_pixel_spacing(
            pixel_spacing
        )
    )

    prediction_numpy = (
        binary_prediction.detach()
        .to(
            device="cpu"
        )
        .numpy()
    )

    target_numpy = (
        binary_target.detach()
        .to(
            device="cpu"
        )
        .numpy()
    )

    collected: dict[
        str,
        list[float],
    ] = {
        "boundary_precision": [],
        "boundary_recall": [],
        "boundary_f1": [],
        "hd95": [],
        "assd": [],
    }

    batch_size = int(
        prediction_numpy.shape[0]
    )

    for sample_index in range(
        batch_size
    ):
        predicted_mask = (
            prediction_numpy[
                sample_index,
                0,
            ]
        )

        target_mask = (
            target_numpy[
                sample_index,
                0,
            ]
        )

        predicted_boundary = (
            extract_binary_boundary(
                predicted_mask
            )
        )

        target_boundary = (
            extract_binary_boundary(
                target_mask
            )
        )

        boundary_scores = (
            boundary_f1_from_boundaries(
                predicted_boundary,
                target_boundary,
                tolerance=(
                    validated_tolerance
                ),
                pixel_spacing=(
                    validated_spacing
                ),
            )
        )

        distance_scores = (
            surface_distances_from_boundaries(
                predicted_boundary,
                target_boundary,
                pixel_spacing=(
                    validated_spacing
                ),
            )
        )

        for name, value in {
            **boundary_scores,
            **distance_scores,
        }.items():
            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    f"Boundary metric {name!r} "
                    "produced a non-finite value "
                    f"for sample {sample_index}."
                )

            collected[
                name
            ].append(
                float(
                    value
                )
            )

    per_image = {
        name: torch.tensor(
            values,
            dtype=torch.float64,
            device=prediction.device,
        )
        for name, values
        in collected.items()
    }

    return {
        name: reduce_metric(
            values,
            reduction=reduction,
        )
        for name, values
        in per_image.items()
    }


def boundary_f1_score(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    tolerance: float = (
        DEFAULT_BOUNDARY_TOLERANCE_PIXELS
    ),
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate Boundary F1."""

    return compute_boundary_metrics(
        prediction,
        target,
        prediction_threshold=(
            prediction_threshold
        ),
        target_threshold=(
            target_threshold
        ),
        from_logits=from_logits,
        boundary_tolerance=tolerance,
        pixel_spacing=pixel_spacing,
        reduction=reduction,
    )["boundary_f1"]


def hd95_score(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate symmetric 95th-percentile Hausdorff distance."""

    return compute_boundary_metrics(
        prediction,
        target,
        prediction_threshold=(
            prediction_threshold
        ),
        target_threshold=(
            target_threshold
        ),
        from_logits=from_logits,
        pixel_spacing=pixel_spacing,
        reduction=reduction,
    )["hd95"]


def assd_score(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    pixel_spacing: Sequence[float] = (
        1.0,
        1.0,
    ),
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate average symmetric surface distance."""

    return compute_boundary_metrics(
        prediction,
        target,
        prediction_threshold=(
            prediction_threshold
        ),
        target_threshold=(
            target_threshold
        ),
        from_logits=from_logits,
        pixel_spacing=pixel_spacing,
        reduction=reduction,
    )["assd"]


def run_boundary_metrics_self_test() -> dict[str, Any]:
    """Run deterministic CPU boundary-metric tests."""

    target = torch.zeros(
        4,
        1,
        8,
        8,
        dtype=torch.float32,
    )

    probability = torch.full(
        (
            4,
            1,
            8,
            8,
        ),
        fill_value=0.1,
        dtype=torch.float32,
    )

    # Sample 0: identical square.
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

    # Sample 1: prediction shifted one pixel to the right.
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

    # Sample 2: both masks empty.

    # Sample 3: target non-empty and prediction empty.
    target[
        3,
        0,
        2:6,
        2:6,
    ] = 1.0

    exact_tolerance_metrics = (
        compute_boundary_metrics(
            probability,
            target,
            from_logits=False,
            boundary_tolerance=0.0,
            reduction="none",
        )
    )

    one_pixel_tolerance_metrics = (
        compute_boundary_metrics(
            probability,
            target,
            from_logits=False,
            boundary_tolerance=1.0,
            reduction="none",
        )
    )

    logits = torch.logit(
        probability.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )

    logits_metrics = (
        compute_boundary_metrics(
            logits,
            target,
            from_logits=True,
            boundary_tolerance=1.0,
            reduction="none",
        )
    )

    expected_penalty = (
        image_diagonal_penalty(
            (
                8,
                8,
            )
        )
    )

    expected_metric_names = (
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "hd95",
        "assd",
    )

    checks = {
        "metric_names": (
            tuple(
                exact_tolerance_metrics
            )
            == expected_metric_names
        ),
        "per_image_shapes": all(
            tuple(
                values.shape
            )
            == (
                4,
            )
            for values
            in exact_tolerance_metrics.values()
        ),
        "identical_boundary_f1": (
            float(
                exact_tolerance_metrics[
                    "boundary_f1"
                ][0].item()
            )
            == 1.0
        ),
        "identical_hd95_zero": (
            float(
                exact_tolerance_metrics[
                    "hd95"
                ][0].item()
            )
            == 0.0
        ),
        "identical_assd_zero": (
            float(
                exact_tolerance_metrics[
                    "assd"
                ][0].item()
            )
            == 0.0
        ),
        "shifted_exact_f1_below_one": (
            float(
                exact_tolerance_metrics[
                    "boundary_f1"
                ][1].item()
            )
            < 1.0
        ),
        "shifted_one_pixel_f1_perfect": (
            float(
                one_pixel_tolerance_metrics[
                    "boundary_f1"
                ][1].item()
            )
            == 1.0
        ),
        "shifted_hd95_positive": (
            float(
                exact_tolerance_metrics[
                    "hd95"
                ][1].item()
            )
            > 0.0
        ),
        "shifted_assd_positive": (
            float(
                exact_tolerance_metrics[
                    "assd"
                ][1].item()
            )
            > 0.0
        ),
        "empty_empty_boundary_f1": (
            float(
                exact_tolerance_metrics[
                    "boundary_f1"
                ][2].item()
            )
            == 1.0
        ),
        "empty_empty_hd95": (
            float(
                exact_tolerance_metrics[
                    "hd95"
                ][2].item()
            )
            == 0.0
        ),
        "empty_empty_assd": (
            float(
                exact_tolerance_metrics[
                    "assd"
                ][2].item()
            )
            == 0.0
        ),
        "one_empty_boundary_f1_zero": (
            float(
                exact_tolerance_metrics[
                    "boundary_f1"
                ][3].item()
            )
            == 0.0
        ),
        "one_empty_hd95_penalty": (
            abs(
                float(
                    exact_tolerance_metrics[
                        "hd95"
                    ][3].item()
                )
                - expected_penalty
            )
            < 1e-12
        ),
        "one_empty_assd_penalty": (
            abs(
                float(
                    exact_tolerance_metrics[
                        "assd"
                    ][3].item()
                )
                - expected_penalty
            )
            < 1e-12
        ),
        "logits_probability_equivalence": all(
            torch.equal(
                logits_metrics[name],
                one_pixel_tolerance_metrics[
                    name
                ],
            )
            for name in expected_metric_names
        ),
        "all_metrics_finite": all(
            torch.isfinite(
                values
            ).all().item()
            for values
            in exact_tolerance_metrics.values()
        ),
        "boundary_scores_bounded": all(
            float(
                exact_tolerance_metrics[
                    name
                ].min().item()
            )
            >= 0.0
            and float(
                exact_tolerance_metrics[
                    name
                ].max().item()
            )
            <= 1.0
            for name in (
                "boundary_precision",
                "boundary_recall",
                "boundary_f1",
            )
        ),
        "distance_scores_nonnegative": (
            float(
                exact_tolerance_metrics[
                    "hd95"
                ].min().item()
            )
            >= 0.0
            and float(
                exact_tolerance_metrics[
                    "assd"
                ].min().item()
            )
            >= 0.0
        ),
        "mean_reduction": (
            torch.allclose(
                boundary_f1_score(
                    probability,
                    target,
                    from_logits=False,
                    tolerance=0.0,
                    reduction="mean",
                ),
                exact_tolerance_metrics[
                    "boundary_f1"
                ].mean(),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "default_tolerance": (
            DEFAULT_BOUNDARY_TOLERANCE_PIXELS
            == 2.0
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
            BOUNDARY_METRICS_PROTOCOL_VERSION
        ),
        "default_boundary_tolerance_pixels": (
            DEFAULT_BOUNDARY_TOLERANCE_PIXELS
        ),
        "checks": checks,
        "exact_tolerance_metrics": {
            name: [
                float(value)
                for value in values.tolist()
            ]
            for name, values
            in exact_tolerance_metrics.items()
        },
        "one_pixel_tolerance_metrics": {
            name: [
                float(value)
                for value in values.tolist()
            ]
            for name, values
            in one_pixel_tolerance_metrics.items()
        },
        "one_empty_distance_penalty": (
            expected_penalty
        ),
    }