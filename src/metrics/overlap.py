"""Binary segmentation overlap metrics.

This module provides deterministic, per-image metrics shared by all baseline
models and the proposed BCS-HCTNet model.

Implemented metrics
-------------------
- Dice coefficient
- Intersection over Union / Jaccard
- Precision
- Recall / sensitivity
- Specificity
- Pixel accuracy

Evaluation policy
-----------------
Predictions may be supplied as logits or probabilities. They are converted to
binary masks with a fixed threshold. Ground-truth targets must represent binary
masks using values in [0, 1].

Metrics are calculated per image before optional reduction. This prevents large
lesions from dominating dataset-level results.

For an empty prediction and empty target:

- Dice = 1
- IoU = 1
- Precision = 1
- Recall = 1
- Specificity = 1
- Accuracy = 1
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor


OVERLAP_METRICS_PROTOCOL_VERSION = (
    "BCS-HCTNet-binary-overlap-metrics-v1"
)

Reduction = Literal[
    "none",
    "mean",
    "sum",
]


def _require_tensor(
    value: object,
    context: str,
) -> Tensor:
    """Require a finite floating-point or integer tensor."""

    if not isinstance(
        value,
        Tensor,
    ):
        raise TypeError(
            f"{context} must be a torch.Tensor."
        )

    if value.ndim not in {
        2,
        3,
        4,
    }:
        raise ValueError(
            f"{context} must have 2, 3, or 4 "
            f"dimensions, received shape "
            f"{tuple(value.shape)}."
        )

    if value.numel() == 0:
        raise ValueError(
            f"{context} cannot be empty."
        )

    if not torch.isfinite(
        value
    ).all():
        raise ValueError(
            f"{context} contains non-finite values."
        )

    return value


def _normalize_binary_shape(
    tensor: Tensor,
    context: str,
) -> Tensor:
    """Normalize a binary segmentation tensor to [B, 1, H, W]."""

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(
            1
        )

    if tensor.shape[1] != 1:
        raise ValueError(
            f"{context} must contain exactly one "
            f"segmentation channel, received "
            f"{tensor.shape[1]} channels."
        )

    if (
        tensor.shape[-2] <= 0
        or tensor.shape[-1] <= 0
    ):
        raise ValueError(
            f"{context} has an invalid spatial shape."
        )

    return tensor


def prepare_binary_target(
    target: Tensor,
    *,
    target_threshold: float = 0.5,
) -> Tensor:
    """Validate and convert a target mask to Boolean form."""

    target = _require_tensor(
        target,
        "target",
    )

    target = _normalize_binary_shape(
        target,
        "target",
    )

    threshold = float(
        target_threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "target_threshold must be in [0, 1]."
        )

    minimum = float(
        target.min().item()
    )

    maximum = float(
        target.max().item()
    )

    tolerance = 1e-6

    if (
        minimum < -tolerance
        or maximum > 1.0 + tolerance
    ):
        raise ValueError(
            "Target mask values must be in [0, 1]. "
            f"Observed range: [{minimum}, {maximum}]."
        )

    return target >= threshold


def prepare_binary_prediction(
    prediction: Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Tensor:
    """Convert model output to a Boolean segmentation mask."""

    prediction = _require_tensor(
        prediction,
        "prediction",
    )

    prediction = _normalize_binary_shape(
        prediction,
        "prediction",
    )

    if not isinstance(
        from_logits,
        bool,
    ):
        raise TypeError(
            "from_logits must be Boolean."
        )

    probability_threshold = float(
        threshold
    )

    if not (
        0.0
        <= probability_threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be in [0, 1]."
        )

    if from_logits:
        probabilities = torch.sigmoid(
            prediction
        )

    else:
        minimum = float(
            prediction.min().item()
        )

        maximum = float(
            prediction.max().item()
        )

        tolerance = 1e-6

        if (
            minimum < -tolerance
            or maximum > 1.0 + tolerance
        ):
            raise ValueError(
                "Probability predictions must be "
                "in [0, 1]. Observed range: "
                f"[{minimum}, {maximum}]."
            )

        probabilities = prediction

    return (
        probabilities
        >= probability_threshold
    )


def prepare_binary_pair(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
) -> tuple[
    Tensor,
    Tensor,
]:
    """Prepare shape-compatible prediction and target masks."""

    binary_prediction = (
        prepare_binary_prediction(
            prediction,
            threshold=threshold,
            from_logits=from_logits,
        )
    )

    binary_target = prepare_binary_target(
        target,
        target_threshold=(
            target_threshold
        ),
    )

    if (
        binary_prediction.shape
        != binary_target.shape
    ):
        raise ValueError(
            "Prediction and target shapes must "
            "match after normalization. "
            f"Prediction: "
            f"{tuple(binary_prediction.shape)}; "
            f"target: {tuple(binary_target.shape)}."
        )

    return (
        binary_prediction,
        binary_target,
    )


def binary_confusion_counts(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, Tensor]:
    """Calculate per-image binary confusion counts."""

    (
        binary_prediction,
        binary_target,
    ) = prepare_binary_pair(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
    )

    reduction_dimensions = (
        1,
        2,
        3,
    )

    true_positive = (
        binary_prediction
        & binary_target
    ).sum(
        dim=reduction_dimensions,
        dtype=torch.float64,
    )

    false_positive = (
        binary_prediction
        & ~binary_target
    ).sum(
        dim=reduction_dimensions,
        dtype=torch.float64,
    )

    false_negative = (
        ~binary_prediction
        & binary_target
    ).sum(
        dim=reduction_dimensions,
        dtype=torch.float64,
    )

    true_negative = (
        ~binary_prediction
        & ~binary_target
    ).sum(
        dim=reduction_dimensions,
        dtype=torch.float64,
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _safe_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    zero_denominator_value: float,
) -> Tensor:
    """Divide safely with an explicit empty-case value."""

    replacement = torch.full_like(
        denominator,
        fill_value=float(
            zero_denominator_value
        ),
        dtype=torch.float64,
    )

    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        replacement,
    )


def reduce_metric(
    values: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Apply an approved metric reduction."""

    if reduction == "none":
        return values

    if reduction == "mean":
        return values.mean()

    if reduction == "sum":
        return values.sum()

    raise ValueError(
        "reduction must be one of "
        "'none', 'mean', or 'sum'."
    )


def metrics_from_confusion_counts(
    counts: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Calculate per-image metrics from confusion counts."""

    required_keys = {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }

    missing = sorted(
        required_keys
        - set(
            counts
        )
    )

    if missing:
        raise KeyError(
            f"Confusion counts are missing: {missing}."
        )

    true_positive = counts[
        "true_positive"
    ].to(
        dtype=torch.float64
    )

    false_positive = counts[
        "false_positive"
    ].to(
        dtype=torch.float64
    )

    false_negative = counts[
        "false_negative"
    ].to(
        dtype=torch.float64
    )

    true_negative = counts[
        "true_negative"
    ].to(
        dtype=torch.float64
    )

    dice_denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    specificity_denominator = (
        true_negative
        + false_positive
    )

    accuracy_denominator = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    return {
        "dice": _safe_ratio(
            2.0 * true_positive,
            dice_denominator,
            zero_denominator_value=1.0,
        ),
        "iou": _safe_ratio(
            true_positive,
            iou_denominator,
            zero_denominator_value=1.0,
        ),
        "precision": _safe_ratio(
            true_positive,
            precision_denominator,
            zero_denominator_value=1.0,
        ),
        "recall": _safe_ratio(
            true_positive,
            recall_denominator,
            zero_denominator_value=1.0,
        ),
        "specificity": _safe_ratio(
            true_negative,
            specificity_denominator,
            zero_denominator_value=1.0,
        ),
        "accuracy": _safe_ratio(
            true_positive
            + true_negative,
            accuracy_denominator,
            zero_denominator_value=1.0,
        ),
    }


def compute_overlap_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> dict[str, Tensor]:
    """Calculate all overlap metrics.

    With ``reduction="none"``, every returned metric has shape ``[B]``.
    With ``mean`` or ``sum``, every returned metric is scalar.
    """

    counts = binary_confusion_counts(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
    )

    metrics = (
        metrics_from_confusion_counts(
            counts
        )
    )

    return {
        name: reduce_metric(
            values,
            reduction=reduction,
        )
        for name, values
        in metrics.items()
    }


def dice_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate the binary Dice coefficient."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["dice"]


def iou_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate binary Intersection over Union."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["iou"]


def precision_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate binary segmentation precision."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["precision"]


def recall_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate binary segmentation recall/sensitivity."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["recall"]


def specificity_score(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate binary segmentation specificity."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["specificity"]


def pixel_accuracy(
    prediction: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate binary pixel accuracy."""

    return compute_overlap_metrics(
        prediction,
        target,
        threshold=threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction=reduction,
    )["accuracy"]


def run_overlap_metrics_self_test() -> dict[str, Any]:
    """Run deterministic CPU tests using known binary examples."""

    target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    probability = torch.tensor(
        [
            [
                [
                    [0.9, 0.8],
                    [0.1, 0.2],
                ]
            ],
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ],
            [
                [
                    [0.9, 0.8],
                    [0.1, 0.2],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    metrics = compute_overlap_metrics(
        probability,
        target,
        threshold=0.5,
        from_logits=False,
        reduction="none",
    )

    expected = {
        "dice": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "iou": torch.tensor(
            [
                1.0,
                1.0,
                1.0 / 3.0,
            ],
            dtype=torch.float64,
        ),
        "precision": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "recall": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "specificity": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "accuracy": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
    }

    logits = torch.logit(
        probability.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )

    logits_metrics = compute_overlap_metrics(
        logits,
        target,
        threshold=0.5,
        from_logits=True,
        reduction="none",
    )

    confusion_counts = binary_confusion_counts(
        probability,
        target,
        threshold=0.5,
        from_logits=False,
    )

    checks = {
        "metric_names": (
            tuple(metrics)
            == (
                "dice",
                "iou",
                "precision",
                "recall",
                "specificity",
                "accuracy",
            )
        ),
        "known_values": all(
            torch.allclose(
                metrics[name],
                expected[name],
                atol=1e-12,
                rtol=0.0,
            )
            for name in expected
        ),
        "logits_probability_equivalence": all(
            torch.equal(
                metrics[name],
                logits_metrics[name],
            )
            for name in metrics
        ),
        "per_image_shape": all(
            tuple(
                value.shape
            )
            == (
                3,
            )
            for value in metrics.values()
        ),
        "all_metrics_finite": all(
            torch.isfinite(
                value
            ).all().item()
            for value in metrics.values()
        ),
        "all_metrics_bounded": all(
            float(
                value.min().item()
            )
            >= 0.0
            and float(
                value.max().item()
            )
            <= 1.0
            for value in metrics.values()
        ),
        "empty_empty_is_perfect": all(
            float(
                metrics[name][1].item()
            )
            == 1.0
            for name in metrics
        ),
        "mean_reduction": (
            torch.allclose(
                dice_score(
                    probability,
                    target,
                    from_logits=False,
                    reduction="mean",
                ),
                expected[
                    "dice"
                ].mean(),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "sum_reduction": (
            torch.allclose(
                iou_score(
                    probability,
                    target,
                    from_logits=False,
                    reduction="sum",
                ),
                expected[
                    "iou"
                ].sum(),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "confusion_counts_correct": (
            torch.equal(
                confusion_counts[
                    "true_positive"
                ],
                torch.tensor(
                    [
                        2.0,
                        0.0,
                        1.0,
                    ],
                    dtype=torch.float64,
                ),
            )
            and torch.equal(
                confusion_counts[
                    "false_positive"
                ],
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=torch.float64,
                ),
            )
            and torch.equal(
                confusion_counts[
                    "false_negative"
                ],
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=torch.float64,
                ),
            )
            and torch.equal(
                confusion_counts[
                    "true_negative"
                ],
                torch.tensor(
                    [
                        2.0,
                        4.0,
                        1.0,
                    ],
                    dtype=torch.float64,
                ),
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
            OVERLAP_METRICS_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_metrics": {
            name: [
                float(item)
                for item in value.tolist()
            ]
            for name, value
            in metrics.items()
        },
        "expected_metrics": {
            name: [
                float(item)
                for item in value.tolist()
            ]
            for name, value
            in expected.items()
        },
        "confusion_counts": {
            name: [
                float(item)
                for item in value.tolist()
            ]
            for name, value
            in confusion_counts.items()
        },
    }