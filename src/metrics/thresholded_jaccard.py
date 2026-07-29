"""ISIC-style thresholded Jaccard metric.

The thresholded Jaccard metric is calculated independently for every image:

1. Convert prediction to a binary mask.
2. Calculate the ordinary Jaccard/IoU score.
3. Retain the IoU when it is at least the quality threshold.
4. Replace it with zero when it is below the quality threshold.
5. Average the resulting per-image values for dataset-level reporting.

The approved ISIC threshold is 0.65.

Empty prediction and empty target are treated as a perfect overlap with
ordinary IoU = 1.0, so their thresholded Jaccard value is also 1.0.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from src.metrics.overlap import (
    iou_score,
    reduce_metric,
)


THRESHOLDED_JACCARD_PROTOCOL_VERSION = (
    "BCS-HCTNet-isic-thresholded-jaccard-v1"
)

DEFAULT_JACCARD_QUALITY_THRESHOLD = 0.65

Reduction = Literal[
    "none",
    "mean",
    "sum",
]


def validate_quality_threshold(
    quality_threshold: float,
) -> float:
    """Validate the IoU quality threshold."""

    if isinstance(
        quality_threshold,
        bool,
    ):
        raise TypeError(
            "quality_threshold must be numeric."
        )

    try:
        threshold = float(
            quality_threshold
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "quality_threshold must be numeric."
        ) from error

    if not torch.isfinite(
        torch.tensor(
            threshold,
            dtype=torch.float64,
        )
    ):
        raise ValueError(
            "quality_threshold must be finite."
        )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "quality_threshold must be in [0, 1]."
        )

    return threshold


def apply_jaccard_quality_threshold(
    jaccard_values: Tensor,
    *,
    quality_threshold: float = (
        DEFAULT_JACCARD_QUALITY_THRESHOLD
    ),
) -> Tensor:
    """Apply the ISIC quality threshold to per-image IoU values."""

    if not isinstance(
        jaccard_values,
        Tensor,
    ):
        raise TypeError(
            "jaccard_values must be a torch.Tensor."
        )

    if jaccard_values.ndim != 1:
        raise ValueError(
            "jaccard_values must have shape [B], "
            f"received {tuple(jaccard_values.shape)}."
        )

    if jaccard_values.numel() == 0:
        raise ValueError(
            "jaccard_values cannot be empty."
        )

    if not torch.isfinite(
        jaccard_values
    ).all():
        raise ValueError(
            "jaccard_values contain non-finite values."
        )

    minimum = float(
        jaccard_values.min().item()
    )

    maximum = float(
        jaccard_values.max().item()
    )

    tolerance = 1e-12

    if (
        minimum < -tolerance
        or maximum > 1.0 + tolerance
    ):
        raise ValueError(
            "Jaccard values must be in [0, 1]. "
            f"Observed range: [{minimum}, {maximum}]."
        )

    threshold = validate_quality_threshold(
        quality_threshold
    )

    return torch.where(
        jaccard_values >= threshold,
        jaccard_values,
        torch.zeros_like(
            jaccard_values
        ),
    )


def thresholded_jaccard_score(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    quality_threshold: float = (
        DEFAULT_JACCARD_QUALITY_THRESHOLD
    ),
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate the per-image ISIC thresholded Jaccard score.

    Parameters
    ----------
    prediction:
        Binary-segmentation logits or probabilities.

    target:
        Ground-truth binary masks with values in [0, 1].

    prediction_threshold:
        Probability threshold used to binarize predictions.

    target_threshold:
        Threshold used to binarize target masks.

    quality_threshold:
        Minimum per-image IoU retained by the metric. The approved ISIC
        value is 0.65.

    from_logits:
        Whether ``prediction`` contains logits.

    reduction:
        ``"none"`` returns one value per image. ``"mean"`` returns the
        dataset/batch mean. ``"sum"`` returns the sum.
    """

    per_image_jaccard = iou_score(
        prediction,
        target,
        threshold=prediction_threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction="none",
    )

    thresholded_values = (
        apply_jaccard_quality_threshold(
            per_image_jaccard,
            quality_threshold=quality_threshold,
        )
    )

    return reduce_metric(
        thresholded_values,
        reduction=reduction,
    )


def compute_thresholded_jaccard_details(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    quality_threshold: float = (
        DEFAULT_JACCARD_QUALITY_THRESHOLD
    ),
    from_logits: bool = True,
) -> dict[str, Tensor | float | int]:
    """Return per-image ordinary and thresholded Jaccard details."""

    validated_quality_threshold = (
        validate_quality_threshold(
            quality_threshold
        )
    )

    ordinary_jaccard = iou_score(
        prediction,
        target,
        threshold=prediction_threshold,
        target_threshold=target_threshold,
        from_logits=from_logits,
        reduction="none",
    )

    thresholded_jaccard = (
        apply_jaccard_quality_threshold(
            ordinary_jaccard,
            quality_threshold=(
                validated_quality_threshold
            ),
        )
    )

    passed_quality_threshold = (
        ordinary_jaccard
        >= validated_quality_threshold
    )

    return {
        "ordinary_jaccard": (
            ordinary_jaccard
        ),
        "thresholded_jaccard": (
            thresholded_jaccard
        ),
        "passed_quality_threshold": (
            passed_quality_threshold
        ),
        "quality_threshold": (
            validated_quality_threshold
        ),
        "number_of_images": int(
            ordinary_jaccard.numel()
        ),
        "number_passing_threshold": int(
            passed_quality_threshold
            .sum()
            .item()
        ),
        "pass_rate": float(
            passed_quality_threshold
            .to(
                dtype=torch.float64
            )
            .mean()
            .item()
        ),
        "mean_ordinary_jaccard": float(
            ordinary_jaccard.mean().item()
        ),
        "mean_thresholded_jaccard": float(
            thresholded_jaccard.mean().item()
        ),
    }


def isic_thresholded_jaccard(
    prediction: Tensor,
    target: Tensor,
    *,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate thresholded Jaccard using the approved 0.65 cutoff."""

    return thresholded_jaccard_score(
        prediction,
        target,
        prediction_threshold=prediction_threshold,
        target_threshold=target_threshold,
        quality_threshold=(
            DEFAULT_JACCARD_QUALITY_THRESHOLD
        ),
        from_logits=from_logits,
        reduction=reduction,
    )


def run_thresholded_jaccard_self_test() -> dict[str, Any]:
    """Run deterministic CPU tests with known IoU values."""

    target = torch.zeros(
        4,
        1,
        4,
        5,
        dtype=torch.float32,
    )

    probability = torch.full(
        (
            4,
            1,
            4,
            5,
        ),
        fill_value=0.1,
        dtype=torch.float32,
    )

    # Sample 1: perfect foreground prediction.
    # IoU = 20 / 20 = 1.0.
    target[0] = 1.0
    probability[0] = 0.9

    # Sample 2: exactly at the approved threshold.
    # IoU = 13 / 20 = 0.65, so it must be retained.
    target[1] = 1.0
    probability[
        1
    ].reshape(
        -1
    )[:13] = 0.9

    # Sample 3: below the approved threshold.
    # IoU = 12 / 20 = 0.60, so it must become zero.
    target[2] = 1.0
    probability[
        2
    ].reshape(
        -1
    )[:12] = 0.9

    # Sample 4: empty prediction and empty target.
    # IoU = 1.0 by the approved empty-empty convention.

    details = (
        compute_thresholded_jaccard_details(
            probability,
            target,
            from_logits=False,
        )
    )

    ordinary = details[
        "ordinary_jaccard"
    ]

    thresholded = details[
        "thresholded_jaccard"
    ]

    passed = details[
        "passed_quality_threshold"
    ]

    if not isinstance(
        ordinary,
        Tensor,
    ):
        raise RuntimeError(
            "ordinary_jaccard must be a tensor."
        )

    if not isinstance(
        thresholded,
        Tensor,
    ):
        raise RuntimeError(
            "thresholded_jaccard must be a tensor."
        )

    if not isinstance(
        passed,
        Tensor,
    ):
        raise RuntimeError(
            "passed_quality_threshold must be a tensor."
        )

    expected_ordinary = torch.tensor(
        [
            1.0,
            0.65,
            0.60,
            1.0,
        ],
        dtype=torch.float64,
    )

    expected_thresholded = torch.tensor(
        [
            1.0,
            0.65,
            0.0,
            1.0,
        ],
        dtype=torch.float64,
    )

    expected_passed = torch.tensor(
        [
            True,
            True,
            False,
            True,
        ],
        dtype=torch.bool,
    )

    logits = torch.logit(
        probability.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )

    logits_result = (
        thresholded_jaccard_score(
            logits,
            target,
            from_logits=True,
            reduction="none",
        )
    )

    mean_result = (
        thresholded_jaccard_score(
            probability,
            target,
            from_logits=False,
            reduction="mean",
        )
    )

    sum_result = (
        thresholded_jaccard_score(
            probability,
            target,
            from_logits=False,
            reduction="sum",
        )
    )

    custom_threshold_result = (
        thresholded_jaccard_score(
            probability,
            target,
            quality_threshold=0.60,
            from_logits=False,
            reduction="none",
        )
    )

    expected_custom_threshold = (
        expected_ordinary.clone()
    )

    checks = {
        "ordinary_values_correct": (
            torch.allclose(
                ordinary,
                expected_ordinary,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "thresholded_values_correct": (
            torch.allclose(
                thresholded,
                expected_thresholded,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "threshold_equality_retained": (
            float(
                thresholded[1].item()
            )
            == 0.65
        ),
        "below_threshold_zeroed": (
            float(
                thresholded[2].item()
            )
            == 0.0
        ),
        "empty_empty_retained": (
            float(
                thresholded[3].item()
            )
            == 1.0
        ),
        "pass_flags_correct": (
            torch.equal(
                passed,
                expected_passed,
            )
        ),
        "pass_count_correct": (
            details[
                "number_passing_threshold"
            ]
            == 3
        ),
        "image_count_correct": (
            details[
                "number_of_images"
            ]
            == 4
        ),
        "pass_rate_correct": (
            abs(
                float(
                    details[
                        "pass_rate"
                    ]
                )
                - 0.75
            )
            < 1e-12
        ),
        "logits_probability_equivalence": (
            torch.equal(
                logits_result,
                thresholded,
            )
        ),
        "mean_reduction_correct": (
            torch.allclose(
                mean_result,
                expected_thresholded.mean(),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "sum_reduction_correct": (
            torch.allclose(
                sum_result,
                expected_thresholded.sum(),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "custom_threshold_supported": (
            torch.allclose(
                custom_threshold_result,
                expected_custom_threshold,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "all_values_finite": (
            torch.isfinite(
                ordinary
            ).all().item()
            and torch.isfinite(
                thresholded
            ).all().item()
        ),
        "all_values_bounded": (
            float(
                ordinary.min().item()
            )
            >= 0.0
            and float(
                ordinary.max().item()
            )
            <= 1.0
            and float(
                thresholded.min().item()
            )
            >= 0.0
            and float(
                thresholded.max().item()
            )
            <= 1.0
        ),
        "official_threshold_constant": (
            DEFAULT_JACCARD_QUALITY_THRESHOLD
            == 0.65
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
            THRESHOLDED_JACCARD_PROTOCOL_VERSION
        ),
        "quality_threshold": (
            DEFAULT_JACCARD_QUALITY_THRESHOLD
        ),
        "checks": checks,
        "ordinary_jaccard": [
            float(value)
            for value in ordinary.tolist()
        ],
        "thresholded_jaccard": [
            float(value)
            for value in thresholded.tolist()
        ],
        "passed_quality_threshold": [
            bool(value)
            for value in passed.tolist()
        ],
        "mean_ordinary_jaccard": (
            details[
                "mean_ordinary_jaccard"
            ]
        ),
        "mean_thresholded_jaccard": (
            details[
                "mean_thresholded_jaccard"
            ]
        ),
        "pass_rate": details[
            "pass_rate"
        ],
    }