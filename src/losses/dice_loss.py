"""Differentiable soft Dice loss for segmentation training.

This module implements the shared Dice component used by:

- U-Net;
- UNet++;
- DeepLabV3+;
- TransUNet;
- Swin-UNet;
- the mask and contour branches of BCS-HCTNet.

The implementation calculates Dice independently for each image and then
applies the requested reduction. Per-image calculation prevents large masks
or large batches from dominating the optimization objective.

For prediction probabilities ``p`` and targets ``t``:

    Dice = (2 * sum(p * t) + smooth)
            / (sum(p) + sum(t) + smooth)

    Dice loss = 1 - Dice

The default empty-empty convention gives a Dice score of one and a loss of
zero when both prediction and target are exactly empty.

The loss remains differentiable. No hard thresholding is applied during
training.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


DICE_LOSS_PROTOCOL_VERSION = (
    "BCS-HCTNet-soft-dice-loss-v1"
)

Reduction = Literal[
    "none",
    "mean",
    "sum",
]

SUPPORTED_REDUCTIONS = (
    "none",
    "mean",
    "sum",
)


def _validate_boolean(
    value: object,
    context: str,
) -> bool:
    """Require a Boolean value."""

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be Boolean."
        )

    return value


def _validate_nonnegative_number(
    value: object,
    context: str,
) -> float:
    """Require a finite non-negative numeric value."""

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


def _validate_positive_number(
    value: object,
    context: str,
) -> float:
    """Require a finite positive numeric value."""

    number = _validate_nonnegative_number(
        value,
        context,
    )

    if number <= 0.0:
        raise ValueError(
            f"{context} must be positive."
        )

    return number


def _validate_reduction(
    reduction: object,
) -> Reduction:
    """Validate a supported reduction mode."""

    normalized = str(
        reduction
    ).strip().lower()

    if normalized not in SUPPORTED_REDUCTIONS:
        raise ValueError(
            "reduction must be one of "
            f"{list(SUPPORTED_REDUCTIONS)}, "
            f"received {reduction!r}."
        )

    return normalized  # type: ignore[return-value]


def _normalize_segmentation_tensor(
    tensor: Tensor,
    *,
    context: str,
) -> Tensor:
    """Normalize a segmentation tensor to ``[B, C, H, W]``."""

    if not isinstance(
        tensor,
        Tensor,
    ):
        raise TypeError(
            f"{context} must be a torch.Tensor."
        )

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

    elif tensor.ndim != 4:
        raise ValueError(
            f"{context} must have shape [H, W], "
            "[B, H, W], or [B, C, H, W]. "
            f"Received {tuple(tensor.shape)}."
        )

    if tensor.shape[0] <= 0:
        raise ValueError(
            f"{context} batch cannot be empty."
        )

    if tensor.shape[1] <= 0:
        raise ValueError(
            f"{context} channel dimension "
            "cannot be empty."
        )

    if (
        tensor.shape[-2] <= 0
        or tensor.shape[-1] <= 0
    ):
        raise ValueError(
            f"{context} spatial dimensions "
            "must be positive."
        )

    if not torch.is_floating_point(
        tensor
    ):
        tensor = tensor.to(
            dtype=torch.float32
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{context} contains non-finite values."
        )

    return tensor


def prepare_dice_target(
    target: Tensor,
) -> Tensor:
    """Normalize and validate a Dice target tensor."""

    normalized_target = (
        _normalize_segmentation_tensor(
            target,
            context="target",
        )
    )

    minimum = float(
        normalized_target.min().item()
    )

    maximum = float(
        normalized_target.max().item()
    )

    tolerance = 1e-6

    if (
        minimum < -tolerance
        or maximum > 1.0 + tolerance
    ):
        raise ValueError(
            "Dice target values must be within "
            f"[0, 1]. Observed range: "
            f"[{minimum}, {maximum}]."
        )

    return normalized_target.clamp(
        min=0.0,
        max=1.0,
    )


def prepare_dice_prediction(
    prediction: Tensor,
    *,
    from_logits: bool,
) -> Tensor:
    """Convert logits or probabilities into valid probabilities."""

    resolved_from_logits = (
        _validate_boolean(
            from_logits,
            "from_logits",
        )
    )

    normalized_prediction = (
        _normalize_segmentation_tensor(
            prediction,
            context="prediction",
        )
    )

    if resolved_from_logits:
        probability = torch.sigmoid(
            normalized_prediction
        )

    else:
        minimum = float(
            normalized_prediction.min().item()
        )

        maximum = float(
            normalized_prediction.max().item()
        )

        tolerance = 1e-6

        if (
            minimum < -tolerance
            or maximum > 1.0 + tolerance
        ):
            raise ValueError(
                "Dice probability predictions "
                "must be within [0, 1]. "
                f"Observed range: "
                f"[{minimum}, {maximum}]."
            )

        probability = (
            normalized_prediction.clamp(
                min=0.0,
                max=1.0,
            )
        )

    if not torch.isfinite(
        probability
    ).all():
        raise RuntimeError(
            "Dice probabilities contain "
            "non-finite values."
        )

    return probability


def _resolve_computation_dtype(
    prediction: Tensor,
    target: Tensor,
) -> torch.dtype:
    """Select a numerically stable Dice computation dtype."""

    if (
        prediction.dtype == torch.float64
        or target.dtype == torch.float64
    ):
        return torch.float64

    return torch.float32


def _soft_dice_per_image(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    smooth: float,
    epsilon: float,
) -> Tensor:
    """Calculate one soft Dice score for every image."""

    resolved_smooth = (
        _validate_nonnegative_number(
            smooth,
            "smooth",
        )
    )

    resolved_epsilon = (
        _validate_positive_number(
            epsilon,
            "epsilon",
        )
    )

    probability = prepare_dice_prediction(
        prediction,
        from_logits=from_logits,
    )

    normalized_target = prepare_dice_target(
        target
    )

    if (
        probability.shape
        != normalized_target.shape
    ):
        raise ValueError(
            "Dice prediction and target shapes "
            "must match exactly after "
            "normalization. Prediction: "
            f"{tuple(probability.shape)}; "
            f"target: "
            f"{tuple(normalized_target.shape)}."
        )

    computation_dtype = (
        _resolve_computation_dtype(
            probability,
            normalized_target,
        )
    )

    probability = probability.to(
        dtype=computation_dtype
    )

    normalized_target = (
        normalized_target.to(
            device=probability.device,
            dtype=computation_dtype,
        )
    )

    reduction_dimensions = tuple(
        range(
            1,
            probability.ndim,
        )
    )

    intersection = torch.sum(
        probability
        * normalized_target,
        dim=reduction_dimensions,
    )

    prediction_mass = torch.sum(
        probability,
        dim=reduction_dimensions,
    )

    target_mass = torch.sum(
        normalized_target,
        dim=reduction_dimensions,
    )

    numerator = (
        2.0 * intersection
        + resolved_smooth
    )

    denominator = (
        prediction_mass
        + target_mass
        + resolved_smooth
    )

    denominator = denominator.clamp_min(
        resolved_epsilon
    )

    score = numerator / denominator

    score = score.clamp(
        min=0.0,
        max=1.0,
    )

    if score.ndim != 1:
        raise RuntimeError(
            "Internal Dice score must contain "
            "one value per image."
        )

    if score.shape[0] != probability.shape[0]:
        raise RuntimeError(
            "Internal Dice score batch size "
            "is invalid."
        )

    if not torch.isfinite(
        score
    ).all():
        raise RuntimeError(
            "Soft Dice score contains "
            "non-finite values."
        )

    return score


def _apply_reduction(
    values: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Apply a supported reduction."""

    if reduction == "none":
        return values

    if reduction == "mean":
        return values.mean()

    if reduction == "sum":
        return values.sum()

    raise AssertionError(
        "Unreachable Dice reduction branch."
    )


def soft_dice_score(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    epsilon: float = 1e-7,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate differentiable soft Dice score.

    Parameters
    ----------
    prediction:
        Segmentation logits when ``from_logits=True`` or probabilities when
        ``from_logits=False``.

    target:
        Binary or soft target values within ``[0, 1]``.

    from_logits:
        Apply sigmoid to the prediction before calculating Dice.

    smooth:
        Additive smoothing used in the numerator and denominator.

    epsilon:
        Minimum permitted denominator.

    reduction:
        ``"none"`` returns one value per image. ``"mean"`` and ``"sum"``
        aggregate those per-image values.
    """

    resolved_reduction = (
        _validate_reduction(
            reduction
        )
    )

    scores = _soft_dice_per_image(
        prediction,
        target,
        from_logits=from_logits,
        smooth=smooth,
        epsilon=epsilon,
    )

    return _apply_reduction(
        scores,
        resolved_reduction,
    )


def dice_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    epsilon: float = 1e-7,
    reduction: Reduction = "mean",
) -> Tensor:
    """Calculate differentiable soft Dice loss."""

    resolved_reduction = (
        _validate_reduction(
            reduction
        )
    )

    scores = _soft_dice_per_image(
        prediction,
        target,
        from_logits=from_logits,
        smooth=smooth,
        epsilon=epsilon,
    )

    losses = 1.0 - scores

    if not torch.isfinite(
        losses
    ).all():
        raise RuntimeError(
            "Soft Dice loss contains "
            "non-finite values."
        )

    return _apply_reduction(
        losses,
        resolved_reduction,
    )


class SoftDiceLoss(nn.Module):
    """PyTorch module wrapper for per-image soft Dice loss."""

    def __init__(
        self,
        *,
        from_logits: bool = True,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialize the Dice loss module."""

        super().__init__()

        self.from_logits = (
            _validate_boolean(
                from_logits,
                "from_logits",
            )
        )

        self.smooth = (
            _validate_nonnegative_number(
                smooth,
                "smooth",
            )
        )

        self.epsilon = (
            _validate_positive_number(
                epsilon,
                "epsilon",
            )
        )

        self.reduction = (
            _validate_reduction(
                reduction
            )
        )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculate soft Dice loss."""

        return dice_loss(
            prediction,
            target,
            from_logits=(
                self.from_logits
            ),
            smooth=self.smooth,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )

    def configuration(
        self,
    ) -> dict[str, object]:
        """Return serializable loss configuration."""

        return {
            "protocol_version": (
                DICE_LOSS_PROTOCOL_VERSION
            ),
            "name": "soft_dice_loss",
            "from_logits": (
                self.from_logits
            ),
            "smooth": self.smooth,
            "epsilon": self.epsilon,
            "reduction": self.reduction,
            "per_image": True,
            "hard_thresholding": False,
        }


DiceLoss = SoftDiceLoss


def run_dice_loss_self_test() -> dict[str, object]:
    """Run deterministic CPU Dice-score and gradient tests."""

    torch.manual_seed(
        42
    )

    perfect_prediction = torch.tensor(
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
        ],
        dtype=torch.float32,
    )

    perfect_target = (
        perfect_prediction.clone()
    )

    perfect_scores = soft_dice_score(
        perfect_prediction,
        perfect_target,
        from_logits=False,
        smooth=1.0,
        reduction="none",
    )

    perfect_losses = dice_loss(
        perfect_prediction,
        perfect_target,
        from_logits=False,
        smooth=1.0,
        reduction="none",
    )

    imperfect_prediction = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    imperfect_target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    imperfect_score = soft_dice_score(
        imperfect_prediction,
        imperfect_target,
        from_logits=False,
        smooth=1.0,
        reduction="mean",
    )

    logits = torch.randn(
        3,
        1,
        8,
        8,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            3,
            1,
            8,
            8,
        )
        > 0.5
    ).to(
        dtype=torch.float32
    )

    loss_module = SoftDiceLoss(
        from_logits=True,
        smooth=1.0,
        epsilon=1e-7,
        reduction="mean",
    )

    training_loss = loss_module(
        logits,
        target,
    )

    training_loss.backward()

    probabilities = torch.sigmoid(
        logits.detach()
    )

    logit_scores = soft_dice_score(
        logits.detach(),
        target,
        from_logits=True,
        reduction="none",
    )

    probability_scores = (
        soft_dice_score(
            probabilities,
            target,
            from_logits=False,
            reduction="none",
        )
    )

    no_reduction_loss = dice_loss(
        logits.detach(),
        target,
        from_logits=True,
        reduction="none",
    )

    mean_loss = dice_loss(
        logits.detach(),
        target,
        from_logits=True,
        reduction="mean",
    )

    sum_loss = dice_loss(
        logits.detach(),
        target,
        from_logits=True,
        reduction="sum",
    )

    invalid_target_rejected = False

    try:
        dice_loss(
            torch.zeros(
                1,
                1,
                2,
                2,
            ),
            torch.full(
                (
                    1,
                    1,
                    2,
                    2,
                ),
                fill_value=255.0,
            ),
        )

    except ValueError:
        invalid_target_rejected = True

    shape_mismatch_rejected = False

    try:
        dice_loss(
            torch.zeros(
                1,
                1,
                4,
                4,
            ),
            torch.zeros(
                1,
                1,
                3,
                4,
            ),
        )

    except ValueError:
        shape_mismatch_rejected = True

    configuration = (
        loss_module.configuration()
    )

    checks = {
        "perfect_scores_equal_one": (
            torch.equal(
                perfect_scores,
                torch.ones_like(
                    perfect_scores
                ),
            )
        ),
        "perfect_losses_equal_zero": (
            torch.equal(
                perfect_losses,
                torch.zeros_like(
                    perfect_losses
                ),
            )
        ),
        "empty_empty_score_is_one": (
            perfect_scores[1].item()
            == 1.0
        ),
        "imperfect_score_bounded": (
            0.0
            < imperfect_score.item()
            < 1.0
        ),
        "training_loss_scalar": (
            training_loss.ndim == 0
        ),
        "training_loss_finite": (
            torch.isfinite(
                training_loss
            ).item()
        ),
        "training_loss_bounded": (
            0.0
            <= training_loss.item()
            <= 1.0
        ),
        "training_loss_requires_gradient": (
            training_loss.requires_grad
        ),
        "logit_gradient_exists": (
            logits.grad is not None
        ),
        "logit_gradient_finite": (
            logits.grad is not None
            and torch.isfinite(
                logits.grad
            ).all().item()
        ),
        "logit_gradient_nonzero": (
            logits.grad is not None
            and float(
                logits.grad.abs().sum().item()
            )
            > 0.0
        ),
        "logits_probabilities_equivalent": (
            torch.allclose(
                logit_scores,
                probability_scores,
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "none_reduction_shape": (
            tuple(
                no_reduction_loss.shape
            )
            == (
                3,
            )
        ),
        "mean_reduction_correct": (
            torch.allclose(
                mean_loss,
                no_reduction_loss.mean(),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "sum_reduction_correct": (
            torch.allclose(
                sum_loss,
                no_reduction_loss.sum(),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "invalid_target_rejected": (
            invalid_target_rejected
        ),
        "shape_mismatch_rejected": (
            shape_mismatch_rejected
        ),
        "configuration_correct": (
            configuration[
                "protocol_version"
            ]
            == DICE_LOSS_PROTOCOL_VERSION
            and configuration[
                "per_image"
            ]
            is True
            and configuration[
                "hard_thresholding"
            ]
            is False
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
            DICE_LOSS_PROTOCOL_VERSION
        ),
        "checks": checks,
        "perfect_scores": (
            perfect_scores.tolist()
        ),
        "imperfect_score": float(
            imperfect_score.item()
        ),
        "training_loss": float(
            training_loss.detach().item()
        ),
        "per_image_losses": (
            no_reduction_loss.tolist()
        ),
        "configuration": configuration,
    }