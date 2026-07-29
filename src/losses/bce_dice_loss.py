"""Combined BCE and soft Dice loss for binary segmentation.

This is the shared mask objective for all supervised baseline models:

- U-Net;
- UNet++;
- DeepLabV3+;
- TransUNet;
- Swin-UNet.

The same mask-loss implementation can later be used inside the composite
BCS-HCTNet objective.

The loss is calculated per image:

    total = bce_weight * BCEWithLogits
            + dice_weight * SoftDiceLoss

The default configuration gives equal importance to BCE and Dice by assigning
both a weight of one. No thresholding is applied during optimization.

The criterion can return either:

    Tensor

or a component mapping compatible with the shared trainer:

    {
        "total_loss": Tensor,
        "bce_loss": Tensor,
        "dice_loss": Tensor,
    }
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.losses.dice_loss import (
    DICE_LOSS_PROTOCOL_VERSION,
    dice_loss,
    prepare_dice_target,
)


BCE_DICE_LOSS_PROTOCOL_VERSION = (
    "BCS-HCTNet-bce-dice-loss-v1"
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
    """Require a finite non-negative number."""

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
    """Require a finite positive number."""

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
    """Validate a loss reduction mode."""

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


def prepare_binary_logits(
    prediction: Tensor,
) -> Tensor:
    """Normalize binary segmentation logits to ``[B, 1, H, W]``."""

    if not isinstance(
        prediction,
        Tensor,
    ):
        raise TypeError(
            "prediction must be a torch.Tensor."
        )

    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(
            0
        ).unsqueeze(
            0
        )

    elif prediction.ndim == 3:
        prediction = prediction.unsqueeze(
            1
        )

    elif prediction.ndim != 4:
        raise ValueError(
            "prediction must have shape [H, W], "
            "[B, H, W], or [B, 1, H, W]. "
            f"Received {tuple(prediction.shape)}."
        )

    if prediction.shape[0] <= 0:
        raise ValueError(
            "Prediction batch cannot be empty."
        )

    if prediction.shape[1] != 1:
        raise ValueError(
            "Binary segmentation logits must "
            "contain exactly one channel."
        )

    if (
        prediction.shape[-2] <= 0
        or prediction.shape[-1] <= 0
    ):
        raise ValueError(
            "Prediction spatial dimensions "
            "must be positive."
        )

    if not torch.is_floating_point(
        prediction
    ):
        prediction = prediction.to(
            dtype=torch.float32
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Prediction logits contain "
            "non-finite values."
        )

    return prediction


def _prepare_binary_target(
    target: Tensor,
    *,
    reference: Tensor,
) -> Tensor:
    """Normalize a target and align it with prediction logits."""

    normalized_target = prepare_dice_target(
        target
    )

    if normalized_target.shape[1] != 1:
        raise ValueError(
            "Binary segmentation target must "
            "contain exactly one channel."
        )

    if (
        normalized_target.shape
        != reference.shape
    ):
        raise ValueError(
            "Prediction and target shapes must "
            "match exactly. Prediction: "
            f"{tuple(reference.shape)}; target: "
            f"{tuple(normalized_target.shape)}."
        )

    normalized_target = (
        normalized_target.to(
            device=reference.device,
            dtype=reference.dtype,
        )
    )

    return normalized_target


def _resolve_pos_weight(
    pos_weight: float | None,
    *,
    reference: Tensor,
) -> Tensor | None:
    """Create the BCE positive-class weight tensor."""

    if pos_weight is None:
        return None

    resolved_weight = (
        _validate_positive_number(
            pos_weight,
            "pos_weight",
        )
    )

    return torch.tensor(
        [
            resolved_weight,
        ],
        device=reference.device,
        dtype=reference.dtype,
    )


def _apply_reduction(
    values: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Apply a supported reduction to per-image values."""

    if reduction == "none":
        return values

    if reduction == "mean":
        return values.mean()

    if reduction == "sum":
        return values.sum()

    raise AssertionError(
        "Unreachable reduction branch."
    )


def binary_cross_entropy_per_image(
    prediction: Tensor,
    target: Tensor,
    *,
    pos_weight: float | None = None,
) -> Tensor:
    """Calculate mean pixel BCE independently for every image."""

    logits = prepare_binary_logits(
        prediction
    )

    normalized_target = (
        _prepare_binary_target(
            target,
            reference=logits,
        )
    )

    resolved_pos_weight = (
        _resolve_pos_weight(
            pos_weight,
            reference=logits,
        )
    )

    pixel_losses = (
        F.binary_cross_entropy_with_logits(
            logits,
            normalized_target,
            pos_weight=(
                resolved_pos_weight
            ),
            reduction="none",
        )
    )

    per_image_losses = (
        pixel_losses.flatten(
            start_dim=1
        ).mean(
            dim=1
        )
    )

    if per_image_losses.ndim != 1:
        raise RuntimeError(
            "Internal BCE calculation must "
            "produce one value per image."
        )

    if not torch.isfinite(
        per_image_losses
    ).all():
        raise RuntimeError(
            "Per-image BCE contains "
            "non-finite values."
        )

    return per_image_losses


def compute_bce_dice_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
    dice_smooth: float = 1.0,
    dice_epsilon: float = 1e-7,
    reduction: Reduction = "mean",
) -> dict[str, Tensor]:
    """Calculate the combined objective and its components.

    Parameters
    ----------
    prediction:
        One-channel binary segmentation logits.

    target:
        Binary or soft mask target with values in ``[0, 1]``.

    bce_weight:
        Multiplicative weight for BCEWithLogits.

    dice_weight:
        Multiplicative weight for soft Dice loss.

    pos_weight:
        Optional positive-class weighting for BCE. The default is ``None``,
        because Dice already helps compensate for foreground imbalance.

    dice_smooth:
        Dice numerator and denominator smoothing.

    dice_epsilon:
        Minimum Dice denominator.

    reduction:
        ``"none"`` returns one value per image. ``"mean"`` and ``"sum"``
        aggregate per-image values.
    """

    resolved_bce_weight = (
        _validate_nonnegative_number(
            bce_weight,
            "bce_weight",
        )
    )

    resolved_dice_weight = (
        _validate_nonnegative_number(
            dice_weight,
            "dice_weight",
        )
    )

    if (
        resolved_bce_weight == 0.0
        and resolved_dice_weight == 0.0
    ):
        raise ValueError(
            "At least one of bce_weight or "
            "dice_weight must be positive."
        )

    resolved_dice_smooth = (
        _validate_nonnegative_number(
            dice_smooth,
            "dice_smooth",
        )
    )

    resolved_dice_epsilon = (
        _validate_positive_number(
            dice_epsilon,
            "dice_epsilon",
        )
    )

    resolved_reduction = (
        _validate_reduction(
            reduction
        )
    )

    logits = prepare_binary_logits(
        prediction
    )

    normalized_target = (
        _prepare_binary_target(
            target,
            reference=logits,
        )
    )

    per_image_bce = (
        binary_cross_entropy_per_image(
            logits,
            normalized_target,
            pos_weight=pos_weight,
        )
    )

    per_image_dice = dice_loss(
        logits,
        normalized_target,
        from_logits=True,
        smooth=resolved_dice_smooth,
        epsilon=resolved_dice_epsilon,
        reduction="none",
    )

    if (
        per_image_bce.shape
        != per_image_dice.shape
    ):
        raise RuntimeError(
            "BCE and Dice per-image loss "
            "shapes do not match."
        )

    per_image_total = (
        resolved_bce_weight
        * per_image_bce
        + resolved_dice_weight
        * per_image_dice
    )

    if not torch.isfinite(
        per_image_total
    ).all():
        raise RuntimeError(
            "Combined BCE-Dice loss contains "
            "non-finite values."
        )

    return {
        "total_loss": _apply_reduction(
            per_image_total,
            resolved_reduction,
        ),
        "bce_loss": _apply_reduction(
            per_image_bce,
            resolved_reduction,
        ),
        "dice_loss": _apply_reduction(
            per_image_dice,
            resolved_reduction,
        ),
    }


def bce_dice_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
    dice_smooth: float = 1.0,
    dice_epsilon: float = 1e-7,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return only the combined BCE-Dice loss tensor."""

    components = compute_bce_dice_loss(
        prediction,
        target,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        dice_smooth=dice_smooth,
        dice_epsilon=dice_epsilon,
        reduction=reduction,
    )

    return components[
        "total_loss"
    ]


class BCEDiceLoss(nn.Module):
    """Combined BCEWithLogits and soft Dice criterion."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1e-7,
        reduction: Reduction = "mean",
        return_components: bool = True,
    ) -> None:
        """Initialize the shared mask criterion."""

        super().__init__()

        self.bce_weight = (
            _validate_nonnegative_number(
                bce_weight,
                "bce_weight",
            )
        )

        self.dice_weight = (
            _validate_nonnegative_number(
                dice_weight,
                "dice_weight",
            )
        )

        if (
            self.bce_weight == 0.0
            and self.dice_weight == 0.0
        ):
            raise ValueError(
                "At least one loss weight "
                "must be positive."
            )

        self.pos_weight = (
            None
            if pos_weight is None
            else _validate_positive_number(
                pos_weight,
                "pos_weight",
            )
        )

        self.dice_smooth = (
            _validate_nonnegative_number(
                dice_smooth,
                "dice_smooth",
            )
        )

        self.dice_epsilon = (
            _validate_positive_number(
                dice_epsilon,
                "dice_epsilon",
            )
        )

        self.reduction = (
            _validate_reduction(
                reduction
            )
        )

        self.return_components = (
            _validate_boolean(
                return_components,
                "return_components",
            )
        )

        if (
            self.return_components
            and self.reduction == "none"
        ):
            raise ValueError(
                "return_components=True requires "
                "'mean' or 'sum' reduction for "
                "shared-trainer compatibility."
            )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor | dict[str, Tensor]:
        """Calculate the configured mask objective."""

        components = compute_bce_dice_loss(
            prediction,
            target,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            pos_weight=self.pos_weight,
            dice_smooth=self.dice_smooth,
            dice_epsilon=(
                self.dice_epsilon
            ),
            reduction=self.reduction,
        )

        if self.return_components:
            return components

        return components[
            "total_loss"
        ]

    def configuration(
        self,
    ) -> dict[str, object]:
        """Return serializable criterion metadata."""

        return {
            "protocol_version": (
                BCE_DICE_LOSS_PROTOCOL_VERSION
            ),
            "dice_loss_protocol_version": (
                DICE_LOSS_PROTOCOL_VERSION
            ),
            "name": "bce_dice_loss",
            "task": (
                "binary_segmentation"
            ),
            "learning_type": (
                "fully_supervised"
            ),
            "bce_weight": (
                self.bce_weight
            ),
            "dice_weight": (
                self.dice_weight
            ),
            "pos_weight": (
                self.pos_weight
            ),
            "dice_smooth": (
                self.dice_smooth
            ),
            "dice_epsilon": (
                self.dice_epsilon
            ),
            "reduction": (
                self.reduction
            ),
            "return_components": (
                self.return_components
            ),
            "per_image": True,
            "from_logits": True,
            "hard_thresholding": False,
            "uses_auxiliary_targets": False,
        }


BCEAndDiceLoss = BCEDiceLoss
BaselineMaskLoss = BCEDiceLoss


def run_bce_dice_loss_self_test() -> dict[str, object]:
    """Run deterministic CPU component and gradient tests."""

    torch.manual_seed(
        42
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

    criterion = BCEDiceLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=None,
        dice_smooth=1.0,
        dice_epsilon=1e-7,
        reduction="mean",
        return_components=True,
    )

    components = criterion(
        logits,
        target,
    )

    if not isinstance(
        components,
        dict,
    ):
        raise RuntimeError(
            "Self-test criterion did not "
            "return components."
        )

    total_loss = components[
        "total_loss"
    ]

    total_loss.backward()

    manual_total = (
        components[
            "bce_loss"
        ]
        + components[
            "dice_loss"
        ]
    )

    scalar_criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        reduction="mean",
        return_components=False,
    )

    scalar_loss = scalar_criterion(
        logits.detach(),
        target,
    )

    per_image = compute_bce_dice_loss(
        logits.detach(),
        target,
        bce_weight=1.0,
        dice_weight=1.0,
        reduction="none",
    )

    mean_components = (
        compute_bce_dice_loss(
            logits.detach(),
            target,
            bce_weight=1.0,
            dice_weight=1.0,
            reduction="mean",
        )
    )

    sum_components = (
        compute_bce_dice_loss(
            logits.detach(),
            target,
            bce_weight=1.0,
            dice_weight=1.0,
            reduction="sum",
        )
    )

    positive_logits = torch.zeros(
        1,
        1,
        2,
        2,
    )

    positive_target = torch.ones_like(
        positive_logits
    )

    unweighted_bce = (
        binary_cross_entropy_per_image(
            positive_logits,
            positive_target,
            pos_weight=None,
        )
    )

    weighted_bce = (
        binary_cross_entropy_per_image(
            positive_logits,
            positive_target,
            pos_weight=2.0,
        )
    )

    invalid_shape_rejected = False

    try:
        bce_dice_loss(
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
        invalid_shape_rejected = True

    zero_weights_rejected = False

    try:
        BCEDiceLoss(
            bce_weight=0.0,
            dice_weight=0.0,
        )

    except ValueError:
        zero_weights_rejected = True

    invalid_pos_weight_rejected = False

    try:
        BCEDiceLoss(
            pos_weight=0.0
        )

    except ValueError:
        invalid_pos_weight_rejected = True

    incompatible_component_mode_rejected = (
        False
    )

    try:
        BCEDiceLoss(
            reduction="none",
            return_components=True,
        )

    except ValueError:
        incompatible_component_mode_rejected = (
            True
        )

    configuration = (
        criterion.configuration()
    )

    checks = {
        "component_keys_correct": (
            tuple(
                components
            )
            == (
                "total_loss",
                "bce_loss",
                "dice_loss",
            )
        ),
        "all_components_scalar": all(
            value.ndim == 0
            for value
            in components.values()
        ),
        "all_components_finite": all(
            torch.isfinite(
                value
            ).item()
            for value
            in components.values()
        ),
        "total_matches_components": (
            torch.allclose(
                total_loss,
                manual_total,
                rtol=0.0,
                atol=(
                    10.0
                    * torch.finfo(
                        total_loss.dtype
                    ).eps
                ),
            )
        ),
        "total_requires_gradient": (
            total_loss.requires_grad
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
                logits.grad
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "scalar_mode_returns_tensor": (
            isinstance(
                scalar_loss,
                Tensor,
            )
            and scalar_loss.ndim == 0
        ),
        "per_image_shape": (
            tuple(
                per_image[
                    "total_loss"
                ].shape
            )
            == (
                3,
            )
        ),
        "mean_reduction_correct": (
            torch.allclose(
                mean_components[
                    "total_loss"
                ],
                per_image[
                    "total_loss"
                ].mean(),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "sum_reduction_correct": (
            torch.allclose(
                sum_components[
                    "total_loss"
                ],
                per_image[
                    "total_loss"
                ].sum(),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "positive_weight_increases_bce": (
            weighted_bce.item()
            > unweighted_bce.item()
        ),
        "invalid_shape_rejected": (
            invalid_shape_rejected
        ),
        "zero_weights_rejected": (
            zero_weights_rejected
        ),
        "invalid_pos_weight_rejected": (
            invalid_pos_weight_rejected
        ),
        "incompatible_component_mode_rejected": (
            incompatible_component_mode_rejected
        ),
        "configuration_protocol_correct": (
            configuration[
                "protocol_version"
            ]
            == BCE_DICE_LOSS_PROTOCOL_VERSION
        ),
        "fully_supervised_configuration": (
            configuration[
                "learning_type"
            ]
            == "fully_supervised"
        ),
        "no_hard_thresholding": (
            configuration[
                "hard_thresholding"
            ]
            is False
        ),
        "no_auxiliary_targets": (
            configuration[
                "uses_auxiliary_targets"
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
            BCE_DICE_LOSS_PROTOCOL_VERSION
        ),
        "checks": checks,
        "total_loss": float(
            total_loss.detach().item()
        ),
        "bce_loss": float(
            components[
                "bce_loss"
            ].detach().item()
        ),
        "dice_loss": float(
            components[
                "dice_loss"
            ].detach().item()
        ),
        "per_image_total_losses": (
            per_image[
                "total_loss"
            ].tolist()
        ),
        "configuration": (
            configuration
        ),
    }