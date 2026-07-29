"""Public API for BCS-HCTNet segmentation losses.

Currently implemented and approved for baseline experiments:

- differentiable soft Dice loss;
- combined BCEWithLogits and soft Dice mask loss.

The following proposed-model losses intentionally remain outside the public
API until they are implemented and independently validated:

- contour loss;
- signed-distance-map loss;
- boundary loss;
- geometric consistency loss;
- full BCS-HCTNet composite loss.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.losses.bce_dice_loss import (
    BCE_DICE_LOSS_PROTOCOL_VERSION,
    BCEAndDiceLoss,
    BCEDiceLoss,
    BaselineMaskLoss,
    bce_dice_loss,
    binary_cross_entropy_per_image,
    compute_bce_dice_loss,
    prepare_binary_logits,
    run_bce_dice_loss_self_test,
)
from src.losses.dice_loss import (
    DICE_LOSS_PROTOCOL_VERSION,
    DiceLoss,
    SoftDiceLoss,
    dice_loss,
    prepare_dice_prediction,
    prepare_dice_target,
    run_dice_loss_self_test,
    soft_dice_score,
)


LOSS_PUBLIC_API_VERSION = (
    "BCS-HCTNet-loss-public-api-v1"
)


__all__ = [
    # Package metadata.
    "LOSS_PUBLIC_API_VERSION",

    # Protocol versions.
    "DICE_LOSS_PROTOCOL_VERSION",
    "BCE_DICE_LOSS_PROTOCOL_VERSION",

    # Dice functions and classes.
    "prepare_dice_target",
    "prepare_dice_prediction",
    "soft_dice_score",
    "dice_loss",
    "SoftDiceLoss",
    "DiceLoss",

    # BCE-Dice functions and classes.
    "prepare_binary_logits",
    "binary_cross_entropy_per_image",
    "compute_bce_dice_loss",
    "bce_dice_loss",
    "BCEDiceLoss",
    "BCEAndDiceLoss",
    "BaselineMaskLoss",

    # Package metadata and validation.
    "loss_public_api_summary",
    "build_baseline_mask_criterion",
    "run_loss_public_api_self_test",

    # Module-level validation.
    "run_dice_loss_self_test",
    "run_bce_dice_loss_self_test",
]


def build_baseline_mask_criterion(
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
    dice_smooth: float = 1.0,
    dice_epsilon: float = 1e-7,
    return_components: bool = True,
) -> BCEDiceLoss:
    """Build the approved mask-only baseline criterion.

    All five baseline architectures must use the same criterion configuration
    during controlled comparisons unless an explicitly documented ablation
    changes it.

    Parameters
    ----------
    bce_weight:
        Weight applied to per-image BCEWithLogits loss.

    dice_weight:
        Weight applied to per-image differentiable Dice loss.

    pos_weight:
        Optional BCE positive-class weight. The default is ``None``.

    dice_smooth:
        Dice numerator and denominator smoothing.

    dice_epsilon:
        Minimum allowed Dice denominator.

    return_components:
        Return ``total_loss``, ``bce_loss``, and ``dice_loss`` as a mapping
        compatible with the shared trainer.
    """

    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        dice_smooth=dice_smooth,
        dice_epsilon=dice_epsilon,
        reduction="mean",
        return_components=return_components,
    )


def loss_public_api_summary() -> dict[str, Any]:
    """Return reproducible metadata for the current loss package."""

    return {
        "protocol_version": (
            LOSS_PUBLIC_API_VERSION
        ),
        "implemented_losses": [
            "soft_dice_loss",
            "bce_dice_loss",
        ],
        "baseline_criterion": (
            "bce_dice_loss"
        ),
        "baseline_task": (
            "binary_lesion_segmentation"
        ),
        "learning_type": (
            "fully_supervised"
        ),
        "prediction_key": (
            "mask_logits"
        ),
        "target_key": "mask",
        "from_logits": True,
        "per_image_calculation": True,
        "hard_thresholding": False,
        "uses_auxiliary_targets": False,
        "reduction": "mean",
        "proposed_losses_pending": [
            "contour_loss",
            "sdm_loss",
            "boundary_loss",
            "consistency_loss",
            "composite_loss",
        ],
        "public_symbol_count": len(
            __all__
        ),
    }


def run_loss_public_api_self_test() -> dict[str, Any]:
    """Validate exports and the baseline criterion contract on CPU."""

    torch.manual_seed(
        42
    )

    required_symbols = {
        "SoftDiceLoss",
        "DiceLoss",
        "BCEDiceLoss",
        "BaselineMaskLoss",
        "soft_dice_score",
        "dice_loss",
        "bce_dice_loss",
        "compute_bce_dice_loss",
        "build_baseline_mask_criterion",
    }

    symbols_are_unique = (
        len(
            __all__
        )
        == len(
            set(
                __all__
            )
        )
    )

    required_symbols_exported = (
        required_symbols.issubset(
            set(
                __all__
            )
        )
    )

    aliases_valid = {
        "dice_alias": (
            DiceLoss is SoftDiceLoss
        ),
        "bce_dice_alias": (
            BCEAndDiceLoss
            is BCEDiceLoss
        ),
        "baseline_mask_alias": (
            BaselineMaskLoss
            is BCEDiceLoss
        ),
    }

    criterion = (
        build_baseline_mask_criterion(
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=None,
            dice_smooth=1.0,
            dice_epsilon=1e-7,
            return_components=True,
        )
    )

    logits = torch.randn(
        2,
        1,
        8,
        8,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            2,
            1,
            8,
            8,
        )
        > 0.5
    ).to(
        dtype=torch.float32
    )

    criterion_output = criterion(
        logits,
        target,
    )

    if not isinstance(
        criterion_output,
        dict,
    ):
        raise RuntimeError(
            "Baseline criterion did not return "
            "the expected component mapping."
        )

    total_loss = criterion_output[
        "total_loss"
    ]

    total_loss.backward()

    configuration = (
        criterion.configuration()
    )

    package_summary = (
        loss_public_api_summary()
    )

    dice_self_test = (
        run_dice_loss_self_test()
    )

    bce_dice_self_test = (
        run_bce_dice_loss_self_test()
    )

    expected_component_keys = (
        "total_loss",
        "bce_loss",
        "dice_loss",
    )

    checks = {
        "public_symbols_unique": (
            symbols_are_unique
        ),
        "required_symbols_exported": (
            required_symbols_exported
        ),
        "aliases_valid": all(
            aliases_valid.values()
        ),
        "criterion_class_correct": (
            isinstance(
                criterion,
                BCEDiceLoss,
            )
        ),
        "criterion_component_keys": (
            tuple(
                criterion_output
            )
            == expected_component_keys
        ),
        "criterion_components_scalar": all(
            value.ndim == 0
            for value
            in criterion_output.values()
        ),
        "criterion_components_finite": all(
            bool(
                torch.isfinite(
                    value
                ).item()
            )
            for value
            in criterion_output.values()
        ),
        "total_loss_requires_gradient": (
            total_loss.requires_grad
        ),
        "logit_gradient_exists": (
            logits.grad is not None
        ),
        "logit_gradient_finite": (
            logits.grad is not None
            and bool(
                torch.isfinite(
                    logits.grad
                ).all().item()
            )
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
        "criterion_configuration_correct": (
            configuration[
                "protocol_version"
            ]
            == BCE_DICE_LOSS_PROTOCOL_VERSION
            and configuration[
                "learning_type"
            ]
            == "fully_supervised"
            and configuration[
                "uses_auxiliary_targets"
            ]
            is False
        ),
        "package_protocol_correct": (
            package_summary[
                "protocol_version"
            ]
            == LOSS_PUBLIC_API_VERSION
        ),
        "package_baseline_criterion_correct": (
            package_summary[
                "baseline_criterion"
            ]
            == "bce_dice_loss"
        ),
        "package_mask_only": (
            package_summary[
                "prediction_key"
            ]
            == "mask_logits"
            and package_summary[
                "target_key"
            ]
            == "mask"
            and package_summary[
                "uses_auxiliary_targets"
            ]
            is False
        ),
        "proposed_losses_still_pending": (
            package_summary[
                "proposed_losses_pending"
            ]
            == [
                "contour_loss",
                "sdm_loss",
                "boundary_loss",
                "consistency_loss",
                "composite_loss",
            ]
        ),
        "dice_self_test_passed": (
            dice_self_test[
                "status"
            ]
            == "passed"
            and all(
                dice_self_test[
                    "checks"
                ].values()
            )
        ),
        "bce_dice_self_test_passed": (
            bce_dice_self_test[
                "status"
            ]
            == "passed"
            and all(
                bce_dice_self_test[
                    "checks"
                ].values()
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
            LOSS_PUBLIC_API_VERSION
        ),
        "checks": checks,
        "aliases": aliases_valid,
        "criterion_output": {
            name: float(
                value.detach().item()
            )
            for name, value
            in criterion_output.items()
        },
        "criterion_configuration": (
            configuration
        ),
        "package_summary": (
            package_summary
        ),
    }