"""Public API for supervised segmentation baseline models.

This package exposes the five approved binary lesion-segmentation baselines:

- U-Net;
- UNet++;
- DeepLabV3+;
- TransUNet;
- Swin-UNet.

All baseline models:

- use fully supervised learning;
- predict one-channel lesion-mask logits;
- use the common ``mask_logits`` output key;
- do not use BCS-HCTNet boundary conditioning;
- do not use contour, boundary-band, or SDM targets.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.models.baselines.deeplabv3plus import (
    DEEPLABV3PLUS_PROTOCOL_VERSION,
    AtrousSpatialPyramidPooling,
    DeepLabV3Plus,
    DeepLabV3plus,
    DeepLabV3PlusDecoder,
    ResNetFeatureEncoder,
    run_deeplabv3plus_self_test,
)
from src.models.baselines.model_factory import (
    BASELINE_FACTORY_PROTOCOL_VERSION,
    MODEL_ALLOWED_PARAMETERS,
    SUPPORTED_BASELINE_NAMES,
    baseline_factory_summary,
    build_baseline_model,
    build_model,
    get_baseline_architecture_summary,
    normalize_baseline_name,
    run_baseline_model_factory_self_test,
)
from src.models.baselines.swin_unet import (
    SWIN_UNET_PROTOCOL_VERSION,
    SwinUNet,
    SwinUnet,
    SwinUNetBaseline,
    run_swin_unet_self_test,
)
from src.models.baselines.transunet import (
    TRANSUNET_PROTOCOL_VERSION,
    TransUNet,
    TransUNetBaseline,
    run_transunet_self_test,
)
from src.models.baselines.unet import (
    UNET_PROTOCOL_VERSION,
    DoubleConvolution,
    DownsamplingBlock,
    OutputConvolution,
    StandardUNet,
    UNet,
    UpsamplingBlock,
    run_unet_self_test,
)
from src.models.baselines.unetpp import (
    UNETPP_PROTOCOL_VERSION,
    NestedUNet,
    UNetPlusPlus,
    UNetPP,
    run_unetpp_self_test,
)


BASELINE_PUBLIC_API_VERSION = (
    "BCS-HCTNet-baseline-public-api-v1"
)


__all__ = [
    # Public API metadata.
    "BASELINE_PUBLIC_API_VERSION",
    "BASELINE_FACTORY_PROTOCOL_VERSION",
    "SUPPORTED_BASELINE_NAMES",
    "MODEL_ALLOWED_PARAMETERS",

    # Main architecture classes.
    "UNet",
    "StandardUNet",
    "UNetPP",
    "UNetPlusPlus",
    "NestedUNet",
    "DeepLabV3Plus",
    "DeepLabV3plus",
    "TransUNet",
    "TransUNetBaseline",
    "SwinUNet",
    "SwinUnet",
    "SwinUNetBaseline",

    # Shared U-Net components.
    "DoubleConvolution",
    "DownsamplingBlock",
    "UpsamplingBlock",
    "OutputConvolution",

    # DeepLabV3+ components.
    "ResNetFeatureEncoder",
    "AtrousSpatialPyramidPooling",
    "DeepLabV3PlusDecoder",

    # Factory functions.
    "normalize_baseline_name",
    "build_baseline_model",
    "build_model",
    "get_baseline_architecture_summary",
    "baseline_factory_summary",

    # Architecture protocol versions.
    "UNET_PROTOCOL_VERSION",
    "UNETPP_PROTOCOL_VERSION",
    "DEEPLABV3PLUS_PROTOCOL_VERSION",
    "TRANSUNET_PROTOCOL_VERSION",
    "SWIN_UNET_PROTOCOL_VERSION",

    # Individual self-tests.
    "run_unet_self_test",
    "run_unetpp_self_test",
    "run_deeplabv3plus_self_test",
    "run_transunet_self_test",
    "run_swin_unet_self_test",
    "run_baseline_model_factory_self_test",

    # Package-level validation.
    "baseline_public_api_summary",
    "run_baseline_public_api_self_test",
]


def baseline_public_api_summary() -> dict[str, Any]:
    """Return metadata describing the baseline package API."""

    return {
        "protocol_version": (
            BASELINE_PUBLIC_API_VERSION
        ),
        "factory_protocol_version": (
            BASELINE_FACTORY_PROTOCOL_VERSION
        ),
        "supported_models": list(
            SUPPORTED_BASELINE_NAMES
        ),
        "model_classes": {
            "unet": UNet.__name__,
            "unetpp": UNetPP.__name__,
            "deeplabv3plus": (
                DeepLabV3Plus.__name__
            ),
            "transunet": (
                TransUNet.__name__
            ),
            "swin_unet": (
                SwinUNet.__name__
            ),
        },
        "task": (
            "binary_lesion_segmentation"
        ),
        "learning_type": (
            "fully_supervised"
        ),
        "output_key": (
            "mask_logits"
        ),
        "output_channels": 1,
        "uses_boundary_conditioning": False,
        "uses_auxiliary_geometric_targets": (
            False
        ),
        "public_symbol_count": len(
            __all__
        ),
    }


def _check_model_output(
    model: nn.Module,
    image: Tensor,
) -> tuple[
    bool,
    tuple[int, ...] | None,
]:
    """Run one model and validate the common output contract."""

    model.eval()

    with torch.inference_mode():
        output = model(
            image
        )

    if not isinstance(
        output,
        dict,
    ):
        return (
            False,
            None,
        )

    mask_logits = output.get(
        "mask_logits"
    )

    if not isinstance(
        mask_logits,
        Tensor,
    ):
        return (
            False,
            None,
        )

    valid = bool(
        mask_logits.ndim == 4
        and mask_logits.shape[0]
        == image.shape[0]
        and mask_logits.shape[1] == 1
        and mask_logits.shape[-2:]
        == image.shape[-2:]
        and torch.isfinite(
            mask_logits
        ).all().item()
    )

    return (
        valid,
        tuple(
            int(value)
            for value
            in mask_logits.shape
        ),
    )


def run_baseline_public_api_self_test() -> dict[str, Any]:
    """Validate package exports and the common model contract on CPU."""

    torch.manual_seed(
        42
    )

    required_public_symbols = {
        "UNet",
        "UNetPP",
        "DeepLabV3Plus",
        "TransUNet",
        "SwinUNet",
        "build_baseline_model",
        "build_model",
        "SUPPORTED_BASELINE_NAMES",
    }

    all_symbols_unique = (
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
        required_public_symbols
        .issubset(
            set(
                __all__
            )
        )
    )

    alias_checks = {
        "standard_unet": (
            StandardUNet is UNet
        ),
        "unet_plus_plus": (
            UNetPlusPlus is UNetPP
        ),
        "nested_unet": (
            NestedUNet is UNetPP
        ),
        "deeplab_alias": (
            DeepLabV3plus
            is DeepLabV3Plus
        ),
        "transunet_alias": (
            TransUNetBaseline
            is TransUNet
        ),
        "swin_alias": (
            SwinUnet is SwinUNet
            and SwinUNetBaseline
            is SwinUNet
        ),
    }

    canonical_name_checks = {
        "unet": (
            normalize_baseline_name(
                "U-Net"
            )
            == "unet"
        ),
        "unetpp": (
            normalize_baseline_name(
                "UNet++"
            )
            == "unetpp"
        ),
        "deeplabv3plus": (
            normalize_baseline_name(
                "DeepLabV3+"
            )
            == "deeplabv3plus"
        ),
        "transunet": (
            normalize_baseline_name(
                "TransUNet"
            )
            == "transunet"
        ),
        "swin_unet": (
            normalize_baseline_name(
                "Swin-Unet"
            )
            == "swin_unet"
        ),
    }

    model = build_baseline_model(
        "unet",
        {
            "base_channels": 4,
            "input_channels": 3,
            "output_channels": 1,
        },
    )

    image = torch.randn(
        1,
        3,
        33,
        35,
        dtype=torch.float32,
    )

    (
        output_contract_valid,
        observed_output_shape,
    ) = _check_model_output(
        model,
        image,
    )

    architecture = (
        get_baseline_architecture_summary(
            model
        )
    )

    package_summary = (
        baseline_public_api_summary()
    )

    factory_summary = (
        baseline_factory_summary()
    )

    checks = {
        "public_symbols_unique": (
            all_symbols_unique
        ),
        "required_symbols_exported": (
            required_symbols_exported
        ),
        "all_aliases_valid": all(
            alias_checks.values()
        ),
        "all_name_aliases_normalize": all(
            canonical_name_checks.values()
        ),
        "supported_model_count": (
            len(
                SUPPORTED_BASELINE_NAMES
            )
            == 5
        ),
        "supported_model_order": (
            SUPPORTED_BASELINE_NAMES
            == (
                "unet",
                "unetpp",
                "deeplabv3plus",
                "transunet",
                "swin_unet",
            )
        ),
        "factory_returns_module": (
            isinstance(
                model,
                nn.Module,
            )
        ),
        "common_output_contract": (
            output_contract_valid
        ),
        "output_shape": (
            observed_output_shape
            == (
                1,
                1,
                33,
                35,
            )
        ),
        "fully_supervised": (
            architecture[
                "learning_type"
            ]
            == "fully_supervised"
        ),
        "binary_segmentation_task": (
            architecture[
                "task"
            ]
            == (
                "binary_lesion_"
                "segmentation"
            )
        ),
        "mask_output_key": (
            architecture[
                "output_keys"
            ]
            == [
                "mask_logits",
            ]
        ),
        "no_boundary_conditioning": (
            architecture[
                "uses_boundary_conditioning"
            ]
            is False
        ),
        "no_auxiliary_targets": (
            architecture[
                "uses_auxiliary_targets"
            ]
            is False
        ),
        "package_summary_valid": (
            package_summary[
                "protocol_version"
            ]
            == BASELINE_PUBLIC_API_VERSION
            and package_summary[
                "supported_models"
            ]
            == list(
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "factory_summary_matches": (
            factory_summary[
                "supported_models"
            ]
            == package_summary[
                "supported_models"
            ]
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
            BASELINE_PUBLIC_API_VERSION
        ),
        "checks": checks,
        "aliases": alias_checks,
        "canonical_names": (
            canonical_name_checks
        ),
        "supported_models": list(
            SUPPORTED_BASELINE_NAMES
        ),
        "observed_output_shape": (
            list(
                observed_output_shape
            )
            if observed_output_shape
            is not None
            else None
        ),
        "architecture": architecture,
        "package_summary": (
            package_summary
        ),
    }