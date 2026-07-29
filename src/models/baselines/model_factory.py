"""Factory for constructing supervised segmentation baseline models.

Supported baselines
-------------------
- U-Net
- UNet++
- DeepLabV3+
- TransUNet
- Swin-UNet

The factory accepts either:

1. A direct model name and parameter mapping:

       build_baseline_model(
           "unet",
           {
               "base_channels": 64,
           },
       )

2. A complete configuration containing a ``model`` section:

       build_baseline_model(
           configuration={
               "model": {
                   "name": "unet",
                   "parameters": {
                       "base_channels": 64,
                   },
               }
           }
       )

Configuration errors are rejected rather than silently ignored. This prevents
misspelled parameter names from changing experimental conditions unnoticed.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.models.baselines.deeplabv3plus import (
    DeepLabV3Plus,
)
from src.models.baselines.swin_unet import (
    SwinUNet,
)
from src.models.baselines.transunet import (
    TransUNet,
)
from src.models.baselines.unet import (
    UNet,
)
from src.models.baselines.unetpp import (
    UNetPP,
)


BASELINE_FACTORY_PROTOCOL_VERSION = (
    "BCS-HCTNet-baseline-model-factory-v1"
)

SUPPORTED_BASELINE_NAMES = (
    "unet",
    "unetpp",
    "deeplabv3plus",
    "transunet",
    "swin_unet",
)


MODEL_NAME_ALIASES = {
    "unet": "unet",
    "u_net": "unet",
    "standard_unet": "unet",
    "unet_baseline": "unet",

    "unetpp": "unetpp",
    "unet_plus_plus": "unetpp",
    "unetplusplus": "unetpp",
    "nested_unet": "unetpp",
    "nestedunet": "unetpp",

    "deeplabv3plus": "deeplabv3plus",
    "deeplab_v3_plus": "deeplabv3plus",
    "deeplabv3_plus": "deeplabv3plus",
    "deeplab_v3plus": "deeplabv3plus",

    "transunet": "transunet",
    "trans_unet": "transunet",
    "transunet_style": "transunet",

    "swin_unet": "swin_unet",
    "swinunet": "swin_unet",
    "swin_unet_baseline": "swin_unet",
}


MODEL_ALLOWED_PARAMETERS = {
    "unet": {
        "input_channels",
        "output_channels",
        "base_channels",
        "bilinear",
        "dropout_probability",
    },
    "unetpp": {
        "input_channels",
        "output_channels",
        "base_channels",
        "deep_supervision",
        "dropout_probability",
    },
    "deeplabv3plus": {
        "input_channels",
        "output_channels",
        "backbone_name",
        "backbone_pretrained",
        "backbone_weights_path",
        "aspp_channels",
        "decoder_channels",
        "low_level_projection_channels",
        "atrous_rates",
        "dropout_probability",
    },
    "transunet": {
        "input_channels",
        "output_channels",
        "base_channels",
        "encoder_blocks",
        "transformer_dimension",
        "transformer_layers",
        "transformer_heads",
        "transformer_mlp_dimension",
        "transformer_dropout",
        "attention_dropout",
        "bottleneck_dropout",
        "bilinear_decoder",
    },
    "swin_unet": {
        "input_channels",
        "output_channels",
        "patch_size",
        "embedding_dimension",
        "depths",
        "number_of_heads",
        "window_size",
        "mlp_ratio",
        "dropout_probability",
        "attention_dropout",
        "drop_path_rate",
    },
}


COMMON_PARAMETER_ALIASES = {
    "in_channels": "input_channels",
    "image_channels": "input_channels",
    "num_input_channels": "input_channels",

    "out_channels": "output_channels",
    "number_of_classes": "output_channels",
    "num_classes": "output_channels",
    "classes": "output_channels",

    "base_features": "base_channels",
    "initial_channels": "base_channels",
}


MODEL_SPECIFIC_PARAMETER_ALIASES = {
    "unet": {
        "bilinear_upsampling": "bilinear",
        "dropout": "dropout_probability",
    },
    "unetpp": {
        "deep_supervision_enabled": (
            "deep_supervision"
        ),
        "dropout": "dropout_probability",
    },
    "deeplabv3plus": {
        "backbone": "backbone_name",
        "pretrained": (
            "backbone_pretrained"
        ),
        "weights_path": (
            "backbone_weights_path"
        ),
        "low_level_channels": (
            "low_level_projection_channels"
        ),
        "dropout": "dropout_probability",
    },
    "transunet": {
        "transformer_dim": (
            "transformer_dimension"
        ),
        "transformer_depth": (
            "transformer_layers"
        ),
        "num_heads": (
            "transformer_heads"
        ),
        "mlp_dimension": (
            "transformer_mlp_dimension"
        ),
        "dropout": (
            "transformer_dropout"
        ),
        "bilinear": (
            "bilinear_decoder"
        ),
    },
    "swin_unet": {
        "embed_dim": (
            "embedding_dimension"
        ),
        "heads": (
            "number_of_heads"
        ),
        "dropout": (
            "dropout_probability"
        ),
        "drop_path": (
            "drop_path_rate"
        ),
    },
}


def normalize_baseline_name(
    name: object,
) -> str:
    """Normalize a baseline name or alias."""

    normalized = str(
        name
    ).strip().lower()

    normalized = (
        normalized
        .replace(
            "+",
            "plus",
        )
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    while "__" in normalized:
        normalized = normalized.replace(
            "__",
            "_",
        )

    if normalized not in MODEL_NAME_ALIASES:
        raise ValueError(
            "Unsupported baseline model "
            f"{name!r}. Supported canonical "
            f"names are "
            f"{list(SUPPORTED_BASELINE_NAMES)}."
        )

    return MODEL_NAME_ALIASES[
        normalized
    ]


def _require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Require a mapping value."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{context} must be a mapping."
        )

    return value


def _merge_nested_parameters(
    model_configuration: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """Merge optional parameters/kwargs sections."""

    configuration = dict(
        model_configuration
    )

    nested_keys = [
        key
        for key in (
            "parameters",
            "kwargs",
            "model_parameters",
        )
        if key in configuration
    ]

    if len(
        nested_keys
    ) > 1:
        raise ValueError(
            "Model configuration may contain only "
            "one of 'parameters', 'kwargs', or "
            "'model_parameters'."
        )

    nested_parameters: dict[
        str,
        Any,
    ] = {}

    if nested_keys:
        nested_key = nested_keys[0]

        nested_mapping = _require_mapping(
            configuration.pop(
                nested_key
            ),
            f"model.{nested_key}",
        )

        nested_parameters = dict(
            nested_mapping
        )

    metadata_keys = {
        "name",
        "architecture",
        "model_name",
        "type",
        "family",
        "task",
        "learning_type",
    }

    direct_parameters = {
        str(key): value
        for key, value
        in configuration.items()
        if str(key) not in metadata_keys
    }

    overlapping = sorted(
        set(
            direct_parameters
        )
        & set(
            nested_parameters
        )
    )

    if overlapping:
        raise ValueError(
            "Model parameters are defined both "
            "directly and inside the nested "
            f"parameter section: {overlapping}."
        )

    return {
        **direct_parameters,
        **nested_parameters,
    }


def _extract_model_section(
    configuration: Mapping[
        str,
        Any,
    ]
    | None,
) -> dict[str, Any]:
    """Extract a direct or nested model configuration."""

    if configuration is None:
        return {}

    resolved_configuration = dict(
        _require_mapping(
            configuration,
            "configuration",
        )
    )

    if "model" not in resolved_configuration:
        return resolved_configuration

    model_section = _require_mapping(
        resolved_configuration[
            "model"
        ],
        "configuration['model']",
    )

    return dict(
        model_section
    )


def _resolve_model_name(
    explicit_name: object | None,
    model_section: Mapping[
        str,
        Any,
    ],
) -> str:
    """Resolve an explicit or configuration model name."""

    configuration_names = [
        model_section[key]
        for key in (
            "name",
            "architecture",
            "model_name",
            "type",
        )
        if key in model_section
        and model_section[key] is not None
    ]

    if explicit_name is None:
        if not configuration_names:
            raise ValueError(
                "No baseline model name was "
                "provided. Supply name or include "
                "model.name in the configuration."
            )

        normalized_names = {
            normalize_baseline_name(
                value
            )
            for value
            in configuration_names
        }

        if len(
            normalized_names
        ) != 1:
            raise ValueError(
                "Model configuration contains "
                "conflicting model names."
            )

        return normalized_names.pop()

    resolved_explicit_name = (
        normalize_baseline_name(
            explicit_name
        )
    )

    for configuration_name in (
        configuration_names
    ):
        resolved_configuration_name = (
            normalize_baseline_name(
                configuration_name
            )
        )

        if (
            resolved_configuration_name
            != resolved_explicit_name
        ):
            raise ValueError(
                "Explicit model name conflicts "
                "with the configuration model "
                "name."
            )

    return resolved_explicit_name


def _normalize_parameter_names(
    *,
    model_name: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize common and model-specific parameter aliases."""

    aliases = {
        **COMMON_PARAMETER_ALIASES,
        **MODEL_SPECIFIC_PARAMETER_ALIASES[
            model_name
        ],
    }

    normalized: dict[
        str,
        Any,
    ] = {}

    for raw_name, value in (
        parameters.items()
    ):
        parameter_name = str(
            raw_name
        ).strip()

        if not parameter_name:
            raise ValueError(
                "Model parameter names cannot "
                "be empty."
            )

        canonical_name = aliases.get(
            parameter_name,
            parameter_name,
        )

        if canonical_name in normalized:
            raise ValueError(
                "Model parameter was supplied "
                "multiple times through aliases: "
                f"{canonical_name!r}."
            )

        normalized[
            canonical_name
        ] = value

    allowed_parameters = (
        MODEL_ALLOWED_PARAMETERS[
            model_name
        ]
    )

    unknown_parameters = sorted(
        set(
            normalized
        )
        - allowed_parameters
    )

    if unknown_parameters:
        raise ValueError(
            f"Unsupported parameters for "
            f"{model_name!r}: "
            f"{unknown_parameters}. Allowed "
            f"parameters are "
            f"{sorted(allowed_parameters)}."
        )

    return normalized


def build_baseline_model(
    name: object | None = None,
    configuration: Mapping[
        str,
        Any,
    ]
    | None = None,
) -> nn.Module:
    """Construct one configured baseline model."""

    model_section = _extract_model_section(
        configuration
    )

    model_name = _resolve_model_name(
        name,
        model_section,
    )

    raw_parameters = (
        _merge_nested_parameters(
            model_section
        )
    )

    parameters = _normalize_parameter_names(
        model_name=model_name,
        parameters=raw_parameters,
    )

    if model_name == "unet":
        model: nn.Module = UNet(
            **parameters
        )

    elif model_name == "unetpp":
        model = UNetPP(
            **parameters
        )

    elif model_name == "deeplabv3plus":
        model = DeepLabV3Plus(
            **parameters
        )

    elif model_name == "transunet":
        model = TransUNet(
            **parameters
        )

    elif model_name == "swin_unet":
        model = SwinUNet(
            **parameters
        )

    else:
        raise AssertionError(
            "Unreachable baseline-model branch."
        )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Baseline factory did not return a "
            "torch.nn.Module."
        )

    return model


def build_model(
    name: object | None = None,
    configuration: Mapping[
        str,
        Any,
    ]
    | None = None,
) -> nn.Module:
    """Compatibility alias for building a baseline model."""

    return build_baseline_model(
        name=name,
        configuration=configuration,
    )


def get_baseline_architecture_summary(
    model: nn.Module,
) -> dict[str, Any]:
    """Return a validated model architecture summary."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    summary_function = getattr(
        model,
        "architecture_summary",
        None,
    )

    if not callable(
        summary_function
    ):
        raise TypeError(
            "Baseline model does not provide "
            "architecture_summary()."
        )

    summary = summary_function()

    if not isinstance(
        summary,
        Mapping,
    ):
        raise TypeError(
            "architecture_summary() must return "
            "a mapping."
        )

    result = dict(
        summary
    )

    required_keys = {
        "architecture",
        "task",
        "learning_type",
        "parameter_count",
        "output_keys",
        "uses_boundary_conditioning",
        "uses_auxiliary_targets",
    }

    missing = sorted(
        required_keys
        - set(
            result
        )
    )

    if missing:
        raise RuntimeError(
            "Baseline architecture summary is "
            f"missing keys: {missing}."
        )

    if (
        result[
            "learning_type"
        ]
        != "fully_supervised"
    ):
        raise RuntimeError(
            "Baseline model is not marked as "
            "fully supervised."
        )

    if (
        result[
            "uses_boundary_conditioning"
        ]
        is not False
    ):
        raise RuntimeError(
            "Baseline model unexpectedly uses "
            "boundary conditioning."
        )

    if (
        result[
            "uses_auxiliary_targets"
        ]
        is not False
    ):
        raise RuntimeError(
            "Baseline model unexpectedly uses "
            "auxiliary geometric targets."
        )

    return result


def baseline_factory_summary() -> dict[str, Any]:
    """Return factory registry metadata."""

    return {
        "protocol_version": (
            BASELINE_FACTORY_PROTOCOL_VERSION
        ),
        "supported_models": list(
            SUPPORTED_BASELINE_NAMES
        ),
        "model_classes": {
            "unet": (
                UNet.__name__
            ),
            "unetpp": (
                UNetPP.__name__
            ),
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
        "learning_type": (
            "fully_supervised"
        ),
        "task": (
            "binary_lesion_segmentation"
        ),
        "output_key": (
            "mask_logits"
        ),
    }


def _run_factory_forward_test(
    *,
    name: str,
    configuration: Mapping[
        str,
        Any,
    ],
    input_shape: tuple[
        int,
        int,
        int,
        int,
    ],
) -> dict[str, Any]:
    """Construct and forward one reduced baseline."""

    model = build_baseline_model(
        name,
        configuration,
    )

    model.eval()

    image = torch.randn(
        *input_shape,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = model(
            image
        )

    if not isinstance(
        output,
        Mapping,
    ):
        raise RuntimeError(
            f"{name} output is not a mapping."
        )

    mask_logits = output.get(
        "mask_logits"
    )

    if not isinstance(
        mask_logits,
        Tensor,
    ):
        raise RuntimeError(
            f"{name} did not return mask_logits."
        )

    summary = (
        get_baseline_architecture_summary(
            model
        )
    )

    result = {
        "class_name": (
            model.__class__.__name__
        ),
        "output_shape": list(
            mask_logits.shape
        ),
        "output_finite": bool(
            torch.isfinite(
                mask_logits
            ).all().item()
        ),
        "parameter_count": int(
            summary[
                "parameter_count"
            ]
        ),
        "architecture": (
            summary[
                "architecture"
            ]
        ),
        "learning_type": (
            summary[
                "learning_type"
            ]
        ),
        "uses_boundary_conditioning": (
            summary[
                "uses_boundary_conditioning"
            ]
        ),
        "uses_auxiliary_targets": (
            summary[
                "uses_auxiliary_targets"
            ]
        ),
    }

    del output
    del mask_logits
    del image
    del model

    gc.collect()

    return result


def run_baseline_model_factory_self_test() -> dict[str, Any]:
    """Run reduced CPU construction and forward tests for every baseline."""

    torch.manual_seed(
        42
    )

    input_shape = (
        1,
        3,
        65,
        67,
    )

    model_tests = {
        "unet": _run_factory_forward_test(
            name="U-Net",
            configuration={
                "base_channels": 8,
            },
            input_shape=input_shape,
        ),
        "unetpp": _run_factory_forward_test(
            name="UNet++",
            configuration={
                "base_channels": 4,
                "deep_supervision": False,
            },
            input_shape=input_shape,
        ),
        "deeplabv3plus": (
            _run_factory_forward_test(
                name="DeepLabV3+",
                configuration={
                    "backbone": "resnet50",
                    "pretrained": False,
                    "aspp_channels": 32,
                    "decoder_channels": 32,
                    "low_level_projection_channels": 12,
                    "atrous_rates": (
                        3,
                        6,
                        9,
                    ),
                    "dropout": 0.0,
                },
                input_shape=input_shape,
            )
        ),
        "transunet": (
            _run_factory_forward_test(
                name="TransUNet",
                configuration={
                    "base_channels": 4,
                    "encoder_blocks": (
                        1,
                        1,
                        1,
                        1,
                    ),
                    "transformer_dim": 32,
                    "transformer_layers": 1,
                    "transformer_heads": 4,
                    "transformer_mlp_dimension": 64,
                    "transformer_dropout": 0.0,
                    "attention_dropout": 0.0,
                    "bottleneck_dropout": 0.0,
                },
                input_shape=input_shape,
            )
        ),
        "swin_unet": (
            _run_factory_forward_test(
                name="Swin-Unet",
                configuration={
                    "patch_size": 4,
                    "embed_dim": 12,
                    "depths": (
                        1,
                        1,
                        1,
                        1,
                    ),
                    "heads": (
                        3,
                        3,
                        6,
                        12,
                    ),
                    "window_size": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "drop_path": 0.0,
                },
                input_shape=input_shape,
            )
        ),
    }

    nested_model = build_baseline_model(
        configuration={
            "model": {
                "name": "standard_unet",
                "parameters": {
                    "in_channels": 3,
                    "num_classes": 1,
                    "base_features": 4,
                },
            }
        }
    )

    nested_configuration_valid = (
        isinstance(
            nested_model,
            UNet,
        )
        and nested_model.input_channels == 3
        and nested_model.output_channels == 1
        and nested_model.base_channels == 4
    )

    del nested_model

    conflicting_name_rejected = False

    try:
        build_baseline_model(
            "unet",
            {
                "name": "transunet",
            },
        )

    except ValueError:
        conflicting_name_rejected = True

    unknown_name_rejected = False

    try:
        build_baseline_model(
            "unknown_model",
            {},
        )

    except ValueError:
        unknown_name_rejected = True

    unknown_parameter_rejected = False

    try:
        build_baseline_model(
            "unet",
            {
                "misspelled_channels": 64,
            },
        )

    except ValueError:
        unknown_parameter_rejected = True

    duplicate_alias_rejected = False

    try:
        build_baseline_model(
            "unet",
            {
                "input_channels": 3,
                "in_channels": 3,
            },
        )

    except ValueError:
        duplicate_alias_rejected = True

    expected_classes = {
        "unet": "UNet",
        "unetpp": "UNetPP",
        "deeplabv3plus": (
            "DeepLabV3Plus"
        ),
        "transunet": "TransUNet",
        "swin_unet": "SwinUNet",
    }

    expected_architectures = {
        "unet": "standard_unet",
        "unetpp": "unet_plus_plus",
        "deeplabv3plus": (
            "deeplabv3plus"
        ),
        "transunet": (
            "transunet_style"
        ),
        "swin_unet": "swin_unet",
    }

    expected_output_shape = [
        1,
        1,
        65,
        67,
    ]

    checks = {
        "all_models_constructed": (
            set(
                model_tests
            )
            == set(
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_classes_correct": all(
            model_tests[name][
                "class_name"
            ]
            == expected_classes[name]
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_architectures_correct": all(
            model_tests[name][
                "architecture"
            ]
            == expected_architectures[
                name
            ]
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_output_shapes_correct": all(
            model_tests[name][
                "output_shape"
            ]
            == expected_output_shape
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_outputs_finite": all(
            model_tests[name][
                "output_finite"
            ]
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_parameter_counts_positive": all(
            model_tests[name][
                "parameter_count"
            ]
            > 0
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "all_fully_supervised": all(
            model_tests[name][
                "learning_type"
            ]
            == "fully_supervised"
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "no_boundary_conditioning": all(
            model_tests[name][
                "uses_boundary_conditioning"
            ]
            is False
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "no_auxiliary_geometric_targets": all(
            model_tests[name][
                "uses_auxiliary_targets"
            ]
            is False
            for name in (
                SUPPORTED_BASELINE_NAMES
            )
        ),
        "nested_configuration_supported": (
            nested_configuration_valid
        ),
        "conflicting_name_rejected": (
            conflicting_name_rejected
        ),
        "unknown_name_rejected": (
            unknown_name_rejected
        ),
        "unknown_parameter_rejected": (
            unknown_parameter_rejected
        ),
        "duplicate_alias_rejected": (
            duplicate_alias_rejected
        ),
        "factory_registry_complete": (
            baseline_factory_summary()[
                "supported_models"
            ]
            == list(
                SUPPORTED_BASELINE_NAMES
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
            BASELINE_FACTORY_PROTOCOL_VERSION
        ),
        "checks": checks,
        "supported_models": list(
            SUPPORTED_BASELINE_NAMES
        ),
        "models": model_tests,
        "factory": (
            baseline_factory_summary()
        ),
    }