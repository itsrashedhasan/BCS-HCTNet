"""Regression tests for supervised segmentation baseline models.

This suite verifies:

- the approved five-model baseline registry;
- architecture-name aliases;
- model construction through the shared factory;
- one-channel mask-logit output compatibility;
- odd-sized input support;
- fully supervised baseline metadata;
- rejection of invalid configurations;
- U-Net forward/backward compatibility;
- UNet++ optional deep-supervision outputs;
- TransUNet positional encoding;
- Swin window partition/reconstruction;
- DeepLabV3+ ASPP behavior.

Large research configurations are not trained here. Reduced CPU-safe
configurations are used to verify software correctness.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.baselines import (
    SUPPORTED_BASELINE_NAMES,
    DeepLabV3Plus,
    SwinUNet,
    TransUNet,
    UNet,
    UNetPP,
    baseline_factory_summary,
    build_baseline_model,
    get_baseline_architecture_summary,
    normalize_baseline_name,
)
from src.models.baselines.deeplabv3plus import (
    AtrousSpatialPyramidPooling,
)
from src.models.baselines.swin_unet import (
    window_partition,
    window_reverse,
)
from src.models.baselines.transunet import (
    build_2d_sinusoidal_position_encoding,
)


EXPECTED_BASELINE_NAMES = (
    "unet",
    "unetpp",
    "deeplabv3plus",
    "transunet",
    "swin_unet",
)


BASELINE_TEST_CASES = (
    (
        "U-Net",
        {
            "base_channels": 4,
            "input_channels": 3,
            "output_channels": 1,
        },
        UNet,
        "standard_unet",
        False,
    ),
    (
        "UNet++",
        {
            "base_channels": 4,
            "input_channels": 3,
            "output_channels": 1,
            "deep_supervision": False,
        },
        UNetPP,
        "unet_plus_plus",
        False,
    ),
    (
        "DeepLabV3+",
        {
            "input_channels": 3,
            "output_channels": 1,
            "backbone": "resnet50",
            "pretrained": False,
            "aspp_channels": 16,
            "decoder_channels": 16,
            "low_level_projection_channels": 8,
            "atrous_rates": (
                3,
                6,
                9,
            ),
            "dropout": 0.0,
        },
        DeepLabV3Plus,
        "deeplabv3plus",
        False,
    ),
    (
        "TransUNet",
        {
            "input_channels": 3,
            "output_channels": 1,
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
            "bilinear_decoder": True,
        },
        TransUNet,
        "transunet_style",
        True,
    ),
    (
        "Swin-Unet",
        {
            "input_channels": 3,
            "output_channels": 1,
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
        SwinUNet,
        "swin_unet",
        True,
    ),
)


def test_supported_baseline_registry_is_complete() -> None:
    """The baseline registry must contain the five approved models."""

    assert (
        SUPPORTED_BASELINE_NAMES
        == EXPECTED_BASELINE_NAMES
    )

    summary = baseline_factory_summary()

    assert summary[
        "supported_models"
    ] == list(
        EXPECTED_BASELINE_NAMES
    )

    assert summary[
        "task"
    ] == "binary_lesion_segmentation"

    assert summary[
        "learning_type"
    ] == "fully_supervised"

    assert summary[
        "output_key"
    ] == "mask_logits"


@pytest.mark.parametrize(
    (
        "alias",
        "expected_name",
    ),
    (
        (
            "U-Net",
            "unet",
        ),
        (
            "UNet++",
            "unetpp",
        ),
        (
            "DeepLabV3+",
            "deeplabv3plus",
        ),
        (
            "TransUNet",
            "transunet",
        ),
        (
            "Swin-Unet",
            "swin_unet",
        ),
    ),
)
def test_baseline_name_aliases(
    alias: str,
    expected_name: str,
) -> None:
    """Human-readable aliases must resolve deterministically."""

    assert (
        normalize_baseline_name(
            alias
        )
        == expected_name
    )


@pytest.mark.parametrize(
    (
        "model_name",
        "configuration",
        "expected_class",
        "expected_architecture",
        "uses_transformer",
    ),
    BASELINE_TEST_CASES,
)
def test_factory_builds_each_baseline(
    model_name: str,
    configuration: dict[
        str,
        object,
    ],
    expected_class: type[nn.Module],
    expected_architecture: str,
    uses_transformer: bool,
) -> None:
    """Every baseline must satisfy the shared output contract."""

    torch.manual_seed(
        42
    )

    model = build_baseline_model(
        model_name,
        configuration,
    )

    assert isinstance(
        model,
        expected_class,
    )

    model.eval()

    image = torch.randn(
        1,
        3,
        33,
        35,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = model(
            image
        )

    assert isinstance(
        output,
        dict,
    )

    assert tuple(
        output
    ) == (
        "mask_logits",
    )

    mask_logits = output[
        "mask_logits"
    ]

    assert isinstance(
        mask_logits,
        torch.Tensor,
    )

    assert tuple(
        mask_logits.shape
    ) == (
        1,
        1,
        33,
        35,
    )

    assert torch.isfinite(
        mask_logits
    ).all()

    architecture = (
        get_baseline_architecture_summary(
            model
        )
    )

    assert architecture[
        "architecture"
    ] == expected_architecture

    assert architecture[
        "task"
    ] == "binary_lesion_segmentation"

    assert architecture[
        "learning_type"
    ] == "fully_supervised"

    assert architecture[
        "output_channels"
    ] == 1

    assert architecture[
        "output_keys"
    ] == [
        "mask_logits",
    ]

    assert architecture[
        "uses_transformer"
    ] is uses_transformer

    assert architecture[
        "uses_boundary_conditioning"
    ] is False

    assert architecture[
        "uses_auxiliary_targets"
    ] is False

    assert int(
        architecture[
            "parameter_count"
        ]
    ) > 0

    del output
    del mask_logits
    del image
    del model

    gc.collect()


def test_nested_model_configuration_is_supported() -> None:
    """The factory must accept the experiment-style model section."""

    model = build_baseline_model(
        configuration={
            "model": {
                "name": "standard_unet",
                "parameters": {
                    "in_channels": 3,
                    "num_classes": 1,
                    "base_features": 8,
                    "bilinear_upsampling": True,
                },
            }
        }
    )

    assert isinstance(
        model,
        UNet,
    )

    assert model.input_channels == 3
    assert model.output_channels == 1
    assert model.base_channels == 8
    assert model.bilinear is True


def test_unknown_model_name_is_rejected() -> None:
    """An unregistered architecture must not be silently accepted."""

    with pytest.raises(
        ValueError,
        match="Unsupported baseline model",
    ):
        build_baseline_model(
            "unknown_network",
            {},
        )


def test_unknown_model_parameter_is_rejected() -> None:
    """Misspelled configuration fields must fail immediately."""

    with pytest.raises(
        ValueError,
        match="Unsupported parameters",
    ):
        build_baseline_model(
            "unet",
            {
                "misspelled_channels": 64,
            },
        )


def test_conflicting_model_names_are_rejected() -> None:
    """Explicit and configuration names must not disagree."""

    with pytest.raises(
        ValueError,
        match="conflicts",
    ):
        build_baseline_model(
            "unet",
            {
                "name": "transunet",
            },
        )


def test_unet_supports_shared_training_contract() -> None:
    """Mask logits must produce gradients under the baseline mask loss."""

    torch.manual_seed(
        42
    )

    model = UNet(
        input_channels=3,
        output_channels=1,
        base_channels=4,
        bilinear=True,
    )

    model.train()

    image = torch.randn(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            2,
            1,
            32,
            32,
        )
        > 0.5
    ).to(
        dtype=torch.float32
    )

    output = model(
        image
    )

    loss = (
        F.binary_cross_entropy_with_logits(
            output[
                "mask_logits"
            ],
            target,
        )
    )

    assert torch.isfinite(
        loss
    )

    assert loss.requires_grad

    loss.backward()

    first_gradient = (
        model
        .input_block
        .block[0]
        .weight
        .grad
    )

    assert image.grad is not None

    assert torch.isfinite(
        image.grad
    ).all()

    assert first_gradient is not None

    assert torch.isfinite(
        first_gradient
    ).all()

    assert float(
        first_gradient
        .abs()
        .sum()
        .item()
    ) > 0.0


def test_unetpp_deep_supervision_contract() -> None:
    """Optional UNet++ auxiliary outputs must retain input resolution."""

    model = UNetPP(
        input_channels=3,
        output_channels=1,
        base_channels=4,
        deep_supervision=True,
    )

    model.eval()

    image = torch.randn(
        1,
        3,
        33,
        35,
    )

    with torch.inference_mode():
        output = model(
            image
        )

    assert set(
        output
    ) == {
        "mask_logits",
        "auxiliary_mask_logits",
    }

    auxiliary_logits = output[
        "auxiliary_mask_logits"
    ]

    assert isinstance(
        auxiliary_logits,
        list,
    )

    assert len(
        auxiliary_logits
    ) == 3

    assert all(
        tuple(
            logits.shape
        )
        == (
            1,
            1,
            33,
            35,
        )
        for logits
        in auxiliary_logits
    )

    assert all(
        torch.isfinite(
            logits
        ).all()
        for logits
        in auxiliary_logits
    )


def test_transunet_position_encoding() -> None:
    """Two-dimensional positional encoding must be stable and finite."""

    encoding_first = (
        build_2d_sinusoidal_position_encoding(
            height=4,
            width=5,
            embedding_dimension=64,
            device=torch.device(
                "cpu"
            ),
            dtype=torch.float32,
        )
    )

    encoding_second = (
        build_2d_sinusoidal_position_encoding(
            height=4,
            width=5,
            embedding_dimension=64,
            device=torch.device(
                "cpu"
            ),
            dtype=torch.float32,
        )
    )

    assert tuple(
        encoding_first.shape
    ) == (
        1,
        20,
        64,
    )

    assert torch.isfinite(
        encoding_first
    ).all()

    assert torch.equal(
        encoding_first,
        encoding_second,
    )

    with pytest.raises(
        ValueError,
        match="divisible by four",
    ):
        build_2d_sinusoidal_position_encoding(
            height=4,
            width=5,
            embedding_dimension=62,
            device=torch.device(
                "cpu"
            ),
            dtype=torch.float32,
        )


def test_swin_window_partition_round_trip() -> None:
    """Window partition and reverse operations must preserve values."""

    torch.manual_seed(
        42
    )

    feature_map = torch.randn(
        2,
        8,
        12,
        3,
    )

    windows = window_partition(
        feature_map,
        window_size=4,
    )

    reconstructed = window_reverse(
        windows,
        window_size=4,
        height=8,
        width=12,
        batch_size=2,
    )

    assert tuple(
        windows.shape
    ) == (
        12,
        16,
        3,
    )

    assert tuple(
        reconstructed.shape
    ) == tuple(
        feature_map.shape
    )

    assert torch.equal(
        reconstructed,
        feature_map,
    )


def test_deeplab_aspp_preserves_spatial_resolution() -> None:
    """ASPP branches must return one aligned feature map."""

    torch.manual_seed(
        42
    )

    aspp = (
        AtrousSpatialPyramidPooling(
            input_channels=64,
            output_channels=16,
            atrous_rates=(
                3,
                6,
                9,
            ),
            dropout_probability=0.0,
        )
    )

    aspp.eval()

    feature_map = torch.randn(
        2,
        64,
        9,
        11,
    )

    with torch.inference_mode():
        output = aspp(
            feature_map
        )

    assert tuple(
        output.shape
    ) == (
        2,
        16,
        9,
        11,
    )

    assert torch.isfinite(
        output
    ).all()