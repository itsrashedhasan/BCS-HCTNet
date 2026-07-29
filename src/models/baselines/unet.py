"""Standard U-Net baseline for binary lesion segmentation.

This module implements the original encoder-decoder U-Net structure with:

- four encoder downsampling stages;
- a convolutional bottleneck;
- four decoder upsampling stages;
- skip connections between matching resolutions;
- one-channel binary segmentation logits.

The model is a conventional supervised baseline. It does not use:

- Transformer features;
- boundary-prior conditioning;
- contour supervision;
- signed-distance-map supervision;
- multi-task prediction.

The output follows the common project interface:

    {
        "mask_logits": Tensor[B, 1, H, W]
    }

The default model uses 64 base channels. Smaller channel counts can be used
for unit tests and smoke tests.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


UNET_PROTOCOL_VERSION = (
    "BCS-HCTNet-standard-unet-baseline-v1"
)


def _require_positive_integer(
    value: object,
    context: str,
) -> int:
    """Require a positive integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{context} must be a positive integer."
        )

    return value


def _validate_dropout(
    value: object,
) -> float:
    """Validate a dropout probability."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            "dropout_probability must be numeric."
        )

    try:
        probability = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "dropout_probability must be numeric."
        ) from error

    if not (
        0.0
        <= probability
        < 1.0
    ):
        raise ValueError(
            "dropout_probability must be in [0, 1)."
        )

    return probability


class DoubleConvolution(nn.Module):
    """Two consecutive convolution-normalization-activation blocks."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        intermediate_channels: int | None = None,
    ) -> None:
        """Initialize the double-convolution block."""

        super().__init__()

        resolved_input_channels = (
            _require_positive_integer(
                input_channels,
                "input_channels",
            )
        )

        resolved_output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        resolved_intermediate_channels = (
            resolved_output_channels
            if intermediate_channels is None
            else _require_positive_integer(
                intermediate_channels,
                "intermediate_channels",
            )
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=(
                    resolved_input_channels
                ),
                out_channels=(
                    resolved_intermediate_channels
                ),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                resolved_intermediate_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                in_channels=(
                    resolved_intermediate_channels
                ),
                out_channels=(
                    resolved_output_channels
                ),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                resolved_output_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply two convolutional transformations."""

        return self.block(
            feature_map
        )


class DownsamplingBlock(nn.Module):
    """Max-pooling followed by double convolution."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
    ) -> None:
        """Initialize one encoder downsampling stage."""

        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            DoubleConvolution(
                input_channels=input_channels,
                output_channels=output_channels,
            ),
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Downsample and refine a feature map."""

        return self.block(
            feature_map
        )


class UpsamplingBlock(nn.Module):
    """Upsample, concatenate a skip feature, and refine."""

    def __init__(
        self,
        *,
        decoder_channels: int,
        skip_channels: int,
        output_channels: int,
        bilinear: bool,
    ) -> None:
        """Initialize one decoder upsampling stage."""

        super().__init__()

        resolved_decoder_channels = (
            _require_positive_integer(
                decoder_channels,
                "decoder_channels",
            )
        )

        resolved_skip_channels = (
            _require_positive_integer(
                skip_channels,
                "skip_channels",
            )
        )

        resolved_output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        if not isinstance(
            bilinear,
            bool,
        ):
            raise TypeError(
                "bilinear must be Boolean."
            )

        self.bilinear = bilinear

        if bilinear:
            self.upsampling = nn.Upsample(
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

            upsampled_channels = (
                resolved_decoder_channels
            )

        else:
            self.upsampling = (
                nn.ConvTranspose2d(
                    in_channels=(
                        resolved_decoder_channels
                    ),
                    out_channels=(
                        resolved_output_channels
                    ),
                    kernel_size=2,
                    stride=2,
                )
            )

            upsampled_channels = (
                resolved_output_channels
            )

        self.refinement = DoubleConvolution(
            input_channels=(
                upsampled_channels
                + resolved_skip_channels
            ),
            output_channels=(
                resolved_output_channels
            ),
        )

    def forward(
        self,
        decoder_feature: Tensor,
        skip_feature: Tensor,
    ) -> Tensor:
        """Upsample and fuse decoder and encoder features."""

        if (
            not isinstance(
                decoder_feature,
                Tensor,
            )
            or not isinstance(
                skip_feature,
                Tensor,
            )
        ):
            raise TypeError(
                "Decoder and skip features must "
                "be torch.Tensor objects."
            )

        if (
            decoder_feature.ndim != 4
            or skip_feature.ndim != 4
        ):
            raise ValueError(
                "Decoder and skip features must "
                "have shape [B, C, H, W]."
            )

        if (
            decoder_feature.shape[0]
            != skip_feature.shape[0]
        ):
            raise RuntimeError(
                "Decoder and skip feature batch "
                "sizes do not match."
            )

        decoder_feature = self.upsampling(
            decoder_feature
        )

        target_size = tuple(
            int(value)
            for value in skip_feature.shape[-2:]
        )

        if (
            decoder_feature.shape[-2:]
            != skip_feature.shape[-2:]
        ):
            decoder_feature = F.interpolate(
                decoder_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        fused_feature = torch.cat(
            [
                skip_feature,
                decoder_feature,
            ],
            dim=1,
        )

        return self.refinement(
            fused_feature
        )


class OutputConvolution(nn.Module):
    """One-by-one convolution producing segmentation logits."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
    ) -> None:
        """Initialize the segmentation output layer."""

        super().__init__()

        self.convolution = nn.Conv2d(
            in_channels=(
                _require_positive_integer(
                    input_channels,
                    "input_channels",
                )
            ),
            out_channels=(
                _require_positive_integer(
                    output_channels,
                    "output_channels",
                )
            ),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Produce output logits."""

        return self.convolution(
            feature_map
        )


class UNet(nn.Module):
    """Standard U-Net binary segmentation baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = False,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize U-Net.

        Parameters
        ----------
        input_channels:
            Number of input image channels. The research protocol uses RGB
            images, so the approved value is three.

        output_channels:
            Number of output logit channels. Binary lesion segmentation uses
            one output channel.

        base_channels:
            Width of the first encoder stage. Subsequent stages use
            ``base_channels * [2, 4, 8, 16]``.

        bilinear:
            Use bilinear decoder upsampling instead of transposed
            convolutions.

        dropout_probability:
            Optional spatial dropout at the bottleneck. The default baseline
            uses no dropout.
        """

        super().__init__()

        self.input_channels = (
            _require_positive_integer(
                input_channels,
                "input_channels",
            )
        )

        self.output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        self.base_channels = (
            _require_positive_integer(
                base_channels,
                "base_channels",
            )
        )

        if not isinstance(
            bilinear,
            bool,
        ):
            raise TypeError(
                "bilinear must be Boolean."
            )

        self.bilinear = bilinear

        self.dropout_probability = (
            _validate_dropout(
                dropout_probability
            )
        )

        channels_1 = self.base_channels
        channels_2 = self.base_channels * 2
        channels_3 = self.base_channels * 4
        channels_4 = self.base_channels * 8
        channels_5 = self.base_channels * 16

        self.input_block = DoubleConvolution(
            input_channels=(
                self.input_channels
            ),
            output_channels=channels_1,
        )

        self.down1 = DownsamplingBlock(
            input_channels=channels_1,
            output_channels=channels_2,
        )

        self.down2 = DownsamplingBlock(
            input_channels=channels_2,
            output_channels=channels_3,
        )

        self.down3 = DownsamplingBlock(
            input_channels=channels_3,
            output_channels=channels_4,
        )

        self.down4 = DownsamplingBlock(
            input_channels=channels_4,
            output_channels=channels_5,
        )

        self.bottleneck_dropout = (
            nn.Dropout2d(
                p=self.dropout_probability
            )
            if self.dropout_probability > 0.0
            else nn.Identity()
        )

        self.up1 = UpsamplingBlock(
            decoder_channels=channels_5,
            skip_channels=channels_4,
            output_channels=channels_4,
            bilinear=self.bilinear,
        )

        self.up2 = UpsamplingBlock(
            decoder_channels=channels_4,
            skip_channels=channels_3,
            output_channels=channels_3,
            bilinear=self.bilinear,
        )

        self.up3 = UpsamplingBlock(
            decoder_channels=channels_3,
            skip_channels=channels_2,
            output_channels=channels_2,
            bilinear=self.bilinear,
        )

        self.up4 = UpsamplingBlock(
            decoder_channels=channels_2,
            skip_channels=channels_1,
            output_channels=channels_1,
            bilinear=self.bilinear,
        )

        self.output_layer = OutputConvolution(
            input_channels=channels_1,
            output_channels=(
                self.output_channels
            ),
        )

        self._initialize_parameters()

    def _initialize_parameters(
        self,
    ) -> None:
        """Initialize convolution and normalization parameters."""

        for module in self.modules():
            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                ),
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward_features(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return encoder, bottleneck, and decoder features."""

        self._validate_input(
            image
        )

        encoder1 = self.input_block(
            image
        )

        encoder2 = self.down1(
            encoder1
        )

        encoder3 = self.down2(
            encoder2
        )

        encoder4 = self.down3(
            encoder3
        )

        bottleneck = self.down4(
            encoder4
        )

        bottleneck = (
            self.bottleneck_dropout(
                bottleneck
            )
        )

        decoder4 = self.up1(
            bottleneck,
            encoder4,
        )

        decoder3 = self.up2(
            decoder4,
            encoder3,
        )

        decoder2 = self.up3(
            decoder3,
            encoder2,
        )

        decoder1 = self.up4(
            decoder2,
            encoder1,
        )

        features = {
            "encoder1": encoder1,
            "encoder2": encoder2,
            "encoder3": encoder3,
            "encoder4": encoder4,
            "bottleneck": bottleneck,
            "decoder4": decoder4,
            "decoder3": decoder3,
            "decoder2": decoder2,
            "decoder1": decoder1,
        }

        for name, feature in features.items():
            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"U-Net feature {name!r} "
                    "contains non-finite values."
                )

        return features

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Predict binary lesion-mask logits."""

        input_size = tuple(
            int(value)
            for value in image.shape[-2:]
        )

        features = self.forward_features(
            image
        )

        mask_logits = self.output_layer(
            features[
                "decoder1"
            ]
        )

        if (
            mask_logits.shape[-2:]
            != input_size
        ):
            mask_logits = F.interpolate(
                mask_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        if not torch.isfinite(
            mask_logits
        ).all():
            raise RuntimeError(
                "U-Net mask logits contain "
                "non-finite values."
            )

        return {
            "mask_logits": (
                mask_logits.contiguous()
            )
        }

    def _validate_input(
        self,
        image: Tensor,
    ) -> None:
        """Validate one U-Net input tensor."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "U-Net input must be a "
                "torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "U-Net input must have shape "
                "[B, C, H, W], received "
                f"{tuple(image.shape)}."
            )

        if (
            image.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "U-Net input channel mismatch: "
                f"expected {self.input_channels}, "
                f"received {image.shape[1]}."
            )

        if (
            image.shape[-2] < 16
            or image.shape[-1] < 16
        ):
            raise ValueError(
                "U-Net input height and width "
                "must each be at least 16 pixels."
            )

        if not torch.isfinite(
            image
        ).all():
            raise ValueError(
                "U-Net input contains "
                "non-finite values."
            )

    def parameter_count(
        self,
        *,
        trainable_only: bool = False,
    ) -> int:
        """Return the model parameter count."""

        if not isinstance(
            trainable_only,
            bool,
        ):
            raise TypeError(
                "trainable_only must be Boolean."
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if (
                parameter.requires_grad
                or not trainable_only
            )
        )

    def architecture_summary(
        self,
    ) -> dict[str, Any]:
        """Return baseline architecture metadata."""

        return {
            "protocol_version": (
                UNET_PROTOCOL_VERSION
            ),
            "architecture": (
                "standard_unet"
            ),
            "task": (
                "binary_lesion_segmentation"
            ),
            "learning_type": (
                "fully_supervised"
            ),
            "input_channels": (
                self.input_channels
            ),
            "output_channels": (
                self.output_channels
            ),
            "base_channels": (
                self.base_channels
            ),
            "bilinear_upsampling": (
                self.bilinear
            ),
            "dropout_probability": (
                self.dropout_probability
            ),
            "parameter_count": (
                self.parameter_count()
            ),
            "trainable_parameter_count": (
                self.parameter_count(
                    trainable_only=True
                )
            ),
            "output_keys": [
                "mask_logits",
            ],
            "uses_transformer": False,
            "uses_boundary_conditioning": (
                False
            ),
            "uses_auxiliary_targets": False,
        }


StandardUNet = UNet


def run_unet_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward test."""

    torch.manual_seed(
        42
    )

    model = UNet(
        input_channels=3,
        output_channels=1,
        base_channels=16,
        bilinear=False,
        dropout_probability=0.0,
    )

    model.train()

    image = torch.randn(
        2,
        3,
        65,
        67,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            2,
            1,
            65,
            67,
        )
        > 0.5
    ).to(
        dtype=torch.float32
    )

    output = model(
        image
    )

    mask_logits = output[
        "mask_logits"
    ]

    loss = (
        F.binary_cross_entropy_with_logits(
            mask_logits,
            target,
        )
    )

    loss.backward()

    first_weight_gradient = (
        model
        .input_block
        .block[0]
        .weight
        .grad
    )

    small_parameter_count = (
        model.parameter_count()
    )

    default_model = UNet(
        input_channels=3,
        output_channels=1,
        base_channels=64,
        bilinear=False,
    )

    default_parameter_count = (
        default_model.parameter_count()
    )

    del default_model

    checks = {
        "output_is_mapping": (
            isinstance(
                output,
                dict,
            )
        ),
        "output_keys": (
            tuple(
                output
            )
            == (
                "mask_logits",
            )
        ),
        "output_shape": (
            tuple(
                mask_logits.shape
            )
            == (
                2,
                1,
                65,
                67,
            )
        ),
        "output_finite": (
            torch.isfinite(
                mask_logits
            ).all().item()
        ),
        "loss_finite": (
            torch.isfinite(
                loss
            ).item()
        ),
        "loss_requires_gradient": (
            loss.requires_grad
        ),
        "input_gradient_exists": (
            image.grad is not None
        ),
        "input_gradient_finite": (
            image.grad is not None
            and torch.isfinite(
                image.grad
            ).all().item()
        ),
        "model_gradient_exists": (
            first_weight_gradient
            is not None
        ),
        "model_gradient_nonzero": (
            first_weight_gradient
            is not None
            and float(
                first_weight_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "small_parameter_count_positive": (
            small_parameter_count > 0
        ),
        "default_parameter_count_plausible": (
            30_000_000
            <= default_parameter_count
            <= 32_000_000
        ),
        "fully_supervised_baseline": (
            model.architecture_summary()[
                "learning_type"
            ]
            == "fully_supervised"
        ),
        "no_transformer": (
            model.architecture_summary()[
                "uses_transformer"
            ]
            is False
        ),
        "no_boundary_conditioning": (
            model.architecture_summary()[
                "uses_boundary_conditioning"
            ]
            is False
        ),
        "no_auxiliary_targets": (
            model.architecture_summary()[
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
            UNET_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_output_shape": list(
            mask_logits.shape
        ),
        "small_model_parameter_count": (
            small_parameter_count
        ),
        "default_model_parameter_count": (
            default_parameter_count
        ),
        "architecture": (
            model.architecture_summary()
        ),
    }