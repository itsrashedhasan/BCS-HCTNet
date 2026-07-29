"""UNet++ baseline for binary lesion segmentation.

UNet++ extends U-Net with nested, densely connected skip pathways. Features
from multiple semantic depths are progressively refined before reaching the
final segmentation head.

This implementation provides:

- five encoder resolutions;
- nested dense decoder nodes;
- bilinear feature alignment for arbitrary image dimensions;
- optional deep-supervision outputs;
- one-channel binary lesion-mask logits;
- the shared project model-output interface.

The default experiment uses the final segmentation output only:

    {
        "mask_logits": Tensor[B, 1, H, W]
    }

When deep supervision is enabled, the output also contains:

    {
        "auxiliary_mask_logits": [
            Tensor[B, 1, H, W],
            ...
        ]
    }

This remains a conventional fully supervised baseline. It does not use
Transformer features, boundary conditioning, contour targets, or signed
distance maps.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.baselines.unet import (
    DoubleConvolution,
)


UNETPP_PROTOCOL_VERSION = (
    "BCS-HCTNet-unet-plus-plus-baseline-v1"
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


def _upsample_to(
    feature_map: Tensor,
    reference_feature: Tensor,
) -> Tensor:
    """Resize one feature map to a reference spatial resolution."""

    if (
        not isinstance(
            feature_map,
            Tensor,
        )
        or not isinstance(
            reference_feature,
            Tensor,
        )
    ):
        raise TypeError(
            "feature_map and reference_feature "
            "must be torch.Tensor objects."
        )

    if (
        feature_map.ndim != 4
        or reference_feature.ndim != 4
    ):
        raise ValueError(
            "Feature maps must have shape "
            "[B, C, H, W]."
        )

    if (
        feature_map.shape[0]
        != reference_feature.shape[0]
    ):
        raise RuntimeError(
            "Feature-map batch sizes do not match."
        )

    target_size = tuple(
        int(value)
        for value
        in reference_feature.shape[-2:]
    )

    if (
        feature_map.shape[-2:]
        == reference_feature.shape[-2:]
    ):
        return feature_map

    return F.interpolate(
        feature_map,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


class UNetPP(nn.Module):
    """Nested U-Net++ binary segmentation baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 64,
        deep_supervision: bool = False,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize UNet++.

        Parameters
        ----------
        input_channels:
            Number of image channels. The approved research input is RGB.

        output_channels:
            Number of segmentation-logit channels. Binary lesion
            segmentation uses one channel.

        base_channels:
            Width of the first encoder stage. The five encoder widths are
            ``base_channels * [1, 2, 4, 8, 16]``.

        deep_supervision:
            Produce auxiliary segmentation logits from intermediate nested
            decoder nodes. The default baseline protocol disables this so
            all baseline models optimize one final mask output.

        dropout_probability:
            Optional spatial dropout applied to the deepest encoder feature.
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
            deep_supervision,
            bool,
        ):
            raise TypeError(
                "deep_supervision must be Boolean."
            )

        self.deep_supervision = (
            deep_supervision
        )

        self.dropout_probability = (
            _validate_dropout(
                dropout_probability
            )
        )

        channels = (
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
            self.base_channels * 16,
        )

        (
            channels_0,
            channels_1,
            channels_2,
            channels_3,
            channels_4,
        ) = channels

        self.encoder_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Encoder nodes: X^{i,0}
        self.conv0_0 = DoubleConvolution(
            input_channels=(
                self.input_channels
            ),
            output_channels=channels_0,
        )

        self.conv1_0 = DoubleConvolution(
            input_channels=channels_0,
            output_channels=channels_1,
        )

        self.conv2_0 = DoubleConvolution(
            input_channels=channels_1,
            output_channels=channels_2,
        )

        self.conv3_0 = DoubleConvolution(
            input_channels=channels_2,
            output_channels=channels_3,
        )

        self.conv4_0 = DoubleConvolution(
            input_channels=channels_3,
            output_channels=channels_4,
        )

        # First nested decoder column: X^{i,1}
        self.conv0_1 = DoubleConvolution(
            input_channels=(
                channels_0
                + channels_1
            ),
            output_channels=channels_0,
        )

        self.conv1_1 = DoubleConvolution(
            input_channels=(
                channels_1
                + channels_2
            ),
            output_channels=channels_1,
        )

        self.conv2_1 = DoubleConvolution(
            input_channels=(
                channels_2
                + channels_3
            ),
            output_channels=channels_2,
        )

        self.conv3_1 = DoubleConvolution(
            input_channels=(
                channels_3
                + channels_4
            ),
            output_channels=channels_3,
        )

        # Second nested decoder column: X^{i,2}
        self.conv0_2 = DoubleConvolution(
            input_channels=(
                channels_0 * 2
                + channels_1
            ),
            output_channels=channels_0,
        )

        self.conv1_2 = DoubleConvolution(
            input_channels=(
                channels_1 * 2
                + channels_2
            ),
            output_channels=channels_1,
        )

        self.conv2_2 = DoubleConvolution(
            input_channels=(
                channels_2 * 2
                + channels_3
            ),
            output_channels=channels_2,
        )

        # Third nested decoder column: X^{i,3}
        self.conv0_3 = DoubleConvolution(
            input_channels=(
                channels_0 * 3
                + channels_1
            ),
            output_channels=channels_0,
        )

        self.conv1_3 = DoubleConvolution(
            input_channels=(
                channels_1 * 3
                + channels_2
            ),
            output_channels=channels_1,
        )

        # Final nested decoder node: X^{0,4}
        self.conv0_4 = DoubleConvolution(
            input_channels=(
                channels_0 * 4
                + channels_1
            ),
            output_channels=channels_0,
        )

        self.bottleneck_dropout = (
            nn.Dropout2d(
                p=self.dropout_probability
            )
            if self.dropout_probability > 0.0
            else nn.Identity()
        )

        self.final_head = nn.Conv2d(
            in_channels=channels_0,
            out_channels=(
                self.output_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.deep_supervision:
            self.auxiliary_heads = (
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=(
                                channels_0
                            ),
                            out_channels=(
                                self.output_channels
                            ),
                            kernel_size=1,
                        )
                        for _ in range(3)
                    ]
                )
            )

        else:
            self.auxiliary_heads = (
                nn.ModuleList()
            )

        self._initialize_parameters()

    def _initialize_parameters(
        self,
    ) -> None:
        """Initialize convolutional and normalization parameters."""

        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
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

    def _validate_input(
        self,
        image: Tensor,
    ) -> None:
        """Validate one UNet++ input tensor."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "UNet++ input must be a "
                "torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "UNet++ input must have shape "
                "[B, C, H, W], received "
                f"{tuple(image.shape)}."
            )

        if image.shape[0] <= 0:
            raise ValueError(
                "UNet++ input batch cannot be empty."
            )

        if (
            image.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "UNet++ input channel mismatch: "
                f"expected {self.input_channels}, "
                f"received {image.shape[1]}."
            )

        if (
            image.shape[-2] < 16
            or image.shape[-1] < 16
        ):
            raise ValueError(
                "UNet++ input height and width "
                "must each be at least 16 pixels."
            )

        if not torch.isfinite(
            image
        ).all():
            raise ValueError(
                "UNet++ input contains "
                "non-finite values."
            )

    def forward_features(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all encoder and nested decoder feature nodes."""

        self._validate_input(
            image
        )

        # Encoder column.
        x0_0 = self.conv0_0(
            image
        )

        x1_0 = self.conv1_0(
            self.encoder_pool(
                x0_0
            )
        )

        x2_0 = self.conv2_0(
            self.encoder_pool(
                x1_0
            )
        )

        x3_0 = self.conv3_0(
            self.encoder_pool(
                x2_0
            )
        )

        x4_0 = self.conv4_0(
            self.encoder_pool(
                x3_0
            )
        )

        x4_0 = self.bottleneck_dropout(
            x4_0
        )

        # First nested column.
        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    _upsample_to(
                        x1_0,
                        x0_0,
                    ),
                ],
                dim=1,
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    _upsample_to(
                        x2_0,
                        x1_0,
                    ),
                ],
                dim=1,
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    _upsample_to(
                        x3_0,
                        x2_0,
                    ),
                ],
                dim=1,
            )
        )

        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    _upsample_to(
                        x4_0,
                        x3_0,
                    ),
                ],
                dim=1,
            )
        )

        # Second nested column.
        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    _upsample_to(
                        x1_1,
                        x0_0,
                    ),
                ],
                dim=1,
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    _upsample_to(
                        x2_1,
                        x1_0,
                    ),
                ],
                dim=1,
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    _upsample_to(
                        x3_1,
                        x2_0,
                    ),
                ],
                dim=1,
            )
        )

        # Third nested column.
        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    _upsample_to(
                        x1_2,
                        x0_0,
                    ),
                ],
                dim=1,
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    _upsample_to(
                        x2_2,
                        x1_0,
                    ),
                ],
                dim=1,
            )
        )

        # Final nested node.
        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    _upsample_to(
                        x1_3,
                        x0_0,
                    ),
                ],
                dim=1,
            )
        )

        features = {
            "x0_0": x0_0,
            "x1_0": x1_0,
            "x2_0": x2_0,
            "x3_0": x3_0,
            "x4_0": x4_0,
            "x0_1": x0_1,
            "x1_1": x1_1,
            "x2_1": x2_1,
            "x3_1": x3_1,
            "x0_2": x0_2,
            "x1_2": x1_2,
            "x2_2": x2_2,
            "x0_3": x0_3,
            "x1_3": x1_3,
            "x0_4": x0_4,
        }

        for name, feature in features.items():
            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"UNet++ feature {name!r} "
                    "contains non-finite values."
                )

        return features

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Any]:
        """Predict lesion-mask logits."""

        input_size = tuple(
            int(value)
            for value
            in image.shape[-2:]
        )

        features = self.forward_features(
            image
        )

        mask_logits = self.final_head(
            features[
                "x0_4"
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
                "UNet++ mask logits contain "
                "non-finite values."
            )

        output: dict[str, Any] = {
            "mask_logits": (
                mask_logits.contiguous()
            )
        }

        if self.deep_supervision:
            auxiliary_features = (
                features[
                    "x0_1"
                ],
                features[
                    "x0_2"
                ],
                features[
                    "x0_3"
                ],
            )

            auxiliary_logits: list[
                Tensor
            ] = []

            for head, feature in zip(
                self.auxiliary_heads,
                auxiliary_features,
                strict=True,
            ):
                logits = head(
                    feature
                )

                if (
                    logits.shape[-2:]
                    != input_size
                ):
                    logits = F.interpolate(
                        logits,
                        size=input_size,
                        mode="bilinear",
                        align_corners=False,
                    )

                if not torch.isfinite(
                    logits
                ).all():
                    raise RuntimeError(
                        "UNet++ auxiliary logits "
                        "contain non-finite values."
                    )

                auxiliary_logits.append(
                    logits.contiguous()
                )

            output[
                "auxiliary_mask_logits"
            ] = auxiliary_logits

        return output

    def parameter_count(
        self,
        *,
        trainable_only: bool = False,
    ) -> int:
        """Return the number of model parameters."""

        if not isinstance(
            trainable_only,
            bool,
        ):
            raise TypeError(
                "trainable_only must be Boolean."
            )

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if (
                parameter.requires_grad
                or not trainable_only
            )
        )

    def architecture_summary(
        self,
    ) -> dict[str, Any]:
        """Return architecture and protocol metadata."""

        return {
            "protocol_version": (
                UNETPP_PROTOCOL_VERSION
            ),
            "architecture": (
                "unet_plus_plus"
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
            "deep_supervision": (
                self.deep_supervision
            ),
            "dropout_probability": (
                self.dropout_probability
            ),
            "encoder_depth": 5,
            "nested_decoder_nodes": 10,
            "parameter_count": (
                self.parameter_count()
            ),
            "trainable_parameter_count": (
                self.parameter_count(
                    trainable_only=True
                )
            ),
            "output_keys": (
                [
                    "mask_logits",
                    "auxiliary_mask_logits",
                ]
                if self.deep_supervision
                else [
                    "mask_logits",
                ]
            ),
            "uses_transformer": False,
            "uses_boundary_conditioning": (
                False
            ),
            "uses_auxiliary_targets": False,
        }


UNetPlusPlus = UNetPP
NestedUNet = UNetPP


def run_unetpp_self_test() -> dict[str, Any]:
    """Run offline CPU forward/backward and deep-supervision tests."""

    torch.manual_seed(
        42
    )

    model = UNetPP(
        input_channels=3,
        output_channels=1,
        base_channels=8,
        deep_supervision=False,
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

    first_gradient = (
        model
        .conv0_0
        .block[0]
        .weight
        .grad
    )

    features = model.forward_features(
        image.detach()
    )

    deep_supervision_model = UNetPP(
        input_channels=3,
        output_channels=1,
        base_channels=4,
        deep_supervision=True,
    )

    deep_supervision_model.eval()

    with torch.inference_mode():
        deep_output = (
            deep_supervision_model(
                torch.randn(
                    1,
                    3,
                    33,
                    35,
                )
            )
        )

    auxiliary_logits = deep_output[
        "auxiliary_mask_logits"
    ]

    architecture = (
        model.architecture_summary()
    )

    parameter_count = (
        model.parameter_count()
    )

    checks = {
        "output_is_mapping": (
            isinstance(
                output,
                dict,
            )
        ),
        "standard_output_keys": (
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
            first_gradient is not None
        ),
        "model_gradient_nonzero": (
            first_gradient is not None
            and float(
                first_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "feature_count": (
            len(
                features
            )
            == 15
        ),
        "final_feature_shape": (
            tuple(
                features[
                    "x0_4"
                ].shape[-2:]
            )
            == (
                65,
                67,
            )
        ),
        "deep_supervision_key": (
            "auxiliary_mask_logits"
            in deep_output
        ),
        "deep_supervision_count": (
            len(
                auxiliary_logits
            )
            == 3
        ),
        "deep_supervision_shapes": all(
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
        ),
        "deep_supervision_finite": all(
            torch.isfinite(
                logits
            ).all().item()
            for logits
            in auxiliary_logits
        ),
        "parameter_count_positive": (
            parameter_count > 0
        ),
        "fully_supervised_baseline": (
            architecture[
                "learning_type"
            ]
            == "fully_supervised"
        ),
        "no_transformer": (
            architecture[
                "uses_transformer"
            ]
            is False
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
            UNETPP_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_output_shape": list(
            mask_logits.shape
        ),
        "feature_names": list(
            features
        ),
        "parameter_count": (
            parameter_count
        ),
        "architecture": (
            architecture
        ),
    }