"""DeepLabV3+ baseline for binary lesion segmentation.

This module implements a conventional DeepLabV3+ architecture using:

- a ResNet-50 or ResNet-101 encoder;
- output-stride-16 dilated feature extraction;
- Atrous Spatial Pyramid Pooling (ASPP);
- low-level feature projection;
- DeepLabV3+ decoder refinement;
- one-channel binary lesion-mask logits.

The model follows the common project output interface:

    {
        "mask_logits": Tensor[B, 1, H, W]
    }

This is a fully supervised segmentation baseline. It does not use:

- Transformer features;
- boundary-prior conditioning;
- contour supervision;
- signed-distance-map supervision;
- auxiliary geometric targets.

ImageNet initialization is optional. The CPU self-test uses no downloaded
weights. A local torchvision-compatible ResNet checkpoint can also be loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEEPLABV3PLUS_PROTOCOL_VERSION = (
    "BCS-HCTNet-deeplabv3plus-baseline-v1"
)

SUPPORTED_BACKBONES = (
    "resnet50",
    "resnet101",
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


def _normalize_backbone_name(
    backbone_name: object,
) -> str:
    """Normalize and validate the ResNet backbone name."""

    normalized = str(
        backbone_name
    ).strip().lower().replace(
        "-",
        "",
    )

    aliases = {
        "resnet50": "resnet50",
        "resnet101": "resnet101",
    }

    if normalized not in aliases:
        raise ValueError(
            "Unsupported DeepLabV3+ backbone "
            f"{backbone_name!r}. Supported values "
            f"are {list(SUPPORTED_BACKBONES)}."
        )

    return aliases[
        normalized
    ]


def _trusted_torch_load(
    checkpoint_path: Path,
) -> object:
    """Load a trusted local PyTorch checkpoint."""

    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


def _extract_state_dict(
    checkpoint: object,
) -> dict[str, Tensor]:
    """Extract a tensor state dictionary from a checkpoint."""

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "Backbone checkpoint must contain "
            "a mapping."
        )

    candidate: object = checkpoint

    for key in (
        "state_dict",
        "model_state_dict",
        "backbone_state_dict",
    ):
        nested = checkpoint.get(
            key
        )

        if isinstance(
            nested,
            Mapping,
        ):
            candidate = nested
            break

    if not isinstance(
        candidate,
        Mapping,
    ):
        raise TypeError(
            "Could not locate a valid backbone "
            "state dictionary."
        )

    state_dict: dict[
        str,
        Tensor,
    ] = {}

    for raw_key, value in candidate.items():
        if not isinstance(
            value,
            Tensor,
        ):
            continue

        key = str(
            raw_key
        )

        for prefix in (
            "module.",
            "model.",
            "backbone.",
        ):
            if key.startswith(
                prefix
            ):
                key = key[
                    len(prefix):
                ]

        state_dict[
            key
        ] = value

    if not state_dict:
        raise RuntimeError(
            "Backbone checkpoint contains no "
            "tensor parameters."
        )

    return state_dict


def _build_torchvision_resnet(
    *,
    backbone_name: str,
    pretrained: bool,
) -> nn.Module:
    """Build a torchvision ResNet with output stride 16."""

    try:
        from torchvision.models import (
            ResNet50_Weights,
            ResNet101_Weights,
            resnet50,
            resnet101,
        )

    except ImportError as error:
        raise ImportError(
            "torchvision is required for the "
            "DeepLabV3+ baseline."
        ) from error

    replace_stride_with_dilation = [
        False,
        False,
        True,
    ]

    if backbone_name == "resnet50":
        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        return resnet50(
            weights=weights,
            replace_stride_with_dilation=(
                replace_stride_with_dilation
            ),
        )

    if backbone_name == "resnet101":
        weights = (
            ResNet101_Weights.DEFAULT
            if pretrained
            else None
        )

        return resnet101(
            weights=weights,
            replace_stride_with_dilation=(
                replace_stride_with_dilation
            ),
        )

    raise AssertionError(
        "Unreachable backbone branch."
    )


class ResNetFeatureEncoder(nn.Module):
    """ResNet feature extractor returning low- and high-level features."""

    def __init__(
        self,
        *,
        backbone_name: str = "resnet50",
        pretrained: bool = False,
        weights_path: str | Path | None = None,
    ) -> None:
        """Initialize the feature encoder."""

        super().__init__()

        self.backbone_name = (
            _normalize_backbone_name(
                backbone_name
            )
        )

        if not isinstance(
            pretrained,
            bool,
        ):
            raise TypeError(
                "pretrained must be Boolean."
            )

        if (
            pretrained
            and weights_path is not None
        ):
            raise ValueError(
                "Use either torchvision pretrained "
                "weights or weights_path, not both."
            )

        backbone = (
            _build_torchvision_resnet(
                backbone_name=(
                    self.backbone_name
                ),
                pretrained=pretrained,
            )
        )

        self.pretrained = pretrained
        self.weights_path: str | None = None

        if weights_path is not None:
            resolved_weights_path = (
                Path(
                    weights_path
                )
                .expanduser()
                .resolve()
            )

            if not resolved_weights_path.is_file():
                raise FileNotFoundError(
                    "ResNet weights file not found: "
                    f"{resolved_weights_path}"
                )

            checkpoint = _trusted_torch_load(
                resolved_weights_path
            )

            state_dict = _extract_state_dict(
                checkpoint
            )

            incompatible = (
                backbone.load_state_dict(
                    state_dict,
                    strict=False,
                )
            )

            allowed_missing = {
                "fc.weight",
                "fc.bias",
            }

            unexpected_keys = list(
                incompatible.unexpected_keys
            )

            invalid_missing = [
                key
                for key
                in incompatible.missing_keys
                if key not in allowed_missing
            ]

            if (
                invalid_missing
                or unexpected_keys
            ):
                raise RuntimeError(
                    "Local ResNet checkpoint is "
                    "incompatible. Missing keys: "
                    f"{invalid_missing}; unexpected "
                    f"keys: {unexpected_keys}."
                )

            self.weights_path = str(
                resolved_weights_path
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.low_level_channels = 256
        self.high_level_channels = 2048
        self.output_stride = 16

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return low-level and high-level ResNet features."""

        stem = self.stem(
            image
        )

        low_level = self.layer1(
            stem
        )

        feature2 = self.layer2(
            low_level
        )

        feature3 = self.layer3(
            feature2
        )

        high_level = self.layer4(
            feature3
        )

        features = {
            "low_level": low_level,
            "high_level": high_level,
        }

        for name, feature in features.items():
            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    "DeepLabV3+ encoder feature "
                    f"{name!r} contains non-finite "
                    "values."
                )

        return features


class ConvNormActivation(nn.Sequential):
    """Convolution, batch normalization, and ReLU."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize the block."""

        super().__init__(
            nn.Conv2d(
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
                kernel_size=(
                    _require_positive_integer(
                        kernel_size,
                        "kernel_size",
                    )
                ),
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                output_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )


class ASPPBranch(nn.Module):
    """One atrous-convolution ASPP branch."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        dilation: int,
    ) -> None:
        """Initialize an atrous branch."""

        super().__init__()

        resolved_dilation = (
            _require_positive_integer(
                dilation,
                "dilation",
            )
        )

        self.block = ConvNormActivation(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=3,
            dilation=resolved_dilation,
            padding=resolved_dilation,
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply atrous convolution."""

        return self.block(
            feature_map
        )


class ASPPPooling(nn.Module):
    """Image-level context branch for ASPP."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
    ) -> None:
        """Initialize global pooling projection."""

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.projection = (
            ConvNormActivation(
                input_channels=(
                    input_channels
                ),
                output_channels=(
                    output_channels
                ),
                kernel_size=1,
            )
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Pool, project, and restore spatial resolution."""

        spatial_size = tuple(
            int(value)
            for value
            in feature_map.shape[-2:]
        )

        pooled = self.pool(
            feature_map
        )

        pooled = self.projection(
            pooled
        )

        return F.interpolate(
            pooled,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )


class AtrousSpatialPyramidPooling(
    nn.Module
):
    """DeepLab atrous spatial pyramid pooling module."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int = 256,
        atrous_rates: tuple[
            int,
            int,
            int,
        ] = (
            6,
            12,
            18,
        ),
        dropout_probability: float = 0.1,
    ) -> None:
        """Initialize ASPP."""

        super().__init__()

        if (
            not isinstance(
                atrous_rates,
                tuple,
            )
            or len(
                atrous_rates
            )
            != 3
        ):
            raise ValueError(
                "atrous_rates must contain "
                "exactly three integer rates."
            )

        resolved_output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        resolved_rates = tuple(
            _require_positive_integer(
                rate,
                "atrous rate",
            )
            for rate in atrous_rates
        )

        resolved_dropout = (
            _validate_dropout(
                dropout_probability
            )
        )

        self.branches = nn.ModuleList(
            [
                ConvNormActivation(
                    input_channels=(
                        input_channels
                    ),
                    output_channels=(
                        resolved_output_channels
                    ),
                    kernel_size=1,
                ),
                ASPPBranch(
                    input_channels=(
                        input_channels
                    ),
                    output_channels=(
                        resolved_output_channels
                    ),
                    dilation=(
                        resolved_rates[0]
                    ),
                ),
                ASPPBranch(
                    input_channels=(
                        input_channels
                    ),
                    output_channels=(
                        resolved_output_channels
                    ),
                    dilation=(
                        resolved_rates[1]
                    ),
                ),
                ASPPBranch(
                    input_channels=(
                        input_channels
                    ),
                    output_channels=(
                        resolved_output_channels
                    ),
                    dilation=(
                        resolved_rates[2]
                    ),
                ),
                ASPPPooling(
                    input_channels=(
                        input_channels
                    ),
                    output_channels=(
                        resolved_output_channels
                    ),
                ),
            ]
        )

        concatenated_channels = (
            resolved_output_channels
            * len(
                self.branches
            )
        )

        self.projection = nn.Sequential(
            ConvNormActivation(
                input_channels=(
                    concatenated_channels
                ),
                output_channels=(
                    resolved_output_channels
                ),
                kernel_size=1,
            ),
            nn.Dropout(
                p=resolved_dropout
            ),
        )

        self.atrous_rates = (
            resolved_rates
        )

        self.output_channels = (
            resolved_output_channels
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Aggregate multi-scale atrous context."""

        branch_features = [
            branch(
                feature_map
            )
            for branch
            in self.branches
        ]

        concatenated = torch.cat(
            branch_features,
            dim=1,
        )

        output = self.projection(
            concatenated
        )

        if not torch.isfinite(
            output
        ).all():
            raise RuntimeError(
                "ASPP output contains non-finite "
                "values."
            )

        return output


class DeepLabV3PlusDecoder(nn.Module):
    """Low-level feature fusion decoder."""

    def __init__(
        self,
        *,
        low_level_channels: int,
        aspp_channels: int,
        decoder_channels: int = 256,
        low_level_projection_channels: int = 48,
        output_channels: int = 1,
        dropout_probability: float = 0.1,
    ) -> None:
        """Initialize the decoder."""

        super().__init__()

        resolved_decoder_channels = (
            _require_positive_integer(
                decoder_channels,
                "decoder_channels",
            )
        )

        resolved_low_projection = (
            _require_positive_integer(
                low_level_projection_channels,
                "low_level_projection_channels",
            )
        )

        resolved_output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        resolved_dropout = (
            _validate_dropout(
                dropout_probability
            )
        )

        self.low_level_projection = (
            ConvNormActivation(
                input_channels=(
                    low_level_channels
                ),
                output_channels=(
                    resolved_low_projection
                ),
                kernel_size=1,
            )
        )

        fused_channels = (
            aspp_channels
            + resolved_low_projection
        )

        self.refinement = nn.Sequential(
            ConvNormActivation(
                input_channels=(
                    fused_channels
                ),
                output_channels=(
                    resolved_decoder_channels
                ),
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(
                p=resolved_dropout
            ),
            ConvNormActivation(
                input_channels=(
                    resolved_decoder_channels
                ),
                output_channels=(
                    resolved_decoder_channels
                ),
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(
                p=resolved_dropout
            ),
        )

        self.classifier = nn.Conv2d(
            in_channels=(
                resolved_decoder_channels
            ),
            out_channels=(
                resolved_output_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        *,
        aspp_feature: Tensor,
        low_level_feature: Tensor,
    ) -> Tensor:
        """Fuse ASPP and low-level encoder features."""

        projected_low_level = (
            self.low_level_projection(
                low_level_feature
            )
        )

        upsampled_aspp = F.interpolate(
            aspp_feature,
            size=(
                projected_low_level.shape[-2:]
            ),
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat(
            [
                upsampled_aspp,
                projected_low_level,
            ],
            dim=1,
        )

        refined = self.refinement(
            fused
        )

        return self.classifier(
            refined
        )


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ binary lesion-segmentation baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        backbone_name: str = "resnet50",
        backbone_pretrained: bool = False,
        backbone_weights_path: (
            str | Path | None
        ) = None,
        aspp_channels: int = 256,
        decoder_channels: int = 256,
        low_level_projection_channels: int = 48,
        atrous_rates: tuple[
            int,
            int,
            int,
        ] = (
            6,
            12,
            18,
        ),
        dropout_probability: float = 0.1,
    ) -> None:
        """Initialize DeepLabV3+."""

        super().__init__()

        self.input_channels = (
            _require_positive_integer(
                input_channels,
                "input_channels",
            )
        )

        if self.input_channels != 3:
            raise ValueError(
                "The torchvision ResNet encoder "
                "requires three RGB input channels."
            )

        self.output_channels = (
            _require_positive_integer(
                output_channels,
                "output_channels",
            )
        )

        self.backbone_name = (
            _normalize_backbone_name(
                backbone_name
            )
        )

        self.backbone_pretrained = (
            backbone_pretrained
        )

        self.dropout_probability = (
            _validate_dropout(
                dropout_probability
            )
        )

        self.encoder = ResNetFeatureEncoder(
            backbone_name=(
                self.backbone_name
            ),
            pretrained=(
                backbone_pretrained
            ),
            weights_path=(
                backbone_weights_path
            ),
        )

        self.aspp = (
            AtrousSpatialPyramidPooling(
                input_channels=(
                    self.encoder
                    .high_level_channels
                ),
                output_channels=(
                    aspp_channels
                ),
                atrous_rates=(
                    atrous_rates
                ),
                dropout_probability=(
                    self.dropout_probability
                ),
            )
        )

        self.decoder = (
            DeepLabV3PlusDecoder(
                low_level_channels=(
                    self.encoder
                    .low_level_channels
                ),
                aspp_channels=(
                    aspp_channels
                ),
                decoder_channels=(
                    decoder_channels
                ),
                low_level_projection_channels=(
                    low_level_projection_channels
                ),
                output_channels=(
                    self.output_channels
                ),
                dropout_probability=(
                    self.dropout_probability
                ),
            )
        )

        self.aspp_channels = (
            aspp_channels
        )

        self.decoder_channels = (
            decoder_channels
        )

        self.low_level_projection_channels = (
            low_level_projection_channels
        )

        self.atrous_rates = tuple(
            atrous_rates
        )

        self._initialize_new_parameters()

    def _initialize_new_parameters(
        self,
    ) -> None:
        """Initialize ASPP and decoder without resetting the encoder."""

        for root_module in (
            self.aspp,
            self.decoder,
        ):
            for module in root_module.modules():
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
        """Validate one model input tensor."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "DeepLabV3+ input must be a "
                "torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "DeepLabV3+ input must have shape "
                "[B, C, H, W], received "
                f"{tuple(image.shape)}."
            )

        if image.shape[0] <= 0:
            raise ValueError(
                "Input batch cannot be empty."
            )

        if (
            image.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "Input channel mismatch: expected "
                f"{self.input_channels}, received "
                f"{image.shape[1]}."
            )

        if (
            image.shape[-2] < 32
            or image.shape[-1] < 32
        ):
            raise ValueError(
                "DeepLabV3+ input height and width "
                "must each be at least 32 pixels."
            )

        if not torch.isfinite(
            image
        ).all():
            raise ValueError(
                "Input contains non-finite values."
            )

    def forward_features(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return encoder, ASPP, and decoder-level features."""

        self._validate_input(
            image
        )

        encoder_features = self.encoder(
            image
        )

        aspp_feature = self.aspp(
            encoder_features[
                "high_level"
            ]
        )

        decoder_logits = self.decoder(
            aspp_feature=(
                aspp_feature
            ),
            low_level_feature=(
                encoder_features[
                    "low_level"
                ]
            ),
        )

        return {
            "low_level": (
                encoder_features[
                    "low_level"
                ]
            ),
            "high_level": (
                encoder_features[
                    "high_level"
                ]
            ),
            "aspp": aspp_feature,
            "decoder_logits": (
                decoder_logits
            ),
        }

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Predict binary lesion-mask logits."""

        input_size = tuple(
            int(value)
            for value
            in image.shape[-2:]
        )

        features = self.forward_features(
            image
        )

        mask_logits = F.interpolate(
            features[
                "decoder_logits"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        if not torch.isfinite(
            mask_logits
        ).all():
            raise RuntimeError(
                "DeepLabV3+ mask logits contain "
                "non-finite values."
            )

        return {
            "mask_logits": (
                mask_logits.contiguous()
            )
        }

    def freeze_batch_normalization(
        self,
    ) -> None:
        """Put all batch-normalization layers in evaluation mode."""

        for module in self.modules():
            if isinstance(
                module,
                nn.BatchNorm2d,
            ):
                module.eval()

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
                DEEPLABV3PLUS_PROTOCOL_VERSION
            ),
            "architecture": (
                "deeplabv3plus"
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
            "backbone": (
                self.backbone_name
            ),
            "backbone_pretrained": (
                self.backbone_pretrained
            ),
            "backbone_weights_path": (
                self.encoder.weights_path
            ),
            "output_stride": (
                self.encoder.output_stride
            ),
            "aspp_channels": (
                self.aspp_channels
            ),
            "decoder_channels": (
                self.decoder_channels
            ),
            "low_level_projection_channels": (
                self.low_level_projection_channels
            ),
            "atrous_rates": list(
                self.atrous_rates
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


DeepLabV3plus = DeepLabV3Plus


def run_deeplabv3plus_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward test."""

    torch.manual_seed(
        42
    )

    model = DeepLabV3Plus(
        input_channels=3,
        output_channels=1,
        backbone_name="resnet50",
        backbone_pretrained=False,
        aspp_channels=64,
        decoder_channels=64,
        low_level_projection_channels=24,
        atrous_rates=(
            3,
            6,
            9,
        ),
        dropout_probability=0.0,
    )

    # Evaluation mode avoids batch-normalization problems in the
    # image-pooling branch when the synthetic batch size is one.
    model.eval()

    image = torch.randn(
        1,
        3,
        33,
        35,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            1,
            1,
            33,
            35,
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

    classifier_gradient = (
        model
        .decoder
        .classifier
        .weight
        .grad
    )

    encoder_gradient = (
        model
        .encoder
        .stem[0]
        .weight
        .grad
    )

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
                1,
                1,
                33,
                35,
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
        "decoder_gradient_exists": (
            classifier_gradient
            is not None
        ),
        "decoder_gradient_nonzero": (
            classifier_gradient
            is not None
            and float(
                classifier_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "encoder_gradient_exists": (
            encoder_gradient
            is not None
        ),
        "encoder_gradient_finite": (
            encoder_gradient
            is not None
            and torch.isfinite(
                encoder_gradient
            ).all().item()
        ),
        "parameter_count_positive": (
            parameter_count > 0
        ),
        "resnet50_backbone": (
            architecture[
                "backbone"
            ]
            == "resnet50"
        ),
        "output_stride_16": (
            architecture[
                "output_stride"
            ]
            == 16
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
        "no_downloaded_weights": (
            architecture[
                "backbone_pretrained"
            ]
            is False
            and architecture[
                "backbone_weights_path"
            ]
            is None
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
            DEEPLABV3PLUS_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_output_shape": list(
            mask_logits.shape
        ),
        "parameter_count": (
            parameter_count
        ),
        "architecture": (
            architecture
        ),
    }