"""TransUNet-style baseline for binary lesion segmentation.

This implementation combines:

- a residual convolutional encoder;
- a Vision Transformer bottleneck;
- a U-Net-style convolutional decoder;
- multi-resolution CNN skip connections;
- one-channel binary lesion-mask logits.

The Transformer operates on the deepest convolutional feature map rather than
directly on raw image patches. Dynamic two-dimensional sinusoidal positional
encoding allows arbitrary valid image dimensions.

The model follows the shared project interface:

    {
        "mask_logits": Tensor[B, 1, H, W]
    }

This is a fully supervised mask-only baseline. It does not use:

- boundary-reliability conditioning;
- contour targets;
- boundary-band targets;
- signed-distance-map targets;
- BCS-HCTNet fusion modules.

The default configuration is deliberately smaller than the original
TransUNet R50-ViT-B/16 model so it remains feasible for the approved
experimental compute budget. All major dimensions remain configurable.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.baselines.unet import (
    DoubleConvolution,
    UpsamplingBlock,
)


TRANSUNET_PROTOCOL_VERSION = (
    "BCS-HCTNet-transunet-style-baseline-v1"
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


def _require_nonnegative_integer(
    value: object,
    context: str,
) -> int:
    """Require a non-negative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{context} must be a non-negative integer."
        )

    return value


def _validate_probability(
    value: object,
    context: str,
) -> float:
    """Validate a probability in [0, 1)."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
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
            f"{context} must be numeric."
        ) from error

    if not (
        0.0
        <= probability
        < 1.0
    ):
        raise ValueError(
            f"{context} must be in [0, 1)."
        )

    return probability


class ResidualConvolutionBlock(nn.Module):
    """Two-convolution residual feature block."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
    ) -> None:
        """Initialize the residual block."""

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

        resolved_stride = (
            _require_positive_integer(
                stride,
                "stride",
            )
        )

        self.convolution1 = nn.Conv2d(
            in_channels=(
                resolved_input_channels
            ),
            out_channels=(
                resolved_output_channels
            ),
            kernel_size=3,
            stride=resolved_stride,
            padding=1,
            bias=False,
        )

        self.normalization1 = nn.BatchNorm2d(
            resolved_output_channels
        )

        self.activation = nn.ReLU(
            inplace=True
        )

        self.convolution2 = nn.Conv2d(
            in_channels=(
                resolved_output_channels
            ),
            out_channels=(
                resolved_output_channels
            ),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.normalization2 = nn.BatchNorm2d(
            resolved_output_channels
        )

        if (
            resolved_stride != 1
            or resolved_input_channels
            != resolved_output_channels
        ):
            self.residual_projection = (
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=(
                            resolved_input_channels
                        ),
                        out_channels=(
                            resolved_output_channels
                        ),
                        kernel_size=1,
                        stride=(
                            resolved_stride
                        ),
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        resolved_output_channels
                    ),
                )
            )

        else:
            self.residual_projection = (
                nn.Identity()
            )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply the residual feature transformation."""

        residual = self.residual_projection(
            feature_map
        )

        output = self.convolution1(
            feature_map
        )

        output = self.normalization1(
            output
        )

        output = self.activation(
            output
        )

        output = self.convolution2(
            output
        )

        output = self.normalization2(
            output
        )

        output = output + residual

        return self.activation(
            output
        )


class ResidualEncoderStage(nn.Module):
    """Downsampling residual encoder stage."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        number_of_blocks: int,
    ) -> None:
        """Initialize one encoder stage."""

        super().__init__()

        resolved_number_of_blocks = (
            _require_positive_integer(
                number_of_blocks,
                "number_of_blocks",
            )
        )

        blocks: list[nn.Module] = [
            ResidualConvolutionBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=2,
            )
        ]

        for _ in range(
            resolved_number_of_blocks - 1
        ):
            blocks.append(
                ResidualConvolutionBlock(
                    input_channels=(
                        output_channels
                    ),
                    output_channels=(
                        output_channels
                    ),
                    stride=1,
                )
            )

        self.blocks = nn.Sequential(
            *blocks
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Downsample and refine a feature map."""

        return self.blocks(
            feature_map
        )


def build_2d_sinusoidal_position_encoding(
    *,
    height: int,
    width: int,
    embedding_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build deterministic two-dimensional sinusoidal positional encoding.

    Returns
    -------
    Tensor
        Position encoding with shape
        ``[1, height * width, embedding_dimension]``.
    """

    resolved_height = (
        _require_positive_integer(
            height,
            "height",
        )
    )

    resolved_width = (
        _require_positive_integer(
            width,
            "width",
        )
    )

    resolved_dimension = (
        _require_positive_integer(
            embedding_dimension,
            "embedding_dimension",
        )
    )

    if (
        resolved_dimension % 4
        != 0
    ):
        raise ValueError(
            "embedding_dimension must be "
            "divisible by four for two-dimensional "
            "sinusoidal positional encoding."
        )

    quarter_dimension = (
        resolved_dimension // 4
    )

    frequency_indices = torch.arange(
        quarter_dimension,
        device=device,
        dtype=torch.float32,
    )

    denominator = torch.pow(
        torch.tensor(
            10_000.0,
            device=device,
            dtype=torch.float32,
        ),
        frequency_indices
        / max(
            quarter_dimension,
            1,
        ),
    )

    y_positions = torch.arange(
        resolved_height,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(
        1
    )

    x_positions = torch.arange(
        resolved_width,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(
        1
    )

    y_angles = (
        y_positions
        / denominator.unsqueeze(
            0
        )
    )

    x_angles = (
        x_positions
        / denominator.unsqueeze(
            0
        )
    )

    y_encoding = torch.cat(
        [
            torch.sin(
                y_angles
            ),
            torch.cos(
                y_angles
            ),
        ],
        dim=1,
    )

    x_encoding = torch.cat(
        [
            torch.sin(
                x_angles
            ),
            torch.cos(
                x_angles
            ),
        ],
        dim=1,
    )

    y_grid = (
        y_encoding[
            :,
            None,
            :,
        ]
        .expand(
            resolved_height,
            resolved_width,
            -1,
        )
    )

    x_grid = (
        x_encoding[
            None,
            :,
            :,
        ]
        .expand(
            resolved_height,
            resolved_width,
            -1,
        )
    )

    encoding = torch.cat(
        [
            y_grid,
            x_grid,
        ],
        dim=-1,
    )

    encoding = encoding.reshape(
        1,
        resolved_height
        * resolved_width,
        resolved_dimension,
    )

    return encoding.to(
        dtype=dtype
    )


class TransformerBottleneck(nn.Module):
    """Transformer encoder operating on deep CNN feature tokens."""

    def __init__(
        self,
        *,
        input_channels: int,
        embedding_dimension: int,
        number_of_layers: int,
        number_of_heads: int,
        mlp_dimension: int,
        dropout_probability: float,
        attention_dropout_probability: float,
    ) -> None:
        """Initialize the Transformer bottleneck."""

        super().__init__()

        self.input_channels = (
            _require_positive_integer(
                input_channels,
                "input_channels",
            )
        )

        self.embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.number_of_layers = (
            _require_positive_integer(
                number_of_layers,
                "number_of_layers",
            )
        )

        self.number_of_heads = (
            _require_positive_integer(
                number_of_heads,
                "number_of_heads",
            )
        )

        self.mlp_dimension = (
            _require_positive_integer(
                mlp_dimension,
                "mlp_dimension",
            )
        )

        self.dropout_probability = (
            _validate_probability(
                dropout_probability,
                "dropout_probability",
            )
        )

        self.attention_dropout_probability = (
            _validate_probability(
                attention_dropout_probability,
                (
                    "attention_dropout_"
                    "probability"
                ),
            )
        )

        if (
            self.embedding_dimension
            % self.number_of_heads
            != 0
        ):
            raise ValueError(
                "embedding_dimension must be "
                "divisible by number_of_heads."
            )

        if (
            self.embedding_dimension
            % 4
            != 0
        ):
            raise ValueError(
                "embedding_dimension must be "
                "divisible by four."
            )

        self.input_projection = nn.Conv2d(
            in_channels=(
                self.input_channels
            ),
            out_channels=(
                self.embedding_dimension
            ),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.input_normalization = (
            nn.BatchNorm2d(
                self.embedding_dimension
            )
        )

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=(
                    self.embedding_dimension
                ),
                nhead=(
                    self.number_of_heads
                ),
                dim_feedforward=(
                    self.mlp_dimension
                ),
                dropout=(
                    self.dropout_probability
                ),
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer=(
                    transformer_layer
                ),
                num_layers=(
                    self.number_of_layers
                ),
                norm=nn.LayerNorm(
                    self.embedding_dimension
                ),
            )
        )

        self.token_dropout = nn.Dropout(
            p=(
                self.attention_dropout_probability
            )
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                in_channels=(
                    self.embedding_dimension
                ),
                out_channels=(
                    self.input_channels
                ),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.input_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """Transform CNN features and return spatial features and tokens."""

        if not isinstance(
            feature_map,
            Tensor,
        ):
            raise TypeError(
                "Transformer input must be a "
                "torch.Tensor."
            )

        if feature_map.ndim != 4:
            raise ValueError(
                "Transformer input must have shape "
                "[B, C, H, W]."
            )

        if (
            feature_map.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "Transformer input channel mismatch."
            )

        projected = self.input_projection(
            feature_map
        )

        projected = self.input_normalization(
            projected
        )

        batch_size = int(
            projected.shape[0]
        )

        height = int(
            projected.shape[-2]
        )

        width = int(
            projected.shape[-1]
        )

        tokens = projected.flatten(
            start_dim=2
        ).transpose(
            1,
            2,
        )

        positional_encoding = (
            build_2d_sinusoidal_position_encoding(
                height=height,
                width=width,
                embedding_dimension=(
                    self.embedding_dimension
                ),
                device=tokens.device,
                dtype=tokens.dtype,
            )
        )

        tokens = (
            tokens
            + positional_encoding
        )

        tokens = self.token_dropout(
            tokens
        )

        transformed_tokens = (
            self.transformer(
                tokens
            )
        )

        transformed_feature = (
            transformed_tokens.transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                self.embedding_dimension,
                height,
                width,
            )
        )

        transformed_feature = (
            self.output_projection(
                transformed_feature
            )
        )

        transformed_feature = (
            transformed_feature
            + feature_map
        )

        if not torch.isfinite(
            transformed_feature
        ).all():
            raise RuntimeError(
                "Transformer bottleneck produced "
                "non-finite spatial features."
            )

        if not torch.isfinite(
            transformed_tokens
        ).all():
            raise RuntimeError(
                "Transformer bottleneck produced "
                "non-finite tokens."
            )

        return (
            transformed_feature,
            transformed_tokens,
        )


class TransUNet(nn.Module):
    """CNN–Transformer U-Net segmentation baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 32,
        encoder_blocks: tuple[
            int,
            int,
            int,
            int,
        ] = (
            1,
            1,
            2,
            2,
        ),
        transformer_dimension: int = 512,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_mlp_dimension: int = 2048,
        transformer_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        bottleneck_dropout: float = 0.0,
        bilinear_decoder: bool = True,
    ) -> None:
        """Initialize the TransUNet-style model."""

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

        if (
            not isinstance(
                encoder_blocks,
                tuple,
            )
            or len(
                encoder_blocks
            )
            != 4
        ):
            raise ValueError(
                "encoder_blocks must contain "
                "four positive integers."
            )

        self.encoder_blocks = tuple(
            _require_positive_integer(
                value,
                "encoder block count",
            )
            for value
            in encoder_blocks
        )

        self.transformer_dimension = (
            _require_positive_integer(
                transformer_dimension,
                "transformer_dimension",
            )
        )

        self.transformer_layers = (
            _require_positive_integer(
                transformer_layers,
                "transformer_layers",
            )
        )

        self.transformer_heads = (
            _require_positive_integer(
                transformer_heads,
                "transformer_heads",
            )
        )

        self.transformer_mlp_dimension = (
            _require_positive_integer(
                transformer_mlp_dimension,
                (
                    "transformer_mlp_"
                    "dimension"
                ),
            )
        )

        self.transformer_dropout = (
            _validate_probability(
                transformer_dropout,
                "transformer_dropout",
            )
        )

        self.attention_dropout = (
            _validate_probability(
                attention_dropout,
                "attention_dropout",
            )
        )

        self.bottleneck_dropout_probability = (
            _validate_probability(
                bottleneck_dropout,
                "bottleneck_dropout",
            )
        )

        if not isinstance(
            bilinear_decoder,
            bool,
        ):
            raise TypeError(
                "bilinear_decoder must be Boolean."
            )

        self.bilinear_decoder = (
            bilinear_decoder
        )

        channels_1 = self.base_channels
        channels_2 = self.base_channels * 2
        channels_3 = self.base_channels * 4
        channels_4 = self.base_channels * 8
        channels_5 = self.base_channels * 16

        self.encoder_channels = (
            channels_1,
            channels_2,
            channels_3,
            channels_4,
            channels_5,
        )

        self.stem = DoubleConvolution(
            input_channels=(
                self.input_channels
            ),
            output_channels=channels_1,
        )

        self.encoder1 = (
            ResidualEncoderStage(
                input_channels=channels_1,
                output_channels=channels_2,
                number_of_blocks=(
                    self.encoder_blocks[0]
                ),
            )
        )

        self.encoder2 = (
            ResidualEncoderStage(
                input_channels=channels_2,
                output_channels=channels_3,
                number_of_blocks=(
                    self.encoder_blocks[1]
                ),
            )
        )

        self.encoder3 = (
            ResidualEncoderStage(
                input_channels=channels_3,
                output_channels=channels_4,
                number_of_blocks=(
                    self.encoder_blocks[2]
                ),
            )
        )

        self.encoder4 = (
            ResidualEncoderStage(
                input_channels=channels_4,
                output_channels=channels_5,
                number_of_blocks=(
                    self.encoder_blocks[3]
                ),
            )
        )

        self.bottleneck_dropout = (
            nn.Dropout2d(
                p=(
                    self.bottleneck_dropout_probability
                )
            )
            if (
                self.bottleneck_dropout_probability
                > 0.0
            )
            else nn.Identity()
        )

        self.transformer_bottleneck = (
            TransformerBottleneck(
                input_channels=channels_5,
                embedding_dimension=(
                    self.transformer_dimension
                ),
                number_of_layers=(
                    self.transformer_layers
                ),
                number_of_heads=(
                    self.transformer_heads
                ),
                mlp_dimension=(
                    self.transformer_mlp_dimension
                ),
                dropout_probability=(
                    self.transformer_dropout
                ),
                attention_dropout_probability=(
                    self.attention_dropout
                ),
            )
        )

        self.decoder4 = UpsamplingBlock(
            decoder_channels=channels_5,
            skip_channels=channels_4,
            output_channels=channels_4,
            bilinear=(
                self.bilinear_decoder
            ),
        )

        self.decoder3 = UpsamplingBlock(
            decoder_channels=channels_4,
            skip_channels=channels_3,
            output_channels=channels_3,
            bilinear=(
                self.bilinear_decoder
            ),
        )

        self.decoder2 = UpsamplingBlock(
            decoder_channels=channels_3,
            skip_channels=channels_2,
            output_channels=channels_2,
            bilinear=(
                self.bilinear_decoder
            ),
        )

        self.decoder1 = UpsamplingBlock(
            decoder_channels=channels_2,
            skip_channels=channels_1,
            output_channels=channels_1,
            bilinear=(
                self.bilinear_decoder
            ),
        )

        self.segmentation_head = nn.Conv2d(
            in_channels=channels_1,
            out_channels=(
                self.output_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
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

    def _validate_input(
        self,
        image: Tensor,
    ) -> None:
        """Validate one input image tensor."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "TransUNet input must be a "
                "torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "TransUNet input must have shape "
                "[B, C, H, W], received "
                f"{tuple(image.shape)}."
            )

        if image.shape[0] <= 0:
            raise ValueError(
                "TransUNet input batch cannot "
                "be empty."
            )

        if (
            image.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "TransUNet input channel mismatch: "
                f"expected {self.input_channels}, "
                f"received {image.shape[1]}."
            )

        if (
            image.shape[-2] < 16
            or image.shape[-1] < 16
        ):
            raise ValueError(
                "TransUNet input height and width "
                "must each be at least 16 pixels."
            )

        if not torch.isfinite(
            image
        ).all():
            raise ValueError(
                "TransUNet input contains "
                "non-finite values."
            )

    def forward_features(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return CNN, Transformer, and decoder features."""

        self._validate_input(
            image
        )

        encoder0 = self.stem(
            image
        )

        encoder1 = self.encoder1(
            encoder0
        )

        encoder2 = self.encoder2(
            encoder1
        )

        encoder3 = self.encoder3(
            encoder2
        )

        encoder4 = self.encoder4(
            encoder3
        )

        encoder4 = self.bottleneck_dropout(
            encoder4
        )

        (
            transformer_feature,
            transformer_tokens,
        ) = self.transformer_bottleneck(
            encoder4
        )

        decoder3 = self.decoder4(
            transformer_feature,
            encoder3,
        )

        decoder2 = self.decoder3(
            decoder3,
            encoder2,
        )

        decoder1 = self.decoder2(
            decoder2,
            encoder1,
        )

        decoder0 = self.decoder1(
            decoder1,
            encoder0,
        )

        features = {
            "encoder0": encoder0,
            "encoder1": encoder1,
            "encoder2": encoder2,
            "encoder3": encoder3,
            "encoder4": encoder4,
            "transformer_feature": (
                transformer_feature
            ),
            "transformer_tokens": (
                transformer_tokens
            ),
            "decoder3": decoder3,
            "decoder2": decoder2,
            "decoder1": decoder1,
            "decoder0": decoder0,
        }

        for name, feature in features.items():
            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"TransUNet feature {name!r} "
                    "contains non-finite values."
                )

        return features

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Predict one-channel lesion-mask logits."""

        input_size = tuple(
            int(value)
            for value
            in image.shape[-2:]
        )

        features = self.forward_features(
            image
        )

        mask_logits = self.segmentation_head(
            features[
                "decoder0"
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
                "TransUNet mask logits contain "
                "non-finite values."
            )

        return {
            "mask_logits": (
                mask_logits.contiguous()
            )
        }

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
                TRANSUNET_PROTOCOL_VERSION
            ),
            "architecture": (
                "transunet_style"
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
            "encoder_channels": list(
                self.encoder_channels
            ),
            "encoder_blocks": list(
                self.encoder_blocks
            ),
            "transformer_dimension": (
                self.transformer_dimension
            ),
            "transformer_layers": (
                self.transformer_layers
            ),
            "transformer_heads": (
                self.transformer_heads
            ),
            "transformer_mlp_dimension": (
                self.transformer_mlp_dimension
            ),
            "transformer_dropout": (
                self.transformer_dropout
            ),
            "attention_dropout": (
                self.attention_dropout
            ),
            "bottleneck_dropout": (
                self.bottleneck_dropout_probability
            ),
            "bilinear_decoder": (
                self.bilinear_decoder
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
            "uses_transformer": True,
            "uses_cnn_encoder": True,
            "uses_boundary_conditioning": (
                False
            ),
            "uses_auxiliary_targets": False,
        }


TransUNetBaseline = TransUNet


def run_transunet_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward test."""

    torch.manual_seed(
        42
    )

    model = TransUNet(
        input_channels=3,
        output_channels=1,
        base_channels=8,
        encoder_blocks=(
            1,
            1,
            1,
            1,
        ),
        transformer_dimension=64,
        transformer_layers=2,
        transformer_heads=4,
        transformer_mlp_dimension=128,
        transformer_dropout=0.0,
        attention_dropout=0.0,
        bottleneck_dropout=0.0,
        bilinear_decoder=True,
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

    encoder_gradient = (
        model
        .stem
        .block[0]
        .weight
        .grad
    )

    transformer_gradient = (
        model
        .transformer_bottleneck
        .input_projection
        .weight
        .grad
    )

    decoder_gradient = (
        model
        .segmentation_head
        .weight
        .grad
    )

    with torch.no_grad():
        features = model.forward_features(
            image.detach()
        )

    transformer_tokens = features[
        "transformer_tokens"
    ]

    transformer_feature = features[
        "transformer_feature"
    ]

    positional_encoding = (
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
        "encoder_gradient_exists": (
            encoder_gradient
            is not None
        ),
        "encoder_gradient_nonzero": (
            encoder_gradient is not None
            and float(
                encoder_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "transformer_gradient_exists": (
            transformer_gradient
            is not None
        ),
        "transformer_gradient_nonzero": (
            transformer_gradient
            is not None
            and float(
                transformer_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "decoder_gradient_exists": (
            decoder_gradient
            is not None
        ),
        "decoder_gradient_nonzero": (
            decoder_gradient
            is not None
            and float(
                decoder_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "transformer_tokens_rank": (
            transformer_tokens.ndim
            == 3
        ),
        "transformer_token_batch": (
            transformer_tokens.shape[0]
            == 2
        ),
        "transformer_token_dimension": (
            transformer_tokens.shape[-1]
            == 64
        ),
        "transformer_feature_channels": (
            transformer_feature.shape[1]
            == 128
        ),
        "position_encoding_shape": (
            tuple(
                positional_encoding.shape
            )
            == (
                1,
                20,
                64,
            )
        ),
        "position_encoding_finite": (
            torch.isfinite(
                positional_encoding
            ).all().item()
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
        "transformer_enabled": (
            architecture[
                "uses_transformer"
            ]
            is True
        ),
        "cnn_encoder_enabled": (
            architecture[
                "uses_cnn_encoder"
            ]
            is True
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
            TRANSUNET_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_output_shape": list(
            mask_logits.shape
        ),
        "transformer_token_shape": list(
            transformer_tokens.shape
        ),
        "transformer_feature_shape": list(
            transformer_feature.shape
        ),
        "parameter_count": (
            parameter_count
        ),
        "architecture": (
            architecture
        ),
    }