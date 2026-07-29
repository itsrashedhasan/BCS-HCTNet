"""PVTv2-B1 transformer encoder for BCS-HCTNet.

This module implements the hierarchical transformer branch used to extract
global-context features at four spatial scales.

Architecture
------------
The encoder follows the official PVTv2-B1 configuration:

- embedding dimensions: 64, 128, 320, 512;
- block depths: 2, 2, 2, 2;
- attention heads: 1, 2, 5, 8;
- MLP ratios: 8, 8, 4, 4;
- spatial-reduction ratios: 8, 4, 2, 1;
- overlapping patch embeddings;
- convolutional feed-forward networks.

For a 352 x 352 input, returned feature maps have these shapes:

- stage1: [B, 64, 88, 88]
- stage2: [B, 128, 44, 44]
- stage3: [B, 320, 22, 22]
- stage4: [B, 512, 11, 11]

No network download is performed. Official or compatible pretrained weights
may be supplied through a local checkpoint path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


TRANSFORMER_ENCODER_PROTOCOL_VERSION = (
    "BCS-HCTNet-pvtv2-b1-transformer-encoder-v1"
)

PVT_V2_B1_EMBED_DIMS = (
    64,
    128,
    320,
    512,
)

PVT_V2_B1_DEPTHS = (
    2,
    2,
    2,
    2,
)

PVT_V2_B1_NUM_HEADS = (
    1,
    2,
    5,
    8,
)

PVT_V2_B1_MLP_RATIOS = (
    8.0,
    8.0,
    4.0,
    4.0,
)

PVT_V2_B1_SR_RATIOS = (
    8,
    4,
    2,
    1,
)

PVT_V2_B1_PATCH_SIZES = (
    7,
    3,
    3,
    3,
)

PVT_V2_B1_STRIDES = (
    4,
    2,
    2,
    2,
)


@dataclass(frozen=True)
class TransformerFeatureSpec:
    """Description of one transformer feature level."""

    name: str
    channels: int
    stride: int
    depth: int
    attention_heads: int
    spatial_reduction_ratio: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "channels": self.channels,
            "stride": self.stride,
            "depth": self.depth,
            "attention_heads": (
                self.attention_heads
            ),
            "spatial_reduction_ratio": (
                self.spatial_reduction_ratio
            ),
        }


TRANSFORMER_FEATURE_SPECS = (
    TransformerFeatureSpec(
        name="stage1",
        channels=64,
        stride=4,
        depth=2,
        attention_heads=1,
        spatial_reduction_ratio=8,
    ),
    TransformerFeatureSpec(
        name="stage2",
        channels=128,
        stride=8,
        depth=2,
        attention_heads=2,
        spatial_reduction_ratio=4,
    ),
    TransformerFeatureSpec(
        name="stage3",
        channels=320,
        stride=16,
        depth=2,
        attention_heads=5,
        spatial_reduction_ratio=2,
    ),
    TransformerFeatureSpec(
        name="stage4",
        channels=512,
        stride=32,
        depth=2,
        attention_heads=8,
        spatial_reduction_ratio=1,
    ),
)


def drop_path(
    tensor: Tensor,
    drop_probability: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Apply stochastic depth independently to each sample."""

    if not (
        0.0
        <= float(drop_probability)
        < 1.0
    ):
        raise ValueError(
            "drop_probability must be in "
            f"[0, 1), received {drop_probability}."
        )

    if (
        drop_probability == 0.0
        or not training
    ):
        return tensor

    keep_probability = (
        1.0
        - float(
            drop_probability
        )
    )

    random_shape = (
        tensor.shape[0],
        *(
            1
            for _ in range(
                tensor.ndim - 1
            )
        ),
    )

    random_tensor = (
        keep_probability
        + torch.rand(
            random_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    )

    random_tensor.floor_()

    return (
        tensor
        / keep_probability
        * random_tensor
    )


class DropPath(nn.Module):
    """Module wrapper for sample-wise stochastic depth."""

    def __init__(
        self,
        drop_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if not (
            0.0
            <= float(drop_probability)
            < 1.0
        ):
            raise ValueError(
                "drop_probability must be in "
                "[0, 1)."
            )

        self.drop_probability = float(
            drop_probability
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """Apply stochastic depth."""

        return drop_path(
            tensor,
            drop_probability=(
                self.drop_probability
            ),
            training=self.training,
        )


class DepthwiseConvolution(nn.Module):
    """Depthwise convolution used inside the PVTv2 MLP."""

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )

    def forward(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Apply depthwise convolution to token features."""

        if tokens.ndim != 3:
            raise ValueError(
                "Depthwise-convolution input must "
                "have shape [B, N, C]."
            )

        batch_size, token_count, channels = (
            tokens.shape
        )

        if token_count != height * width:
            raise RuntimeError(
                "Token count does not match "
                f"height x width: {token_count} "
                f"versus {height} x {width}."
            )

        feature_map = (
            tokens.transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        feature_map = self.dwconv(
            feature_map
        )

        return (
            feature_map.flatten(2)
            .transpose(
                1,
                2,
            )
            .contiguous()
        )


class PVTMLP(nn.Module):
    """Convolutional feed-forward network used by PVTv2."""

    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        output_features: int | None = None,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError(
                "input_features must be positive."
            )

        if hidden_features <= 0:
            raise ValueError(
                "hidden_features must be positive."
            )

        if not (
            0.0
            <= dropout_probability
            < 1.0
        ):
            raise ValueError(
                "dropout_probability must be "
                "in [0, 1)."
            )

        resolved_output_features = (
            input_features
            if output_features is None
            else output_features
        )

        if resolved_output_features <= 0:
            raise ValueError(
                "output_features must be positive."
            )

        self.fc1 = nn.Linear(
            input_features,
            hidden_features,
        )

        self.dwconv = DepthwiseConvolution(
            hidden_features
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_features,
            resolved_output_features,
        )

        self.drop = nn.Dropout(
            dropout_probability
        )

    def forward(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Apply the convolutional MLP."""

        tokens = self.fc1(
            tokens
        )

        tokens = self.dwconv(
            tokens,
            height=height,
            width=width,
        )

        tokens = self.act(
            tokens
        )

        tokens = self.drop(
            tokens
        )

        tokens = self.fc2(
            tokens
        )

        return self.drop(
            tokens
        )


class SpatialReductionAttention(nn.Module):
    """Multi-head attention with spatially reduced keys and values."""

    def __init__(
        self,
        *,
        dimension: int,
        number_of_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        spatial_reduction_ratio: int = 1,
    ) -> None:
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be positive."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be positive."
            )

        if (
            dimension
            % number_of_heads
            != 0
        ):
            raise ValueError(
                f"dimension {dimension} must be "
                "divisible by number_of_heads "
                f"{number_of_heads}."
            )

        if spatial_reduction_ratio <= 0:
            raise ValueError(
                "spatial_reduction_ratio must "
                "be positive."
            )

        for name, value in {
            "attention_dropout": (
                attention_dropout
            ),
            "projection_dropout": (
                projection_dropout
            ),
        }.items():
            if not (
                0.0
                <= float(value)
                < 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0, 1)."
                )

        self.dimension = dimension

        self.num_heads = number_of_heads

        head_dimension = (
            dimension
            // number_of_heads
        )

        self.scale = (
            head_dimension ** -0.5
        )

        self.q = nn.Linear(
            dimension,
            dimension,
            bias=qkv_bias,
        )

        self.kv = nn.Linear(
            dimension,
            dimension * 2,
            bias=qkv_bias,
        )

        self.attn_drop = nn.Dropout(
            attention_dropout
        )

        self.proj = nn.Linear(
            dimension,
            dimension,
        )

        self.proj_drop = nn.Dropout(
            projection_dropout
        )

        self.sr_ratio = int(
            spatial_reduction_ratio
        )

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=dimension,
                out_channels=dimension,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
            )

            self.norm = nn.LayerNorm(
                dimension,
                eps=1e-6,
            )

    def forward(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Apply spatial-reduction attention."""

        if tokens.ndim != 3:
            raise ValueError(
                "Attention input must have shape "
                "[B, N, C]."
            )

        batch_size, token_count, channels = (
            tokens.shape
        )

        if channels != self.dimension:
            raise RuntimeError(
                "Attention channel mismatch: "
                f"expected {self.dimension}, "
                f"found {channels}."
            )

        if token_count != height * width:
            raise RuntimeError(
                "Attention token count does not "
                "match height x width."
            )

        query = (
            self.q(
                tokens
            )
            .reshape(
                batch_size,
                token_count,
                self.num_heads,
                channels
                // self.num_heads,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
        )

        if self.sr_ratio > 1:
            if (
                height < self.sr_ratio
                or width < self.sr_ratio
            ):
                raise RuntimeError(
                    "Feature map is smaller than "
                    "the configured spatial-"
                    "reduction ratio."
                )

            reduced = (
                tokens.permute(
                    0,
                    2,
                    1,
                )
                .reshape(
                    batch_size,
                    channels,
                    height,
                    width,
                )
            )

            reduced = self.sr(
                reduced
            )

            reduced = (
                reduced.reshape(
                    batch_size,
                    channels,
                    -1,
                )
                .permute(
                    0,
                    2,
                    1,
                )
            )

            reduced = self.norm(
                reduced
            )

        else:
            reduced = tokens

        key_value = (
            self.kv(
                reduced
            )
            .reshape(
                batch_size,
                -1,
                2,
                self.num_heads,
                channels
                // self.num_heads,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        key = key_value[0]
        value = key_value[1]

        attention = (
            query
            @ key.transpose(
                -2,
                -1,
            )
        ) * self.scale

        attention = attention.softmax(
            dim=-1
        )

        attention = self.attn_drop(
            attention
        )

        output = (
            attention
            @ value
        )

        output = (
            output.transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                token_count,
                channels,
            )
        )

        output = self.proj(
            output
        )

        return self.proj_drop(
            output
        )


class PVTBlock(nn.Module):
    """One PVTv2 transformer block."""

    def __init__(
        self,
        *,
        dimension: int,
        number_of_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        dropout_probability: float,
        attention_dropout_probability: float,
        drop_path_probability: float,
        spatial_reduction_ratio: int,
    ) -> None:
        super().__init__()

        if mlp_ratio <= 0:
            raise ValueError(
                "mlp_ratio must be positive."
            )

        self.norm1 = nn.LayerNorm(
            dimension,
            eps=1e-6,
        )

        self.attn = SpatialReductionAttention(
            dimension=dimension,
            number_of_heads=(
                number_of_heads
            ),
            qkv_bias=qkv_bias,
            attention_dropout=(
                attention_dropout_probability
            ),
            projection_dropout=(
                dropout_probability
            ),
            spatial_reduction_ratio=(
                spatial_reduction_ratio
            ),
        )

        self.drop_path = (
            DropPath(
                drop_path_probability
            )
            if drop_path_probability > 0.0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(
            dimension,
            eps=1e-6,
        )

        hidden_dimension = int(
            dimension
            * mlp_ratio
        )

        self.mlp = PVTMLP(
            input_features=dimension,
            hidden_features=(
                hidden_dimension
            ),
            output_features=dimension,
            dropout_probability=(
                dropout_probability
            ),
        )

    def forward(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Apply attention and convolutional MLP."""

        tokens = (
            tokens
            + self.drop_path(
                self.attn(
                    self.norm1(
                        tokens
                    ),
                    height=height,
                    width=width,
                )
            )
        )

        tokens = (
            tokens
            + self.drop_path(
                self.mlp(
                    self.norm2(
                        tokens
                    ),
                    height=height,
                    width=width,
                )
            )
        )

        return tokens


class OverlapPatchEmbedding(nn.Module):
    """Overlapping image-to-token patch embedding."""

    def __init__(
        self,
        *,
        patch_size: int,
        stride: int,
        input_channels: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()

        if patch_size <= stride:
            raise ValueError(
                "patch_size must be greater "
                "than stride."
            )

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must "
                "be positive."
            )

        self.patch_size = int(
            patch_size
        )

        self.stride = int(
            stride
        )

        self.proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=(
                embedding_dimension
            ),
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2,
        )

        self.norm = nn.LayerNorm(
            embedding_dimension,
            eps=1e-6,
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> tuple[
        Tensor,
        int,
        int,
    ]:
        """Convert one feature map to normalized tokens."""

        if feature_map.ndim != 4:
            raise ValueError(
                "Patch-embedding input must have "
                "shape [B, C, H, W]."
            )

        feature_map = self.proj(
            feature_map
        )

        height = int(
            feature_map.shape[-2]
        )

        width = int(
            feature_map.shape[-1]
        )

        tokens = (
            feature_map.flatten(2)
            .transpose(
                1,
                2,
            )
        )

        tokens = self.norm(
            tokens
        )

        return (
            tokens,
            height,
            width,
        )


def initialize_pvt_module(
    module: nn.Module,
) -> None:
    """Initialize a PVTv2 module."""

    if isinstance(
        module,
        nn.Linear,
    ):
        nn.init.trunc_normal_(
            module.weight,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(
                module.bias
            )

    elif isinstance(
        module,
        nn.LayerNorm,
    ):
        nn.init.ones_(
            module.weight
        )

        nn.init.zeros_(
            module.bias
        )

    elif isinstance(
        module,
        nn.Conv2d,
    ):
        kernel_height = int(
            module.kernel_size[0]
        )

        kernel_width = int(
            module.kernel_size[1]
        )

        fan_out = (
            kernel_height
            * kernel_width
            * module.out_channels
        )

        fan_out //= module.groups

        standard_deviation = (
            2.0
            / float(
                fan_out
            )
        ) ** 0.5

        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=standard_deviation,
        )

        if module.bias is not None:
            nn.init.zeros_(
                module.bias
            )


def extract_checkpoint_state_dict(
    checkpoint: object,
) -> dict[str, Tensor]:
    """Extract and normalize a PVTv2 state dictionary."""

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "PVTv2 checkpoint must contain "
            "a mapping."
        )

    candidate: object = checkpoint

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
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
            "Could not locate a state dictionary "
            "inside the PVTv2 checkpoint."
        )

    state_dict: dict[
        str,
        Tensor,
    ] = {}

    prefixes = (
        "module.",
        "backbone.",
        "transformer_encoder.",
        "encoder.",
    )

    for key, value in candidate.items():
        if (
            not isinstance(
                key,
                str,
            )
            or not isinstance(
                value,
                Tensor,
            )
        ):
            continue

        normalized_key = key

        prefix_removed = True

        while prefix_removed:
            prefix_removed = False

            for prefix in prefixes:
                if normalized_key.startswith(
                    prefix
                ):
                    normalized_key = (
                        normalized_key[
                            len(prefix):
                        ]
                    )

                    prefix_removed = True
                    break

        state_dict[
            normalized_key
        ] = value

    if not state_dict:
        raise RuntimeError(
            "PVTv2 checkpoint contains no "
            "tensor parameters."
        )

    return state_dict


def load_checkpoint(
    checkpoint_path: Path,
) -> object:
    """Load a local checkpoint across PyTorch versions."""

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


class PVTv2B1TransformerEncoder(nn.Module):
    """Hierarchical PVTv2-B1 feature encoder."""

    feature_specs = TRANSFORMER_FEATURE_SPECS

    def __init__(
        self,
        *,
        pretrained: bool = False,
        weights_path: str | Path | None = None,
        dropout_probability: float = 0.0,
        attention_dropout_probability: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        """Initialize the PVTv2-B1 encoder.

        ``pretrained=True`` requires a local ``weights_path``. Automatic
        downloads are intentionally disabled for reproducible Kaggle runs.
        """

        super().__init__()

        if not isinstance(
            pretrained,
            bool,
        ):
            raise TypeError(
                "pretrained must be Boolean."
            )

        for name, value in {
            "dropout_probability": (
                dropout_probability
            ),
            "attention_dropout_probability": (
                attention_dropout_probability
            ),
            "drop_path_rate": (
                drop_path_rate
            ),
        }.items():
            if not (
                0.0
                <= float(value)
                < 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0, 1)."
                )

        resolved_weights_path = (
            Path(
                weights_path
            )
            .expanduser()
            .resolve()
            if weights_path is not None
            else None
        )

        if (
            resolved_weights_path is not None
            and not resolved_weights_path.is_file()
        ):
            raise FileNotFoundError(
                "PVTv2-B1 checkpoint not found: "
                f"{resolved_weights_path}"
            )

        if (
            pretrained
            and resolved_weights_path is None
        ):
            raise RuntimeError(
                "pretrained=True requires a local "
                "PVTv2-B1 weights_path. Automatic "
                "downloads are disabled."
            )

        self.patch_embed1 = (
            OverlapPatchEmbedding(
                patch_size=7,
                stride=4,
                input_channels=3,
                embedding_dimension=64,
            )
        )

        self.patch_embed2 = (
            OverlapPatchEmbedding(
                patch_size=3,
                stride=2,
                input_channels=64,
                embedding_dimension=128,
            )
        )

        self.patch_embed3 = (
            OverlapPatchEmbedding(
                patch_size=3,
                stride=2,
                input_channels=128,
                embedding_dimension=320,
            )
        )

        self.patch_embed4 = (
            OverlapPatchEmbedding(
                patch_size=3,
                stride=2,
                input_channels=320,
                embedding_dimension=512,
            )
        )

        total_blocks = sum(
            PVT_V2_B1_DEPTHS
        )

        drop_path_values = (
            torch.linspace(
                0.0,
                float(
                    drop_path_rate
                ),
                total_blocks,
            )
            .tolist()
        )

        current_block = 0

        stage_blocks: list[
            nn.ModuleList
        ] = []

        for stage_index in range(4):
            blocks = nn.ModuleList(
                [
                    PVTBlock(
                        dimension=(
                            PVT_V2_B1_EMBED_DIMS[
                                stage_index
                            ]
                        ),
                        number_of_heads=(
                            PVT_V2_B1_NUM_HEADS[
                                stage_index
                            ]
                        ),
                        mlp_ratio=(
                            PVT_V2_B1_MLP_RATIOS[
                                stage_index
                            ]
                        ),
                        qkv_bias=True,
                        dropout_probability=(
                            dropout_probability
                        ),
                        attention_dropout_probability=(
                            attention_dropout_probability
                        ),
                        drop_path_probability=float(
                            drop_path_values[
                                current_block
                                + block_index
                            ]
                        ),
                        spatial_reduction_ratio=(
                            PVT_V2_B1_SR_RATIOS[
                                stage_index
                            ]
                        ),
                    )
                    for block_index in range(
                        PVT_V2_B1_DEPTHS[
                            stage_index
                        ]
                    )
                ]
            )

            stage_blocks.append(
                blocks
            )

            current_block += (
                PVT_V2_B1_DEPTHS[
                    stage_index
                ]
            )

        self.block1 = stage_blocks[0]
        self.block2 = stage_blocks[1]
        self.block3 = stage_blocks[2]
        self.block4 = stage_blocks[3]

        self.norm1 = nn.LayerNorm(
            64,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(
            128,
            eps=1e-6,
        )

        self.norm3 = nn.LayerNorm(
            320,
            eps=1e-6,
        )

        self.norm4 = nn.LayerNorm(
            512,
            eps=1e-6,
        )

        self.apply(
            initialize_pvt_module
        )

        if resolved_weights_path is not None:
            checkpoint = load_checkpoint(
                resolved_weights_path
            )

            state_dict = (
                extract_checkpoint_state_dict(
                    checkpoint
                )
            )

            incompatible = self.load_state_dict(
                state_dict,
                strict=False,
            )

            missing_keys = set(
                incompatible.missing_keys
            )

            unexpected_keys = set(
                incompatible.unexpected_keys
            )

            allowed_unexpected = {
                "head.weight",
                "head.bias",
            }

            unexpected_keys -= (
                allowed_unexpected
            )

            if (
                missing_keys
                or unexpected_keys
            ):
                raise RuntimeError(
                    "PVTv2-B1 checkpoint is "
                    "incompatible. Missing keys: "
                    f"{sorted(missing_keys)}; "
                    "unexpected keys: "
                    f"{sorted(unexpected_keys)}."
                )

        self.pretrained_requested = (
            pretrained
        )

        self.weights_source = (
            str(
                resolved_weights_path
            )
            if resolved_weights_path
            is not None
            else "random_initialization"
        )

        self.drop_path_rate = float(
            drop_path_rate
        )

    @property
    def output_channels(
        self,
    ) -> dict[str, int]:
        """Return channels for every feature level."""

        return {
            specification.name: (
                specification.channels
            )
            for specification
            in self.feature_specs
        }

    @property
    def output_strides(
        self,
    ) -> dict[str, int]:
        """Return spatial strides for every feature level."""

        return {
            specification.name: (
                specification.stride
            )
            for specification
            in self.feature_specs
        }

    @staticmethod
    def _run_stage(
        *,
        feature_map: Tensor,
        patch_embedding: OverlapPatchEmbedding,
        blocks: nn.ModuleList,
        normalization: nn.LayerNorm,
    ) -> Tensor:
        """Run one hierarchical PVTv2 stage."""

        (
            tokens,
            height,
            width,
        ) = patch_embedding(
            feature_map
        )

        for block in blocks:
            tokens = block(
                tokens,
                height=height,
                width=width,
            )

        tokens = normalization(
            tokens
        )

        batch_size = int(
            tokens.shape[0]
        )

        channels = int(
            tokens.shape[-1]
        )

        return (
            tokens.reshape(
                batch_size,
                height,
                width,
                channels,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Extract four hierarchical transformer feature maps."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "Transformer encoder input must "
                "be a torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "Transformer encoder input must "
                "have shape [B, C, H, W], "
                f"received {tuple(image.shape)}."
            )

        if image.shape[1] != 3:
            raise ValueError(
                "Transformer encoder requires "
                "three RGB channels."
            )

        if (
            image.shape[-2] < 32
            or image.shape[-1] < 32
        ):
            raise ValueError(
                "Transformer encoder input height "
                "and width must each be at least "
                "32 pixels."
            )

        stage1 = self._run_stage(
            feature_map=image,
            patch_embedding=(
                self.patch_embed1
            ),
            blocks=self.block1,
            normalization=self.norm1,
        )

        stage2 = self._run_stage(
            feature_map=stage1,
            patch_embedding=(
                self.patch_embed2
            ),
            blocks=self.block2,
            normalization=self.norm2,
        )

        stage3 = self._run_stage(
            feature_map=stage2,
            patch_embedding=(
                self.patch_embed3
            ),
            blocks=self.block3,
            normalization=self.norm3,
        )

        stage4 = self._run_stage(
            feature_map=stage3,
            patch_embedding=(
                self.patch_embed4
            ),
            blocks=self.block4,
            normalization=self.norm4,
        )

        features = {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }

        self._validate_features(
            image=image,
            features=features,
        )

        return features

    def _validate_features(
        self,
        *,
        image: Tensor,
        features: Mapping[str, Tensor],
    ) -> None:
        """Validate feature-map invariants."""

        expected_names = [
            specification.name
            for specification
            in self.feature_specs
        ]

        if list(
            features
        ) != expected_names:
            raise RuntimeError(
                "Transformer feature names or "
                "order are invalid."
            )

        batch_size = int(
            image.shape[0]
        )

        previous_height = int(
            image.shape[-2]
        )

        previous_width = int(
            image.shape[-1]
        )

        for specification in (
            self.feature_specs
        ):
            feature = features[
                specification.name
            ]

            if feature.ndim != 4:
                raise RuntimeError(
                    f"{specification.name} must "
                    "be four-dimensional."
                )

            if (
                feature.shape[0]
                != batch_size
            ):
                raise RuntimeError(
                    f"{specification.name} batch "
                    "size differs from input."
                )

            if (
                feature.shape[1]
                != specification.channels
            ):
                raise RuntimeError(
                    f"{specification.name} expected "
                    f"{specification.channels} "
                    f"channels, found "
                    f"{feature.shape[1]}."
                )

            if (
                feature.shape[-2]
                > previous_height
                or feature.shape[-1]
                > previous_width
            ):
                raise RuntimeError(
                    f"{specification.name} increased "
                    "spatial resolution."
                )

            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"{specification.name} contains "
                    "non-finite values."
                )

            previous_height = int(
                feature.shape[-2]
            )

            previous_width = int(
                feature.shape[-1]
            )

    def parameter_count(
        self,
        trainable_only: bool = False,
    ) -> int:
        """Return transformer encoder parameter count."""

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
        """Return architecture metadata."""

        return {
            "protocol_version": (
                TRANSFORMER_ENCODER_PROTOCOL_VERSION
            ),
            "architecture": "pvt_v2_b1",
            "weights_source": (
                self.weights_source
            ),
            "parameter_count": (
                self.parameter_count()
            ),
            "trainable_parameter_count": (
                self.parameter_count(
                    trainable_only=True
                )
            ),
            "drop_path_rate": (
                self.drop_path_rate
            ),
            "embed_dimensions": list(
                PVT_V2_B1_EMBED_DIMS
            ),
            "depths": list(
                PVT_V2_B1_DEPTHS
            ),
            "attention_heads": list(
                PVT_V2_B1_NUM_HEADS
            ),
            "mlp_ratios": list(
                PVT_V2_B1_MLP_RATIOS
            ),
            "spatial_reduction_ratios": list(
                PVT_V2_B1_SR_RATIOS
            ),
            "features": [
                specification.to_dict()
                for specification
                in self.feature_specs
            ],
        }


def run_transformer_encoder_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward self-test."""

    torch.manual_seed(
        42
    )

    encoder = (
        PVTv2B1TransformerEncoder(
            pretrained=False,
            dropout_probability=0.0,
            attention_dropout_probability=0.0,
            drop_path_rate=0.0,
        )
    )

    encoder.eval()

    image = torch.randn(
        1,
        3,
        64,
        64,
        dtype=torch.float32,
        requires_grad=True,
    )

    features = encoder(
        image
    )

    expected_shapes = {
        "stage1": (
            1,
            64,
            16,
            16,
        ),
        "stage2": (
            1,
            128,
            8,
            8,
        ),
        "stage3": (
            1,
            320,
            4,
            4,
        ),
        "stage4": (
            1,
            512,
            2,
            2,
        ),
    }

    loss = sum(
        feature.square().mean()
        for feature in features.values()
    )

    loss.backward()

    first_projection_gradient = (
        encoder
        .patch_embed1
        .proj
        .weight
        .grad
    )

    parameter_count = (
        encoder.parameter_count()
    )

    checks = {
        "feature_names": (
            list(features)
            == [
                "stage1",
                "stage2",
                "stage3",
                "stage4",
            ]
        ),
        "feature_shapes": all(
            tuple(
                features[name].shape
            )
            == expected_shape
            for name, expected_shape
            in expected_shapes.items()
        ),
        "all_features_finite": all(
            bool(
                torch.isfinite(
                    feature
                ).all().item()
            )
            for feature
            in features.values()
        ),
        "loss_finite": bool(
            torch.isfinite(
                loss
            ).item()
        ),
        "input_gradient_exists": (
            image.grad is not None
        ),
        "input_gradient_finite": (
            image.grad is not None
            and bool(
                torch.isfinite(
                    image.grad
                ).all().item()
            )
        ),
        "encoder_gradient_exists": (
            first_projection_gradient
            is not None
        ),
        "encoder_gradient_nonzero": (
            first_projection_gradient
            is not None
            and float(
                first_projection_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "output_channels_correct": (
            encoder.output_channels
            == {
                "stage1": 64,
                "stage2": 128,
                "stage3": 320,
                "stage4": 512,
            }
        ),
        "output_strides_correct": (
            encoder.output_strides
            == {
                "stage1": 4,
                "stage2": 8,
                "stage3": 16,
                "stage4": 32,
            }
        ),
        "parameter_count_plausible": (
            13_000_000
            <= parameter_count
            <= 14_500_000
        ),
        "offline_weights_used": (
            encoder.weights_source
            == "random_initialization"
        ),
        "official_depths": (
            PVT_V2_B1_DEPTHS
            == (
                2,
                2,
                2,
                2,
            )
        ),
        "official_embed_dimensions": (
            PVT_V2_B1_EMBED_DIMS
            == (
                64,
                128,
                320,
                512,
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
            TRANSFORMER_ENCODER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "expected_shapes": {
            name: list(shape)
            for name, shape
            in expected_shapes.items()
        },
        "observed_shapes": {
            name: list(
                feature.shape
            )
            for name, feature
            in features.items()
        },
        "architecture": (
            encoder.architecture_summary()
        ),
    }