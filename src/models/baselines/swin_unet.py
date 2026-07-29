"""Swin-UNet baseline for binary lesion segmentation.

This module implements a hierarchical Swin Transformer encoder with a
U-Net-style multi-resolution decoder.

Architecture
------------
- convolutional patch embedding;
- hierarchical shifted-window Transformer encoder;
- four feature resolutions;
- patch-merging downsampling;
- decoder upsampling with encoder skip connections;
- one-channel binary lesion-mask logits.

The output follows the shared project interface:

    {
        "mask_logits": Tensor[B, 1, H, W]
    }

This is a conventional fully supervised baseline. It does not use:

- boundary-reliability conditioning;
- contour supervision;
- boundary-band supervision;
- signed-distance-map supervision;
- BCS-HCTNet fusion modules.

The model handles arbitrary image dimensions by padding internally and
restoring the original input resolution before returning logits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.baselines.unet import (
    DoubleConvolution,
)


SWIN_UNET_PROTOCOL_VERSION = (
    "BCS-HCTNet-swin-unet-baseline-v1"
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


def _validate_integer_sequence(
    values: object,
    *,
    context: str,
    expected_length: int,
) -> tuple[int, ...]:
    """Validate a fixed-length sequence of positive integers."""

    if (
        isinstance(
            values,
            (
                str,
                bytes,
            ),
        )
        or not isinstance(
            values,
            Sequence,
        )
    ):
        raise TypeError(
            f"{context} must be a sequence."
        )

    resolved = tuple(
        _require_positive_integer(
            value,
            f"{context} value",
        )
        for value in values
    )

    if len(
        resolved
    ) != expected_length:
        raise ValueError(
            f"{context} must contain exactly "
            f"{expected_length} values."
        )

    return resolved


class DropPath(nn.Module):
    """Stochastic-depth residual regularization."""

    def __init__(
        self,
        probability: float = 0.0,
    ) -> None:
        """Initialize stochastic depth."""

        super().__init__()

        self.probability = (
            _validate_probability(
                probability,
                "drop_path_probability",
            )
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """Drop complete residual paths during training."""

        if (
            self.probability == 0.0
            or not self.training
        ):
            return tensor

        keep_probability = (
            1.0
            - self.probability
        )

        shape = (
            tensor.shape[0],
        ) + (
            1,
        ) * (
            tensor.ndim - 1
        )

        random_tensor = (
            keep_probability
            + torch.rand(
                shape,
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


class FeedForwardNetwork(nn.Module):
    """Transformer feed-forward network."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        hidden_dimension: int,
        dropout_probability: float,
    ) -> None:
        """Initialize the feed-forward network."""

        super().__init__()

        resolved_embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        resolved_hidden_dimension = (
            _require_positive_integer(
                hidden_dimension,
                "hidden_dimension",
            )
        )

        resolved_dropout = (
            _validate_probability(
                dropout_probability,
                "dropout_probability",
            )
        )

        self.network = nn.Sequential(
            nn.Linear(
                resolved_embedding_dimension,
                resolved_hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                p=resolved_dropout
            ),
            nn.Linear(
                resolved_hidden_dimension,
                resolved_embedding_dimension,
            ),
            nn.Dropout(
                p=resolved_dropout
            ),
        )

    def forward(
        self,
        tokens: Tensor,
    ) -> Tensor:
        """Apply the feed-forward transformation."""

        return self.network(
            tokens
        )


def window_partition(
    feature_map: Tensor,
    window_size: int,
) -> Tensor:
    """Partition a padded BHWC feature map into windows.

    Parameters
    ----------
    feature_map:
        Tensor with shape ``[B, H, W, C]``.

    window_size:
        Square window width and height.

    Returns
    -------
    Tensor
        Windows with shape
        ``[B * number_of_windows, window_size**2, C]``.
    """

    if not isinstance(
        feature_map,
        Tensor,
    ):
        raise TypeError(
            "feature_map must be a torch.Tensor."
        )

    if feature_map.ndim != 4:
        raise ValueError(
            "feature_map must have shape "
            "[B, H, W, C]."
        )

    resolved_window_size = (
        _require_positive_integer(
            window_size,
            "window_size",
        )
    )

    batch_size, height, width, channels = (
        feature_map.shape
    )

    if (
        height % resolved_window_size
        != 0
        or width % resolved_window_size
        != 0
    ):
        raise ValueError(
            "Feature-map height and width must "
            "be divisible by window_size."
        )

    windows = feature_map.view(
        batch_size,
        height // resolved_window_size,
        resolved_window_size,
        width // resolved_window_size,
        resolved_window_size,
        channels,
    )

    windows = windows.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    return windows.view(
        -1,
        resolved_window_size
        * resolved_window_size,
        channels,
    )


def window_reverse(
    windows: Tensor,
    *,
    window_size: int,
    height: int,
    width: int,
    batch_size: int,
) -> Tensor:
    """Reverse window partitioning into a BHWC feature map."""

    resolved_window_size = (
        _require_positive_integer(
            window_size,
            "window_size",
        )
    )

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

    resolved_batch_size = (
        _require_positive_integer(
            batch_size,
            "batch_size",
        )
    )

    if windows.ndim != 3:
        raise ValueError(
            "windows must have shape "
            "[B*nW, window_size**2, C]."
        )

    channels = int(
        windows.shape[-1]
    )

    feature_map = windows.view(
        resolved_batch_size,
        resolved_height
        // resolved_window_size,
        resolved_width
        // resolved_window_size,
        resolved_window_size,
        resolved_window_size,
        channels,
    )

    feature_map = feature_map.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    return feature_map.view(
        resolved_batch_size,
        resolved_height,
        resolved_width,
        channels,
    )


def build_shifted_window_mask(
    *,
    padded_height: int,
    padded_width: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> Tensor | None:
    """Build the attention mask for shifted-window attention."""

    if shift_size == 0:
        return None

    resolved_height = (
        _require_positive_integer(
            padded_height,
            "padded_height",
        )
    )

    resolved_width = (
        _require_positive_integer(
            padded_width,
            "padded_width",
        )
    )

    resolved_window_size = (
        _require_positive_integer(
            window_size,
            "window_size",
        )
    )

    resolved_shift_size = (
        _require_positive_integer(
            shift_size,
            "shift_size",
        )
    )

    if (
        resolved_shift_size
        >= resolved_window_size
    ):
        raise ValueError(
            "shift_size must be smaller than "
            "window_size."
        )

    mask = torch.zeros(
        (
            1,
            resolved_height,
            resolved_width,
            1,
        ),
        device=device,
        dtype=torch.float32,
    )

    height_slices = (
        slice(
            0,
            -resolved_window_size,
        ),
        slice(
            -resolved_window_size,
            -resolved_shift_size,
        ),
        slice(
            -resolved_shift_size,
            None,
        ),
    )

    width_slices = (
        slice(
            0,
            -resolved_window_size,
        ),
        slice(
            -resolved_window_size,
            -resolved_shift_size,
        ),
        slice(
            -resolved_shift_size,
            None,
        ),
    )

    region_index = 0

    for height_slice in height_slices:
        for width_slice in width_slices:
            mask[
                :,
                height_slice,
                width_slice,
                :,
            ] = region_index

            region_index += 1

    mask_windows = window_partition(
        mask,
        resolved_window_size,
    ).squeeze(
        -1
    )

    attention_mask = (
        mask_windows.unsqueeze(
            1
        )
        - mask_windows.unsqueeze(
            2
        )
    )

    attention_mask = (
        attention_mask.masked_fill(
            attention_mask != 0,
            -100.0,
        )
        .masked_fill(
            attention_mask == 0,
            0.0,
        )
    )

    return attention_mask


class WindowAttention(nn.Module):
    """Multi-head self-attention within local windows."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        window_size: int,
        number_of_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        """Initialize window attention."""

        super().__init__()

        self.embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.window_size = (
            _require_positive_integer(
                window_size,
                "window_size",
            )
        )

        self.number_of_heads = (
            _require_positive_integer(
                number_of_heads,
                "number_of_heads",
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

        self.head_dimension = (
            self.embedding_dimension
            // self.number_of_heads
        )

        self.scale = (
            self.head_dimension
            ** -0.5
        )

        self.query_key_value = nn.Linear(
            self.embedding_dimension,
            self.embedding_dimension * 3,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(
            p=_validate_probability(
                attention_dropout,
                "attention_dropout",
            )
        )

        self.projection = nn.Linear(
            self.embedding_dimension,
            self.embedding_dimension,
        )

        self.projection_dropout = nn.Dropout(
            p=_validate_probability(
                projection_dropout,
                "projection_dropout",
            )
        )

        number_of_relative_positions = (
            (
                2 * self.window_size
                - 1
            )
            * (
                2 * self.window_size
                - 1
            )
        )

        self.relative_position_bias_table = (
            nn.Parameter(
                torch.zeros(
                    number_of_relative_positions,
                    self.number_of_heads,
                )
            )
        )

        coordinate_height = torch.arange(
            self.window_size
        )

        coordinate_width = torch.arange(
            self.window_size
        )

        coordinates = torch.stack(
            torch.meshgrid(
                coordinate_height,
                coordinate_width,
                indexing="ij",
            )
        )

        coordinates_flat = torch.flatten(
            coordinates,
            start_dim=1,
        )

        relative_coordinates = (
            coordinates_flat[
                :,
                :,
                None,
            ]
            - coordinates_flat[
                :,
                None,
                :,
            ]
        )

        relative_coordinates = (
            relative_coordinates.permute(
                1,
                2,
                0,
            ).contiguous()
        )

        relative_coordinates[
            :,
            :,
            0,
        ] += (
            self.window_size - 1
        )

        relative_coordinates[
            :,
            :,
            1,
        ] += (
            self.window_size - 1
        )

        relative_coordinates[
            :,
            :,
            0,
        ] *= (
            2 * self.window_size
            - 1
        )

        relative_position_index = (
            relative_coordinates.sum(
                dim=-1
            )
        )

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

    def forward(
        self,
        window_tokens: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Apply local-window attention."""

        if window_tokens.ndim != 3:
            raise ValueError(
                "window_tokens must have shape "
                "[B*nW, N, C]."
            )

        number_of_windows_batch = int(
            window_tokens.shape[0]
        )

        number_of_tokens = int(
            window_tokens.shape[1]
        )

        channels = int(
            window_tokens.shape[2]
        )

        expected_tokens = (
            self.window_size
            * self.window_size
        )

        if number_of_tokens != expected_tokens:
            raise ValueError(
                "Window token count does not match "
                "window_size."
            )

        if channels != self.embedding_dimension:
            raise ValueError(
                "Window-token channel dimension "
                "does not match attention module."
            )

        query_key_value = (
            self.query_key_value(
                window_tokens
            )
            .reshape(
                number_of_windows_batch,
                number_of_tokens,
                3,
                self.number_of_heads,
                self.head_dimension,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        query = query_key_value[
            0
        ]

        key = query_key_value[
            1
        ]

        value = query_key_value[
            2
        ]

        query = query * self.scale

        attention = (
            query
            @ key.transpose(
                -2,
                -1,
            )
        )

        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.view(
                    -1
                )
            ]
            .view(
                number_of_tokens,
                number_of_tokens,
                self.number_of_heads,
            )
            .permute(
                2,
                0,
                1,
            )
            .contiguous()
        )

        attention = (
            attention
            + relative_position_bias.unsqueeze(
                0
            )
        )

        if attention_mask is not None:
            number_of_windows = int(
                attention_mask.shape[0]
            )

            if (
                number_of_windows_batch
                % number_of_windows
                != 0
            ):
                raise RuntimeError(
                    "Attention-mask window count "
                    "does not divide the batch."
                )

            attention = attention.view(
                number_of_windows_batch
                // number_of_windows,
                number_of_windows,
                self.number_of_heads,
                number_of_tokens,
                number_of_tokens,
            )

            attention = (
                attention
                + attention_mask[
                    None,
                    :,
                    None,
                    :,
                    :,
                ].to(
                    dtype=attention.dtype
                )
            )

            attention = attention.view(
                -1,
                self.number_of_heads,
                number_of_tokens,
                number_of_tokens,
            )

        attention = F.softmax(
            attention,
            dim=-1,
        )

        attention = self.attention_dropout(
            attention
        )

        output = (
            attention
            @ value
        )

        output = output.transpose(
            1,
            2,
        ).reshape(
            number_of_windows_batch,
            number_of_tokens,
            channels,
        )

        output = self.projection(
            output
        )

        return self.projection_dropout(
            output
        )


class SwinTransformerBlock(nn.Module):
    """One regular or shifted-window Transformer block."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        number_of_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout_probability: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_probability: float = 0.0,
    ) -> None:
        """Initialize a Swin Transformer block."""

        super().__init__()

        self.embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.window_size = (
            _require_positive_integer(
                window_size,
                "window_size",
            )
        )

        if (
            isinstance(shift_size, bool)
            or not isinstance(shift_size, int)
            or shift_size < 0
            or shift_size >= self.window_size
        ):
            raise ValueError(
                "shift_size must be an integer in "
                "[0, window_size)."
            )

        self.shift_size = shift_size

        if isinstance(
            mlp_ratio,
            bool,
        ):
            raise TypeError(
                "mlp_ratio must be numeric."
            )

        resolved_mlp_ratio = float(
            mlp_ratio
        )

        if (
            not math.isfinite(
                resolved_mlp_ratio
            )
            or resolved_mlp_ratio <= 0.0
        ):
            raise ValueError(
                "mlp_ratio must be positive "
                "and finite."
            )

        hidden_dimension = max(
            1,
            int(
                self.embedding_dimension
                * resolved_mlp_ratio
            ),
        )

        self.normalization1 = nn.LayerNorm(
            self.embedding_dimension
        )

        self.attention = WindowAttention(
            embedding_dimension=(
                self.embedding_dimension
            ),
            window_size=(
                self.window_size
            ),
            number_of_heads=(
                number_of_heads
            ),
            attention_dropout=(
                attention_dropout
            ),
            projection_dropout=(
                dropout_probability
            ),
        )

        self.drop_path = DropPath(
            probability=(
                drop_path_probability
            )
        )

        self.normalization2 = nn.LayerNorm(
            self.embedding_dimension
        )

        self.feed_forward = (
            FeedForwardNetwork(
                embedding_dimension=(
                    self.embedding_dimension
                ),
                hidden_dimension=(
                    hidden_dimension
                ),
                dropout_probability=(
                    dropout_probability
                ),
            )
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply regular or shifted-window attention."""

        if feature_map.ndim != 4:
            raise ValueError(
                "Swin block input must have shape "
                "[B, C, H, W]."
            )

        batch_size = int(
            feature_map.shape[0]
        )

        channels = int(
            feature_map.shape[1]
        )

        height = int(
            feature_map.shape[2]
        )

        width = int(
            feature_map.shape[3]
        )

        if channels != self.embedding_dimension:
            raise ValueError(
                "Swin block channel mismatch."
            )

        tokens = feature_map.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        shortcut = tokens

        tokens = self.normalization1(
            tokens
        )

        if self.shift_size > 0:
            shifted_tokens = torch.roll(
                tokens,
                shifts=(
                    -self.shift_size,
                    -self.shift_size,
                ),
                dims=(
                    1,
                    2,
                ),
            )

        else:
            shifted_tokens = tokens

        padding_height = (
            self.window_size
            - height % self.window_size
        ) % self.window_size

        padding_width = (
            self.window_size
            - width % self.window_size
        ) % self.window_size

        shifted_tokens = F.pad(
            shifted_tokens,
            (
                0,
                0,
                0,
                padding_width,
                0,
                padding_height,
            ),
        )

        padded_height = (
            height + padding_height
        )

        padded_width = (
            width + padding_width
        )

        windows = window_partition(
            shifted_tokens,
            self.window_size,
        )

        attention_mask = (
            build_shifted_window_mask(
                padded_height=(
                    padded_height
                ),
                padded_width=(
                    padded_width
                ),
                window_size=(
                    self.window_size
                ),
                shift_size=(
                    self.shift_size
                ),
                device=(
                    feature_map.device
                ),
            )
        )

        attended_windows = self.attention(
            windows,
            attention_mask,
        )

        shifted_tokens = window_reverse(
            attended_windows,
            window_size=(
                self.window_size
            ),
            height=padded_height,
            width=padded_width,
            batch_size=batch_size,
        )

        if (
            padding_height > 0
            or padding_width > 0
        ):
            shifted_tokens = shifted_tokens[
                :,
                :height,
                :width,
                :,
            ].contiguous()

        if self.shift_size > 0:
            tokens = torch.roll(
                shifted_tokens,
                shifts=(
                    self.shift_size,
                    self.shift_size,
                ),
                dims=(
                    1,
                    2,
                ),
            )

        else:
            tokens = shifted_tokens

        tokens = (
            shortcut
            + self.drop_path(
                tokens
            )
        )

        tokens = (
            tokens
            + self.drop_path(
                self.feed_forward(
                    self.normalization2(
                        tokens
                    )
                )
            )
        )

        output = tokens.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        if not torch.isfinite(
            output
        ).all():
            raise RuntimeError(
                "Swin Transformer block produced "
                "non-finite features."
            )

        return output


class PatchEmbedding(nn.Module):
    """Convert RGB images into non-overlapping patch embeddings."""

    def __init__(
        self,
        *,
        input_channels: int,
        embedding_dimension: int,
        patch_size: int,
    ) -> None:
        """Initialize patch embedding."""

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

        self.patch_size = (
            _require_positive_integer(
                patch_size,
                "patch_size",
            )
        )

        self.projection = nn.Conv2d(
            in_channels=(
                self.input_channels
            ),
            out_channels=(
                self.embedding_dimension
            ),
            kernel_size=(
                self.patch_size
            ),
            stride=(
                self.patch_size
            ),
            padding=0,
            bias=True,
        )

        self.normalization = nn.LayerNorm(
            self.embedding_dimension
        )

    def forward(
        self,
        image: Tensor,
    ) -> tuple[
        Tensor,
        tuple[int, int],
    ]:
        """Embed padded image patches."""

        original_height = int(
            image.shape[-2]
        )

        original_width = int(
            image.shape[-1]
        )

        padding_height = (
            self.patch_size
            - original_height
            % self.patch_size
        ) % self.patch_size

        padding_width = (
            self.patch_size
            - original_width
            % self.patch_size
        ) % self.patch_size

        padded_image = F.pad(
            image,
            (
                0,
                padding_width,
                0,
                padding_height,
            ),
        )

        feature_map = self.projection(
            padded_image
        )

        feature_map = feature_map.permute(
            0,
            2,
            3,
            1,
        )

        feature_map = self.normalization(
            feature_map
        )

        feature_map = feature_map.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return (
            feature_map,
            (
                original_height
                + padding_height,
                original_width
                + padding_width,
            ),
        )


class PatchMerging(nn.Module):
    """Merge neighboring 2×2 patches and double channels."""

    def __init__(
        self,
        embedding_dimension: int,
    ) -> None:
        """Initialize patch merging."""

        super().__init__()

        self.embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.normalization = nn.LayerNorm(
            4 * self.embedding_dimension
        )

        self.reduction = nn.Linear(
            4 * self.embedding_dimension,
            2 * self.embedding_dimension,
            bias=False,
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Downsample the feature map by two."""

        if feature_map.ndim != 4:
            raise ValueError(
                "Patch-merging input must have "
                "shape [B, C, H, W]."
            )

        batch_size, channels, height, width = (
            feature_map.shape
        )

        if channels != self.embedding_dimension:
            raise ValueError(
                "Patch-merging channel mismatch."
            )

        padding_height = (
            height % 2
        )

        padding_width = (
            width % 2
        )

        if (
            padding_height > 0
            or padding_width > 0
        ):
            feature_map = F.pad(
                feature_map,
                (
                    0,
                    padding_width,
                    0,
                    padding_height,
                ),
            )

        feature_map = feature_map.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        top_left = feature_map[
            :,
            0::2,
            0::2,
            :,
        ]

        bottom_left = feature_map[
            :,
            1::2,
            0::2,
            :,
        ]

        top_right = feature_map[
            :,
            0::2,
            1::2,
            :,
        ]

        bottom_right = feature_map[
            :,
            1::2,
            1::2,
            :,
        ]

        merged = torch.cat(
            [
                top_left,
                bottom_left,
                top_right,
                bottom_right,
            ],
            dim=-1,
        )

        merged = self.normalization(
            merged
        )

        merged = self.reduction(
            merged
        )

        return merged.permute(
            0,
            3,
            1,
            2,
        ).contiguous()


class SwinStage(nn.Module):
    """Sequence of alternating regular and shifted-window blocks."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        depth: int,
        number_of_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout_probability: float,
        attention_dropout: float,
        drop_path_probabilities: Sequence[
            float
        ],
    ) -> None:
        """Initialize one hierarchical Swin stage."""

        super().__init__()

        resolved_depth = (
            _require_positive_integer(
                depth,
                "depth",
            )
        )

        if len(
            drop_path_probabilities
        ) != resolved_depth:
            raise ValueError(
                "drop_path_probabilities length "
                "must match stage depth."
            )

        blocks: list[nn.Module] = []

        for block_index in range(
            resolved_depth
        ):
            shift_size = (
                0
                if block_index % 2 == 0
                else window_size // 2
            )

            blocks.append(
                SwinTransformerBlock(
                    embedding_dimension=(
                        embedding_dimension
                    ),
                    number_of_heads=(
                        number_of_heads
                    ),
                    window_size=(
                        window_size
                    ),
                    shift_size=(
                        shift_size
                    ),
                    mlp_ratio=(
                        mlp_ratio
                    ),
                    dropout_probability=(
                        dropout_probability
                    ),
                    attention_dropout=(
                        attention_dropout
                    ),
                    drop_path_probability=float(
                        drop_path_probabilities[
                            block_index
                        ]
                    ),
                )
            )

        self.blocks = nn.Sequential(
            *blocks
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply all stage blocks."""

        return self.blocks(
            feature_map
        )


class DecoderFusionBlock(nn.Module):
    """Upsample and fuse a decoder feature with an encoder skip."""

    def __init__(
        self,
        *,
        decoder_channels: int,
        skip_channels: int,
        output_channels: int,
    ) -> None:
        """Initialize one decoder fusion stage."""

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

        self.decoder_projection = nn.Conv2d(
            in_channels=(
                resolved_decoder_channels
            ),
            out_channels=(
                resolved_output_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.skip_projection = nn.Conv2d(
            in_channels=(
                resolved_skip_channels
            ),
            out_channels=(
                resolved_output_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.refinement = DoubleConvolution(
            input_channels=(
                resolved_output_channels
                * 2
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
        """Upsample, project, concatenate, and refine."""

        decoder_feature = F.interpolate(
            decoder_feature,
            size=(
                skip_feature.shape[-2:]
            ),
            mode="bilinear",
            align_corners=False,
        )

        decoder_feature = (
            self.decoder_projection(
                decoder_feature
            )
        )

        skip_feature = self.skip_projection(
            skip_feature
        )

        fused = torch.cat(
            [
                decoder_feature,
                skip_feature,
            ],
            dim=1,
        )

        return self.refinement(
            fused
        )


class SwinUNet(nn.Module):
    """Hierarchical Swin-UNet segmentation baseline."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        patch_size: int = 4,
        embedding_dimension: int = 96,
        depths: tuple[
            int,
            int,
            int,
            int,
        ] = (
            2,
            2,
            6,
            2,
        ),
        number_of_heads: tuple[
            int,
            int,
            int,
            int,
        ] = (
            3,
            6,
            12,
            24,
        ),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout_probability: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        """Initialize Swin-UNet."""

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

        self.patch_size = (
            _require_positive_integer(
                patch_size,
                "patch_size",
            )
        )

        self.embedding_dimension = (
            _require_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.depths = (
            _validate_integer_sequence(
                depths,
                context="depths",
                expected_length=4,
            )
        )

        self.number_of_heads = (
            _validate_integer_sequence(
                number_of_heads,
                context="number_of_heads",
                expected_length=4,
            )
        )

        self.window_size = (
            _require_positive_integer(
                window_size,
                "window_size",
            )
        )

        if isinstance(
            mlp_ratio,
            bool,
        ):
            raise TypeError(
                "mlp_ratio must be numeric."
            )

        self.mlp_ratio = float(
            mlp_ratio
        )

        if (
            not math.isfinite(
                self.mlp_ratio
            )
            or self.mlp_ratio <= 0.0
        ):
            raise ValueError(
                "mlp_ratio must be positive "
                "and finite."
            )

        self.dropout_probability = (
            _validate_probability(
                dropout_probability,
                "dropout_probability",
            )
        )

        self.attention_dropout = (
            _validate_probability(
                attention_dropout,
                "attention_dropout",
            )
        )

        self.drop_path_rate = (
            _validate_probability(
                drop_path_rate,
                "drop_path_rate",
            )
        )

        stage_dimensions = (
            self.embedding_dimension,
            self.embedding_dimension * 2,
            self.embedding_dimension * 4,
            self.embedding_dimension * 8,
        )

        self.stage_dimensions = (
            stage_dimensions
        )

        for dimension, heads in zip(
            stage_dimensions,
            self.number_of_heads,
            strict=True,
        ):
            if dimension % heads != 0:
                raise ValueError(
                    "Each stage embedding dimension "
                    "must be divisible by its "
                    "number of attention heads."
                )

        total_blocks = sum(
            self.depths
        )

        drop_path_probabilities = (
            torch.linspace(
                0.0,
                self.drop_path_rate,
                total_blocks,
            ).tolist()
        )

        block_offset = 0

        self.patch_embedding = PatchEmbedding(
            input_channels=(
                self.input_channels
            ),
            embedding_dimension=(
                stage_dimensions[0]
            ),
            patch_size=(
                self.patch_size
            ),
        )

        self.stage1 = SwinStage(
            embedding_dimension=(
                stage_dimensions[0]
            ),
            depth=self.depths[0],
            number_of_heads=(
                self.number_of_heads[0]
            ),
            window_size=(
                self.window_size
            ),
            mlp_ratio=(
                self.mlp_ratio
            ),
            dropout_probability=(
                self.dropout_probability
            ),
            attention_dropout=(
                self.attention_dropout
            ),
            drop_path_probabilities=(
                drop_path_probabilities[
                    block_offset:
                    block_offset
                    + self.depths[0]
                ]
            ),
        )

        block_offset += self.depths[
            0
        ]

        self.merge1 = PatchMerging(
            stage_dimensions[0]
        )

        self.stage2 = SwinStage(
            embedding_dimension=(
                stage_dimensions[1]
            ),
            depth=self.depths[1],
            number_of_heads=(
                self.number_of_heads[1]
            ),
            window_size=(
                self.window_size
            ),
            mlp_ratio=(
                self.mlp_ratio
            ),
            dropout_probability=(
                self.dropout_probability
            ),
            attention_dropout=(
                self.attention_dropout
            ),
            drop_path_probabilities=(
                drop_path_probabilities[
                    block_offset:
                    block_offset
                    + self.depths[1]
                ]
            ),
        )

        block_offset += self.depths[
            1
        ]

        self.merge2 = PatchMerging(
            stage_dimensions[1]
        )

        self.stage3 = SwinStage(
            embedding_dimension=(
                stage_dimensions[2]
            ),
            depth=self.depths[2],
            number_of_heads=(
                self.number_of_heads[2]
            ),
            window_size=(
                self.window_size
            ),
            mlp_ratio=(
                self.mlp_ratio
            ),
            dropout_probability=(
                self.dropout_probability
            ),
            attention_dropout=(
                self.attention_dropout
            ),
            drop_path_probabilities=(
                drop_path_probabilities[
                    block_offset:
                    block_offset
                    + self.depths[2]
                ]
            ),
        )

        block_offset += self.depths[
            2
        ]

        self.merge3 = PatchMerging(
            stage_dimensions[2]
        )

        self.stage4 = SwinStage(
            embedding_dimension=(
                stage_dimensions[3]
            ),
            depth=self.depths[3],
            number_of_heads=(
                self.number_of_heads[3]
            ),
            window_size=(
                self.window_size
            ),
            mlp_ratio=(
                self.mlp_ratio
            ),
            dropout_probability=(
                self.dropout_probability
            ),
            attention_dropout=(
                self.attention_dropout
            ),
            drop_path_probabilities=(
                drop_path_probabilities[
                    block_offset:
                    block_offset
                    + self.depths[3]
                ]
            ),
        )

        self.decoder3 = DecoderFusionBlock(
            decoder_channels=(
                stage_dimensions[3]
            ),
            skip_channels=(
                stage_dimensions[2]
            ),
            output_channels=(
                stage_dimensions[2]
            ),
        )

        self.decoder2 = DecoderFusionBlock(
            decoder_channels=(
                stage_dimensions[2]
            ),
            skip_channels=(
                stage_dimensions[1]
            ),
            output_channels=(
                stage_dimensions[1]
            ),
        )

        self.decoder1 = DecoderFusionBlock(
            decoder_channels=(
                stage_dimensions[1]
            ),
            skip_channels=(
                stage_dimensions[0]
            ),
            output_channels=(
                stage_dimensions[0]
            ),
        )

        final_channels = max(
            self.embedding_dimension // 2,
            16,
        )

        self.final_refinement = (
            DoubleConvolution(
                input_channels=(
                    stage_dimensions[0]
                ),
                output_channels=(
                    final_channels
                ),
            )
        )

        self.segmentation_head = nn.Conv2d(
            in_channels=(
                final_channels
            ),
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
        """Initialize trainable parameters."""

        for module in self.modules():
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
        """Validate a Swin-UNet input tensor."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "Swin-UNet input must be a "
                "torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "Swin-UNet input must have shape "
                "[B, C, H, W], received "
                f"{tuple(image.shape)}."
            )

        if image.shape[0] <= 0:
            raise ValueError(
                "Swin-UNet input batch cannot "
                "be empty."
            )

        if (
            image.shape[1]
            != self.input_channels
        ):
            raise ValueError(
                "Swin-UNet input channel mismatch: "
                f"expected {self.input_channels}, "
                f"received {image.shape[1]}."
            )

        minimum_dimension = (
            self.patch_size * 8
        )

        if (
            image.shape[-2]
            < minimum_dimension
            or image.shape[-1]
            < minimum_dimension
        ):
            raise ValueError(
                "Swin-UNet input height and width "
                f"must each be at least "
                f"{minimum_dimension} pixels."
            )

        if not torch.isfinite(
            image
        ).all():
            raise ValueError(
                "Swin-UNet input contains "
                "non-finite values."
            )

    def forward_features(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Compute hierarchical encoder and decoder features."""

        self._validate_input(
            image
        )

        (
            patch_features,
            _,
        ) = self.patch_embedding(
            image
        )

        encoder1 = self.stage1(
            patch_features
        )

        encoder2 = self.stage2(
            self.merge1(
                encoder1
            )
        )

        encoder3 = self.stage3(
            self.merge2(
                encoder2
            )
        )

        encoder4 = self.stage4(
            self.merge3(
                encoder3
            )
        )

        decoder3 = self.decoder3(
            encoder4,
            encoder3,
        )

        decoder2 = self.decoder2(
            decoder3,
            encoder2,
        )

        decoder1 = self.decoder1(
            decoder2,
            encoder1,
        )

        features = {
            "patch_embedding": (
                patch_features
            ),
            "encoder1": encoder1,
            "encoder2": encoder2,
            "encoder3": encoder3,
            "encoder4": encoder4,
            "decoder3": decoder3,
            "decoder2": decoder2,
            "decoder1": decoder1,
        }

        for name, feature in features.items():
            if not torch.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"Swin-UNet feature {name!r} "
                    "contains non-finite values."
                )

        return features

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Predict one-channel binary lesion-mask logits."""

        input_size = tuple(
            int(value)
            for value
            in image.shape[-2:]
        )

        features = self.forward_features(
            image
        )

        full_resolution_feature = (
            F.interpolate(
                features[
                    "decoder1"
                ],
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        )

        full_resolution_feature = (
            self.final_refinement(
                full_resolution_feature
            )
        )

        mask_logits = (
            self.segmentation_head(
                full_resolution_feature
            )
        )

        if not torch.isfinite(
            mask_logits
        ).all():
            raise RuntimeError(
                "Swin-UNet mask logits contain "
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
                SWIN_UNET_PROTOCOL_VERSION
            ),
            "architecture": (
                "swin_unet"
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
            "patch_size": (
                self.patch_size
            ),
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "stage_dimensions": list(
                self.stage_dimensions
            ),
            "depths": list(
                self.depths
            ),
            "number_of_heads": list(
                self.number_of_heads
            ),
            "window_size": (
                self.window_size
            ),
            "mlp_ratio": (
                self.mlp_ratio
            ),
            "dropout_probability": (
                self.dropout_probability
            ),
            "attention_dropout": (
                self.attention_dropout
            ),
            "drop_path_rate": (
                self.drop_path_rate
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
            "uses_shifted_windows": True,
            "uses_cnn_encoder": False,
            "uses_boundary_conditioning": (
                False
            ),
            "uses_auxiliary_targets": False,
        }


SwinUnet = SwinUNet
SwinUNetBaseline = SwinUNet


def run_swin_unet_self_test() -> dict[str, Any]:
    """Run a reduced offline CPU forward/backward test."""

    torch.manual_seed(
        42
    )

    model = SwinUNet(
        input_channels=3,
        output_channels=1,
        patch_size=4,
        embedding_dimension=24,
        depths=(
            1,
            2,
            1,
            1,
        ),
        number_of_heads=(
            3,
            3,
            6,
            12,
        ),
        window_size=4,
        mlp_ratio=2.0,
        dropout_probability=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )

    model.train()

    image = torch.randn(
        1,
        3,
        65,
        67,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = (
        torch.rand(
            1,
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

    patch_gradient = (
        model
        .patch_embedding
        .projection
        .weight
        .grad
    )

    attention_gradient = (
        model
        .stage2
        .blocks[1]
        .attention
        .query_key_value
        .weight
        .grad
    )

    decoder_gradient = (
        model
        .segmentation_head
        .weight
        .grad
    )

    model.eval()

    with torch.inference_mode():
        features = model.forward_features(
            image.detach()
        )

    architecture = (
        model.architecture_summary()
    )

    parameter_count = (
        model.parameter_count()
    )

    shifted_mask = (
        build_shifted_window_mask(
            padded_height=8,
            padded_width=8,
            window_size=4,
            shift_size=2,
            device=torch.device(
                "cpu"
            ),
        )
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
        "patch_gradient_exists": (
            patch_gradient is not None
        ),
        "patch_gradient_nonzero": (
            patch_gradient is not None
            and float(
                patch_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "attention_gradient_exists": (
            attention_gradient is not None
        ),
        "attention_gradient_nonzero": (
            attention_gradient is not None
            and float(
                attention_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "decoder_gradient_exists": (
            decoder_gradient is not None
        ),
        "decoder_gradient_nonzero": (
            decoder_gradient is not None
            and float(
                decoder_gradient
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
            == 8
        ),
        "encoder_channel_progression": (
            [
                features[
                    "encoder1"
                ].shape[1],
                features[
                    "encoder2"
                ].shape[1],
                features[
                    "encoder3"
                ].shape[1],
                features[
                    "encoder4"
                ].shape[1],
            ]
            == [
                24,
                48,
                96,
                192,
            ]
        ),
        "decoder1_channels": (
            features[
                "decoder1"
            ].shape[1]
            == 24
        ),
        "shifted_mask_exists": (
            shifted_mask is not None
        ),
        "shifted_mask_shape": (
            shifted_mask is not None
            and tuple(
                shifted_mask.shape
            )
            == (
                4,
                16,
                16,
            )
        ),
        "shifted_mask_finite": (
            shifted_mask is not None
            and torch.isfinite(
                shifted_mask
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
        "shifted_windows_enabled": (
            architecture[
                "uses_shifted_windows"
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
            SWIN_UNET_PROTOCOL_VERSION
        ),
        "checks": checks,
        "observed_output_shape": list(
            mask_logits.shape
        ),
        "feature_shapes": {
            name: list(
                feature.shape
            )
            for name, feature
            in features.items()
        },
        "parameter_count": (
            parameter_count
        ),
        "architecture": (
            architecture
        ),
    }