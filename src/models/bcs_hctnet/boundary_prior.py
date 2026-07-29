"""Boundary-prior estimator for BCS-HCTNet.

The boundary-prior branch predicts where lesion-boundary evidence is likely
to be reliable. The resulting map conditions CNN–Transformer fusion:

- near likely lesion boundaries, local CNN information can receive greater
  emphasis;
- away from boundaries, global Transformer context can receive greater
  emphasis.

Evidence sources
----------------
The estimator combines:

1. high-resolution CNN stem features;
2. CNN Stage 1 local-detail features;
3. Transformer Stage 1 contextual features;
4. absolute disagreement between projected CNN and Transformer features.

Outputs
-------
boundary_logits:
    Unnormalized boundary-prior logits at Stage 1 resolution.

boundary_probability:
    Sigmoid probability map in [0, 1].

reliability_map:
    Alias of the boundary probability used by the fusion system.

reliability_pyramid:
    Reliability maps resized to Transformer stages 1 through 4.

For a 352 x 352 input, the native boundary-prior resolution is 88 x 88.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BOUNDARY_PRIOR_PROTOCOL_VERSION = (
    "BCS-HCTNet-boundary-prior-v1"
)

DEFAULT_PYRAMID_LEVELS = (
    "stage1",
    "stage2",
    "stage3",
    "stage4",
)


def choose_group_count(
    channels: int,
    maximum_groups: int = 8,
) -> int:
    """Choose the largest valid GroupNorm group count."""

    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels <= 0
    ):
        raise ValueError(
            "channels must be a positive integer."
        )

    if (
        isinstance(maximum_groups, bool)
        or not isinstance(maximum_groups, int)
        or maximum_groups <= 0
    ):
        raise ValueError(
            "maximum_groups must be a "
            "positive integer."
        )

    for group_count in range(
        min(
            channels,
            maximum_groups,
        ),
        0,
        -1,
    ):
        if channels % group_count == 0:
            return group_count

    return 1


class ConvNormActivation(nn.Module):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        for name, value in {
            "input_channels": input_channels,
            "output_channels": output_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "groups": groups,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive "
                    f"integer, received {value!r}."
                )

        if input_channels % groups != 0:
            raise ValueError(
                "input_channels must be divisible "
                "by groups."
            )

        if output_channels % groups != 0:
            raise ValueError(
                "output_channels must be divisible "
                "by groups."
            )

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=choose_group_count(
                    output_channels
                ),
                num_channels=output_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Apply convolution, normalization, and activation."""

        return self.block(
            feature_map
        )


class FeatureProjection(nn.Module):
    """Project one feature level into a common hidden dimension."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.projection = ConvNormActivation(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Project the input feature map."""

        return self.projection(
            feature_map
        )


class BoundaryRefinementBlock(nn.Module):
    """Residual spatial-refinement block."""

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError(
                "channels must be a positive integer."
            )

        self.conv1 = ConvNormActivation(
            input_channels=channels,
            output_channels=channels,
            kernel_size=3,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=choose_group_count(
                    channels
                ),
                num_channels=channels,
            ),
        )

        self.activation = nn.GELU()

    def forward(
        self,
        feature_map: Tensor,
    ) -> Tensor:
        """Refine features through a residual path."""

        residual = feature_map

        feature_map = self.conv1(
            feature_map
        )

        feature_map = self.conv2(
            feature_map
        )

        return self.activation(
            feature_map + residual
        )


class BoundaryPriorEstimator(nn.Module):
    """Predict boundary reliability from early hybrid features."""

    def __init__(
        self,
        *,
        cnn_stem_channels: int = 64,
        cnn_stage1_channels: int = 64,
        transformer_stage1_channels: int = 64,
        hidden_channels: int = 64,
        refinement_blocks: int = 2,
        pyramid_levels: Sequence[str] = (
            DEFAULT_PYRAMID_LEVELS
        ),
        initial_boundary_bias: float = -2.0,
    ) -> None:
        """Initialize the boundary-prior estimator.

        ``initial_boundary_bias`` is negative because boundary pixels are
        normally sparse. It prevents the initial map from assigning high
        boundary probability everywhere while preserving trainable gradients.
        """

        super().__init__()

        for name, value in {
            "cnn_stem_channels": (
                cnn_stem_channels
            ),
            "cnn_stage1_channels": (
                cnn_stage1_channels
            ),
            "transformer_stage1_channels": (
                transformer_stage1_channels
            ),
            "hidden_channels": (
                hidden_channels
            ),
            "refinement_blocks": (
                refinement_blocks
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive "
                    f"integer, received {value!r}."
                )

        if not isinstance(
            pyramid_levels,
            Sequence,
        ) or isinstance(
            pyramid_levels,
            (
                str,
                bytes,
            ),
        ):
            raise TypeError(
                "pyramid_levels must be a "
                "non-string sequence."
            )

        normalized_levels = tuple(
            str(level).strip()
            for level in pyramid_levels
        )

        if not normalized_levels:
            raise ValueError(
                "pyramid_levels cannot be empty."
            )

        if any(
            not level
            for level in normalized_levels
        ):
            raise ValueError(
                "pyramid_levels cannot contain "
                "empty names."
            )

        if len(
            set(normalized_levels)
        ) != len(
            normalized_levels
        ):
            raise ValueError(
                "pyramid_levels must be unique."
            )

        if normalized_levels[0] != "stage1":
            raise ValueError(
                "The first pyramid level must "
                "be 'stage1'."
            )

        if not torch.isfinite(
            torch.tensor(
                float(
                    initial_boundary_bias
                )
            )
        ):
            raise ValueError(
                "initial_boundary_bias must "
                "be finite."
            )

        self.hidden_channels = int(
            hidden_channels
        )

        self.pyramid_levels = (
            normalized_levels
        )

        self.cnn_stem_projection = (
            FeatureProjection(
                input_channels=(
                    cnn_stem_channels
                ),
                output_channels=(
                    hidden_channels
                ),
            )
        )

        self.cnn_stage1_projection = (
            FeatureProjection(
                input_channels=(
                    cnn_stage1_channels
                ),
                output_channels=(
                    hidden_channels
                ),
            )
        )

        self.transformer_stage1_projection = (
            FeatureProjection(
                input_channels=(
                    transformer_stage1_channels
                ),
                output_channels=(
                    hidden_channels
                ),
            )
        )

        evidence_channels = (
            hidden_channels * 4
        )

        self.evidence_fusion = (
            ConvNormActivation(
                input_channels=(
                    evidence_channels
                ),
                output_channels=(
                    hidden_channels
                ),
                kernel_size=1,
            )
        )

        self.refinement = nn.Sequential(
            *[
                BoundaryRefinementBlock(
                    hidden_channels
                )
                for _ in range(
                    refinement_blocks
                )
            ]
        )

        intermediate_channels = max(
            16,
            hidden_channels // 2,
        )

        self.boundary_head = nn.Sequential(
            ConvNormActivation(
                input_channels=(
                    hidden_channels
                ),
                output_channels=(
                    intermediate_channels
                ),
                kernel_size=3,
            ),
            nn.Conv2d(
                in_channels=(
                    intermediate_channels
                ),
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self._initialize_parameters(
            initial_boundary_bias=float(
                initial_boundary_bias
            )
        )

    def _initialize_parameters(
        self,
        *,
        initial_boundary_bias: float,
    ) -> None:
        """Initialize convolution and normalization parameters."""

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
                nn.GroupNorm,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        output_layer = (
            self.boundary_head[-1]
        )

        if not isinstance(
            output_layer,
            nn.Conv2d,
        ):
            raise RuntimeError(
                "Boundary head output layer "
                "must be Conv2d."
            )

        nn.init.normal_(
            output_layer.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            output_layer.bias,
            initial_boundary_bias,
        )

    @staticmethod
    def _require_feature(
        features: Mapping[str, Tensor],
        name: str,
        context: str,
    ) -> Tensor:
        """Retrieve and validate one feature map."""

        if not isinstance(
            features,
            Mapping,
        ):
            raise TypeError(
                f"{context} must be a mapping."
            )

        if name not in features:
            raise KeyError(
                f"{context} is missing feature "
                f"{name!r}."
            )

        feature = features[name]

        if not isinstance(
            feature,
            Tensor,
        ):
            raise TypeError(
                f"{context}[{name!r}] must be "
                "a torch.Tensor."
            )

        if feature.ndim != 4:
            raise ValueError(
                f"{context}[{name!r}] must have "
                "shape [B, C, H, W], received "
                f"{tuple(feature.shape)}."
            )

        if not torch.isfinite(
            feature
        ).all():
            raise RuntimeError(
                f"{context}[{name!r}] contains "
                "non-finite values."
            )

        return feature

    def forward(
        self,
        cnn_features: Mapping[str, Tensor],
        transformer_features: Mapping[
            str,
            Tensor,
        ],
    ) -> dict[
        str,
        Tensor | dict[str, Tensor],
    ]:
        """Predict the boundary-prior reliability pyramid."""

        cnn_stem = self._require_feature(
            cnn_features,
            "stem",
            "cnn_features",
        )

        cnn_stage1 = self._require_feature(
            cnn_features,
            "stage1",
            "cnn_features",
        )

        transformer_stage1 = (
            self._require_feature(
                transformer_features,
                "stage1",
                "transformer_features",
            )
        )

        batch_sizes = {
            int(
                cnn_stem.shape[0]
            ),
            int(
                cnn_stage1.shape[0]
            ),
            int(
                transformer_stage1.shape[0]
            ),
        }

        if len(batch_sizes) != 1:
            raise RuntimeError(
                "Boundary-prior feature batch "
                "sizes do not match."
            )

        stage1_size = tuple(
            int(value)
            for value in (
                transformer_stage1.shape[-2:]
            )
        )

        if tuple(
            cnn_stage1.shape[-2:]
        ) != stage1_size:
            raise RuntimeError(
                "CNN Stage 1 and Transformer "
                "Stage 1 spatial sizes must match. "
                f"Found {tuple(cnn_stage1.shape[-2:])} "
                f"and {stage1_size}."
            )

        projected_stem = (
            self.cnn_stem_projection(
                cnn_stem
            )
        )

        projected_stem = F.interpolate(
            projected_stem,
            size=stage1_size,
            mode="bilinear",
            align_corners=False,
        )

        projected_cnn = (
            self.cnn_stage1_projection(
                cnn_stage1
            )
        )

        projected_transformer = (
            self.transformer_stage1_projection(
                transformer_stage1
            )
        )

        disagreement = torch.abs(
            projected_cnn
            - projected_transformer
        )

        evidence = torch.cat(
            [
                projected_stem,
                projected_cnn,
                projected_transformer,
                disagreement,
            ],
            dim=1,
        )

        fused_evidence = (
            self.evidence_fusion(
                evidence
            )
        )

        refined_evidence = self.refinement(
            fused_evidence
        )

        boundary_logits = (
            self.boundary_head(
                refined_evidence
            )
        )

        boundary_probability = (
            torch.sigmoid(
                boundary_logits
            )
        )

        reliability_pyramid: dict[
            str,
            Tensor,
        ] = {}

        for level_name in (
            self.pyramid_levels
        ):
            target_feature = (
                self._require_feature(
                    transformer_features,
                    level_name,
                    "transformer_features",
                )
            )

            target_size = tuple(
                int(value)
                for value in (
                    target_feature.shape[-2:]
                )
            )

            if target_size == stage1_size:
                reliability = (
                    boundary_probability
                )

            else:
                reliability = F.interpolate(
                    boundary_probability,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            reliability_pyramid[
                level_name
            ] = reliability.contiguous()

        outputs: dict[
            str,
            Tensor | dict[str, Tensor],
        ] = {
            "boundary_logits": (
                boundary_logits.contiguous()
            ),
            "boundary_probability": (
                boundary_probability.contiguous()
            ),
            "reliability_map": (
                boundary_probability.contiguous()
            ),
            "reliability_pyramid": (
                reliability_pyramid
            ),
            "boundary_features": (
                refined_evidence.contiguous()
            ),
        }

        self._validate_outputs(
            outputs=outputs,
            transformer_features=(
                transformer_features
            ),
        )

        return outputs

    def _validate_outputs(
        self,
        *,
        outputs: Mapping[
            str,
            Tensor | dict[str, Tensor],
        ],
        transformer_features: Mapping[
            str,
            Tensor,
        ],
    ) -> None:
        """Validate all boundary-prior outputs."""

        required_tensor_outputs = (
            "boundary_logits",
            "boundary_probability",
            "reliability_map",
            "boundary_features",
        )

        for name in (
            required_tensor_outputs
        ):
            output = outputs.get(
                name
            )

            if not isinstance(
                output,
                Tensor,
            ):
                raise RuntimeError(
                    f"Boundary-prior output "
                    f"{name!r} is not a tensor."
                )

            if not torch.isfinite(
                output
            ).all():
                raise RuntimeError(
                    f"Boundary-prior output "
                    f"{name!r} contains "
                    "non-finite values."
                )

        boundary_logits = outputs[
            "boundary_logits"
        ]

        boundary_probability = outputs[
            "boundary_probability"
        ]

        reliability_map = outputs[
            "reliability_map"
        ]

        if not isinstance(
            boundary_logits,
            Tensor,
        ) or not isinstance(
            boundary_probability,
            Tensor,
        ) or not isinstance(
            reliability_map,
            Tensor,
        ):
            raise RuntimeError(
                "Boundary-prior tensors are invalid."
            )

        expected_size = tuple(
            transformer_features[
                "stage1"
            ].shape[-2:]
        )

        expected_shape = (
            int(
                transformer_features[
                    "stage1"
                ].shape[0]
            ),
            1,
            int(
                expected_size[0]
            ),
            int(
                expected_size[1]
            ),
        )

        for name, tensor in {
            "boundary_logits": (
                boundary_logits
            ),
            "boundary_probability": (
                boundary_probability
            ),
            "reliability_map": (
                reliability_map
            ),
        }.items():
            if tuple(
                tensor.shape
            ) != expected_shape:
                raise RuntimeError(
                    f"{name} expected shape "
                    f"{expected_shape}, found "
                    f"{tuple(tensor.shape)}."
                )

        probability_min = float(
            boundary_probability.min().item()
        )

        probability_max = float(
            boundary_probability.max().item()
        )

        if (
            probability_min < 0.0
            or probability_max > 1.0
        ):
            raise RuntimeError(
                "Boundary probabilities are "
                "outside [0, 1]."
            )

        pyramid = outputs.get(
            "reliability_pyramid"
        )

        if not isinstance(
            pyramid,
            Mapping,
        ):
            raise RuntimeError(
                "reliability_pyramid must be "
                "a mapping."
            )

        if tuple(
            pyramid
        ) != self.pyramid_levels:
            raise RuntimeError(
                "Reliability-pyramid names or "
                "order are invalid."
            )

        for level_name in (
            self.pyramid_levels
        ):
            reliability = pyramid[
                level_name
            ]

            expected_level_shape = (
                int(
                    transformer_features[
                        level_name
                    ].shape[0]
                ),
                1,
                int(
                    transformer_features[
                        level_name
                    ].shape[-2]
                ),
                int(
                    transformer_features[
                        level_name
                    ].shape[-1]
                ),
            )

            if tuple(
                reliability.shape
            ) != expected_level_shape:
                raise RuntimeError(
                    f"Reliability map "
                    f"{level_name!r} expected "
                    f"{expected_level_shape}, found "
                    f"{tuple(reliability.shape)}."
                )

            if not torch.isfinite(
                reliability
            ).all():
                raise RuntimeError(
                    f"Reliability map "
                    f"{level_name!r} contains "
                    "non-finite values."
                )

            if (
                float(
                    reliability.min().item()
                )
                < 0.0
                or float(
                    reliability.max().item()
                )
                > 1.0
            ):
                raise RuntimeError(
                    f"Reliability map "
                    f"{level_name!r} is outside "
                    "[0, 1]."
                )

    def parameter_count(
        self,
        trainable_only: bool = False,
    ) -> int:
        """Return parameter count."""

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
        """Return architecture metadata."""

        return {
            "protocol_version": (
                BOUNDARY_PRIOR_PROTOCOL_VERSION
            ),
            "architecture": (
                "early_hybrid_boundary_prior"
            ),
            "hidden_channels": (
                self.hidden_channels
            ),
            "pyramid_levels": list(
                self.pyramid_levels
            ),
            "parameter_count": (
                self.parameter_count()
            ),
            "trainable_parameter_count": (
                self.parameter_count(
                    trainable_only=True
                )
            ),
            "evidence_sources": [
                "cnn_stem",
                "cnn_stage1",
                "transformer_stage1",
                (
                    "absolute_cnn_transformer_"
                    "disagreement"
                ),
            ],
            "native_output_stride": 4,
        }


def run_boundary_prior_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward self-test."""

    torch.manual_seed(
        42
    )

    estimator = BoundaryPriorEstimator(
        cnn_stem_channels=64,
        cnn_stage1_channels=64,
        transformer_stage1_channels=64,
        hidden_channels=32,
        refinement_blocks=2,
    )

    estimator.eval()

    cnn_stem = torch.randn(
        2,
        64,
        32,
        32,
        dtype=torch.float32,
        requires_grad=True,
    )

    cnn_stage1 = torch.randn(
        2,
        64,
        16,
        16,
        dtype=torch.float32,
        requires_grad=True,
    )

    transformer_stage1 = torch.randn(
        2,
        64,
        16,
        16,
        dtype=torch.float32,
        requires_grad=True,
    )

    cnn_features = {
        "stem": cnn_stem,
        "stage1": cnn_stage1,
    }

    transformer_features = {
        "stage1": transformer_stage1,
        "stage2": torch.randn(
            2,
            128,
            8,
            8,
        ),
        "stage3": torch.randn(
            2,
            320,
            4,
            4,
        ),
        "stage4": torch.randn(
            2,
            512,
            2,
            2,
        ),
    }

    outputs = estimator(
        cnn_features,
        transformer_features,
    )

    boundary_logits = outputs[
        "boundary_logits"
    ]

    boundary_probability = outputs[
        "boundary_probability"
    ]

    reliability_map = outputs[
        "reliability_map"
    ]

    boundary_features = outputs[
        "boundary_features"
    ]

    reliability_pyramid = outputs[
        "reliability_pyramid"
    ]

    if not isinstance(
        boundary_logits,
        Tensor,
    ) or not isinstance(
        boundary_probability,
        Tensor,
    ) or not isinstance(
        reliability_map,
        Tensor,
    ) or not isinstance(
        boundary_features,
        Tensor,
    ) or not isinstance(
        reliability_pyramid,
        Mapping,
    ):
        raise RuntimeError(
            "Boundary-prior self-test outputs "
            "have invalid types."
        )

    expected_pyramid_shapes = {
        "stage1": (
            2,
            1,
            16,
            16,
        ),
        "stage2": (
            2,
            1,
            8,
            8,
        ),
        "stage3": (
            2,
            1,
            4,
            4,
        ),
        "stage4": (
            2,
            1,
            2,
            2,
        ),
    }

    loss = (
        boundary_logits.square().mean()
        + boundary_probability.mean()
        + boundary_features.square().mean()
        + sum(
            reliability.square().mean()
            for reliability
            in reliability_pyramid.values()
        )
    )

    loss.backward()

    output_layer = (
        estimator.boundary_head[-1]
    )

    if not isinstance(
        output_layer,
        nn.Conv2d,
    ):
        raise RuntimeError(
            "Unexpected boundary-head output type."
        )

    output_gradient = (
        output_layer.weight.grad
    )

    checks = {
        "boundary_logits_shape": (
            tuple(
                boundary_logits.shape
            )
            == (
                2,
                1,
                16,
                16,
            )
        ),
        "boundary_probability_shape": (
            tuple(
                boundary_probability.shape
            )
            == (
                2,
                1,
                16,
                16,
            )
        ),
        "reliability_map_shape": (
            tuple(
                reliability_map.shape
            )
            == (
                2,
                1,
                16,
                16,
            )
        ),
        "boundary_features_shape": (
            tuple(
                boundary_features.shape
            )
            == (
                2,
                32,
                16,
                16,
            )
        ),
        "pyramid_names": (
            tuple(
                reliability_pyramid
            )
            == DEFAULT_PYRAMID_LEVELS
        ),
        "pyramid_shapes": all(
            tuple(
                reliability_pyramid[
                    level_name
                ].shape
            )
            == expected_shape
            for level_name, expected_shape
            in expected_pyramid_shapes.items()
        ),
        "probability_matches_logits": (
            torch.allclose(
                boundary_probability,
                torch.sigmoid(
                    boundary_logits
                ),
                atol=1e-7,
                rtol=1e-6,
            )
        ),
        "reliability_matches_probability": (
            torch.equal(
                reliability_map,
                boundary_probability,
            )
        ),
        "all_outputs_finite": (
            torch.isfinite(
                boundary_logits
            ).all().item()
            and torch.isfinite(
                boundary_probability
            ).all().item()
            and torch.isfinite(
                boundary_features
            ).all().item()
            and all(
                torch.isfinite(
                    reliability
                ).all().item()
                for reliability
                in reliability_pyramid.values()
            )
        ),
        "probabilities_bounded": (
            float(
                boundary_probability.min().item()
            )
            >= 0.0
            and float(
                boundary_probability.max().item()
            )
            <= 1.0
        ),
        "cnn_stem_gradient_exists": (
            cnn_stem.grad is not None
        ),
        "cnn_stem_gradient_nonzero": (
            cnn_stem.grad is not None
            and float(
                cnn_stem.grad.abs().sum().item()
            )
            > 0.0
        ),
        "cnn_stage1_gradient_exists": (
            cnn_stage1.grad is not None
        ),
        "cnn_stage1_gradient_nonzero": (
            cnn_stage1.grad is not None
            and float(
                cnn_stage1.grad.abs().sum().item()
            )
            > 0.0
        ),
        "transformer_gradient_exists": (
            transformer_stage1.grad
            is not None
        ),
        "transformer_gradient_nonzero": (
            transformer_stage1.grad
            is not None
            and float(
                transformer_stage1
                .grad
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "output_head_gradient_exists": (
            output_gradient is not None
        ),
        "output_head_gradient_nonzero": (
            output_gradient is not None
            and float(
                output_gradient
                .abs()
                .sum()
                .item()
            )
            > 0.0
        ),
        "loss_finite": (
            torch.isfinite(
                loss
            ).item()
        ),
        "parameter_count_positive": (
            estimator.parameter_count()
            > 0
        ),
        "parameter_count_compact": (
            estimator.parameter_count()
            < 1_000_000
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
            BOUNDARY_PRIOR_PROTOCOL_VERSION
        ),
        "checks": checks,
        "expected_pyramid_shapes": {
            level_name: list(shape)
            for level_name, shape
            in expected_pyramid_shapes.items()
        },
        "observed_pyramid_shapes": {
            level_name: list(
                reliability.shape
            )
            for level_name, reliability
            in reliability_pyramid.items()
        },
        "boundary_probability_range": [
            float(
                boundary_probability.min().item()
            ),
            float(
                boundary_probability.max().item()
            ),
        ],
        "architecture": (
            estimator.architecture_summary()
        ),
    }