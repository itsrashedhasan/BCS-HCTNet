"""ResNet-34 CNN encoder for BCS-HCTNet.

The CNN branch extracts local texture, edge, and lesion-boundary features
at multiple spatial resolutions.

Returned features
-----------------
stem:
    1/2 input resolution, 64 channels.

stage1:
    1/4 input resolution, 64 channels.

stage2:
    1/8 input resolution, 128 channels.

stage3:
    1/16 input resolution, 256 channels.

stage4:
    1/32 input resolution, 512 channels.

For the approved 352 x 352 input resolution, the expected spatial sizes are:

- stem:   176 x 176
- stage1:  88 x 88
- stage2:  44 x 44
- stage3:  22 x 22
- stage4:  11 x 11

The module supports random initialization, torchvision ImageNet weights,
or a local ResNet-34 checkpoint. The self-test never downloads weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNet34_Weights,
    resnet34,
)


CNN_ENCODER_PROTOCOL_VERSION = (
    "BCS-HCTNet-resnet34-cnn-encoder-v1"
)


@dataclass(frozen=True)
class EncoderFeatureSpec:
    """Description of one encoder feature level."""

    name: str
    channels: int
    stride: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "channels": self.channels,
            "stride": self.stride,
        }


CNN_FEATURE_SPECS = (
    EncoderFeatureSpec(
        name="stem",
        channels=64,
        stride=2,
    ),
    EncoderFeatureSpec(
        name="stage1",
        channels=64,
        stride=4,
    ),
    EncoderFeatureSpec(
        name="stage2",
        channels=128,
        stride=8,
    ),
    EncoderFeatureSpec(
        name="stage3",
        channels=256,
        stride=16,
    ),
    EncoderFeatureSpec(
        name="stage4",
        channels=512,
        stride=32,
    ),
)


def _extract_state_dict(
    checkpoint: object,
) -> Mapping[str, Tensor]:
    """Extract a model state dictionary from a checkpoint."""

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "ResNet checkpoint must contain "
            "a mapping."
        )

    candidate: object = checkpoint

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
    ):
        nested = checkpoint.get(key)

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
            "inside the checkpoint."
        )

    state_dict: dict[str, Tensor] = {}

    for key, value in candidate.items():
        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                "Checkpoint parameter names "
                "must be strings."
            )

        if not isinstance(
            value,
            Tensor,
        ):
            continue

        normalized_key = key

        for prefix in (
            "module.",
            "backbone.",
            "encoder.",
        ):
            if normalized_key.startswith(
                prefix
            ):
                normalized_key = (
                    normalized_key[
                        len(prefix):
                    ]
                )

        state_dict[
            normalized_key
        ] = value

    if not state_dict:
        raise RuntimeError(
            "Checkpoint contains no tensor "
            "parameters."
        )

    return state_dict


def _load_checkpoint(
    checkpoint_path: Path,
) -> object:
    """Load a checkpoint safely across PyTorch versions."""

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


class ResNet34CNNEncoder(nn.Module):
    """Multi-scale ResNet-34 CNN encoder."""

    feature_specs = CNN_FEATURE_SPECS

    def __init__(
        self,
        *,
        pretrained: bool = False,
        allow_weight_download: bool = False,
        weights_path: str | Path | None = None,
    ) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        pretrained:
            Request torchvision ImageNet weights.

        allow_weight_download:
            Allow torchvision to download ImageNet weights when they are
            not already cached. This must be explicitly enabled.

        weights_path:
            Optional local ResNet-34 checkpoint. When supplied, this takes
            precedence over torchvision ImageNet weights.
        """

        super().__init__()

        for name, value in {
            "pretrained": pretrained,
            "allow_weight_download": (
                allow_weight_download
            ),
        }.items():
            if not isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{name} must be Boolean."
                )

        resolved_weights_path = (
            Path(weights_path)
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
                "Local ResNet-34 checkpoint "
                f"not found: {resolved_weights_path}"
            )

        if (
            pretrained
            and resolved_weights_path is None
            and not allow_weight_download
        ):
            raise RuntimeError(
                "ImageNet pretrained weights were "
                "requested, but downloading is "
                "disabled and no local weights_path "
                "was provided."
            )

        torchvision_weights = None

        if (
            pretrained
            and resolved_weights_path is None
        ):
            torchvision_weights = (
                ResNet34_Weights.DEFAULT
            )

        backbone = resnet34(
            weights=torchvision_weights
        )

        if resolved_weights_path is not None:
            checkpoint = _load_checkpoint(
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

            unexpected = set(
                incompatible.unexpected_keys
            )

            missing = (
                set(
                    incompatible.missing_keys
                )
                - allowed_missing
            )

            if missing or unexpected:
                raise RuntimeError(
                    "Local ResNet-34 checkpoint "
                    "is incompatible. "
                    f"Missing keys: {sorted(missing)}; "
                    f"unexpected keys: "
                    f"{sorted(unexpected)}."
                )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.max_pool = backbone.maxpool
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self.pretrained_requested = pretrained

        self.weights_source = (
            str(resolved_weights_path)
            if resolved_weights_path is not None
            else (
                "torchvision_imagenet"
                if torchvision_weights is not None
                else "random_initialization"
            )
        )

    @property
    def output_channels(
        self,
    ) -> dict[str, int]:
        """Return output channels for every feature level."""

        return {
            specification.name: (
                specification.channels
            )
            for specification in self.feature_specs
        }

    @property
    def output_strides(
        self,
    ) -> dict[str, int]:
        """Return spatial stride for every feature level."""

        return {
            specification.name: (
                specification.stride
            )
            for specification in self.feature_specs
        }

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Extract multi-scale CNN features."""

        if not isinstance(
            image,
            Tensor,
        ):
            raise TypeError(
                "CNN encoder input must be "
                "a torch.Tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                "CNN encoder input must have "
                "shape [batch, channels, height, "
                f"width], received {tuple(image.shape)}."
            )

        if image.shape[1] != 3:
            raise ValueError(
                "CNN encoder requires three RGB "
                f"channels, received {image.shape[1]}."
            )

        if (
            image.shape[-2] < 32
            or image.shape[-1] < 32
        ):
            raise ValueError(
                "CNN encoder input height and width "
                "must each be at least 32 pixels."
            )

        stem = self.stem(
            image
        )

        stage1 = self.stage1(
            self.max_pool(
                stem
            )
        )

        stage2 = self.stage2(
            stage1
        )

        stage3 = self.stage3(
            stage2
        )

        stage4 = self.stage4(
            stage3
        )

        features = {
            "stem": stem,
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
        """Validate feature names, channels, and values."""

        expected_names = [
            specification.name
            for specification in self.feature_specs
        ]

        if list(
            features
        ) != expected_names:
            raise RuntimeError(
                "CNN feature names or order are "
                f"invalid: {list(features)}."
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

        for specification in self.feature_specs:
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
                    "size differs from the input."
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
                    "spatial resolution unexpectedly."
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
        """Return the encoder parameter count."""

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
        """Return encoder architecture metadata."""

        return {
            "protocol_version": (
                CNN_ENCODER_PROTOCOL_VERSION
            ),
            "architecture": "resnet34",
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
            "features": [
                specification.to_dict()
                for specification
                in self.feature_specs
            ],
        }


def run_cnn_encoder_self_test() -> dict[str, Any]:
    """Run an offline CPU forward/backward self-test."""

    torch.manual_seed(
        42
    )

    encoder = ResNet34CNNEncoder(
        pretrained=False,
        allow_weight_download=False,
    )

    encoder.eval()

    image = torch.randn(
        1,
        3,
        128,
        128,
        dtype=torch.float32,
        requires_grad=True,
    )

    features = encoder(
        image
    )

    expected_shapes = {
        "stem": (
            1,
            64,
            64,
            64,
        ),
        "stage1": (
            1,
            64,
            32,
            32,
        ),
        "stage2": (
            1,
            128,
            16,
            16,
        ),
        "stage3": (
            1,
            256,
            8,
            8,
        ),
        "stage4": (
            1,
            512,
            4,
            4,
        ),
    }

    loss = sum(
        feature.square().mean()
        for feature in features.values()
    )

    loss.backward()

    first_convolution = (
        encoder.stem[0]
    )

    gradient = (
        first_convolution.weight.grad
    )

    checks = {
        "feature_names": (
            list(features)
            == [
                "stem",
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
            torch.isfinite(
                feature
            ).all().item()
            for feature in features.values()
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
            and torch.isfinite(
                image.grad
            ).all().item()
        ),
        "encoder_gradient_exists": (
            gradient is not None
        ),
        "encoder_gradient_nonzero": (
            gradient is not None
            and float(
                gradient.abs().sum().item()
            )
            > 0.0
        ),
        "output_channels_correct": (
            encoder.output_channels
            == {
                "stem": 64,
                "stage1": 64,
                "stage2": 128,
                "stage3": 256,
                "stage4": 512,
            }
        ),
        "output_strides_correct": (
            encoder.output_strides
            == {
                "stem": 2,
                "stage1": 4,
                "stage2": 8,
                "stage3": 16,
                "stage4": 32,
            }
        ),
        "parameter_count_positive": (
            encoder.parameter_count()
            > 0
        ),
        "offline_weights_used": (
            encoder.weights_source
            == "random_initialization"
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
            CNN_ENCODER_PROTOCOL_VERSION
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