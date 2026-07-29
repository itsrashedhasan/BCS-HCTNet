"""Synchronized multi-target transforms for BCS-HCTNet.

The same geometric transformation is applied to:

- RGB image;
- binary lesion mask;
- contour target;
- boundary-band target;
- signed-distance map.

Photometric transformations are applied only to the RGB image.

Interpolation policy
--------------------
- image: bilinear;
- binary mask, contour, and boundary band: nearest;
- signed-distance map: bilinear.

Validation and test transforms are deterministic and perform only resizing,
tensor conversion, and ImageNet normalization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


TRANSFORM_PROTOCOL_VERSION = (
    "BCS-HCTNet-multitarget-transform-v1"
)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for synchronized segmentation transforms."""

    height: int = 352
    width: int = 352

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5

    affine_probability: float = 0.5
    rotation_min_degrees: float = -20.0
    rotation_max_degrees: float = 20.0
    scale_min: float = 0.90
    scale_max: float = 1.10
    translation_min_fraction: float = -0.05
    translation_max_fraction: float = 0.05

    brightness_contrast_probability: float = 0.4
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15

    hue_saturation_probability: float = 0.2
    hue_shift_limit: float = 0.02
    saturation_shift_limit: float = 0.10

    gaussian_noise_probability: float = 0.15
    gaussian_noise_std_min: float = 0.0
    gaussian_noise_std_max: float = 0.02

    gaussian_blur_probability: float = 0.10
    gaussian_blur_kernel_sizes: tuple[int, ...] = (
        3,
        5,
    )

    coarse_dropout_probability: float = 0.10
    coarse_dropout_holes_min: int = 1
    coarse_dropout_holes_max: int = 4
    coarse_dropout_size_fraction_min: float = 0.03
    coarse_dropout_size_fraction_max: float = 0.10

    image_mean: tuple[float, float, float] = (
        IMAGENET_MEAN
    )

    image_std: tuple[float, float, float] = (
        IMAGENET_STD
    )

    def validate(self) -> None:
        """Validate transform settings."""

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                "Transform height and width "
                "must be positive."
            )

        probability_values = {
            "horizontal_flip_probability": (
                self.horizontal_flip_probability
            ),
            "vertical_flip_probability": (
                self.vertical_flip_probability
            ),
            "affine_probability": (
                self.affine_probability
            ),
            "brightness_contrast_probability": (
                self.brightness_contrast_probability
            ),
            "hue_saturation_probability": (
                self.hue_saturation_probability
            ),
            "gaussian_noise_probability": (
                self.gaussian_noise_probability
            ),
            "gaussian_blur_probability": (
                self.gaussian_blur_probability
            ),
            "coarse_dropout_probability": (
                self.coarse_dropout_probability
            ),
        }

        for name, value in probability_values.items():
            if not (
                0.0
                <= float(value)
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0, 1], "
                    f"received {value}."
                )

        if (
            self.rotation_min_degrees
            > self.rotation_max_degrees
        ):
            raise ValueError(
                "rotation_min_degrees cannot "
                "exceed rotation_max_degrees."
            )

        if (
            self.scale_min <= 0
            or self.scale_min > self.scale_max
        ):
            raise ValueError(
                "Invalid affine scale range."
            )

        if (
            self.translation_min_fraction
            > self.translation_max_fraction
        ):
            raise ValueError(
                "Invalid translation range."
            )

        if (
            self.translation_min_fraction < -1.0
            or self.translation_max_fraction > 1.0
        ):
            raise ValueError(
                "Translation fractions must "
                "remain in [-1, 1]."
            )

        if (
            self.gaussian_noise_std_min < 0.0
            or self.gaussian_noise_std_min
            > self.gaussian_noise_std_max
        ):
            raise ValueError(
                "Invalid Gaussian-noise range."
            )

        if (
            self.coarse_dropout_holes_min < 1
            or self.coarse_dropout_holes_min
            > self.coarse_dropout_holes_max
        ):
            raise ValueError(
                "Invalid coarse-dropout hole count."
            )

        if (
            self.coarse_dropout_size_fraction_min
            <= 0.0
            or self.coarse_dropout_size_fraction_min
            > self.coarse_dropout_size_fraction_max
        ):
            raise ValueError(
                "Invalid coarse-dropout size range."
            )

        if len(self.image_mean) != 3:
            raise ValueError(
                "image_mean must contain "
                "three values."
            )

        if len(self.image_std) != 3:
            raise ValueError(
                "image_std must contain "
                "three values."
            )

        if any(
            float(value) <= 0.0
            for value in self.image_std
        ):
            raise ValueError(
                "Every image standard deviation "
                "must be positive."
            )

        for kernel_size in (
            self.gaussian_blur_kernel_sizes
        ):
            if (
                kernel_size <= 0
                or kernel_size % 2 == 0
            ):
                raise ValueError(
                    "Gaussian-blur kernel sizes "
                    "must be positive odd integers."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        self.validate()

        return {
            "protocol_version": (
                TRANSFORM_PROTOCOL_VERSION
            ),
            "height": self.height,
            "width": self.width,
            "horizontal_flip_probability": (
                self.horizontal_flip_probability
            ),
            "vertical_flip_probability": (
                self.vertical_flip_probability
            ),
            "affine_probability": (
                self.affine_probability
            ),
            "rotation_degrees": [
                self.rotation_min_degrees,
                self.rotation_max_degrees,
            ],
            "scale": [
                self.scale_min,
                self.scale_max,
            ],
            "translation_fraction": [
                self.translation_min_fraction,
                self.translation_max_fraction,
            ],
            "brightness_contrast_probability": (
                self.brightness_contrast_probability
            ),
            "hue_saturation_probability": (
                self.hue_saturation_probability
            ),
            "gaussian_noise_probability": (
                self.gaussian_noise_probability
            ),
            "gaussian_blur_probability": (
                self.gaussian_blur_probability
            ),
            "coarse_dropout_probability": (
                self.coarse_dropout_probability
            ),
            "image_mean": list(
                self.image_mean
            ),
            "image_std": list(
                self.image_std
            ),
        }


def _require_pil_image(
    value: object,
    name: str,
) -> Image.Image:
    """Require a PIL image."""

    if not isinstance(
        value,
        Image.Image,
    ):
        raise TypeError(
            f"{name} must be a PIL image, "
            f"received {type(value).__name__}."
        )

    return value


def _prepare_rgb_image(
    image: Image.Image,
) -> Image.Image:
    """Convert an input image to RGB."""

    return _require_pil_image(
        image,
        "image",
    ).convert(
        "RGB"
    )


def _prepare_binary_image(
    image: Image.Image,
    name: str,
) -> Image.Image:
    """Convert a binary target to 8-bit grayscale."""

    return _require_pil_image(
        image,
        name,
    ).convert(
        "L"
    )


def _prepare_sdm_tensor(
    sdm: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Convert an SDM to a [1, H, W] float tensor."""

    if isinstance(
        sdm,
        torch.Tensor,
    ):
        tensor = sdm.detach().clone().to(
            dtype=torch.float32
        )
    else:
        array = np.asarray(
            sdm,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "SDM contains non-finite values."
            )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                array
            )
        ).to(
            dtype=torch.float32
        )

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if (
        tensor.ndim != 3
        or tensor.shape[0] != 1
    ):
        raise ValueError(
            "SDM must have shape [H, W] or "
            f"[1, H, W], received "
            f"{tuple(tensor.shape)}."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            "SDM contains non-finite values."
        )

    return tensor


def _binary_pil_to_tensor(
    image: Image.Image,
) -> torch.Tensor:
    """Convert a binary PIL target to [1, H, W]."""

    tensor = TF.pil_to_tensor(
        image
    ).to(
        dtype=torch.float32
    )

    return (
        tensor >= 127.5
    ).to(
        dtype=torch.float32
    )


def _resize_sdm(
    sdm: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize an SDM with bilinear interpolation."""

    if tuple(
        sdm.shape[-2:]
    ) == (
        height,
        width,
    ):
        return sdm

    resized = TF.resize(
        sdm,
        size=[
            height,
            width,
        ],
        interpolation=(
            InterpolationMode.BILINEAR
        ),
        antialias=True,
    )

    return torch.clamp(
        resized,
        min=-1.0,
        max=1.0,
    )


def _random_uniform(
    minimum: float,
    maximum: float,
) -> float:
    """Sample a Python-random uniform value."""

    return random.uniform(
        float(minimum),
        float(maximum),
    )


class MultiTargetTransform:
    """Synchronized transform for image and all targets."""

    def __init__(
        self,
        config: TransformConfig,
        training: bool,
    ) -> None:
        config.validate()

        if not isinstance(
            training,
            bool,
        ):
            raise TypeError(
                "training must be Boolean."
            )

        self.config = config
        self.training = training

    def _resize_inputs(
        self,
        image: Image.Image,
        mask: Image.Image,
        contour: Image.Image,
        boundary_band: Image.Image,
        sdm: torch.Tensor,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        Image.Image,
        torch.Tensor,
    ]:
        """Resize all inputs to the locked resolution."""

        size = [
            self.config.height,
            self.config.width,
        ]

        image = TF.resize(
            image,
            size=size,
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )

        mask = TF.resize(
            mask,
            size=size,
            interpolation=(
                InterpolationMode.NEAREST
            ),
        )

        contour = TF.resize(
            contour,
            size=size,
            interpolation=(
                InterpolationMode.NEAREST
            ),
        )

        boundary_band = TF.resize(
            boundary_band,
            size=size,
            interpolation=(
                InterpolationMode.NEAREST
            ),
        )

        sdm = _resize_sdm(
            sdm,
            height=self.config.height,
            width=self.config.width,
        )

        return (
            image,
            mask,
            contour,
            boundary_band,
            sdm,
        )

    def _apply_horizontal_flip(
        self,
        image: Image.Image,
        mask: Image.Image,
        contour: Image.Image,
        boundary_band: Image.Image,
        sdm: torch.Tensor,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        Image.Image,
        torch.Tensor,
    ]:
        """Apply a synchronized horizontal flip."""

        return (
            TF.hflip(image),
            TF.hflip(mask),
            TF.hflip(contour),
            TF.hflip(boundary_band),
            torch.flip(
                sdm,
                dims=(-1,),
            ),
        )

    def _apply_vertical_flip(
        self,
        image: Image.Image,
        mask: Image.Image,
        contour: Image.Image,
        boundary_band: Image.Image,
        sdm: torch.Tensor,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        Image.Image,
        torch.Tensor,
    ]:
        """Apply a synchronized vertical flip."""

        return (
            TF.vflip(image),
            TF.vflip(mask),
            TF.vflip(contour),
            TF.vflip(boundary_band),
            torch.flip(
                sdm,
                dims=(-2,),
            ),
        )

    def _sample_affine_parameters(
        self,
    ) -> tuple[
        float,
        list[int],
        float,
        list[float],
    ]:
        """Sample synchronized affine parameters."""

        angle = _random_uniform(
            self.config.rotation_min_degrees,
            self.config.rotation_max_degrees,
        )

        scale = _random_uniform(
            self.config.scale_min,
            self.config.scale_max,
        )

        horizontal_fraction = (
            _random_uniform(
                self.config
                .translation_min_fraction,
                self.config
                .translation_max_fraction,
            )
        )

        vertical_fraction = (
            _random_uniform(
                self.config
                .translation_min_fraction,
                self.config
                .translation_max_fraction,
            )
        )

        translate = [
            int(
                round(
                    horizontal_fraction
                    * self.config.width
                )
            ),
            int(
                round(
                    vertical_fraction
                    * self.config.height
                )
            ),
        ]

        shear = [
            0.0,
            0.0,
        ]

        return (
            angle,
            translate,
            scale,
            shear,
        )

    def _apply_affine(
        self,
        image: Image.Image,
        mask: Image.Image,
        contour: Image.Image,
        boundary_band: Image.Image,
        sdm: torch.Tensor,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        Image.Image,
        torch.Tensor,
    ]:
        """Apply one synchronized affine transform."""

        (
            angle,
            translate,
            scale,
            shear,
        ) = self._sample_affine_parameters()

        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            fill=[
                0,
                0,
                0,
            ],
        )

        binary_arguments = {
            "angle": angle,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "interpolation": (
                InterpolationMode.NEAREST
            ),
            "fill": 0,
        }

        mask = TF.affine(
            mask,
            **binary_arguments,
        )

        contour = TF.affine(
            contour,
            **binary_arguments,
        )

        boundary_band = TF.affine(
            boundary_band,
            **binary_arguments,
        )

        sdm = TF.affine(
            sdm,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            fill=-1.0,
        )

        sdm = torch.clamp(
            sdm,
            min=-1.0,
            max=1.0,
        )

        return (
            image,
            mask,
            contour,
            boundary_band,
            sdm,
        )

    def _apply_brightness_contrast(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply random brightness and contrast."""

        brightness_factor = (
            1.0
            + _random_uniform(
                -self.config.brightness_limit,
                self.config.brightness_limit,
            )
        )

        contrast_factor = (
            1.0
            + _random_uniform(
                -self.config.contrast_limit,
                self.config.contrast_limit,
            )
        )

        image = TF.adjust_brightness(
            image,
            brightness_factor,
        )

        image = TF.adjust_contrast(
            image,
            contrast_factor,
        )

        return image

    def _apply_hue_saturation(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply random hue and saturation shifts."""

        hue_factor = _random_uniform(
            -self.config.hue_shift_limit,
            self.config.hue_shift_limit,
        )

        saturation_factor = (
            1.0
            + _random_uniform(
                -self.config
                .saturation_shift_limit,
                self.config
                .saturation_shift_limit,
            )
        )

        image = TF.adjust_hue(
            image,
            hue_factor,
        )

        image = TF.adjust_saturation(
            image,
            saturation_factor,
        )

        return image

    def _apply_gaussian_blur(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply Gaussian blur using an odd kernel."""

        kernel_size = random.choice(
            self.config
            .gaussian_blur_kernel_sizes
        )

        radius = max(
            0.1,
            (
                float(kernel_size)
                - 1.0
            )
            / 2.0,
        )

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )

    def _apply_gaussian_noise(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply zero-mean Gaussian noise."""

        standard_deviation = (
            _random_uniform(
                self.config
                .gaussian_noise_std_min,
                self.config
                .gaussian_noise_std_max,
            )
        )

        if standard_deviation == 0.0:
            return image_tensor

        noise = torch.randn_like(
            image_tensor
        ) * float(
            standard_deviation
        )

        return torch.clamp(
            image_tensor + noise,
            min=0.0,
            max=1.0,
        )

    def _apply_coarse_dropout(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply image-only rectangular dropout."""

        output = image_tensor.clone()

        hole_count = random.randint(
            self.config
            .coarse_dropout_holes_min,
            self.config
            .coarse_dropout_holes_max,
        )

        height = int(
            output.shape[-2]
        )

        width = int(
            output.shape[-1]
        )

        channel_fill = torch.tensor(
            self.config.image_mean,
            dtype=output.dtype,
            device=output.device,
        ).view(
            3,
            1,
            1,
        )

        for _ in range(
            hole_count
        ):
            size_fraction = (
                _random_uniform(
                    self.config
                    .coarse_dropout_size_fraction_min,
                    self.config
                    .coarse_dropout_size_fraction_max,
                )
            )

            hole_height = max(
                1,
                int(
                    round(
                        size_fraction
                        * height
                    )
                ),
            )

            hole_width = max(
                1,
                int(
                    round(
                        size_fraction
                        * width
                    )
                ),
            )

            top = random.randint(
                0,
                max(
                    0,
                    height - hole_height,
                ),
            )

            left = random.randint(
                0,
                max(
                    0,
                    width - hole_width,
                ),
            )

            output[
                :,
                top:
                top + hole_height,
                left:
                left + hole_width,
            ] = channel_fill

        return output

    def __call__(
        self,
        sample: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Transform one multi-target sample."""

        required_keys = {
            "image",
            "mask",
            "contour",
            "boundary_band",
            "sdm",
        }

        missing = sorted(
            required_keys
            - set(sample)
        )

        if missing:
            raise KeyError(
                "Transform sample is missing "
                f"keys: {missing}."
            )

        image = _prepare_rgb_image(
            sample["image"]
        )

        mask = _prepare_binary_image(
            sample["mask"],
            "mask",
        )

        contour = _prepare_binary_image(
            sample["contour"],
            "contour",
        )

        boundary_band = (
            _prepare_binary_image(
                sample["boundary_band"],
                "boundary_band",
            )
        )

        sdm = _prepare_sdm_tensor(
            sample["sdm"]
        )

        (
            image,
            mask,
            contour,
            boundary_band,
            sdm,
        ) = self._resize_inputs(
            image,
            mask,
            contour,
            boundary_band,
            sdm,
        )

        if self.training:
            if (
                random.random()
                < self.config
                .horizontal_flip_probability
            ):
                (
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                ) = self._apply_horizontal_flip(
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                )

            if (
                random.random()
                < self.config
                .vertical_flip_probability
            ):
                (
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                ) = self._apply_vertical_flip(
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                )

            if (
                random.random()
                < self.config
                .affine_probability
            ):
                (
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                ) = self._apply_affine(
                    image,
                    mask,
                    contour,
                    boundary_band,
                    sdm,
                )

            if (
                random.random()
                < self.config
                .brightness_contrast_probability
            ):
                image = (
                    self._apply_brightness_contrast(
                        image
                    )
                )

            if (
                random.random()
                < self.config
                .hue_saturation_probability
            ):
                image = (
                    self._apply_hue_saturation(
                        image
                    )
                )

            if (
                random.random()
                < self.config
                .gaussian_blur_probability
            ):
                image = (
                    self._apply_gaussian_blur(
                        image
                    )
                )

        image_tensor = TF.to_tensor(
            image
        ).to(
            dtype=torch.float32
        )

        if (
            self.training
            and random.random()
            < self.config
            .gaussian_noise_probability
        ):
            image_tensor = (
                self._apply_gaussian_noise(
                    image_tensor
                )
            )

        if (
            self.training
            and random.random()
            < self.config
            .coarse_dropout_probability
        ):
            image_tensor = (
                self._apply_coarse_dropout(
                    image_tensor
                )
            )

        image_tensor = TF.normalize(
            image_tensor,
            mean=list(
                self.config.image_mean
            ),
            std=list(
                self.config.image_std
            ),
        )

        mask_tensor = (
            _binary_pil_to_tensor(
                mask
            )
        )

        contour_tensor = (
            _binary_pil_to_tensor(
                contour
            )
        )

        boundary_band_tensor = (
            _binary_pil_to_tensor(
                boundary_band
            )
        )

        sdm = torch.clamp(
            sdm.to(
                dtype=torch.float32
            ),
            min=-1.0,
            max=1.0,
        )

        output = dict(
            sample
        )

        output.update(
            {
                "image": (
                    image_tensor.contiguous()
                ),
                "mask": (
                    mask_tensor.contiguous()
                ),
                "contour": (
                    contour_tensor.contiguous()
                ),
                "boundary_band": (
                    boundary_band_tensor
                    .contiguous()
                ),
                "sdm": (
                    sdm.contiguous()
                ),
            }
        )

        self._validate_output(
            output
        )

        return output

    def _validate_output(
        self,
        sample: Mapping[str, Any],
    ) -> None:
        """Validate transformed tensor invariants."""

        expected_spatial_shape = (
            self.config.height,
            self.config.width,
        )

        image = sample["image"]

        if (
            not isinstance(
                image,
                torch.Tensor,
            )
            or tuple(
                image.shape
            )
            != (
                3,
                *expected_spatial_shape,
            )
        ):
            raise RuntimeError(
                "Transformed image has invalid "
                f"shape: {getattr(image, 'shape', None)}."
            )

        for name in [
            "mask",
            "contour",
            "boundary_band",
            "sdm",
        ]:
            tensor = sample[name]

            if (
                not isinstance(
                    tensor,
                    torch.Tensor,
                )
                or tuple(
                    tensor.shape
                )
                != (
                    1,
                    *expected_spatial_shape,
                )
            ):
                raise RuntimeError(
                    f"Transformed {name} has "
                    "invalid shape: "
                    f"{getattr(tensor, 'shape', None)}."
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise RuntimeError(
                    f"Transformed {name} contains "
                    "non-finite values."
                )

        for name in [
            "mask",
            "contour",
            "boundary_band",
        ]:
            values = torch.unique(
                sample[name]
            )

            if not all(
                float(value)
                in {
                    0.0,
                    1.0,
                }
                for value in values
            ):
                raise RuntimeError(
                    f"Transformed {name} is not "
                    "binary."
                )

        sdm = sample["sdm"]

        if (
            float(
                torch.min(sdm)
            )
            < -1.000001
            or float(
                torch.max(sdm)
            )
            > 1.000001
        ):
            raise RuntimeError(
                "Transformed SDM is outside "
                "the range [-1, 1]."
            )


def build_transform_config(
    experiment_payload: Mapping[str, Any],
) -> TransformConfig:
    """Build TransformConfig from the validated experiment YAML."""

    data = experiment_payload[
        "data"
    ]

    augmentation = experiment_payload[
        "augmentation"
    ]

    image_size = data[
        "image_size"
    ]

    normalization = (
        data[
            "preprocessing"
        ][
            "normalization"
        ]
    )

    geometric = augmentation[
        "geometric"
    ]

    photometric = augmentation[
        "photometric"
    ]

    regularization = augmentation[
        "regularization"
    ]

    affine = geometric[
        "affine"
    ]

    noise = photometric[
        "gaussian_noise"
    ]

    blur = photometric[
        "gaussian_blur"
    ]

    dropout = regularization[
        "coarse_dropout"
    ]

    config = TransformConfig(
        height=int(
            image_size["height"]
        ),
        width=int(
            image_size["width"]
        ),
        horizontal_flip_probability=(
            float(
                geometric[
                    "horizontal_flip"
                ][
                    "probability"
                ]
            )
            if geometric[
                "horizontal_flip"
            ][
                "enabled"
            ]
            else 0.0
        ),
        vertical_flip_probability=(
            float(
                geometric[
                    "vertical_flip"
                ][
                    "probability"
                ]
            )
            if geometric[
                "vertical_flip"
            ][
                "enabled"
            ]
            else 0.0
        ),
        affine_probability=(
            float(
                affine["probability"]
            )
            if affine["enabled"]
            else 0.0
        ),
        rotation_min_degrees=float(
            affine[
                "rotation_degrees"
            ][
                "minimum"
            ]
        ),
        rotation_max_degrees=float(
            affine[
                "rotation_degrees"
            ][
                "maximum"
            ]
        ),
        scale_min=float(
            affine["scale"][
                "minimum"
            ]
        ),
        scale_max=float(
            affine["scale"][
                "maximum"
            ]
        ),
        translation_min_fraction=float(
            affine[
                "translation_fraction"
            ][
                "minimum"
            ]
        ),
        translation_max_fraction=float(
            affine[
                "translation_fraction"
            ][
                "maximum"
            ]
        ),
        brightness_contrast_probability=(
            float(
                photometric[
                    "brightness_contrast"
                ][
                    "probability"
                ]
            )
            if photometric[
                "brightness_contrast"
            ][
                "enabled"
            ]
            else 0.0
        ),
        brightness_limit=float(
            photometric[
                "brightness_contrast"
            ][
                "brightness_limit"
            ]
        ),
        contrast_limit=float(
            photometric[
                "brightness_contrast"
            ][
                "contrast_limit"
            ]
        ),
        hue_saturation_probability=(
            float(
                photometric[
                    "hue_saturation"
                ][
                    "probability"
                ]
            )
            if photometric[
                "hue_saturation"
            ][
                "enabled"
            ]
            else 0.0
        ),
        hue_shift_limit=float(
            photometric[
                "hue_saturation"
            ][
                "hue_shift_limit"
            ]
        ),
        saturation_shift_limit=float(
            photometric[
                "hue_saturation"
            ][
                "saturation_shift_limit"
            ]
        ),
        gaussian_noise_probability=(
            float(
                noise["probability"]
            )
            if noise["enabled"]
            else 0.0
        ),
        gaussian_noise_std_min=float(
            noise[
                "standard_deviation"
            ][
                "minimum"
            ]
        ),
        gaussian_noise_std_max=float(
            noise[
                "standard_deviation"
            ][
                "maximum"
            ]
        ),
        gaussian_blur_probability=(
            float(
                blur["probability"]
            )
            if blur["enabled"]
            else 0.0
        ),
        gaussian_blur_kernel_sizes=tuple(
            int(value)
            for value in blur[
                "kernel_sizes"
            ]
        ),
        coarse_dropout_probability=(
            float(
                dropout["probability"]
            )
            if dropout["enabled"]
            else 0.0
        ),
        coarse_dropout_holes_min=int(
            dropout["holes"][
                "minimum"
            ]
        ),
        coarse_dropout_holes_max=int(
            dropout["holes"][
                "maximum"
            ]
        ),
        coarse_dropout_size_fraction_min=float(
            dropout[
                "hole_size_fraction"
            ][
                "minimum"
            ]
        ),
        coarse_dropout_size_fraction_max=float(
            dropout[
                "hole_size_fraction"
            ][
                "maximum"
            ]
        ),
        image_mean=tuple(
            float(value)
            for value in normalization[
                "mean"
            ]
        ),
        image_std=tuple(
            float(value)
            for value in normalization[
                "std"
            ]
        ),
    )

    config.validate()

    return config


def build_transforms(
    experiment_payload: Mapping[str, Any],
) -> dict[str, MultiTargetTransform]:
    """Build train and deterministic evaluation transforms."""

    transform_config = (
        build_transform_config(
            experiment_payload
        )
    )

    return {
        "train": MultiTargetTransform(
            transform_config,
            training=True,
        ),
        "validation": (
            MultiTargetTransform(
                transform_config,
                training=False,
            )
        ),
        "internal_test": (
            MultiTargetTransform(
                transform_config,
                training=False,
            )
        ),
        "external": (
            MultiTargetTransform(
                transform_config,
                training=False,
            )
        ),
    }


def run_transform_self_test() -> dict[str, Any]:
    """Run deterministic shape and synchronization checks."""

    height = 96
    width = 96

    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    binary = (
        (
            xx - 48
        ) ** 2
        + (
            yy - 48
        ) ** 2
        <= 24**2
    )

    contour = np.logical_xor(
        binary,
        (
            (
                xx - 48
            ) ** 2
            + (
                yy - 48
            ) ** 2
            <= 22**2
        ),
    )

    boundary_band = (
        (
            xx - 48
        ) ** 2
        + (
            yy - 48
        ) ** 2
        <= 28**2
    ) & (
        (
            xx - 48
        ) ** 2
        + (
            yy - 48
        ) ** 2
        >= 19**2
    )

    image_array = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    image_array[
        binary
    ] = [
        180,
        90,
        60,
    ]

    approximate_sdm = np.where(
        binary,
        0.5,
        -0.5,
    ).astype(
        np.float32
    )

    sample = {
        "image": Image.fromarray(
            image_array,
            mode="RGB",
        ),
        "mask": Image.fromarray(
            binary.astype(
                np.uint8
            ) * 255,
            mode="L",
        ),
        "contour": Image.fromarray(
            contour.astype(
                np.uint8
            ) * 255,
            mode="L",
        ),
        "boundary_band": Image.fromarray(
            boundary_band.astype(
                np.uint8
            ) * 255,
            mode="L",
        ),
        "sdm": approximate_sdm,
        "image_id": "SELF_TEST",
    }

    evaluation_config = TransformConfig(
        height=128,
        width=128,
    )

    evaluation_transform = (
        MultiTargetTransform(
            evaluation_config,
            training=False,
        )
    )

    first = evaluation_transform(
        sample
    )

    second = evaluation_transform(
        sample
    )

    checks = {
        "image_shape": (
            tuple(
                first["image"].shape
            )
            == (
                3,
                128,
                128,
            )
        ),
        "mask_shape": (
            tuple(
                first["mask"].shape
            )
            == (
                1,
                128,
                128,
            )
        ),
        "contour_shape": (
            tuple(
                first["contour"].shape
            )
            == (
                1,
                128,
                128,
            )
        ),
        "boundary_band_shape": (
            tuple(
                first[
                    "boundary_band"
                ].shape
            )
            == (
                1,
                128,
                128,
            )
        ),
        "sdm_shape": (
            tuple(
                first["sdm"].shape
            )
            == (
                1,
                128,
                128,
            )
        ),
        "evaluation_image_deterministic": (
            torch.equal(
                first["image"],
                second["image"],
            )
        ),
        "evaluation_mask_deterministic": (
            torch.equal(
                first["mask"],
                second["mask"],
            )
        ),
        "evaluation_sdm_deterministic": (
            torch.equal(
                first["sdm"],
                second["sdm"],
            )
        ),
        "binary_mask_preserved": (
            set(
                float(value)
                for value in torch.unique(
                    first["mask"]
                )
            )
            <= {
                0.0,
                1.0,
            }
        ),
        "sdm_bounded": (
            float(
                first["sdm"].min()
            )
            >= -1.0
            and float(
                first["sdm"].max()
            )
            <= 1.0
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
            TRANSFORM_PROTOCOL_VERSION
        ),
        "checks": checks,
    }