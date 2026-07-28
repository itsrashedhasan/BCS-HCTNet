"""Geometry utilities for BCS-HCTNet derived segmentation targets.

The project generates all supervision targets after nearest-neighbour resizing
of the binary ground-truth mask to the fixed training resolution. This avoids
interpolating an already-computed signed-distance map.

Primary protocol defaults
-------------------------
- output size: 352 x 352
- contour width: 2 pixels
- boundary-band radius: 3 pixels
- signed-distance clipping distance: 20 pixels

All values are configurable and must be recorded in the generated artifact
report. The signed-distance convention is positive inside the lesion and
negative outside the lesion.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)


TARGET_PROTOCOL_VERSION = (
    "BCS-HCTNet-target-geometry-v1"
)


@dataclass(frozen=True)
class TargetGeometryConfig:
    """Configuration for mask-derived target generation."""

    output_size: int = 352
    contour_width: int = 2
    boundary_band_radius: int = 3
    sdm_clip_distance: float = 20.0
    connectivity: int = 8

    def validate(self) -> None:
        """Reject unsupported or ambiguous settings."""

        if self.output_size < 32:
            raise ValueError(
                "output_size must be at least "
                f"32 pixels, received {self.output_size}."
            )

        if self.contour_width not in {
            1,
            2,
            3,
        }:
            raise ValueError(
                "contour_width must be one of "
                "{1, 2, 3}, received "
                f"{self.contour_width}."
            )

        if self.boundary_band_radius < 0:
            raise ValueError(
                "boundary_band_radius must be "
                "non-negative, received "
                f"{self.boundary_band_radius}."
            )

        if self.boundary_band_radius > 32:
            raise ValueError(
                "boundary_band_radius above "
                "32 pixels is not supported."
            )

        if not np.isfinite(
            self.sdm_clip_distance
        ):
            raise ValueError(
                "sdm_clip_distance must be finite."
            )

        if self.sdm_clip_distance <= 0:
            raise ValueError(
                "sdm_clip_distance must be positive, "
                f"received {self.sdm_clip_distance}."
            )

        if self.connectivity not in {
            4,
            8,
        }:
            raise ValueError(
                "connectivity must be 4 or 8, "
                f"received {self.connectivity}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""

        self.validate()

        return {
            "protocol_version": (
                TARGET_PROTOCOL_VERSION
            ),
            **asdict(self),
            "sdm_sign_convention": (
                "positive_inside_negative_outside"
            ),
            "mask_resize_interpolation": (
                "nearest"
            ),
            "sdm_generation_resolution": (
                "generated_after_mask_resize"
            ),
        }

    def fingerprint(self) -> str:
        """Return a stable configuration hash."""

        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()


def _as_mask_array(
    mask: np.ndarray | Image.Image,
) -> np.ndarray:
    """Convert an input mask to a 2D array."""

    if isinstance(mask, Image.Image):
        array = np.asarray(
            mask.convert("L")
        )
    else:
        array = np.asarray(mask)

    if array.ndim == 3:
        if array.shape[2] == 0:
            raise ValueError(
                "Mask has an empty channel dimension."
            )

        array = np.max(
            array,
            axis=2,
        )

    if array.ndim != 2:
        raise ValueError(
            "Mask must be two-dimensional after "
            "channel reduction, received shape "
            f"{array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            "Mask array is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Mask contains non-finite values."
        )

    return array


def to_binary_mask(
    mask: np.ndarray | Image.Image,
) -> np.ndarray:
    """Convert a mask to a Boolean lesion mask."""

    array = _as_mask_array(mask)

    maximum = float(
        np.max(array)
    )

    minimum = float(
        np.min(array)
    )

    if (
        maximum <= 1.0
        and minimum >= 0.0
    ):
        binary = array > 0.5
    else:
        binary = array >= 127.5

    binary = np.asarray(
        binary,
        dtype=bool,
    )

    if not np.any(binary):
        raise ValueError(
            "Mask contains no lesion foreground."
        )

    return binary


def resize_binary_mask(
    mask: np.ndarray | Image.Image,
    output_size: int,
) -> np.ndarray:
    """Resize a mask using nearest interpolation."""

    if output_size < 1:
        raise ValueError(
            "output_size must be positive."
        )

    binary = to_binary_mask(mask)

    image = Image.fromarray(
        binary.astype(np.uint8) * 255,
        mode="L",
    )

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    )

    resized = image.resize(
        (
            output_size,
            output_size,
        ),
        resample=resampling.NEAREST,
    )

    resized_binary = np.asarray(
        resized,
        dtype=np.uint8,
    ) > 0

    if not np.any(resized_binary):
        raise ValueError(
            "Nearest-neighbour resizing removed "
            "all lesion foreground."
        )

    return resized_binary


def connectivity_structure(
    connectivity: int,
) -> np.ndarray:
    """Return a 3x3 morphology structure."""

    if connectivity == 4:
        return np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )

    if connectivity == 8:
        return np.ones(
            (
                3,
                3,
            ),
            dtype=bool,
        )

    raise ValueError(
        "connectivity must be 4 or 8."
    )


def disk_structure(
    radius: int,
) -> np.ndarray:
    """Return a filled Euclidean disk."""

    if radius < 0:
        raise ValueError(
            "radius must be non-negative."
        )

    if radius == 0:
        return np.ones(
            (
                1,
                1,
            ),
            dtype=bool,
        )

    coordinates = np.arange(
        -radius,
        radius + 1,
    )

    yy, xx = np.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    return (
        xx * xx + yy * yy
        <= radius * radius
    )


def generate_contour_target(
    binary_mask: np.ndarray,
    contour_width: int = 2,
    connectivity: int = 8,
) -> np.ndarray:
    """Generate a controlled-width contour target.

    Width semantics:

    - width 1: one inward boundary layer;
    - width 2: one outward and one inward layer;
    - width 3: one outward and two inward layers.

    At the image border, pixels outside the image cannot
    be represented, so the available in-frame contour is
    retained.
    """

    mask = np.asarray(
        binary_mask,
        dtype=bool,
    )

    if mask.ndim != 2:
        raise ValueError(
            "binary_mask must be two-dimensional."
        )

    if not np.any(mask):
        raise ValueError(
            "Cannot generate a contour from "
            "an empty mask."
        )

    if contour_width not in {
        1,
        2,
        3,
    }:
        raise ValueError(
            "contour_width must be one of "
            "{1, 2, 3}."
        )

    structure = connectivity_structure(
        connectivity
    )

    outer_iterations = (
        contour_width // 2
    )

    inner_iterations = (
        contour_width
        - outer_iterations
    )

    if outer_iterations:
        outer = binary_dilation(
            mask,
            structure=structure,
            iterations=outer_iterations,
            border_value=0,
        )
    else:
        outer = mask.copy()

    inner = binary_erosion(
        mask,
        structure=structure,
        iterations=inner_iterations,
        border_value=0,
    )

    contour = np.logical_xor(
        outer,
        inner,
    )

    if not np.any(contour):
        raise RuntimeError(
            "Generated contour is empty."
        )

    return contour


def generate_boundary_band_target(
    contour: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """Dilate a contour into a boundary band."""

    contour_binary = np.asarray(
        contour,
        dtype=bool,
    )

    if contour_binary.ndim != 2:
        raise ValueError(
            "contour must be two-dimensional."
        )

    if not np.any(contour_binary):
        raise ValueError(
            "Cannot generate a boundary band "
            "from an empty contour."
        )

    if radius == 0:
        return contour_binary.copy()

    band = binary_dilation(
        contour_binary,
        structure=disk_structure(
            radius
        ),
        iterations=1,
        border_value=0,
    )

    return np.asarray(
        band,
        dtype=bool,
    )


def generate_signed_distance_map(
    binary_mask: np.ndarray,
    clip_distance: float = 20.0,
) -> np.ndarray:
    """Generate a normalized signed distance map.

    Positive values are inside the lesion.
    Negative values are outside the lesion.

    A one-pixel background frame is added before
    computing the distance transforms. This ensures
    that a full-foreground mask has a valid distance
    to the image boundary.
    """

    mask = np.asarray(
        binary_mask,
        dtype=bool,
    )

    if mask.ndim != 2:
        raise ValueError(
            "binary_mask must be two-dimensional."
        )

    if not np.any(mask):
        raise ValueError(
            "Cannot generate an SDM from "
            "an empty mask."
        )

    if not np.isfinite(
        clip_distance
    ):
        raise ValueError(
            "clip_distance must be finite."
        )

    if clip_distance <= 0:
        raise ValueError(
            "clip_distance must be positive."
        )

    padded = np.pad(
        mask,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    inside_distance = (
        distance_transform_edt(
            padded
        )
    )

    outside_distance = (
        distance_transform_edt(
            np.logical_not(
                padded
            )
        )
    )

    signed_distance = (
        inside_distance
        - outside_distance
    )[
        1:-1,
        1:-1,
    ]

    signed_distance = np.clip(
        signed_distance,
        -clip_distance,
        clip_distance,
    )

    normalized = (
        signed_distance
        / float(
            clip_distance
        )
    ).astype(
        np.float32,
        copy=False,
    )

    return normalized


def generate_target_bundle(
    mask: np.ndarray | Image.Image,
    config: TargetGeometryConfig | None = None,
) -> dict[str, np.ndarray]:
    """Resize a mask and generate all targets."""

    active_config = (
        config
        if config is not None
        else TargetGeometryConfig()
    )

    active_config.validate()

    resized_mask = resize_binary_mask(
        mask,
        active_config.output_size,
    )

    contour = generate_contour_target(
        resized_mask,
        contour_width=(
            active_config.contour_width
        ),
        connectivity=(
            active_config.connectivity
        ),
    )

    boundary_band = (
        generate_boundary_band_target(
            contour,
            radius=(
                active_config
                .boundary_band_radius
            ),
        )
    )

    sdm = generate_signed_distance_map(
        resized_mask,
        clip_distance=(
            active_config
            .sdm_clip_distance
        ),
    )

    bundle = {
        "mask": (
            resized_mask.astype(
                np.uint8
            )
            * 255
        ),
        "contour": (
            contour.astype(
                np.uint8
            )
            * 255
        ),
        "boundary_band": (
            boundary_band.astype(
                np.uint8
            )
            * 255
        ),
        "sdm": sdm,
    }

    validate_target_bundle(
        bundle,
        active_config,
    )

    return bundle


def validate_target_bundle(
    bundle: Mapping[
        str,
        np.ndarray,
    ],
    config: TargetGeometryConfig,
) -> None:
    """Validate target shape and geometry."""

    config.validate()

    required = {
        "mask",
        "contour",
        "boundary_band",
        "sdm",
    }

    missing = sorted(
        required - set(bundle)
    )

    if missing:
        raise ValueError(
            "Target bundle is missing keys: "
            f"{missing}."
        )

    expected_shape = (
        config.output_size,
        config.output_size,
    )

    for key in [
        "mask",
        "contour",
        "boundary_band",
        "sdm",
    ]:
        array = np.asarray(
            bundle[key]
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"{key} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

    for key in [
        "mask",
        "contour",
        "boundary_band",
    ]:
        values = set(
            np.unique(
                np.asarray(
                    bundle[key]
                )
            ).tolist()
        )

        if not values <= {
            0,
            255,
        }:
            raise ValueError(
                f"{key} is not binary 0/255; "
                "found values "
                f"{sorted(values)[:20]}."
            )

    mask = np.asarray(
        bundle["mask"]
    ) > 0

    contour = np.asarray(
        bundle["contour"]
    ) > 0

    boundary_band = np.asarray(
        bundle["boundary_band"]
    ) > 0

    sdm = np.asarray(
        bundle["sdm"]
    )

    if sdm.dtype != np.float32:
        raise ValueError(
            "sdm must be float32, "
            f"received {sdm.dtype}."
        )

    if not np.all(
        np.isfinite(sdm)
    ):
        raise ValueError(
            "sdm contains non-finite values."
        )

    tolerance = 1e-6

    if (
        float(
            np.min(sdm)
        )
        < -1.0 - tolerance
    ):
        raise ValueError(
            "sdm contains values below -1."
        )

    if (
        float(
            np.max(sdm)
        )
        > 1.0 + tolerance
    ):
        raise ValueError(
            "sdm contains values above 1."
        )

    if not np.any(mask):
        raise ValueError(
            "Resized mask contains "
            "no foreground."
        )

    if not np.any(contour):
        raise ValueError(
            "Contour target contains "
            "no foreground."
        )

    contour_is_inside_band = np.all(
        np.logical_or(
            np.logical_not(
                contour
            ),
            boundary_band,
        )
    )

    if not contour_is_inside_band:
        raise ValueError(
            "Boundary band does not contain "
            "every contour pixel."
        )

    if not np.all(
        sdm[mask] > 0
    ):
        raise ValueError(
            "SDM sign invariant failed "
            "inside the lesion."
        )

    background = np.logical_not(
        mask
    )

    if (
        np.any(background)
        and not np.all(
            sdm[background] < 0
        )
    ):
        raise ValueError(
            "SDM sign invariant failed "
            "outside the lesion."
        )


def target_statistics(
    bundle: Mapping[
        str,
        np.ndarray,
    ],
    config: TargetGeometryConfig,
) -> dict[str, Any]:
    """Return numerical QC statistics."""

    validate_target_bundle(
        bundle,
        config,
    )

    mask = np.asarray(
        bundle["mask"]
    ) > 0

    contour = np.asarray(
        bundle["contour"]
    ) > 0

    boundary_band = np.asarray(
        bundle["boundary_band"]
    ) > 0

    sdm = np.asarray(
        bundle["sdm"],
        dtype=np.float32,
    )

    _, component_count = label(
        mask,
        structure=(
            connectivity_structure(
                config.connectivity
            ).astype(
                np.uint8
            )
        ),
    )

    background = np.logical_not(
        mask
    )

    return {
        "target_protocol_version": (
            TARGET_PROTOCOL_VERSION
        ),
        "target_config_sha256": (
            config.fingerprint()
        ),
        "height": int(
            mask.shape[0]
        ),
        "width": int(
            mask.shape[1]
        ),
        "mask_foreground_pixels": int(
            np.count_nonzero(
                mask
            )
        ),
        "mask_foreground_ratio": float(
            np.mean(mask)
        ),
        "mask_is_full_foreground": bool(
            np.all(mask)
        ),
        "connected_component_count": int(
            component_count
        ),
        "contour_pixels": int(
            np.count_nonzero(
                contour
            )
        ),
        "contour_ratio": float(
            np.mean(contour)
        ),
        "boundary_band_pixels": int(
            np.count_nonzero(
                boundary_band
            )
        ),
        "boundary_band_ratio": float(
            np.mean(
                boundary_band
            )
        ),
        "sdm_min": float(
            np.min(sdm)
        ),
        "sdm_max": float(
            np.max(sdm)
        ),
        "sdm_mean": float(
            np.mean(sdm)
        ),
        "sdm_inside_mean": float(
            np.mean(
                sdm[mask]
            )
        ),
        "sdm_outside_mean": (
            float(
                np.mean(
                    sdm[
                        background
                    ]
                )
            )
            if np.any(
                background
            )
            else None
        ),
    }


def save_binary_png(
    array: np.ndarray,
    output_path: Path,
) -> None:
    """Atomically save a binary PNG."""

    values = np.asarray(
        array
    )

    unique_values = set(
        np.unique(
            values
        ).tolist()
    )

    if not unique_values <= {
        0,
        255,
    }:
        raise ValueError(
            "Binary PNG input must contain "
            "only 0 and 255."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".png",
        prefix=(
            f".{output_path.stem}."
        ),
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

    try:
        Image.fromarray(
            values.astype(
                np.uint8
            ),
            mode="L",
        ).save(
            temporary_path,
            format="PNG",
            optimize=True,
        )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_float32_npy(
    array: np.ndarray,
    output_path: Path,
) -> None:
    """Atomically save a float32 NPY file."""

    values = np.asarray(
        array,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Cannot save a float array "
            "containing non-finite values."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".npy",
        prefix=(
            f".{output_path.stem}."
        ),
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

        np.save(
            temporary_file,
            values,
            allow_pickle=False,
        )

        temporary_file.flush()

        os.fsync(
            temporary_file.fileno()
        )

    try:
        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_file(
    path: Path,
    chunk_size: int = (
        1024 * 1024
    ),
) -> str:
    """Return a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as input_file:
        while True:
            chunk = input_file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def run_geometry_self_test() -> dict[str, Any]:
    """Test ordinary, border-touching, and full masks."""

    config = (
        TargetGeometryConfig()
    )

    config.validate()

    size = 96

    yy, xx = np.ogrid[
        :size,
        :size,
    ]

    centered_circle = (
        (xx - 48) ** 2
        + (yy - 48) ** 2
        <= 24 ** 2
    )

    border_touching_circle = (
        (xx - 10) ** 2
        + (yy - 48) ** 2
        <= 24 ** 2
    )

    full_foreground = np.ones(
        (
            size,
            size,
        ),
        dtype=bool,
    )

    cases = {
        "centered_circle": (
            centered_circle
        ),
        "border_touching_circle": (
            border_touching_circle
        ),
        "full_foreground": (
            full_foreground
        ),
    }

    results: dict[
        str,
        Any,
    ] = {}

    for name, mask in cases.items():
        bundle = (
            generate_target_bundle(
                mask,
                config,
            )
        )

        results[name] = (
            target_statistics(
                bundle,
                config,
            )
        )

    return {
        "status": "passed",
        "protocol_version": (
            TARGET_PROTOCOL_VERSION
        ),
        "config": config.to_dict(),
        "cases": results,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run_geometry_self_test(),
            indent=2,
        )
    )