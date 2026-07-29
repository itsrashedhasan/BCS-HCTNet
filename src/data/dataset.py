"""Manifest-driven PyTorch dataset for BCS-HCTNet.

The dataset joins an approved split manifest with the Step 05A target
manifest using only ``image_id``.

Each returned sample contains:

- normalized RGB image;
- binary lesion mask;
- binary contour target;
- binary boundary-band target;
- normalized signed-distance map;
- sample identifier and provenance fields.

Path resolution is designed for persistent Kaggle artifacts. Absolute paths
recorded during earlier Kaggle sessions may no longer exist, so validated
relative paths are resolved against explicitly supplied persistent roots.
"""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import (
    MultiTargetTransform,
    TransformConfig,
)


DATASET_PROTOCOL_VERSION = (
    "BCS-HCTNet-manifest-dataset-v1"
)

LOCKED_JOIN_KEY = "image_id"

DEFAULT_TARGET_HEIGHT = 352
DEFAULT_TARGET_WIDTH = 352
DEFAULT_SDM_CLIP_DISTANCE_PIXELS = 20.0


@dataclass(frozen=True)
class DatasetRecord:
    """One fully resolved dataset sample."""

    index: int
    image_id: str
    split: str

    image_path: Path
    mask_path: Path
    contour_path: Path
    boundary_band_path: Path
    sdm_path: Path

    mask_foreground_ratio: float
    mask_is_full_foreground: bool
    connected_component_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "index": self.index,
            "image_id": self.image_id,
            "split": self.split,
            "image_path": str(
                self.image_path
            ),
            "mask_path": str(
                self.mask_path
            ),
            "contour_path": str(
                self.contour_path
            ),
            "boundary_band_path": str(
                self.boundary_band_path
            ),
            "sdm_path": str(
                self.sdm_path
            ),
            "mask_foreground_ratio": (
                self.mask_foreground_ratio
            ),
            "mask_is_full_foreground": (
                self.mask_is_full_foreground
            ),
            "connected_component_count": (
                self.connected_component_count
            ),
        }


def _normalized_text(
    value: object,
) -> str:
    """Normalize a CSV value."""

    if value is None:
        return ""

    return str(value).strip()


def _parse_optional_float(
    value: object,
    default: float = float("nan"),
) -> float:
    """Parse an optional finite floating-point value."""

    text = _normalized_text(
        value
    )

    if not text:
        return default

    try:
        parsed = float(text)

    except ValueError as error:
        raise ValueError(
            f"Cannot parse floating-point "
            f"value {value!r}."
        ) from error

    if not np.isfinite(parsed):
        raise ValueError(
            f"Numeric value is not finite: "
            f"{value!r}."
        )

    return parsed


def _parse_optional_integer(
    value: object,
    default: int = -1,
) -> int:
    """Parse an optional integer value."""

    text = _normalized_text(
        value
    )

    if not text:
        return default

    try:
        return int(
            float(text)
        )

    except ValueError as error:
        raise ValueError(
            f"Cannot parse integer value "
            f"{value!r}."
        ) from error


def _parse_optional_boolean(
    value: object,
    default: bool = False,
) -> bool:
    """Parse a Boolean-like CSV value."""

    text = _normalized_text(
        value
    ).lower()

    if not text:
        return default

    if text in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"Cannot parse Boolean value "
        f"{value!r}."
    )


def read_manifest(
    path: str | Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    """Read a UTF-8 CSV manifest."""

    manifest_path = Path(
        path
    ).expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"Manifest has no header: "
                f"{manifest_path}"
            )

        fieldnames = [
            str(name)
            for name in reader.fieldnames
        ]

        rows = [
            {
                str(key): (
                    ""
                    if value is None
                    else str(value)
                )
                for key, value in row.items()
            }
            for row in reader
        ]

    if not rows:
        raise RuntimeError(
            f"Manifest is empty: "
            f"{manifest_path}"
        )

    return fieldnames, rows


def build_unique_index(
    rows: Sequence[Mapping[str, str]],
    key: str,
    context: str,
) -> dict[str, Mapping[str, str]]:
    """Build a unique row index."""

    index: dict[
        str,
        Mapping[str, str],
    ] = {}

    missing_rows: list[int] = []
    duplicate_values: list[str] = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        value = _normalized_text(
            row.get(
                key,
                "",
            )
        )

        if not value:
            missing_rows.append(
                row_number
            )

            continue

        if value in index:
            duplicate_values.append(
                value
            )

            continue

        index[value] = row

    if missing_rows:
        raise RuntimeError(
            f"{context} contains empty "
            f"{key!r} values at CSV rows "
            f"{missing_rows[:10]}."
        )

    if duplicate_values:
        unique_duplicates = sorted(
            set(
                duplicate_values
            )
        )

        raise RuntimeError(
            f"{context} contains duplicate "
            f"{key!r} values: "
            f"{unique_duplicates[:10]}."
        )

    return index


def _safe_relative_candidate(
    root: Path,
    relative_value: str,
    context: str,
) -> Path:
    """Resolve a relative path without allowing root escape."""

    root = root.expanduser().resolve()

    relative_path = Path(
        relative_value
    )

    if relative_path.is_absolute():
        raise ValueError(
            f"{context} must be relative, "
            f"received {relative_value!r}."
        )

    candidate = (
        root
        / relative_path
    ).resolve()

    try:
        candidate.relative_to(
            root
        )

    except ValueError as error:
        raise RuntimeError(
            f"{context} escapes its permitted "
            f"root: {relative_value!r}."
        ) from error

    return candidate


def resolve_existing_file(
    *,
    label: str,
    absolute_values: Sequence[object],
    relative_values: Sequence[object],
    roots: Sequence[Path],
) -> Path:
    """Resolve one file from approved path candidates."""

    candidates: list[Path] = []

    for value in absolute_values:
        text = _normalized_text(
            value
        )

        if not text:
            continue

        candidate = Path(
            text
        ).expanduser()

        if candidate.is_absolute():
            candidates.append(
                candidate.resolve()
            )

    for value in relative_values:
        text = _normalized_text(
            value
        )

        if not text:
            continue

        relative_path = Path(
            text
        )

        if relative_path.is_absolute():
            candidates.append(
                relative_path.resolve()
            )

            continue

        for root in roots:
            candidates.append(
                _safe_relative_candidate(
                    root=root,
                    relative_value=text,
                    context=label,
                )
            )

    unique_candidates: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = str(
            candidate
        )

        if key not in seen:
            seen.add(
                key
            )

            unique_candidates.append(
                candidate
            )

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    attempted = [
        str(path)
        for path in unique_candidates
    ]

    raise FileNotFoundError(
        f"Could not resolve {label}. "
        f"Attempted paths: {attempted}"
    )


def _require_target_path_fields(
    target_row: Mapping[str, str],
    image_id: str,
) -> None:
    """Require absolute or relative fields for every target."""

    target_pairs = {
        "mask": (
            "target_mask_path",
            "target_mask_relative_path",
        ),
        "contour": (
            "target_contour_path",
            "target_contour_relative_path",
        ),
        "boundary_band": (
            "target_boundary_band_path",
            "target_boundary_band_relative_path",
        ),
        "sdm": (
            "target_sdm_path",
            "target_sdm_relative_path",
        ),
    }

    for target_name, (
        absolute_column,
        relative_column,
    ) in target_pairs.items():
        absolute_value = _normalized_text(
            target_row.get(
                absolute_column,
                "",
            )
        )

        relative_value = _normalized_text(
            target_row.get(
                relative_column,
                "",
            )
        )

        if not (
            absolute_value
            or relative_value
        ):
            raise RuntimeError(
                f"{image_id}: target manifest "
                f"contains no {target_name} path."
            )


def _resolve_source_image(
    split_row: Mapping[str, str],
    target_row: Mapping[str, str],
    source_roots: Sequence[Path],
    image_id: str,
) -> Path:
    """Resolve a source image path."""

    return resolve_existing_file(
        label=(
            f"source image for {image_id}"
        ),
        absolute_values=[
            target_row.get(
                "source_image_path"
            ),
            split_row.get(
                "source_image_path"
            ),
            split_row.get(
                "image_path"
            ),
        ],
        relative_values=[
            target_row.get(
                "source_image_relative_path"
            ),
            split_row.get(
                "source_image_relative_path"
            ),
            split_row.get(
                "image_relative_path"
            ),
        ],
        roots=source_roots,
    )


def _resolve_target(
    target_row: Mapping[str, str],
    target_root: Path,
    image_id: str,
    target_name: str,
) -> Path:
    """Resolve one Step 05A target file."""

    column_prefixes = {
        "mask": "target_mask",
        "contour": "target_contour",
        "boundary_band": (
            "target_boundary_band"
        ),
        "sdm": "target_sdm",
    }

    if target_name not in column_prefixes:
        raise KeyError(
            f"Unknown target type "
            f"{target_name!r}."
        )

    prefix = column_prefixes[
        target_name
    ]

    return resolve_existing_file(
        label=(
            f"{target_name} target "
            f"for {image_id}"
        ),
        absolute_values=[
            target_row.get(
                f"{prefix}_path"
            )
        ],
        relative_values=[
            target_row.get(
                f"{prefix}_relative_path"
            )
        ],
        roots=[
            target_root
        ],
    )


def _load_rgb_image(
    path: Path,
) -> Image.Image:
    """Load a source image safely as RGB."""

    try:
        with Image.open(
            path
        ) as input_image:
            image = input_image.convert(
                "RGB"
            ).copy()

    except Exception as error:
        raise RuntimeError(
            f"Failed to load RGB image: "
            f"{path}"
        ) from error

    if (
        image.width <= 0
        or image.height <= 0
    ):
        raise RuntimeError(
            f"Image has invalid dimensions: "
            f"{path}"
        )

    return image


def _load_binary_target(
    path: Path,
    expected_height: int,
    expected_width: int,
    target_name: str,
) -> Image.Image:
    """Load and validate a binary PNG target."""

    try:
        with Image.open(
            path
        ) as input_image:
            array = np.asarray(
                input_image.convert(
                    "L"
                ),
                dtype=np.uint8,
            ).copy()

    except Exception as error:
        raise RuntimeError(
            f"Failed to load {target_name}: "
            f"{path}"
        ) from error

    expected_shape = (
        expected_height,
        expected_width,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"{target_name} has shape "
            f"{array.shape}, expected "
            f"{expected_shape}: {path}"
        )

    unique_values = set(
        int(value)
        for value in np.unique(
            array
        )
    )

    if not unique_values.issubset(
        {
            0,
            255,
        }
    ):
        raise RuntimeError(
            f"{target_name} is not encoded as "
            f"binary 0/255 PNG. Values: "
            f"{sorted(unique_values)[:20]}; "
            f"path: {path}"
        )

    return Image.fromarray(
        array,
        mode="L",
    )


def _load_normalized_sdm(
    path: Path,
    expected_height: int,
    expected_width: int,
    clip_distance_pixels: float,
) -> np.ndarray:
    """Load a raw clipped SDM and normalize it to [-1, 1]."""

    try:
        array = np.load(
            path,
            allow_pickle=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to load SDM: {path}"
        ) from error

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if (
        array.ndim == 3
        and array.shape[0] == 1
    ):
        array = array[0]

    elif (
        array.ndim == 3
        and array.shape[-1] == 1
    ):
        array = array[..., 0]

    expected_shape = (
        expected_height,
        expected_width,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"SDM has shape {array.shape}, "
            f"expected {expected_shape}: "
            f"{path}"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise RuntimeError(
            f"SDM contains non-finite values: "
            f"{path}"
        )

    tolerance = 1e-4

    observed_min = float(
        np.min(
            array
        )
    )

    observed_max = float(
        np.max(
            array
        )
    )

    if (
        observed_min
        < -clip_distance_pixels
        - tolerance
        or observed_max
        > clip_distance_pixels
        + tolerance
    ):
        raise RuntimeError(
            "SDM exceeds the approved clipping "
            f"range ±{clip_distance_pixels}: "
            f"min={observed_min}, "
            f"max={observed_max}, "
            f"path={path}"
        )

    normalized = np.clip(
        array,
        -clip_distance_pixels,
        clip_distance_pixels,
    ) / float(
        clip_distance_pixels
    )

    return np.ascontiguousarray(
        normalized.astype(
            np.float32,
            copy=False,
        )
    )


class BCSHCTNetDataset(Dataset):
    """Manifest-driven multi-target segmentation dataset."""

    def __init__(
        self,
        *,
        split_manifest_path: str | Path,
        target_manifest_path: str | Path,
        target_artifact_root: str | Path,
        source_roots: Sequence[
            str | Path
        ],
        transform: Callable[
            [Mapping[str, Any]],
            dict[str, Any],
        ] | None,
        expected_target_height: int = (
            DEFAULT_TARGET_HEIGHT
        ),
        expected_target_width: int = (
            DEFAULT_TARGET_WIDTH
        ),
        sdm_clip_distance_pixels: float = (
            DEFAULT_SDM_CLIP_DISTANCE_PIXELS
        ),
        expected_rows: int | None = None,
        expected_split: str | None = None,
    ) -> None:
        """Build a validated dataset index."""

        super().__init__()

        if (
            expected_target_height <= 0
            or expected_target_width <= 0
        ):
            raise ValueError(
                "Expected target dimensions "
                "must be positive."
            )

        if (
            not np.isfinite(
                sdm_clip_distance_pixels
            )
            or sdm_clip_distance_pixels <= 0
        ):
            raise ValueError(
                "SDM clipping distance must be "
                "positive and finite."
            )

        self.split_manifest_path = Path(
            split_manifest_path
        ).expanduser().resolve()

        self.target_manifest_path = Path(
            target_manifest_path
        ).expanduser().resolve()

        self.target_artifact_root = Path(
            target_artifact_root
        ).expanduser().resolve()

        self.source_roots = tuple(
            Path(root)
            .expanduser()
            .resolve()
            for root in source_roots
        )

        self.transform = transform

        self.expected_target_height = int(
            expected_target_height
        )

        self.expected_target_width = int(
            expected_target_width
        )

        self.sdm_clip_distance_pixels = float(
            sdm_clip_distance_pixels
        )

        self.expected_split = (
            None
            if expected_split is None
            else str(
                expected_split
            ).strip()
        )

        if not self.target_artifact_root.is_dir():
            raise FileNotFoundError(
                "Target artifact root not found: "
                f"{self.target_artifact_root}"
            )

        if not self.source_roots:
            raise ValueError(
                "At least one source image root "
                "must be supplied."
            )

        for root in self.source_roots:
            if not root.is_dir():
                raise FileNotFoundError(
                    "Source image root not found: "
                    f"{root}"
                )

        (
            split_columns,
            split_rows,
        ) = read_manifest(
            self.split_manifest_path
        )

        (
            target_columns,
            target_rows,
        ) = read_manifest(
            self.target_manifest_path
        )

        if (
            LOCKED_JOIN_KEY
            not in split_columns
        ):
            raise KeyError(
                f"Split manifest is missing "
                f"{LOCKED_JOIN_KEY!r}."
            )

        if (
            LOCKED_JOIN_KEY
            not in target_columns
        ):
            raise KeyError(
                f"Target manifest is missing "
                f"{LOCKED_JOIN_KEY!r}."
            )

        if (
            expected_rows is not None
            and len(split_rows)
            != expected_rows
        ):
            raise RuntimeError(
                f"Split manifest expected "
                f"{expected_rows} rows, found "
                f"{len(split_rows)}."
            )

        split_index = build_unique_index(
            split_rows,
            key=LOCKED_JOIN_KEY,
            context="split manifest",
        )

        target_index = build_unique_index(
            target_rows,
            key=LOCKED_JOIN_KEY,
            context="target manifest",
        )

        missing_target_ids = sorted(
            set(
                split_index
            )
            - set(
                target_index
            )
        )

        if missing_target_ids:
            raise RuntimeError(
                "Split samples are missing from "
                "the target manifest: "
                f"{missing_target_ids[:20]}."
            )

        self.records = self._build_records(
            split_rows=split_rows,
            target_index=target_index,
        )

        if len(
            self.records
        ) != len(
            split_rows
        ):
            raise RuntimeError(
                "Dataset record count differs "
                "from split-manifest row count."
            )

    def _build_records(
        self,
        *,
        split_rows: Sequence[
            Mapping[str, str]
        ],
        target_index: Mapping[
            str,
            Mapping[str, str],
        ],
    ) -> list[DatasetRecord]:
        """Resolve every sample and target path."""

        records: list[
            DatasetRecord
        ] = []

        for index, split_row in enumerate(
            split_rows
        ):
            image_id = _normalized_text(
                split_row.get(
                    LOCKED_JOIN_KEY,
                    "",
                )
            )

            target_row = target_index[
                image_id
            ]

            _require_target_path_fields(
                target_row,
                image_id,
            )

            split_name = (
                _normalized_text(
                    split_row.get(
                        "split",
                        "",
                    )
                )
                or _normalized_text(
                    target_row.get(
                        "split",
                        "",
                    )
                )
                or "unknown"
            )

            target_split = _normalized_text(
                target_row.get(
                    "split",
                    "",
                )
            )

            if (
                target_split
                and split_name
                and target_split
                != split_name
            ):
                raise RuntimeError(
                    f"{image_id}: split mismatch "
                    f"between split manifest "
                    f"({split_name!r}) and target "
                    f"manifest "
                    f"({target_split!r})."
                )

            if (
                self.expected_split
                and split_name
                != self.expected_split
            ):
                raise RuntimeError(
                    f"{image_id}: expected split "
                    f"{self.expected_split!r}, "
                    f"found {split_name!r}."
                )

            image_path = (
                _resolve_source_image(
                    split_row=split_row,
                    target_row=target_row,
                    source_roots=(
                        self.source_roots
                    ),
                    image_id=image_id,
                )
            )

            mask_path = _resolve_target(
                target_row=target_row,
                target_root=(
                    self.target_artifact_root
                ),
                image_id=image_id,
                target_name="mask",
            )

            contour_path = _resolve_target(
                target_row=target_row,
                target_root=(
                    self.target_artifact_root
                ),
                image_id=image_id,
                target_name="contour",
            )

            boundary_band_path = (
                _resolve_target(
                    target_row=target_row,
                    target_root=(
                        self.target_artifact_root
                    ),
                    image_id=image_id,
                    target_name=(
                        "boundary_band"
                    ),
                )
            )

            sdm_path = _resolve_target(
                target_row=target_row,
                target_root=(
                    self.target_artifact_root
                ),
                image_id=image_id,
                target_name="sdm",
            )

            records.append(
                DatasetRecord(
                    index=index,
                    image_id=image_id,
                    split=split_name,
                    image_path=image_path,
                    mask_path=mask_path,
                    contour_path=(
                        contour_path
                    ),
                    boundary_band_path=(
                        boundary_band_path
                    ),
                    sdm_path=sdm_path,
                    mask_foreground_ratio=(
                        _parse_optional_float(
                            target_row.get(
                                "mask_foreground_ratio"
                            )
                        )
                    ),
                    mask_is_full_foreground=(
                        _parse_optional_boolean(
                            target_row.get(
                                "mask_is_full_foreground"
                            )
                        )
                    ),
                    connected_component_count=(
                        _parse_optional_integer(
                            target_row.get(
                                "connected_component_count"
                            )
                        )
                    ),
                )
            )

        return records

    def __len__(self) -> int:
        """Return the number of samples."""

        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """Load and transform one sample."""

        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "Dataset index must be an integer."
            )

        record = self.records[
            index
        ]

        try:
            image = _load_rgb_image(
                record.image_path
            )

            mask = _load_binary_target(
                path=record.mask_path,
                expected_height=(
                    self.expected_target_height
                ),
                expected_width=(
                    self.expected_target_width
                ),
                target_name="mask",
            )

            contour = _load_binary_target(
                path=record.contour_path,
                expected_height=(
                    self.expected_target_height
                ),
                expected_width=(
                    self.expected_target_width
                ),
                target_name="contour",
            )

            boundary_band = (
                _load_binary_target(
                    path=(
                        record.boundary_band_path
                    ),
                    expected_height=(
                        self.expected_target_height
                    ),
                    expected_width=(
                        self.expected_target_width
                    ),
                    target_name=(
                        "boundary_band"
                    ),
                )
            )

            sdm = _load_normalized_sdm(
                path=record.sdm_path,
                expected_height=(
                    self.expected_target_height
                ),
                expected_width=(
                    self.expected_target_width
                ),
                clip_distance_pixels=(
                    self.sdm_clip_distance_pixels
                ),
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset sample "
                f"index={index}, "
                f"image_id={record.image_id}."
            ) from error

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "contour": contour,
            "boundary_band": (
                boundary_band
            ),
            "sdm": sdm,
            "index": record.index,
            "image_id": record.image_id,
            "split": record.split,
            "source_image_path": str(
                record.image_path
            ),
            "target_mask_path": str(
                record.mask_path
            ),
            "target_contour_path": str(
                record.contour_path
            ),
            "target_boundary_band_path": str(
                record.boundary_band_path
            ),
            "target_sdm_path": str(
                record.sdm_path
            ),
            "mask_foreground_ratio": (
                record.mask_foreground_ratio
            ),
            "mask_is_full_foreground": (
                record.mask_is_full_foreground
            ),
            "connected_component_count": (
                record.connected_component_count
            ),
        }

        if self.transform is not None:
            sample = self.transform(
                sample
            )

        return sample

    def record(
        self,
        index: int,
    ) -> DatasetRecord:
        """Return one resolved record without loading files."""

        return self.records[
            index
        ]

    def summary(
        self,
    ) -> dict[str, Any]:
        """Return dataset provenance and counts."""

        split_counts: dict[
            str,
            int,
        ] = {}

        full_foreground_count = 0

        for record in self.records:
            split_counts[
                record.split
            ] = (
                split_counts.get(
                    record.split,
                    0,
                )
                + 1
            )

            if (
                record.mask_is_full_foreground
            ):
                full_foreground_count += 1

        return {
            "protocol_version": (
                DATASET_PROTOCOL_VERSION
            ),
            "join_key": (
                LOCKED_JOIN_KEY
            ),
            "split_manifest_path": str(
                self.split_manifest_path
            ),
            "target_manifest_path": str(
                self.target_manifest_path
            ),
            "target_artifact_root": str(
                self.target_artifact_root
            ),
            "source_roots": [
                str(path)
                for path in self.source_roots
            ],
            "rows": len(
                self.records
            ),
            "split_counts": (
                split_counts
            ),
            "full_foreground_count": (
                full_foreground_count
            ),
            "target_size": {
                "height": (
                    self.expected_target_height
                ),
                "width": (
                    self.expected_target_width
                ),
            },
            "sdm_clip_distance_pixels": (
                self.sdm_clip_distance_pixels
            ),
            "sdm_output_range": [
                -1.0,
                1.0,
            ],
        }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    """Write a small UTF-8 CSV file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                fieldnames
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                dict(row)
            )


def run_dataset_self_test() -> dict[str, Any]:
    """Run a synthetic dataset and path-resolution test."""

    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(
            directory
        )

        source_root = (
            temporary_root
            / "source_dataset"
        )

        target_root = (
            temporary_root
            / "step05a_artifact"
        )

        image_directory = (
            source_root
            / "images"
        )

        mask_directory = (
            target_root
            / "targets"
            / "masks"
        )

        contour_directory = (
            target_root
            / "targets"
            / "contours"
        )

        boundary_directory = (
            target_root
            / "targets"
            / "boundary_bands"
        )

        sdm_directory = (
            target_root
            / "targets"
            / "sdm"
        )

        for path in [
            image_directory,
            mask_directory,
            contour_directory,
            boundary_directory,
            sdm_directory,
        ]:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

        split_rows: list[
            dict[str, Any]
        ] = []

        target_rows: list[
            dict[str, Any]
        ] = []

        for index in range(2):
            image_id = (
                f"SELF_TEST_{index:03d}"
            )

            image_array = np.zeros(
                (
                    40,
                    48,
                    3,
                ),
                dtype=np.uint8,
            )

            image_array[
                :,
                :,
                0,
            ] = 80 + index * 20

            image_array[
                8:32,
                10:38,
                1,
            ] = 160

            image_path = (
                image_directory
                / f"{image_id}.jpg"
            )

            Image.fromarray(
                image_array,
                mode="RGB",
            ).save(
                image_path
            )

            yy, xx = np.ogrid[
                :32,
                :32,
            ]

            mask_array = (
                (
                    xx - 16
                ) ** 2
                + (
                    yy - 16
                ) ** 2
                <= (
                    9 + index
                ) ** 2
            ).astype(
                np.uint8
            ) * 255

            inner_array = (
                (
                    xx - 16
                ) ** 2
                + (
                    yy - 16
                ) ** 2
                <= (
                    7 + index
                ) ** 2
            ).astype(
                np.uint8
            ) * 255

            contour_array = (
                (
                    mask_array > 0
                )
                ^ (
                    inner_array > 0
                )
            ).astype(
                np.uint8
            ) * 255

            boundary_array = (
                (
                    (
                        xx - 16
                    ) ** 2
                    + (
                        yy - 16
                    ) ** 2
                    <= (
                        12 + index
                    ) ** 2
                )
                & (
                    (
                        xx - 16
                    ) ** 2
                    + (
                        yy - 16
                    ) ** 2
                    >= (
                        6 + index
                    ) ** 2
                )
            ).astype(
                np.uint8
            ) * 255

            raw_sdm = np.where(
                mask_array > 0,
                10.0,
                -20.0,
            ).astype(
                np.float32
            )

            mask_path = (
                mask_directory
                / f"{image_id}.png"
            )

            contour_path = (
                contour_directory
                / f"{image_id}.png"
            )

            boundary_path = (
                boundary_directory
                / f"{image_id}.png"
            )

            sdm_path = (
                sdm_directory
                / f"{image_id}.npy"
            )

            Image.fromarray(
                mask_array,
                mode="L",
            ).save(
                mask_path
            )

            Image.fromarray(
                contour_array,
                mode="L",
            ).save(
                contour_path
            )

            Image.fromarray(
                boundary_array,
                mode="L",
            ).save(
                boundary_path
            )

            np.save(
                sdm_path,
                raw_sdm,
                allow_pickle=False,
            )

            stale_root = Path(
                "/nonexistent/old-kaggle-session"
            )

            split_rows.append(
                {
                    "image_id": image_id,
                    "split": "train",
                    "image_path": str(
                        stale_root
                        / "images"
                        / image_path.name
                    ),
                    "image_relative_path": (
                        f"images/{image_path.name}"
                    ),
                }
            )

            target_rows.append(
                {
                    "image_id": image_id,
                    "split": "train",
                    "source_image_path": str(
                        stale_root
                        / "images"
                        / image_path.name
                    ),
                    "source_image_relative_path": (
                        f"images/{image_path.name}"
                    ),
                    "target_mask_path": str(
                        stale_root
                        / "targets"
                        / "masks"
                        / mask_path.name
                    ),
                    "target_mask_relative_path": (
                        "targets/masks/"
                        f"{mask_path.name}"
                    ),
                    "target_contour_path": str(
                        stale_root
                        / "targets"
                        / "contours"
                        / contour_path.name
                    ),
                    "target_contour_relative_path": (
                        "targets/contours/"
                        f"{contour_path.name}"
                    ),
                    "target_boundary_band_path": str(
                        stale_root
                        / "targets"
                        / "boundary_bands"
                        / boundary_path.name
                    ),
                    (
                        "target_boundary_band_"
                        "relative_path"
                    ): (
                        "targets/boundary_bands/"
                        f"{boundary_path.name}"
                    ),
                    "target_sdm_path": str(
                        stale_root
                        / "targets"
                        / "sdm"
                        / sdm_path.name
                    ),
                    "target_sdm_relative_path": (
                        "targets/sdm/"
                        f"{sdm_path.name}"
                    ),
                    "mask_foreground_ratio": (
                        float(
                            np.mean(
                                mask_array > 0
                            )
                        )
                    ),
                    "mask_is_full_foreground": (
                        False
                    ),
                    "connected_component_count": 1,
                }
            )

        split_manifest = (
            temporary_root
            / "train.csv"
        )

        target_manifest = (
            temporary_root
            / "targets.csv"
        )

        _write_csv(
            split_manifest,
            fieldnames=[
                "image_id",
                "split",
                "image_path",
                "image_relative_path",
            ],
            rows=split_rows,
        )

        _write_csv(
            target_manifest,
            fieldnames=[
                "image_id",
                "split",
                "source_image_path",
                "source_image_relative_path",
                "target_mask_path",
                "target_mask_relative_path",
                "target_contour_path",
                "target_contour_relative_path",
                "target_boundary_band_path",
                (
                    "target_boundary_band_"
                    "relative_path"
                ),
                "target_sdm_path",
                "target_sdm_relative_path",
                "mask_foreground_ratio",
                "mask_is_full_foreground",
                "connected_component_count",
            ],
            rows=target_rows,
        )

        transform = MultiTargetTransform(
            TransformConfig(
                height=64,
                width=64,
            ),
            training=False,
        )

        dataset = BCSHCTNetDataset(
            split_manifest_path=(
                split_manifest
            ),
            target_manifest_path=(
                target_manifest
            ),
            target_artifact_root=(
                target_root
            ),
            source_roots=[
                source_root
            ],
            transform=transform,
            expected_target_height=32,
            expected_target_width=32,
            sdm_clip_distance_pixels=20.0,
            expected_rows=2,
            expected_split="train",
        )

        first = dataset[0]

        repeated = dataset[0]

        summary = dataset.summary()

        checks = {
            "dataset_length": (
                len(dataset) == 2
            ),
            "join_key_locked": (
                summary["join_key"]
                == "image_id"
            ),
            "relative_image_path_resolved": (
                dataset.record(
                    0
                ).image_path
                == (
                    source_root
                    / "images"
                    / "SELF_TEST_000.jpg"
                ).resolve()
            ),
            "relative_target_path_resolved": (
                dataset.record(
                    0
                ).mask_path
                == (
                    target_root
                    / "targets"
                    / "masks"
                    / "SELF_TEST_000.png"
                ).resolve()
            ),
            "image_shape": (
                tuple(
                    first[
                        "image"
                    ].shape
                )
                == (
                    3,
                    64,
                    64,
                )
            ),
            "mask_shape": (
                tuple(
                    first[
                        "mask"
                    ].shape
                )
                == (
                    1,
                    64,
                    64,
                )
            ),
            "contour_shape": (
                tuple(
                    first[
                        "contour"
                    ].shape
                )
                == (
                    1,
                    64,
                    64,
                )
            ),
            "boundary_shape": (
                tuple(
                    first[
                        "boundary_band"
                    ].shape
                )
                == (
                    1,
                    64,
                    64,
                )
            ),
            "sdm_shape": (
                tuple(
                    first[
                        "sdm"
                    ].shape
                )
                == (
                    1,
                    64,
                    64,
                )
            ),
            "sdm_normalized": (
                float(
                    first[
                        "sdm"
                    ].min()
                )
                >= -1.0
                and float(
                    first[
                        "sdm"
                    ].max()
                )
                <= 1.0
            ),
            "binary_targets_preserved": all(
                set(
                    float(value)
                    for value
                    in torch.unique(
                        first[
                            target_name
                        ]
                    )
                )
                <= {
                    0.0,
                    1.0,
                }
                for target_name in [
                    "mask",
                    "contour",
                    "boundary_band",
                ]
            ),
            "evaluation_is_deterministic": (
                torch.equal(
                    first["image"],
                    repeated["image"],
                )
                and torch.equal(
                    first["mask"],
                    repeated["mask"],
                )
                and torch.equal(
                    first["sdm"],
                    repeated["sdm"],
                )
            ),
            "image_id_preserved": (
                first["image_id"]
                == "SELF_TEST_000"
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
                DATASET_PROTOCOL_VERSION
            ),
            "checks": checks,
            "summary": summary,
        }


if __name__ == "__main__":
    result = run_dataset_self_test()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if result[
        "status"
    ] != "passed":
        raise SystemExit(1)