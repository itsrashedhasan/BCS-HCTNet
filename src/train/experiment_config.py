"""Load and validate reproducible BCS-HCTNet experiment configurations.

This module performs CPU-only configuration validation before a data loader,
model, optimizer, or GPU session is created.

It validates:

- YAML structure and supported schema version;
- persistent artifact roots;
- Step 05C training-readiness approval;
- manifest existence and expected row counts;
- fixed 352 x 352 target geometry;
- data-loader settings;
- augmentation restrictions;
- loss-weight consistency;
- model-selection and evaluation safeguards;
- reproducibility settings;
- stable configuration fingerprinting.

No persistent input artifact is modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SUPPORTED_SCHEMA_VERSION = 1

EXPECTED_TARGET_SIZE = {
    "height": 352,
    "width": 352,
}

EXPECTED_TARGET_PROTOCOL = {
    "contour_width_pixels": 2,
    "boundary_band_radius_pixels": 3,
    "sdm_clip_distance_pixels": 20.0,
    "sdm_sign": (
        "positive_inside_negative_outside"
    ),
}

EXPECTED_SPLIT_ROWS = {
    "train": 2594,
    "validation": 100,
    "internal_test_primary": 1000,
    "internal_test_excluding_full_foreground": 999,
    "internal_test_excluding_foreground_ge_0_98": 992,
    "derived_targets": 3694,
}

EXPECTED_TRAINING_STATUS = (
    "training_unblocked"
)

CONFIG_VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-experiment-config-validation-v1"
)


@dataclass(frozen=True)
class ResolvedManifest:
    """A validated manifest reference."""

    name: str
    artifact_name: str
    artifact_root: Path
    relative_path: str
    absolute_path: Path
    expected_rows: int
    observed_rows: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "artifact_name": self.artifact_name,
            "artifact_root": str(
                self.artifact_root
            ),
            "relative_path": self.relative_path,
            "absolute_path": str(
                self.absolute_path
            ),
            "expected_rows": (
                self.expected_rows
            ),
            "observed_rows": (
                self.observed_rows
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ValidatedExperimentConfig:
    """Validated configuration plus resolved inputs."""

    source_path: Path
    payload: dict[str, Any]
    canonical_sha256: str
    source_file_sha256: str
    resolved_artifacts: dict[str, Path]
    resolved_manifests: dict[
        str,
        ResolvedManifest,
    ]
    training_readiness_path: Path
    training_readiness: dict[str, Any]
    validation_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return {
            "validation_protocol_version": (
                CONFIG_VALIDATION_PROTOCOL_VERSION
            ),
            "source_path": str(
                self.source_path
            ),
            "canonical_configuration_sha256": (
                self.canonical_sha256
            ),
            "source_file_sha256": (
                self.source_file_sha256
            ),
            "resolved_artifacts": {
                name: str(path)
                for name, path
                in self.resolved_artifacts.items()
            },
            "resolved_manifests": {
                name: manifest.to_dict()
                for name, manifest
                in self.resolved_manifests.items()
            },
            "training_readiness_path": str(
                self.training_readiness_path
            ),
            "training_readiness": (
                self.training_readiness
            ),
            "validation_summary": (
                self.validation_summary
            ),
        }


def require_mapping(
    value: object,
    context: str,
) -> dict[str, Any]:
    """Require a dictionary-like configuration section."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{context} must be a mapping, "
            f"received {type(value).__name__}."
        )

    return dict(value)


def require_sequence(
    value: object,
    context: str,
) -> list[Any]:
    """Require a non-string sequence."""

    if isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            f"{context} must be a sequence, "
            "not a string."
        )

    if not isinstance(
        value,
        Sequence,
    ):
        raise TypeError(
            f"{context} must be a sequence, "
            f"received {type(value).__name__}."
        )

    return list(value)


def require_keys(
    mapping: Mapping[str, Any],
    required_keys: Sequence[str],
    context: str,
) -> None:
    """Require configuration keys."""

    missing = sorted(
        key
        for key in required_keys
        if key not in mapping
    )

    if missing:
        raise KeyError(
            f"{context} is missing required "
            f"keys: {missing}."
        )


def require_boolean(
    value: object,
    context: str,
) -> bool:
    """Require a real Boolean value."""

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be Boolean, "
            f"received {value!r}."
        )

    return value


def require_positive_integer(
    value: object,
    context: str,
) -> int:
    """Require a positive integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{context} must be an integer, "
            f"received {value!r}."
        )

    if value <= 0:
        raise ValueError(
            f"{context} must be positive, "
            f"received {value}."
        )

    return value


def require_nonnegative_integer(
    value: object,
    context: str,
) -> int:
    """Require a non-negative integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{context} must be an integer, "
            f"received {value!r}."
        )

    if value < 0:
        raise ValueError(
            f"{context} must be non-negative, "
            f"received {value}."
        )

    return value


def require_positive_number(
    value: object,
    context: str,
) -> float:
    """Require a positive finite numeric value."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric, "
            f"received {value!r}."
        ) from error

    if not (
        number > 0.0
        and number < float("inf")
    ):
        raise ValueError(
            f"{context} must be positive "
            f"and finite, received {number}."
        )

    return number


def require_probability(
    value: object,
    context: str,
) -> float:
    """Require a value in the inclusive range [0, 1]."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        probability = float(value)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric, "
            f"received {value!r}."
        ) from error

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise ValueError(
            f"{context} must be in [0, 1], "
            f"received {probability}."
        )

    return probability


def sha256_file(
    path: Path,
    chunk_size: int = (
        1024 * 1024
    ),
) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def csv_row_count(
    path: Path,
) -> int:
    """Count data rows in a UTF-8 CSV file."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {path}"
            )

        return sum(
            1
            for _ in reader
        )


def canonical_configuration_sha256(
    payload: Mapping[str, Any],
) -> str:
    """Hash parsed configuration content deterministically."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def load_yaml_payload(
    config_path: Path,
) -> dict[str, Any]:
    """Load a YAML configuration safely."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Experiment configuration "
            f"not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        payload = yaml.safe_load(
            input_file
        )

    return require_mapping(
        payload,
        "experiment configuration",
    )


def resolve_artifact_roots(
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Resolve and validate configured artifact roots."""

    artifacts = require_mapping(
        payload.get("artifacts"),
        "artifacts",
    )

    require_keys(
        artifacts,
        [
            "step04",
            "step05a",
            "step05c",
            "require_training_readiness",
            "expected_training_status",
        ],
        "artifacts",
    )

    resolved: dict[
        str,
        Path,
    ] = {}

    for artifact_name in [
        "step04",
        "step05a",
        "step05c",
    ]:
        artifact_section = require_mapping(
            artifacts[
                artifact_name
            ],
            (
                f"artifacts."
                f"{artifact_name}"
            ),
        )

        require_keys(
            artifact_section,
            ["root"],
            (
                f"artifacts."
                f"{artifact_name}"
            ),
        )

        root_text = str(
            artifact_section["root"]
        ).strip()

        if not root_text:
            raise ValueError(
                "Artifact root cannot be empty: "
                f"{artifact_name}."
            )

        root_path = Path(
            os.path.expandvars(
                os.path.expanduser(
                    root_text
                )
            )
        ).resolve()

        if not root_path.is_dir():
            raise FileNotFoundError(
                f"Artifact root does not exist "
                f"for {artifact_name}: "
                f"{root_path}"
            )

        resolved[
            artifact_name
        ] = root_path

    return resolved


def validate_training_readiness(
    payload: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
) -> tuple[
    Path,
    dict[str, Any],
]:
    """Validate the persistent Step 05C approval."""

    artifacts = require_mapping(
        payload["artifacts"],
        "artifacts",
    )

    readiness_required = (
        require_boolean(
            artifacts[
                "require_training_readiness"
            ],
            (
                "artifacts."
                "require_training_readiness"
            ),
        )
    )

    expected_status = str(
        artifacts[
            "expected_training_status"
        ]
    ).strip()

    if not expected_status:
        raise ValueError(
            "artifacts.expected_training_status "
            "cannot be empty."
        )

    readiness_path = (
        artifact_roots["step05c"]
        / "outputs"
        / "reports"
        / "TRAINING_READINESS.json"
    )

    if not readiness_path.is_file():
        if readiness_required:
            raise FileNotFoundError(
                "Required training-readiness "
                f"file not found: {readiness_path}"
            )

        return (
            readiness_path,
            {},
        )

    readiness = require_mapping(
        json.loads(
            readiness_path.read_text(
                encoding="utf-8"
            )
        ),
        "TRAINING_READINESS.json",
    )

    observed_status = str(
        readiness.get(
            "status",
            "",
        )
    ).strip()

    training_allowed = (
        readiness.get(
            "training_allowed"
        )
    )

    if (
        observed_status
        != expected_status
    ):
        raise RuntimeError(
            "Training-readiness status mismatch: "
            f"expected {expected_status!r}, "
            f"found {observed_status!r}."
        )

    if training_allowed is not True:
        raise RuntimeError(
            "TRAINING_READINESS.json does not "
            "authorize training."
        )

    if (
        expected_status
        != EXPECTED_TRAINING_STATUS
    ):
        raise RuntimeError(
            "Foundation configuration must require "
            f"{EXPECTED_TRAINING_STATUS!r}, "
            f"received {expected_status!r}."
        )

    return (
        readiness_path,
        readiness,
    )


def resolve_manifests(
    payload: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
) -> dict[str, ResolvedManifest]:
    """Resolve configured manifests and verify row counts."""

    manifests = require_mapping(
        payload.get("manifests"),
        "manifests",
    )

    require_keys(
        manifests,
        list(
            EXPECTED_SPLIT_ROWS
        ),
        "manifests",
    )

    resolved: dict[
        str,
        ResolvedManifest,
    ] = {}

    for manifest_name, locked_count in (
        EXPECTED_SPLIT_ROWS.items()
    ):
        section = require_mapping(
            manifests[
                manifest_name
            ],
            (
                f"manifests."
                f"{manifest_name}"
            ),
        )

        require_keys(
            section,
            [
                "artifact",
                "relative_path",
                "expected_rows",
            ],
            (
                f"manifests."
                f"{manifest_name}"
            ),
        )

        artifact_name = str(
            section["artifact"]
        ).strip()

        if artifact_name not in (
            artifact_roots
        ):
            raise KeyError(
                "Unknown artifact reference "
                f"{artifact_name!r} in "
                f"manifest {manifest_name!r}."
            )

        relative_path = str(
            section[
                "relative_path"
            ]
        ).strip()

        if not relative_path:
            raise ValueError(
                "Manifest relative path cannot "
                f"be empty: {manifest_name}."
            )

        relative_object = Path(
            relative_path
        )

        if relative_object.is_absolute():
            raise ValueError(
                "Manifest path must be relative "
                f"to its artifact root: "
                f"{manifest_name}."
            )

        expected_rows = (
            require_positive_integer(
                section[
                    "expected_rows"
                ],
                (
                    f"manifests."
                    f"{manifest_name}."
                    "expected_rows"
                ),
            )
        )

        if expected_rows != locked_count:
            raise RuntimeError(
                "Configured row count differs "
                "from the approved protocol for "
                f"{manifest_name}: expected "
                f"{locked_count}, configured "
                f"{expected_rows}."
            )

        artifact_root = (
            artifact_roots[
                artifact_name
            ]
        )

        absolute_path = (
            artifact_root
            / relative_object
        ).resolve()

        try:
            absolute_path.relative_to(
                artifact_root
            )

        except ValueError as error:
            raise RuntimeError(
                "Manifest path escapes its "
                f"artifact root: {absolute_path}"
            ) from error

        if not absolute_path.is_file():
            raise FileNotFoundError(
                "Configured manifest not found: "
                f"{absolute_path}"
            )

        observed_rows = csv_row_count(
            absolute_path
        )

        if (
            observed_rows
            != expected_rows
        ):
            raise RuntimeError(
                f"{manifest_name} row-count "
                "mismatch: expected "
                f"{expected_rows}, found "
                f"{observed_rows}."
            )

        resolved[
            manifest_name
        ] = ResolvedManifest(
            name=manifest_name,
            artifact_name=(
                artifact_name
            ),
            artifact_root=(
                artifact_root
            ),
            relative_path=(
                relative_object.as_posix()
            ),
            absolute_path=(
                absolute_path
            ),
            expected_rows=(
                expected_rows
            ),
            observed_rows=(
                observed_rows
            ),
            sha256=sha256_file(
                absolute_path
            ),
        )

    return resolved


def validate_data_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate image, target, and loader configuration."""

    data = require_mapping(
        payload.get("data"),
        "data",
    )

    require_keys(
        data,
        [
            "dataset_name",
            "task",
            "image_size",
            "image_channels",
            "output_classes",
            "image_mode",
            "mask_mode",
            "preprocessing",
            "targets",
            "loader",
        ],
        "data",
    )

    image_size = require_mapping(
        data["image_size"],
        "data.image_size",
    )

    if image_size != EXPECTED_TARGET_SIZE:
        raise RuntimeError(
            "Image size differs from the "
            "approved target resolution: "
            f"{image_size}."
        )

    if (
        require_positive_integer(
            data[
                "image_channels"
            ],
            "data.image_channels",
        )
        != 3
    ):
        raise RuntimeError(
            "BCS-HCTNet requires three-channel "
            "RGB inputs."
        )

    if (
        require_positive_integer(
            data[
                "output_classes"
            ],
            "data.output_classes",
        )
        != 1
    ):
        raise RuntimeError(
            "The approved task is binary "
            "segmentation with one output class."
        )

    if str(
        data["image_mode"]
    ).upper() != "RGB":
        raise RuntimeError(
            "data.image_mode must be RGB."
        )

    if str(
        data["mask_mode"]
    ).lower() != "binary":
        raise RuntimeError(
            "data.mask_mode must be binary."
        )

    preprocessing = require_mapping(
        data[
            "preprocessing"
        ],
        "data.preprocessing",
    )

    require_keys(
        preprocessing,
        [
            "image_resize_interpolation",
            (
                "binary_target_"
                "resize_interpolation"
            ),
            "sdm_resize_policy",
            "normalization",
        ],
        "data.preprocessing",
    )

    if str(
        preprocessing[
            "image_resize_interpolation"
        ]
    ).lower() != "bilinear":
        raise RuntimeError(
            "Images must use bilinear resizing."
        )

    if str(
        preprocessing[
            (
                "binary_target_"
                "resize_interpolation"
            )
        ]
    ).lower() != "nearest":
        raise RuntimeError(
            "Binary targets must use nearest "
            "interpolation."
        )

    expected_sdm_policy = (
        "no_resize_precomputed_"
        "at_final_resolution"
    )

    if str(
        preprocessing[
            "sdm_resize_policy"
        ]
    ) != expected_sdm_policy:
        raise RuntimeError(
            "Unexpected SDM resize policy: "
            f"{preprocessing['sdm_resize_policy']!r}."
        )

    normalization = require_mapping(
        preprocessing[
            "normalization"
        ],
        (
            "data.preprocessing."
            "normalization"
        ),
    )

    require_keys(
        normalization,
        [
            "type",
            "input_scale",
            "mean",
            "std",
        ],
        (
            "data.preprocessing."
            "normalization"
        ),
    )

    if str(
        normalization["type"]
    ) != "imagenet_mean_std":
        raise RuntimeError(
            "Foundation configuration requires "
            "ImageNet mean/std normalization."
        )

    if normalization[
        "input_scale"
    ] != "0_1":
        raise RuntimeError(
            "normalization.input_scale must be "
            "the string '0_1'."
        )

    mean = [
        float(value)
        for value in require_sequence(
            normalization["mean"],
            (
                "data.preprocessing."
                "normalization.mean"
            ),
        )
    ]

    standard_deviation = [
        float(value)
        for value in require_sequence(
            normalization["std"],
            (
                "data.preprocessing."
                "normalization.std"
            ),
        )
    ]

    if mean != [
        0.485,
        0.456,
        0.406,
    ]:
        raise RuntimeError(
            "Unexpected ImageNet mean values."
        )

    if standard_deviation != [
        0.229,
        0.224,
        0.225,
    ]:
        raise RuntimeError(
            "Unexpected ImageNet standard "
            "deviation values."
        )

    targets = require_mapping(
        data["targets"],
        "data.targets",
    )

    require_keys(
        targets,
        [
            "mask",
            "contour",
            "boundary_band",
            "signed_distance_map",
        ],
        "data.targets",
    )

    for target_name in [
        "mask",
        "contour",
        "boundary_band",
        "signed_distance_map",
    ]:
        target_section = require_mapping(
            targets[target_name],
            (
                f"data.targets."
                f"{target_name}"
            ),
        )

        if require_boolean(
            target_section.get(
                "enabled"
            ),
            (
                f"data.targets."
                f"{target_name}.enabled"
            ),
        ) is not True:
            raise RuntimeError(
                "Every approved supervision "
                f"target must be enabled: "
                f"{target_name}."
            )

    contour = require_mapping(
        targets["contour"],
        "data.targets.contour",
    )

    if (
        contour.get(
            "contour_width_pixels"
        )
        != EXPECTED_TARGET_PROTOCOL[
            "contour_width_pixels"
        ]
    ):
        raise RuntimeError(
            "Contour width differs from the "
            "approved target protocol."
        )

    boundary_band = require_mapping(
        targets["boundary_band"],
        (
            "data.targets."
            "boundary_band"
        ),
    )

    if (
        boundary_band.get(
            "radius_pixels"
        )
        != EXPECTED_TARGET_PROTOCOL[
            "boundary_band_radius_pixels"
        ]
    ):
        raise RuntimeError(
            "Boundary-band radius differs from "
            "the approved target protocol."
        )

    sdm = require_mapping(
        targets[
            "signed_distance_map"
        ],
        (
            "data.targets."
            "signed_distance_map"
        ),
    )

    if float(
        sdm.get(
            "clip_distance_pixels"
        )
    ) != EXPECTED_TARGET_PROTOCOL[
        "sdm_clip_distance_pixels"
    ]:
        raise RuntimeError(
            "SDM clipping distance differs from "
            "the approved target protocol."
        )

    if str(
        sdm.get(
            "sign_convention"
        )
    ) != EXPECTED_TARGET_PROTOCOL[
        "sdm_sign"
    ]:
        raise RuntimeError(
            "SDM sign convention differs from "
            "the approved target protocol."
        )

    loader = require_mapping(
        data["loader"],
        "data.loader",
    )

    require_keys(
        loader,
        [
            "train_batch_size",
            "evaluation_batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "drop_last_train",
            "drop_last_evaluation",
            "shuffle_train",
            "shuffle_evaluation",
        ],
        "data.loader",
    )

    require_positive_integer(
        loader[
            "train_batch_size"
        ],
        (
            "data.loader."
            "train_batch_size"
        ),
    )

    require_positive_integer(
        loader[
            "evaluation_batch_size"
        ],
        (
            "data.loader."
            "evaluation_batch_size"
        ),
    )

    require_nonnegative_integer(
        loader["num_workers"],
        "data.loader.num_workers",
    )

    require_positive_integer(
        loader[
            "prefetch_factor"
        ],
        (
            "data.loader."
            "prefetch_factor"
        ),
    )

    for boolean_key in [
        "pin_memory",
        "persistent_workers",
        "drop_last_train",
        "drop_last_evaluation",
        "shuffle_train",
        "shuffle_evaluation",
    ]:
        require_boolean(
            loader[boolean_key],
            (
                f"data.loader."
                f"{boolean_key}"
            ),
        )

    if (
        loader[
            "shuffle_train"
        ]
        is not True
    ):
        raise RuntimeError(
            "Training data must be shuffled."
        )

    if (
        loader[
            "shuffle_evaluation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Evaluation data must not be shuffled."
        )

    if (
        loader[
            "drop_last_evaluation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Evaluation batches must preserve "
            "every sample."
        )

    return {
        "image_size": image_size,
        "target_protocol": (
            EXPECTED_TARGET_PROTOCOL
        ),
        "train_batch_size": (
            loader[
                "train_batch_size"
            ]
        ),
        "evaluation_batch_size": (
            loader[
                "evaluation_batch_size"
            ]
        ),
        "num_workers": (
            loader[
                "num_workers"
            ]
        ),
    }


def validate_augmentation_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate training-only augmentation safeguards."""

    augmentation = require_mapping(
        payload.get(
            "augmentation"
        ),
        "augmentation",
    )

    require_keys(
        augmentation,
        [
            "enabled_for_train",
            "enabled_for_validation",
            "enabled_for_internal_test",
            "enabled_for_external_datasets",
            "profile_name",
            "geometric",
            "photometric",
            "regularization",
            "prohibited",
        ],
        "augmentation",
    )

    if require_boolean(
        augmentation[
            "enabled_for_train"
        ],
        (
            "augmentation."
            "enabled_for_train"
        ),
    ) is not True:
        raise RuntimeError(
            "Training augmentation must be enabled."
        )

    for key in [
        "enabled_for_validation",
        "enabled_for_internal_test",
        "enabled_for_external_datasets",
    ]:
        if require_boolean(
            augmentation[key],
            f"augmentation.{key}",
        ) is not False:
            raise RuntimeError(
                "Random augmentation is prohibited "
                f"for {key}."
            )

    prohibited = require_mapping(
        augmentation[
            "prohibited"
        ],
        "augmentation.prohibited",
    )

    required_prohibitions = [
        "random_augmentation_on_validation",
        "random_augmentation_on_test",
        "hidden_clahe_default",
        "hidden_hair_removal_default",
        "aggressive_crop",
        "elastic_deformation",
    ]

    require_keys(
        prohibited,
        required_prohibitions,
        "augmentation.prohibited",
    )

    for prohibition in (
        required_prohibitions
    ):
        if require_boolean(
            prohibited[prohibition],
            (
                "augmentation.prohibited."
                f"{prohibition}"
            ),
        ) is not True:
            raise RuntimeError(
                "Required prohibition is not "
                f"enabled: {prohibition}."
            )

    geometric = require_mapping(
        augmentation[
            "geometric"
        ],
        "augmentation.geometric",
    )

    for operation_name in [
        "horizontal_flip",
        "vertical_flip",
        "affine",
    ]:
        operation = require_mapping(
            geometric[
                operation_name
            ],
            (
                "augmentation.geometric."
                f"{operation_name}"
            ),
        )

        require_probability(
            operation.get(
                "probability"
            ),
            (
                "augmentation.geometric."
                f"{operation_name}."
                "probability"
            ),
        )

    photometric = require_mapping(
        augmentation[
            "photometric"
        ],
        "augmentation.photometric",
    )

    for operation_name in [
        "brightness_contrast",
        "hue_saturation",
        "gaussian_noise",
        "gaussian_blur",
    ]:
        operation = require_mapping(
            photometric[
                operation_name
            ],
            (
                "augmentation.photometric."
                f"{operation_name}"
            ),
        )

        require_probability(
            operation.get(
                "probability"
            ),
            (
                "augmentation.photometric."
                f"{operation_name}."
                "probability"
            ),
        )

    regularization = require_mapping(
        augmentation[
            "regularization"
        ],
        "augmentation.regularization",
    )

    coarse_dropout = require_mapping(
        regularization[
            "coarse_dropout"
        ],
        (
            "augmentation.regularization."
            "coarse_dropout"
        ),
    )

    require_probability(
        coarse_dropout.get(
            "probability"
        ),
        (
            "augmentation.regularization."
            "coarse_dropout.probability"
        ),
    )

    return {
        "profile_name": str(
            augmentation[
                "profile_name"
            ]
        ),
        "train_only": True,
        "required_prohibitions": (
            required_prohibitions
        ),
    }


def validate_model_and_loss(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate required model outputs and composite loss."""

    model = require_mapping(
        payload.get("model"),
        "model",
    )

    loss = require_mapping(
        payload.get("loss"),
        "loss",
    )

    require_keys(
        model,
        [
            "name",
            "cnn_encoder",
            "transformer_encoder",
            "decoder",
            "components",
            "outputs",
        ],
        "model",
    )

    if str(
        model["name"]
    ) != "bcs_hctnet":
        raise RuntimeError(
            "Unexpected model name."
        )

    components = require_mapping(
        model["components"],
        "model.components",
    )

    required_components = [
        "boundary_conditioned_fusion",
        "boundary_prior",
        "contour_head",
        "signed_distance_head",
        "contour_distance_consistency",
        "return_fusion_maps",
    ]

    require_keys(
        components,
        required_components,
        "model.components",
    )

    for component in required_components:
        if require_boolean(
            components[component],
            (
                f"model.components."
                f"{component}"
            ),
        ) is not True:
            raise RuntimeError(
                "Required model component is "
                f"disabled: {component}."
            )

    outputs = require_mapping(
        model["outputs"],
        "model.outputs",
    )

    required_outputs = [
        "mask_logits",
        "contour_logits",
        "sdm_prediction",
        "boundary_prior",
        "fusion_maps",
    ]

    require_keys(
        outputs,
        required_outputs,
        "model.outputs",
    )

    for output_name in required_outputs:
        if require_boolean(
            outputs[output_name],
            (
                f"model.outputs."
                f"{output_name}"
            ),
        ) is not True:
            raise RuntimeError(
                "Required model output is "
                f"disabled: {output_name}."
            )

    require_keys(
        loss,
        [
            "name",
            "weights",
            "mask",
            "contour",
            "signed_distance",
            "boundary",
            "consistency",
        ],
        "loss",
    )

    weights = require_mapping(
        loss["weights"],
        "loss.weights",
    )

    expected_weight_keys = [
        "mask",
        "contour",
        "signed_distance",
        "boundary",
        "consistency",
    ]

    require_keys(
        weights,
        expected_weight_keys,
        "loss.weights",
    )

    numeric_weights = {
        name: require_positive_number(
            weights[name],
            f"loss.weights.{name}",
        )
        for name in expected_weight_keys
    }

    if numeric_weights["mask"] != 1.0:
        raise RuntimeError(
            "Primary mask-loss weight must "
            "remain 1.0."
        )

    return {
        "model_name": model["name"],
        "required_components": (
            required_components
        ),
        "required_outputs": (
            required_outputs
        ),
        "loss_weights": (
            numeric_weights
        ),
    }


def validate_training_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate optimizer, scheduler, and checkpoint policy."""

    training = require_mapping(
        payload.get("training"),
        "training",
    )

    require_keys(
        training,
        [
            "maximum_epochs",
            "early_stopping_patience",
            "optimizer",
            "scheduler",
            "automatic_mixed_precision",
            "gradient_accumulation_steps",
            "gradient_clip_norm",
            "validation_every_epochs",
            "checkpoint",
            "resume",
        ],
        "training",
    )

    maximum_epochs = (
        require_positive_integer(
            training[
                "maximum_epochs"
            ],
            (
                "training."
                "maximum_epochs"
            ),
        )
    )

    patience = (
        require_positive_integer(
            training[
                "early_stopping_patience"
            ],
            (
                "training."
                "early_stopping_patience"
            ),
        )
    )

    if patience >= maximum_epochs:
        raise RuntimeError(
            "Early-stopping patience must be "
            "smaller than maximum_epochs."
        )

    optimizer = require_mapping(
        training["optimizer"],
        "training.optimizer",
    )

    if str(
        optimizer.get("name")
    ).lower() != "adamw":
        raise RuntimeError(
            "The approved optimizer is AdamW."
        )

    learning_rate = (
        require_positive_number(
            optimizer.get(
                "learning_rate"
            ),
            (
                "training.optimizer."
                "learning_rate"
            ),
        )
    )

    weight_decay = (
        require_positive_number(
            optimizer.get(
                "weight_decay"
            ),
            (
                "training.optimizer."
                "weight_decay"
            ),
        )
    )

    scheduler = require_mapping(
        training["scheduler"],
        "training.scheduler",
    )

    if str(
        scheduler.get("name")
    ).lower() != "cosine_annealing":
        raise RuntimeError(
            "The approved scheduler is "
            "cosine annealing."
        )

    checkpoint = require_mapping(
        training["checkpoint"],
        "training.checkpoint",
    )

    if str(
        checkpoint.get("monitor")
    ) != "validation_dice":
        raise RuntimeError(
            "Checkpoint selection must monitor "
            "validation_dice."
        )

    if str(
        checkpoint.get("mode")
    ).lower() != "maximum":
        raise RuntimeError(
            "validation_dice checkpoint mode "
            "must be maximum."
        )

    require_boolean(
        training[
            "automatic_mixed_precision"
        ],
        (
            "training."
            "automatic_mixed_precision"
        ),
    )

    require_positive_integer(
        training[
            "gradient_accumulation_steps"
        ],
        (
            "training."
            "gradient_accumulation_steps"
        ),
    )

    require_positive_number(
        training[
            "gradient_clip_norm"
        ],
        (
            "training."
            "gradient_clip_norm"
        ),
    )

    return {
        "maximum_epochs": (
            maximum_epochs
        ),
        "early_stopping_patience": (
            patience
        ),
        "optimizer": "adamw",
        "learning_rate": (
            learning_rate
        ),
        "weight_decay": (
            weight_decay
        ),
        "scheduler": (
            "cosine_annealing"
        ),
        "checkpoint_monitor": (
            "validation_dice"
        ),
    }


def validate_evaluation_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate model-selection and sensitivity rules."""

    evaluation = require_mapping(
        payload.get("evaluation"),
        "evaluation",
    )

    require_keys(
        evaluation,
        [
            "model_selection_partition",
            "primary_internal_test",
            "required_sensitivity_analyses",
            "prohibit_internal_test_tuning",
            "prohibit_external_dataset_tuning",
        ],
        "evaluation",
    )

    expected_selection = (
        "official_isic2018_"
        "validation_only"
    )

    if str(
        evaluation[
            "model_selection_partition"
        ]
    ) != expected_selection:
        raise RuntimeError(
            "Model selection must use only the "
            "official ISIC 2018 validation set."
        )

    if require_boolean(
        evaluation[
            "prohibit_internal_test_tuning"
        ],
        (
            "evaluation."
            "prohibit_internal_test_tuning"
        ),
    ) is not True:
        raise RuntimeError(
            "Internal-test tuning must be "
            "explicitly prohibited."
        )

    if require_boolean(
        evaluation[
            "prohibit_external_dataset_tuning"
        ],
        (
            "evaluation."
            "prohibit_external_dataset_tuning"
        ),
    ) is not True:
        raise RuntimeError(
            "External-dataset tuning must be "
            "explicitly prohibited."
        )

    primary_test = require_mapping(
        evaluation[
            "primary_internal_test"
        ],
        (
            "evaluation."
            "primary_internal_test"
        ),
    )

    if (
        primary_test.get(
            "expected_images"
        )
        != 1000
    ):
        raise RuntimeError(
            "Primary internal-test cohort must "
            "contain 1,000 images."
        )

    sensitivity = require_sequence(
        evaluation[
            "required_sensitivity_analyses"
        ],
        (
            "evaluation."
            "required_sensitivity_analyses"
        ),
    )

    observed_sensitivity = {}

    for item in sensitivity:
        section = require_mapping(
            item,
            (
                "evaluation.required_"
                "sensitivity_analyses item"
            ),
        )

        name = str(
            section.get(
                "name",
                "",
            )
        )

        expected_images = (
            require_positive_integer(
                section.get(
                    "expected_images"
                ),
                (
                    "evaluation sensitivity "
                    f"{name}.expected_images"
                ),
            )
        )

        observed_sensitivity[
            name
        ] = expected_images

    expected_sensitivity = {
        "excluding_exact_full_foreground": 999,
        (
            "excluding_foreground_"
            "ratio_ge_0_98"
        ): 992,
    }

    if (
        observed_sensitivity
        != expected_sensitivity
    ):
        raise RuntimeError(
            "Required sensitivity analyses "
            "do not match the approved protocol: "
            f"{observed_sensitivity}."
        )

    return {
        "model_selection_partition": (
            expected_selection
        ),
        "primary_internal_test_images": (
            1000
        ),
        "sensitivity_analyses": (
            expected_sensitivity
        ),
        "internal_test_tuning": False,
        "external_tuning": False,
    }


def validate_reproducibility_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate deterministic execution settings."""

    reproducibility = require_mapping(
        payload.get(
            "reproducibility"
        ),
        "reproducibility",
    )

    seed_keys = [
        "seed_python",
        "seed_numpy",
        "seed_torch",
        "seed_cuda",
    ]

    require_keys(
        reproducibility,
        [
            *seed_keys,
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "record",
        ],
        "reproducibility",
    )

    seeds = {
        key: require_nonnegative_integer(
            reproducibility[key],
            f"reproducibility.{key}",
        )
        for key in seed_keys
    }

    if len(
        set(
            seeds.values()
        )
    ) != 1:
        raise RuntimeError(
            "Foundation reproducibility seeds "
            "must be identical."
        )

    if require_boolean(
        reproducibility[
            "deterministic_algorithms"
        ],
        (
            "reproducibility."
            "deterministic_algorithms"
        ),
    ) is not True:
        raise RuntimeError(
            "Deterministic algorithms must be "
            "enabled."
        )

    if require_boolean(
        reproducibility[
            "cudnn_deterministic"
        ],
        (
            "reproducibility."
            "cudnn_deterministic"
        ),
    ) is not True:
        raise RuntimeError(
            "cuDNN deterministic mode must "
            "be enabled."
        )

    if require_boolean(
        reproducibility[
            "cudnn_benchmark"
        ],
        (
            "reproducibility."
            "cudnn_benchmark"
        ),
    ) is not False:
        raise RuntimeError(
            "cuDNN benchmark must be disabled "
            "for deterministic execution."
        )

    return {
        "seed": next(
            iter(
                seeds.values()
            )
        ),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def validate_smoke_test_section(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the initial bounded smoke-test protocol."""

    smoke_test = require_mapping(
        payload.get("smoke_test"),
        "smoke_test",
    )

    require_keys(
        smoke_test,
        [
            "enabled",
            "train_samples",
            "validation_samples",
            "batch_size",
            "epochs",
            "require_loss_finite",
            "require_backward_pass",
            "require_all_output_shapes",
            "require_nonzero_gradients",
            "save_debug_visualizations",
        ],
        "smoke_test",
    )

    if require_boolean(
        smoke_test["enabled"],
        "smoke_test.enabled",
    ) is not True:
        raise RuntimeError(
            "The E00 smoke test must be enabled."
        )

    train_samples = (
        require_positive_integer(
            smoke_test[
                "train_samples"
            ],
            (
                "smoke_test."
                "train_samples"
            ),
        )
    )

    validation_samples = (
        require_positive_integer(
            smoke_test[
                "validation_samples"
            ],
            (
                "smoke_test."
                "validation_samples"
            ),
        )
    )

    batch_size = (
        require_positive_integer(
            smoke_test[
                "batch_size"
            ],
            (
                "smoke_test."
                "batch_size"
            ),
        )
    )

    epochs = (
        require_positive_integer(
            smoke_test[
                "epochs"
            ],
            "smoke_test.epochs",
        )
    )

    if train_samples % batch_size != 0:
        raise RuntimeError(
            "Smoke-test train sample count "
            "must be divisible by batch size."
        )

    if (
        validation_samples
        % batch_size
        != 0
    ):
        raise RuntimeError(
            "Smoke-test validation sample count "
            "must be divisible by batch size."
        )

    if epochs != 1:
        raise RuntimeError(
            "The foundation smoke test must "
            "run exactly one epoch."
        )

    for requirement in [
        "require_loss_finite",
        "require_backward_pass",
        "require_all_output_shapes",
        "require_nonzero_gradients",
        "save_debug_visualizations",
    ]:
        if require_boolean(
            smoke_test[
                requirement
            ],
            (
                f"smoke_test."
                f"{requirement}"
            ),
        ) is not True:
            raise RuntimeError(
                "Smoke-test requirement must "
                f"remain enabled: {requirement}."
            )

    return {
        "train_samples": train_samples,
        "validation_samples": (
            validation_samples
        ),
        "batch_size": batch_size,
        "epochs": epochs,
    }


def validate_experiment_configuration(
    config_path: str | Path,
) -> ValidatedExperimentConfig:
    """Load and fully validate an experiment YAML file."""

    source_path = Path(
        config_path
    ).expanduser().resolve()

    payload = load_yaml_payload(
        source_path
    )

    require_keys(
        payload,
        [
            "schema_version",
            "experiment",
            "artifacts",
            "manifests",
            "data",
            "augmentation",
            "model",
            "training",
            "loss",
            "inference",
            "evaluation",
            "reproducibility",
            "smoke_test",
            "outputs",
        ],
        "experiment configuration",
    )

    schema_version = (
        payload[
            "schema_version"
        ]
    )

    if (
        schema_version
        != SUPPORTED_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Unsupported experiment schema "
            f"version {schema_version!r}; "
            f"supported version is "
            f"{SUPPORTED_SCHEMA_VERSION}."
        )

    experiment = require_mapping(
        payload["experiment"],
        "experiment",
    )

    require_keys(
        experiment,
        [
            "id",
            "name",
            "purpose",
            "primary_seed",
            "deterministic",
        ],
        "experiment",
    )

    if str(
        experiment["id"]
    ) != "E00":
        raise RuntimeError(
            "Foundation configuration must "
            "use experiment ID E00."
        )

    if str(
        experiment["name"]
    ) != "bcs_hctnet_foundation":
        raise RuntimeError(
            "Unexpected foundation "
            "experiment name."
        )

    if require_boolean(
        experiment[
            "deterministic"
        ],
        "experiment.deterministic",
    ) is not True:
        raise RuntimeError(
            "The foundation experiment must "
            "be deterministic."
        )

    artifact_roots = (
        resolve_artifact_roots(
            payload
        )
    )

    (
        training_readiness_path,
        training_readiness,
    ) = validate_training_readiness(
        payload,
        artifact_roots,
    )

    resolved_manifests = (
        resolve_manifests(
            payload,
            artifact_roots,
        )
    )

    validation_summary = {
        "schema_version": (
            schema_version
        ),
        "experiment": {
            "id": experiment["id"],
            "name": (
                experiment["name"]
            ),
            "purpose": (
                experiment["purpose"]
            ),
            "primary_seed": (
                experiment[
                    "primary_seed"
                ]
            ),
            "deterministic": (
                experiment[
                    "deterministic"
                ]
            ),
        },
        "data": validate_data_section(
            payload
        ),
        "augmentation": (
            validate_augmentation_section(
                payload
            )
        ),
        "model_and_loss": (
            validate_model_and_loss(
                payload
            )
        ),
        "training": (
            validate_training_section(
                payload
            )
        ),
        "evaluation": (
            validate_evaluation_section(
                payload
            )
        ),
        "reproducibility": (
            validate_reproducibility_section(
                payload
            )
        ),
        "smoke_test": (
            validate_smoke_test_section(
                payload
            )
        ),
        "training_readiness": {
            "status": (
                training_readiness[
                    "status"
                ]
            ),
            "training_allowed": (
                training_readiness[
                    "training_allowed"
                ]
            ),
        },
        "manifest_count": len(
            resolved_manifests
        ),
        "all_checks_passed": True,
    }

    return ValidatedExperimentConfig(
        source_path=source_path,
        payload=payload,
        canonical_sha256=(
            canonical_configuration_sha256(
                payload
            )
        ),
        source_file_sha256=(
            sha256_file(
                source_path
            )
        ),
        resolved_artifacts=(
            artifact_roots
        ),
        resolved_manifests=(
            resolved_manifests
        ),
        training_readiness_path=(
            training_readiness_path
        ),
        training_readiness=(
            training_readiness
        ),
        validation_summary=(
            validation_summary
        ),
    )


def save_validated_configuration(
    validated: ValidatedExperimentConfig,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Persist the validated YAML copy and validation report."""

    output_root = Path(
        output_directory
    ).expanduser().resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration_copy_path = (
        output_root
        / validated.source_path.name
    )

    report_path = (
        output_root
        / "CONFIG_VALIDATION.json"
    )

    source_text = (
        validated.source_path.read_text(
            encoding="utf-8"
        )
    )

    configuration_copy_path.write_text(
        source_text,
        encoding="utf-8",
    )

    copied_hash = sha256_file(
        configuration_copy_path
    )

    if (
        copied_hash
        != validated.source_file_sha256
    ):
        raise RuntimeError(
            "Saved configuration copy does "
            "not match the source file."
        )

    report_path.write_text(
        json.dumps(
            validated.to_dict(),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "configuration_copy": (
            configuration_copy_path
        ),
        "validation_report": (
            report_path
        ),
    }


def summarize_validated_configuration(
    validated: ValidatedExperimentConfig,
) -> str:
    """Return a readable validation summary."""

    lines = [
        (
            "Experiment configuration: "
            f"{validated.source_path}"
        ),
        (
            "Experiment ID           : "
            f"{validated.payload['experiment']['id']}"
        ),
        (
            "Experiment name         : "
            f"{validated.payload['experiment']['name']}"
        ),
        (
            "Configuration SHA-256   : "
            f"{validated.canonical_sha256}"
        ),
        (
            "Training readiness      : "
            f"{validated.training_readiness['status']}"
        ),
        (
            "Training allowed        : "
            f"{validated.training_readiness['training_allowed']}"
        ),
        (
            "Resolved artifacts      : "
            f"{len(validated.resolved_artifacts)}"
        ),
        (
            "Validated manifests     : "
            f"{len(validated.resolved_manifests)}"
        ),
    ]

    for name, manifest in (
        validated.resolved_manifests.items()
    ):
        lines.append(
            (
                f" - {name}: "
                f"{manifest.observed_rows} rows"
            )
        )

    lines.append(
        "All configuration checks : True"
    )

    return "\n".join(lines)