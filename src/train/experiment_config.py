"""Experiment configuration loading and validation for BCS-HCTNet.

This module validates the E00 foundation YAML before data loaders, models,
optimizers, or GPU resources are created.

Validation includes:

- approved YAML schema and experiment identity;
- persistent Kaggle artifact roots;
- Step 05C training-readiness authorization;
- required manifest files and fixed row counts;
- approved 352 x 352 target protocol;
- preprocessing and data-loader safeguards;
- training-only augmentation policy;
- required BCS-HCTNet components and outputs;
- composite-loss configuration;
- optimizer, scheduler, and checkpoint policy;
- evaluation and sensitivity-analysis safeguards;
- reproducibility and smoke-test settings.

Persistent input artifacts are read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-experiment-config-validation-v1"
)

SUPPORTED_SCHEMA_VERSION = 1

EXPECTED_EXPERIMENT_ID = "E00"

EXPECTED_EXPERIMENT_NAME = (
    "bcs_hctnet_foundation"
)

EXPECTED_TRAINING_STATUS = (
    "training_unblocked"
)

EXPECTED_MANIFEST_ROWS = {
    "train": 2594,
    "validation": 100,
    "internal_test_primary": 1000,
    "internal_test_excluding_full_foreground": 999,
    "internal_test_excluding_foreground_ge_0_98": 992,
    "derived_targets": 3694,
}

EXPECTED_IMAGE_SIZE = {
    "height": 352,
    "width": 352,
}

EXPECTED_LOSS_WEIGHTS = {
    "mask": 1.0,
    "contour": 0.5,
    "signed_distance": 0.3,
    "boundary": 0.3,
    "consistency": 0.2,
}

EXPECTED_SENSITIVITY_COUNTS = {
    "excluding_exact_full_foreground": 999,
    "excluding_foreground_ratio_ge_0_98": 992,
}


@dataclass(frozen=True)
class ResolvedManifest:
    """A validated experiment manifest."""

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
            "expected_rows": self.expected_rows,
            "observed_rows": self.observed_rows,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ValidatedExperimentConfig:
    """Fully validated experiment configuration."""

    source_path: Path
    payload: dict[str, Any]
    source_file_sha256: str
    canonical_sha256: str
    artifact_roots: dict[str, Path]
    manifests: dict[
        str,
        ResolvedManifest,
    ]
    training_readiness_path: Path
    training_readiness: dict[str, Any]
    validation_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""

        return {
            "validation_protocol_version": (
                CONFIG_VALIDATION_PROTOCOL_VERSION
            ),
            "source_path": str(
                self.source_path
            ),
            "source_file_sha256": (
                self.source_file_sha256
            ),
            "canonical_sha256": (
                self.canonical_sha256
            ),
            "artifact_roots": {
                name: str(path)
                for name, path
                in self.artifact_roots.items()
            },
            "manifests": {
                name: manifest.to_dict()
                for name, manifest
                in self.manifests.items()
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
    """Require a mapping configuration value."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{context} must be a mapping, "
            f"received {type(value).__name__}."
        )

    return dict(value)


def require_keys(
    mapping: Mapping[str, Any],
    keys: list[str],
    context: str,
) -> None:
    """Require all listed keys."""

    missing = sorted(
        key
        for key in keys
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
        isinstance(value, bool)
        or not isinstance(value, int)
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
        isinstance(value, bool)
        or not isinstance(value, int)
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
    """Require a finite positive number."""

    if isinstance(value, bool):
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
            f"{context} must be positive and "
            f"finite, received {number}."
        )

    return number


def require_probability(
    value: object,
    context: str,
) -> float:
    """Require a probability in [0, 1]."""

    if isinstance(value, bool):
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
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a file SHA-256 checksum."""

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


def count_csv_rows(
    path: Path,
) -> int:
    """Count non-header CSV rows."""

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


def canonical_payload_sha256(
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


def load_yaml(
    config_path: Path,
) -> dict[str, Any]:
    """Load a YAML configuration safely."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
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
    """Resolve required persistent artifact roots."""

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

    resolved: dict[str, Path] = {}

    for artifact_name in [
        "step04",
        "step05a",
        "step05c",
    ]:
        section = require_mapping(
            artifacts[artifact_name],
            f"artifacts.{artifact_name}",
        )

        require_keys(
            section,
            ["root"],
            f"artifacts.{artifact_name}",
        )

        root_text = str(
            section["root"]
        ).strip()

        if not root_text:
            raise ValueError(
                f"Artifact root is empty: "
                f"{artifact_name}."
            )

        root_path = Path(
            root_text
        ).expanduser().resolve()

        if not root_path.is_dir():
            raise FileNotFoundError(
                f"Artifact root not found for "
                f"{artifact_name}: {root_path}"
            )

        resolved[artifact_name] = root_path

    return resolved


def validate_training_readiness(
    payload: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
) -> tuple[Path, dict[str, Any]]:
    """Verify Step 05C training authorization."""

    artifacts = require_mapping(
        payload["artifacts"],
        "artifacts",
    )

    if require_boolean(
        artifacts[
            "require_training_readiness"
        ],
        (
            "artifacts."
            "require_training_readiness"
        ),
    ) is not True:
        raise RuntimeError(
            "Training-readiness validation "
            "must remain required."
        )

    expected_status = str(
        artifacts[
            "expected_training_status"
        ]
    ).strip()

    if (
        expected_status
        != EXPECTED_TRAINING_STATUS
    ):
        raise RuntimeError(
            "Expected training status must be "
            f"{EXPECTED_TRAINING_STATUS!r}."
        )

    readiness_path = (
        artifact_roots["step05c"]
        / "outputs"
        / "reports"
        / "TRAINING_READINESS.json"
    )

    if not readiness_path.is_file():
        raise FileNotFoundError(
            "Training-readiness file not found: "
            f"{readiness_path}"
        )

    readiness = require_mapping(
        json.loads(
            readiness_path.read_text(
                encoding="utf-8"
            )
        ),
        "TRAINING_READINESS.json",
    )

    if (
        readiness.get("status")
        != expected_status
    ):
        raise RuntimeError(
            "Training-readiness status mismatch: "
            f"{readiness.get('status')!r}."
        )

    if (
        readiness.get(
            "training_allowed"
        )
        is not True
    ):
        raise RuntimeError(
            "Step 05C does not authorize training."
        )

    return readiness_path, readiness


def resolve_manifests(
    payload: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
) -> dict[str, ResolvedManifest]:
    """Resolve manifests and verify fixed row counts."""

    manifests = require_mapping(
        payload.get("manifests"),
        "manifests",
    )

    require_keys(
        manifests,
        list(
            EXPECTED_MANIFEST_ROWS
        ),
        "manifests",
    )

    resolved: dict[
        str,
        ResolvedManifest,
    ] = {}

    for manifest_name, protocol_count in (
        EXPECTED_MANIFEST_ROWS.items()
    ):
        section = require_mapping(
            manifests[manifest_name],
            f"manifests.{manifest_name}",
        )

        require_keys(
            section,
            [
                "artifact",
                "relative_path",
                "expected_rows",
            ],
            f"manifests.{manifest_name}",
        )

        artifact_name = str(
            section["artifact"]
        ).strip()

        if artifact_name not in artifact_roots:
            raise KeyError(
                f"Unknown artifact "
                f"{artifact_name!r} for "
                f"{manifest_name}."
            )

        configured_count = (
            require_positive_integer(
                section["expected_rows"],
                (
                    f"manifests."
                    f"{manifest_name}."
                    "expected_rows"
                ),
            )
        )

        if configured_count != protocol_count:
            raise RuntimeError(
                f"{manifest_name} must contain "
                f"{protocol_count} rows, not "
                f"{configured_count}."
            )

        relative_path = str(
            section["relative_path"]
        ).strip()

        relative_object = Path(
            relative_path
        )

        if (
            not relative_path
            or relative_object.is_absolute()
        ):
            raise ValueError(
                f"{manifest_name} must use a "
                "non-empty relative path."
            )

        artifact_root = (
            artifact_roots[artifact_name]
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
                f"Manifest path escapes its "
                f"artifact root: {absolute_path}"
            ) from error

        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{absolute_path}"
            )

        observed_count = count_csv_rows(
            absolute_path
        )

        if observed_count != configured_count:
            raise RuntimeError(
                f"{manifest_name}: expected "
                f"{configured_count} rows, found "
                f"{observed_count}."
            )

        resolved[manifest_name] = (
            ResolvedManifest(
                name=manifest_name,
                artifact_name=artifact_name,
                artifact_root=artifact_root,
                relative_path=(
                    relative_object.as_posix()
                ),
                absolute_path=absolute_path,
                expected_rows=(
                    configured_count
                ),
                observed_rows=(
                    observed_count
                ),
                sha256=sha256_file(
                    absolute_path
                ),
            )
        )

    return resolved


def validate_data_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate preprocessing, targets, and loader settings."""

    data = require_mapping(
        payload.get("data"),
        "data",
    )

    require_keys(
        data,
        [
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

    if image_size != EXPECTED_IMAGE_SIZE:
        raise RuntimeError(
            "Approved image size is "
            "352 x 352."
        )

    if data["image_channels"] != 3:
        raise RuntimeError(
            "Image channels must equal 3."
        )

    if data["output_classes"] != 1:
        raise RuntimeError(
            "Output classes must equal 1."
        )

    if str(
        data["image_mode"]
    ).upper() != "RGB":
        raise RuntimeError(
            "Image mode must be RGB."
        )

    if str(
        data["mask_mode"]
    ).lower() != "binary":
        raise RuntimeError(
            "Mask mode must be binary."
        )

    preprocessing = require_mapping(
        data["preprocessing"],
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
            "binary_target_resize_interpolation"
        ]
    ).lower() != "nearest":
        raise RuntimeError(
            "Binary targets must use nearest "
            "interpolation."
        )

    if (
        preprocessing[
            "sdm_resize_policy"
        ]
        != (
            "no_resize_precomputed_"
            "at_final_resolution"
        )
    ):
        raise RuntimeError(
            "Unexpected SDM resize policy."
        )

    normalization = require_mapping(
        preprocessing["normalization"],
        (
            "data.preprocessing."
            "normalization"
        ),
    )

    if (
        normalization["type"]
        != "imagenet_mean_std"
    ):
        raise RuntimeError(
            "ImageNet normalization is required."
        )

    if (
        normalization["input_scale"]
        != "0_1"
    ):
        raise RuntimeError(
            "input_scale must be the string "
            "'0_1'."
        )

    if normalization["mean"] != [
        0.485,
        0.456,
        0.406,
    ]:
        raise RuntimeError(
            "Unexpected normalization mean."
        )

    if normalization["std"] != [
        0.229,
        0.224,
        0.225,
    ]:
        raise RuntimeError(
            "Unexpected normalization std."
        )

    targets = require_mapping(
        data["targets"],
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
            target_section["enabled"],
            (
                f"data.targets."
                f"{target_name}.enabled"
            ),
        ) is not True:
            raise RuntimeError(
                f"Target is disabled: "
                f"{target_name}."
            )

    if (
        targets["contour"][
            "contour_width_pixels"
        ]
        != 2
    ):
        raise RuntimeError(
            "Contour width must equal 2."
        )

    if (
        targets["boundary_band"][
            "radius_pixels"
        ]
        != 3
    ):
        raise RuntimeError(
            "Boundary-band radius must equal 3."
        )

    sdm = targets[
        "signed_distance_map"
    ]

    if (
        float(
            sdm[
                "clip_distance_pixels"
            ]
        )
        != 20.0
    ):
        raise RuntimeError(
            "SDM clipping distance must "
            "equal 20."
        )

    if (
        sdm["sign_convention"]
        != (
            "positive_inside_"
            "negative_outside"
        )
    ):
        raise RuntimeError(
            "Unexpected SDM sign convention."
        )

    loader = require_mapping(
        data["loader"],
        "data.loader",
    )

    train_batch_size = (
        require_positive_integer(
            loader["train_batch_size"],
            (
                "data.loader."
                "train_batch_size"
            ),
        )
    )

    evaluation_batch_size = (
        require_positive_integer(
            loader[
                "evaluation_batch_size"
            ],
            (
                "data.loader."
                "evaluation_batch_size"
            ),
        )
    )

    num_workers = (
        require_nonnegative_integer(
            loader["num_workers"],
            "data.loader.num_workers",
        )
    )

    require_positive_integer(
        loader["prefetch_factor"],
        "data.loader.prefetch_factor",
    )

    if (
        require_boolean(
            loader["shuffle_train"],
            "data.loader.shuffle_train",
        )
        is not True
    ):
        raise RuntimeError(
            "Training loader must shuffle."
        )

    if (
        require_boolean(
            loader["shuffle_evaluation"],
            (
                "data.loader."
                "shuffle_evaluation"
            ),
        )
        is not False
    ):
        raise RuntimeError(
            "Evaluation loader must not shuffle."
        )

    if (
        require_boolean(
            loader[
                "drop_last_evaluation"
            ],
            (
                "data.loader."
                "drop_last_evaluation"
            ),
        )
        is not False
    ):
        raise RuntimeError(
            "Evaluation must preserve all samples."
        )

    persistent_workers = (
        require_boolean(
            loader[
                "persistent_workers"
            ],
            (
                "data.loader."
                "persistent_workers"
            ),
        )
    )

    if (
        num_workers == 0
        and persistent_workers
    ):
        raise RuntimeError(
            "persistent_workers cannot be true "
            "when num_workers is zero."
        )

    return {
        "image_size": image_size,
        "train_batch_size": (
            train_batch_size
        ),
        "evaluation_batch_size": (
            evaluation_batch_size
        ),
        "num_workers": num_workers,
        "targets": [
            "mask",
            "contour",
            "boundary_band",
            "signed_distance_map",
        ],
    }


def validate_augmentation_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate training-only augmentation policy."""

    augmentation = require_mapping(
        payload.get("augmentation"),
        "augmentation",
    )

    if (
        require_boolean(
            augmentation[
                "enabled_for_train"
            ],
            (
                "augmentation."
                "enabled_for_train"
            ),
        )
        is not True
    ):
        raise RuntimeError(
            "Training augmentation must be enabled."
        )

    for key in [
        "enabled_for_validation",
        "enabled_for_internal_test",
        "enabled_for_external_datasets",
    ]:
        if (
            require_boolean(
                augmentation[key],
                f"augmentation.{key}",
            )
            is not False
        ):
            raise RuntimeError(
                f"{key} must remain false."
            )

    prohibited = require_mapping(
        augmentation["prohibited"],
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

    for name in required_prohibitions:
        if (
            require_boolean(
                prohibited[name],
                (
                    "augmentation.prohibited."
                    f"{name}"
                ),
            )
            is not True
        ):
            raise RuntimeError(
                f"Required prohibition is "
                f"disabled: {name}."
            )

    for section_name, operation_names in {
        "geometric": [
            "horizontal_flip",
            "vertical_flip",
            "affine",
        ],
        "photometric": [
            "brightness_contrast",
            "hue_saturation",
            "gaussian_noise",
            "gaussian_blur",
        ],
    }.items():
        section = require_mapping(
            augmentation[section_name],
            f"augmentation.{section_name}",
        )

        for operation_name in operation_names:
            operation = require_mapping(
                section[operation_name],
                (
                    f"augmentation."
                    f"{section_name}."
                    f"{operation_name}"
                ),
            )

            require_boolean(
                operation["enabled"],
                (
                    f"augmentation."
                    f"{section_name}."
                    f"{operation_name}."
                    "enabled"
                ),
            )

            require_probability(
                operation["probability"],
                (
                    f"augmentation."
                    f"{section_name}."
                    f"{operation_name}."
                    "probability"
                ),
            )

    coarse_dropout = require_mapping(
        augmentation[
            "regularization"
        ]["coarse_dropout"],
        (
            "augmentation.regularization."
            "coarse_dropout"
        ),
    )

    require_probability(
        coarse_dropout["probability"],
        (
            "augmentation.regularization."
            "coarse_dropout.probability"
        ),
    )

    return {
        "profile_name": (
            augmentation["profile_name"]
        ),
        "train_only": True,
        "prohibitions": (
            required_prohibitions
        ),
    }


def validate_model_and_loss(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate required model and loss settings."""

    model = require_mapping(
        payload.get("model"),
        "model",
    )

    if model["name"] != "bcs_hctnet":
        raise RuntimeError(
            "Model name must be bcs_hctnet."
        )

    required_components = [
        "boundary_conditioned_fusion",
        "boundary_prior",
        "contour_head",
        "signed_distance_head",
        "contour_distance_consistency",
        "return_fusion_maps",
    ]

    components = require_mapping(
        model["components"],
        "model.components",
    )

    for component in required_components:
        if (
            require_boolean(
                components[component],
                (
                    f"model.components."
                    f"{component}"
                ),
            )
            is not True
        ):
            raise RuntimeError(
                f"Required component disabled: "
                f"{component}."
            )

    required_outputs = [
        "mask_logits",
        "contour_logits",
        "sdm_prediction",
        "boundary_prior",
        "fusion_maps",
    ]

    outputs = require_mapping(
        model["outputs"],
        "model.outputs",
    )

    for output_name in required_outputs:
        if (
            require_boolean(
                outputs[output_name],
                (
                    f"model.outputs."
                    f"{output_name}"
                ),
            )
            is not True
        ):
            raise RuntimeError(
                f"Required output disabled: "
                f"{output_name}."
            )

    loss = require_mapping(
        payload.get("loss"),
        "loss",
    )

    weights = require_mapping(
        loss["weights"],
        "loss.weights",
    )

    observed_weights = {
        name: float(weights[name])
        for name in EXPECTED_LOSS_WEIGHTS
    }

    if (
        observed_weights
        != EXPECTED_LOSS_WEIGHTS
    ):
        raise RuntimeError(
            "Composite-loss weights differ "
            "from the locked E00 protocol: "
            f"{observed_weights}."
        )

    return {
        "model_name": model["name"],
        "cnn_backbone": (
            model["cnn_encoder"][
                "backbone"
            ]
        ),
        "transformer_backbone": (
            model["transformer_encoder"][
                "backbone"
            ]
        ),
        "components": required_components,
        "outputs": required_outputs,
        "loss_weights": observed_weights,
    }


def validate_training_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate optimizer and checkpoint policy."""

    training = require_mapping(
        payload.get("training"),
        "training",
    )

    maximum_epochs = (
        require_positive_integer(
            training["maximum_epochs"],
            "training.maximum_epochs",
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
            "less than maximum epochs."
        )

    optimizer = require_mapping(
        training["optimizer"],
        "training.optimizer",
    )

    if (
        str(
            optimizer["name"]
        ).lower()
        != "adamw"
    ):
        raise RuntimeError(
            "Optimizer must be AdamW."
        )

    learning_rate = (
        require_positive_number(
            optimizer["learning_rate"],
            (
                "training.optimizer."
                "learning_rate"
            ),
        )
    )

    weight_decay = (
        require_positive_number(
            optimizer["weight_decay"],
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

    if (
        scheduler["name"]
        != "cosine_annealing"
    ):
        raise RuntimeError(
            "Scheduler must be cosine_annealing."
        )

    checkpoint = require_mapping(
        training["checkpoint"],
        "training.checkpoint",
    )

    if (
        checkpoint["monitor"]
        != "validation_dice"
    ):
        raise RuntimeError(
            "Checkpoint monitor must be "
            "validation_dice."
        )

    if checkpoint["mode"] != "maximum":
        raise RuntimeError(
            "Checkpoint mode must be maximum."
        )

    return {
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "cosine_annealing",
        "checkpoint_monitor": (
            "validation_dice"
        ),
    }


def validate_evaluation_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate model-selection and test safeguards."""

    evaluation = require_mapping(
        payload.get("evaluation"),
        "evaluation",
    )

    expected_selection = (
        "official_isic2018_"
        "validation_only"
    )

    if (
        evaluation[
            "model_selection_partition"
        ]
        != expected_selection
    ):
        raise RuntimeError(
            "Model selection must use only "
            "official ISIC 2018 validation."
        )

    if (
        require_boolean(
            evaluation[
                "prohibit_internal_test_tuning"
            ],
            (
                "evaluation."
                "prohibit_internal_test_tuning"
            ),
        )
        is not True
    ):
        raise RuntimeError(
            "Internal-test tuning must be "
            "prohibited."
        )

    if (
        require_boolean(
            evaluation[
                "prohibit_external_dataset_tuning"
            ],
            (
                "evaluation."
                "prohibit_external_dataset_tuning"
            ),
        )
        is not True
    ):
        raise RuntimeError(
            "External-dataset tuning must be "
            "prohibited."
        )

    primary = require_mapping(
        evaluation[
            "primary_internal_test"
        ],
        (
            "evaluation."
            "primary_internal_test"
        ),
    )

    if primary["expected_images"] != 1000:
        raise RuntimeError(
            "Primary internal test must contain "
            "1,000 images."
        )

    sensitivity_items = evaluation[
        "required_sensitivity_analyses"
    ]

    if not isinstance(
        sensitivity_items,
        list,
    ):
        raise TypeError(
            "required_sensitivity_analyses "
            "must be a list."
        )

    observed_sensitivity = {
        str(item["name"]): int(
            item["expected_images"]
        )
        for item in sensitivity_items
    }

    if (
        observed_sensitivity
        != EXPECTED_SENSITIVITY_COUNTS
    ):
        raise RuntimeError(
            "Sensitivity cohorts differ from "
            "the approved Step 05C protocol."
        )

    return {
        "model_selection_partition": (
            expected_selection
        ),
        "primary_internal_test": 1000,
        "sensitivity_analyses": (
            observed_sensitivity
        ),
        "internal_test_tuning": False,
        "external_dataset_tuning": False,
    }


def validate_reproducibility_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate deterministic execution settings."""

    reproducibility = require_mapping(
        payload.get("reproducibility"),
        "reproducibility",
    )

    seed_keys = [
        "seed_python",
        "seed_numpy",
        "seed_torch",
        "seed_cuda",
    ]

    seeds = {
        key: require_nonnegative_integer(
            reproducibility[key],
            f"reproducibility.{key}",
        )
        for key in seed_keys
    }

    if set(
        seeds.values()
    ) != {42}:
        raise RuntimeError(
            "E00 reproducibility seeds must "
            "all equal 42."
        )

    if (
        require_boolean(
            reproducibility[
                "deterministic_algorithms"
            ],
            (
                "reproducibility."
                "deterministic_algorithms"
            ),
        )
        is not True
    ):
        raise RuntimeError(
            "Deterministic algorithms must "
            "be enabled."
        )

    if (
        require_boolean(
            reproducibility[
                "cudnn_deterministic"
            ],
            (
                "reproducibility."
                "cudnn_deterministic"
            ),
        )
        is not True
    ):
        raise RuntimeError(
            "cuDNN deterministic mode must "
            "be enabled."
        )

    if (
        require_boolean(
            reproducibility[
                "cudnn_benchmark"
            ],
            (
                "reproducibility."
                "cudnn_benchmark"
            ),
        )
        is not False
    ):
        raise RuntimeError(
            "cuDNN benchmark must be disabled."
        )

    return {
        "seed": 42,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def validate_smoke_test_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate bounded E00 smoke-test settings."""

    smoke_test = require_mapping(
        payload.get("smoke_test"),
        "smoke_test",
    )

    if (
        require_boolean(
            smoke_test["enabled"],
            "smoke_test.enabled",
        )
        is not True
    ):
        raise RuntimeError(
            "E00 smoke test must be enabled."
        )

    expected_values = {
        "train_samples": 8,
        "validation_samples": 4,
        "batch_size": 2,
        "epochs": 1,
    }

    for key, expected_value in (
        expected_values.items()
    ):
        observed_value = (
            require_positive_integer(
                smoke_test[key],
                f"smoke_test.{key}",
            )
        )

        if observed_value != expected_value:
            raise RuntimeError(
                f"smoke_test.{key} must equal "
                f"{expected_value}, found "
                f"{observed_value}."
            )

    for requirement in [
        "require_loss_finite",
        "require_backward_pass",
        "require_all_output_shapes",
        "require_nonzero_gradients",
        "save_debug_visualizations",
    ]:
        if (
            require_boolean(
                smoke_test[requirement],
                (
                    f"smoke_test."
                    f"{requirement}"
                ),
            )
            is not True
        ):
            raise RuntimeError(
                f"Smoke-test requirement is "
                f"disabled: {requirement}."
            )

    return expected_values


def validate_experiment_configuration(
    config_path: str | Path,
) -> ValidatedExperimentConfig:
    """Load and fully validate an experiment YAML."""

    source_path = Path(
        config_path
    ).expanduser().resolve()

    payload = load_yaml(
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

    if (
        payload["schema_version"]
        != SUPPORTED_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Unsupported configuration schema: "
            f"{payload['schema_version']!r}."
        )

    experiment = require_mapping(
        payload["experiment"],
        "experiment",
    )

    if (
        str(experiment["id"])
        != EXPECTED_EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Foundation experiment ID must "
            "be E00."
        )

    if (
        experiment["name"]
        != EXPECTED_EXPERIMENT_NAME
    ):
        raise RuntimeError(
            "Unexpected foundation "
            "experiment name."
        )

    if (
        require_boolean(
            experiment["deterministic"],
            "experiment.deterministic",
        )
        is not True
    ):
        raise RuntimeError(
            "E00 must be deterministic."
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

    manifests = resolve_manifests(
        payload,
        artifact_roots,
    )

    validation_summary = {
        "schema_version": (
            SUPPORTED_SCHEMA_VERSION
        ),
        "experiment": {
            "id": experiment["id"],
            "name": experiment["name"],
            "purpose": experiment["purpose"],
            "primary_seed": (
                experiment["primary_seed"]
            ),
            "deterministic": (
                experiment["deterministic"]
            ),
        },
        "training_readiness": {
            "status": (
                training_readiness["status"]
            ),
            "training_allowed": (
                training_readiness[
                    "training_allowed"
                ]
            ),
        },
        "data": validate_data_protocol(
            payload
        ),
        "augmentation": (
            validate_augmentation_protocol(
                payload
            )
        ),
        "model_and_loss": (
            validate_model_and_loss(
                payload
            )
        ),
        "training": (
            validate_training_protocol(
                payload
            )
        ),
        "evaluation": (
            validate_evaluation_protocol(
                payload
            )
        ),
        "reproducibility": (
            validate_reproducibility_protocol(
                payload
            )
        ),
        "smoke_test": (
            validate_smoke_test_protocol(
                payload
            )
        ),
        "manifest_count": len(
            manifests
        ),
        "all_checks_passed": True,
    }

    return ValidatedExperimentConfig(
        source_path=source_path,
        payload=payload,
        source_file_sha256=sha256_file(
            source_path
        ),
        canonical_sha256=(
            canonical_payload_sha256(
                payload
            )
        ),
        artifact_roots=artifact_roots,
        manifests=manifests,
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


def save_validation_bundle(
    validated: ValidatedExperimentConfig,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save a configuration copy and validation report."""

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

    validation_report_path = (
        output_root
        / "CONFIG_VALIDATION.json"
    )

    shutil.copy2(
        validated.source_path,
        configuration_copy_path,
    )

    if (
        sha256_file(
            configuration_copy_path
        )
        != validated.source_file_sha256
    ):
        raise RuntimeError(
            "Saved configuration copy does not "
            "match the source configuration."
        )

    validation_report_path.write_text(
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
            validation_report_path
        ),
    }


def summarize_validated_configuration(
    validated: ValidatedExperimentConfig,
) -> str:
    """Return a readable validation summary."""

    lines = [
        (
            "Experiment configuration : "
            f"{validated.source_path}"
        ),
        (
            "Experiment ID            : "
            f"{validated.payload['experiment']['id']}"
        ),
        (
            "Experiment name          : "
            f"{validated.payload['experiment']['name']}"
        ),
        (
            "Configuration SHA-256    : "
            f"{validated.canonical_sha256}"
        ),
        (
            "Training readiness       : "
            f"{validated.training_readiness['status']}"
        ),
        (
            "Training allowed         : "
            f"{validated.training_readiness['training_allowed']}"
        ),
        (
            "Resolved artifacts       : "
            f"{len(validated.artifact_roots)}"
        ),
        (
            "Validated manifests      : "
            f"{len(validated.manifests)}"
        ),
    ]

    for name, manifest in (
        validated.manifests.items()
    ):
        lines.append(
            f" - {name}: "
            f"{manifest.observed_rows} rows"
        )

    lines.append(
        "All configuration checks  : True"
    )

    return "\n".join(lines)