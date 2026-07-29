"""Validation for E01-E05 supervised baseline experiment configurations.

The original :mod:`src.train.experiment_config` module intentionally locks the
E00 BCS-HCTNet foundation configuration. It rejects every baseline model and
therefore cannot be used by ``train_baseline.py``.

This module adds a separate validator for the five approved baseline runs while
reusing the already validated data, artifact, augmentation, training,
evaluation, reproducibility, and smoke-test protocols.

Approved experiment mapping
---------------------------
- E01 -> U-Net
- E02 -> UNet++
- E03 -> DeepLabV3+
- E04 -> TransUNet
- E05 -> Swin-UNet

All baseline configurations are mask-only, fully supervised experiments using
the shared BCE-Dice criterion. They must not enable contour, boundary-band,
SDM, consistency, or BCS-HCTNet-specific outputs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.baselines.model_factory import (
    MODEL_ALLOWED_PARAMETERS,
    normalize_baseline_name,
)
from src.train.experiment_config import (
    EXPECTED_TRAINING_STATUS,
    SUPPORTED_SCHEMA_VERSION,
    ResolvedManifest,
    canonical_payload_sha256,
    load_yaml,
    require_boolean,
    require_keys,
    require_mapping,
    require_nonnegative_integer,
    require_positive_integer,
    require_positive_number,
    require_probability,
    resolve_artifact_roots,
    resolve_manifests,
    sha256_file,
    validate_augmentation_protocol,
    validate_data_protocol,
    validate_evaluation_protocol,
    validate_reproducibility_protocol,
    validate_smoke_test_protocol,
    validate_training_protocol,
    validate_training_readiness,
)


BASELINE_CONFIG_VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-baseline-config-validation-v1"
)

BASELINE_EXPERIMENT_MODEL_MAP = {
    "E01": "unet",
    "E02": "unetpp",
    "E03": "deeplabv3plus",
    "E04": "transunet",
    "E05": "swin_unet",
}

BASELINE_REQUIRED_MODEL_PARAMETERS = {
    "unet": {
        "input_channels",
        "output_channels",
        "base_channels",
        "bilinear",
        "dropout_probability",
    },
    "unetpp": {
        "input_channels",
        "output_channels",
        "base_channels",
        "deep_supervision",
        "dropout_probability",
    },
    "deeplabv3plus": {
        "input_channels",
        "output_channels",
        "backbone_name",
        "backbone_pretrained",
        "backbone_weights_path",
        "aspp_channels",
        "decoder_channels",
        "low_level_projection_channels",
        "atrous_rates",
        "dropout_probability",
    },
    "transunet": {
        "input_channels",
        "output_channels",
        "base_channels",
        "encoder_blocks",
        "transformer_dimension",
        "transformer_layers",
        "transformer_heads",
        "transformer_mlp_dimension",
        "transformer_dropout",
        "attention_dropout",
        "bottleneck_dropout",
        "bilinear_decoder",
    },
    "swin_unet": {
        "input_channels",
        "output_channels",
        "patch_size",
        "embedding_dimension",
        "depths",
        "number_of_heads",
        "window_size",
        "mlp_ratio",
        "dropout_probability",
        "attention_dropout",
        "drop_path_rate",
    },
}

BASELINE_REQUIRED_OUTPUT_DIRECTORIES = (
    "checkpoints",
    "logs",
    "histories",
    "predictions",
    "probability_maps",
    "visualizations",
    "configurations",
    "environment",
)

BASELINE_REQUIRED_TOP_LEVEL_KEYS = (
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
)


@dataclass(frozen=True)
class ValidatedBaselineExperimentConfig:
    """Fully validated E01-E05 baseline configuration.

    The attributes intentionally match the interface consumed by
    :func:`src.data.dataloader.build_loader_bundle`.
    """

    source_path: Path
    payload: dict[str, Any]
    source_file_sha256: str
    canonical_sha256: str
    artifact_roots: dict[str, Path]
    manifests: dict[str, ResolvedManifest]
    training_readiness_path: Path
    training_readiness: dict[str, Any]
    validation_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation report."""

        return {
            "validation_protocol_version": (
                BASELINE_CONFIG_VALIDATION_PROTOCOL_VERSION
            ),
            "source_path": str(self.source_path),
            "source_file_sha256": self.source_file_sha256,
            "canonical_sha256": self.canonical_sha256,
            "artifact_roots": {
                name: str(path)
                for name, path in self.artifact_roots.items()
            },
            "manifests": {
                name: manifest.to_dict()
                for name, manifest in self.manifests.items()
            },
            "training_readiness_path": str(
                self.training_readiness_path
            ),
            "training_readiness": self.training_readiness,
            "validation_summary": self.validation_summary,
        }


def _require_nonempty_text(
    value: object,
    context: str,
) -> str:
    """Require non-empty text."""

    text = str(value).strip()

    if not text:
        raise ValueError(f"{context} cannot be empty.")

    return text


def _require_optional_path_text(
    value: object,
    context: str,
) -> str | None:
    """Validate an optional local checkpoint path string."""

    if value is None:
        return None

    text = _require_nonempty_text(value, context)

    if Path(text).is_absolute() and ".." in Path(text).parts:
        raise ValueError(f"{context} contains an unsafe path.")

    return text


def _require_positive_float(
    value: object,
    context: str,
) -> float:
    """Return one finite positive floating-point value."""

    return float(require_positive_number(value, context))


def _require_nonnegative_float(
    value: object,
    context: str,
) -> float:
    """Return one finite non-negative floating-point value."""

    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric.")

    number = float(value)

    if number < 0.0 or number == float("inf") or number != number:
        raise ValueError(
            f"{context} must be finite and non-negative."
        )

    return number


def _require_open_upper_probability(
    value: object,
    context: str,
) -> float:
    """Require a probability in the half-open interval [0, 1)."""

    probability = float(require_probability(value, context))

    if probability >= 1.0:
        raise ValueError(f"{context} must be in [0, 1).")

    return probability


def _require_positive_integer_sequence(
    value: object,
    *,
    context: str,
    expected_length: int,
) -> tuple[int, ...]:
    """Validate a fixed-length sequence of positive integers."""

    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise TypeError(f"{context} must be a sequence.")

    resolved = tuple(
        require_positive_integer(item, f"{context} value")
        for item in value
    )

    if len(resolved) != expected_length:
        raise ValueError(
            f"{context} must contain exactly "
            f"{expected_length} values."
        )

    return resolved


def validate_baseline_experiment_identity(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate experiment identity and the E01-E05 model assignment."""

    experiment = require_mapping(
        payload.get("experiment"),
        "experiment",
    )

    require_keys(
        experiment,
        [
            "id",
            "name",
            "purpose",
            "primary_seed",
            "additional_reproducibility_seeds",
            "deterministic",
        ],
        "experiment",
    )

    experiment_id = _require_nonempty_text(
        experiment["id"],
        "experiment.id",
    ).upper()

    if experiment_id not in BASELINE_EXPERIMENT_MODEL_MAP:
        raise RuntimeError(
            "Baseline experiment ID must be one of "
            f"{list(BASELINE_EXPERIMENT_MODEL_MAP)}, "
            f"received {experiment_id!r}."
        )

    experiment_name = _require_nonempty_text(
        experiment["name"],
        "experiment.name",
    )

    purpose = _require_nonempty_text(
        experiment["purpose"],
        "experiment.purpose",
    )

    primary_seed = require_nonnegative_integer(
        experiment["primary_seed"],
        "experiment.primary_seed",
    )

    if primary_seed != 42:
        raise RuntimeError(
            "Primary controlled baseline runs must use seed 42."
        )

    additional_seeds_value = experiment[
        "additional_reproducibility_seeds"
    ]

    if (
        isinstance(additional_seeds_value, (str, bytes))
        or not isinstance(additional_seeds_value, Sequence)
    ):
        raise TypeError(
            "experiment.additional_reproducibility_seeds "
            "must be a sequence."
        )

    additional_seeds = tuple(
        require_nonnegative_integer(
            value,
            "experiment.additional_reproducibility_seeds value",
        )
        for value in additional_seeds_value
    )

    if len(set(additional_seeds)) != len(additional_seeds):
        raise ValueError(
            "Additional reproducibility seeds must be unique."
        )

    if primary_seed in additional_seeds:
        raise ValueError(
            "Additional reproducibility seeds must not repeat "
            "the primary seed."
        )

    if (
        require_boolean(
            experiment["deterministic"],
            "experiment.deterministic",
        )
        is not True
    ):
        raise RuntimeError(
            "Controlled baseline experiments must be deterministic."
        )

    return {
        "id": experiment_id,
        "name": experiment_name,
        "purpose": purpose,
        "primary_seed": primary_seed,
        "additional_reproducibility_seeds": list(additional_seeds),
        "expected_model": BASELINE_EXPERIMENT_MODEL_MAP[
            experiment_id
        ],
        "deterministic": True,
    }


def _validate_common_model_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate model settings shared by every baseline."""

    input_channels = require_positive_integer(
        parameters["input_channels"],
        "model.parameters.input_channels",
    )

    output_channels = require_positive_integer(
        parameters["output_channels"],
        "model.parameters.output_channels",
    )

    if input_channels != 3:
        raise RuntimeError(
            "All baseline models must use three-channel RGB input."
        )

    if output_channels != 1:
        raise RuntimeError(
            "All baseline models must return one binary mask logit channel."
        )

    return {
        "input_channels": input_channels,
        "output_channels": output_channels,
    }


def _validate_model_specific_parameters(
    model_name: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate architecture-specific baseline parameters."""

    result: dict[str, Any] = {}

    if model_name == "unet":
        result["base_channels"] = require_positive_integer(
            parameters["base_channels"],
            "model.parameters.base_channels",
        )
        result["bilinear"] = require_boolean(
            parameters["bilinear"],
            "model.parameters.bilinear",
        )
        result["dropout_probability"] = _require_open_upper_probability(
            parameters["dropout_probability"],
            "model.parameters.dropout_probability",
        )

    elif model_name == "unetpp":
        result["base_channels"] = require_positive_integer(
            parameters["base_channels"],
            "model.parameters.base_channels",
        )
        deep_supervision = require_boolean(
            parameters["deep_supervision"],
            "model.parameters.deep_supervision",
        )

        if deep_supervision:
            raise RuntimeError(
                "UNet++ deep supervision must remain disabled for the "
                "mask-only baseline comparison."
            )

        result["deep_supervision"] = False
        result["dropout_probability"] = _require_open_upper_probability(
            parameters["dropout_probability"],
            "model.parameters.dropout_probability",
        )

    elif model_name == "deeplabv3plus":
        backbone_name = _require_nonempty_text(
            parameters["backbone_name"],
            "model.parameters.backbone_name",
        ).lower()

        if backbone_name not in {"resnet50", "resnet101"}:
            raise ValueError(
                "DeepLabV3+ backbone_name must be resnet50 or resnet101."
            )

        pretrained = require_boolean(
            parameters["backbone_pretrained"],
            "model.parameters.backbone_pretrained",
        )

        weights_path = _require_optional_path_text(
            parameters["backbone_weights_path"],
            "model.parameters.backbone_weights_path",
        )

        if pretrained and weights_path is not None:
            raise ValueError(
                "Use torchvision pretrained weights or a local backbone "
                "checkpoint, not both."
            )

        result.update(
            {
                "backbone_name": backbone_name,
                "backbone_pretrained": pretrained,
                "backbone_weights_path": weights_path,
                "aspp_channels": require_positive_integer(
                    parameters["aspp_channels"],
                    "model.parameters.aspp_channels",
                ),
                "decoder_channels": require_positive_integer(
                    parameters["decoder_channels"],
                    "model.parameters.decoder_channels",
                ),
                "low_level_projection_channels": (
                    require_positive_integer(
                        parameters["low_level_projection_channels"],
                        "model.parameters.low_level_projection_channels",
                    )
                ),
                "atrous_rates": list(
                    _require_positive_integer_sequence(
                        parameters["atrous_rates"],
                        context="model.parameters.atrous_rates",
                        expected_length=3,
                    )
                ),
                "dropout_probability": _require_open_upper_probability(
                    parameters["dropout_probability"],
                    "model.parameters.dropout_probability",
                ),
            }
        )

    elif model_name == "transunet":
        encoder_blocks = _require_positive_integer_sequence(
            parameters["encoder_blocks"],
            context="model.parameters.encoder_blocks",
            expected_length=4,
        )

        transformer_dimension = require_positive_integer(
            parameters["transformer_dimension"],
            "model.parameters.transformer_dimension",
        )

        transformer_heads = require_positive_integer(
            parameters["transformer_heads"],
            "model.parameters.transformer_heads",
        )

        if transformer_dimension % transformer_heads != 0:
            raise ValueError(
                "TransUNet transformer_dimension must be divisible by "
                "transformer_heads."
            )

        if transformer_dimension % 4 != 0:
            raise ValueError(
                "TransUNet transformer_dimension must be divisible by four."
            )

        result.update(
            {
                "base_channels": require_positive_integer(
                    parameters["base_channels"],
                    "model.parameters.base_channels",
                ),
                "encoder_blocks": list(encoder_blocks),
                "transformer_dimension": transformer_dimension,
                "transformer_layers": require_positive_integer(
                    parameters["transformer_layers"],
                    "model.parameters.transformer_layers",
                ),
                "transformer_heads": transformer_heads,
                "transformer_mlp_dimension": require_positive_integer(
                    parameters["transformer_mlp_dimension"],
                    "model.parameters.transformer_mlp_dimension",
                ),
                "transformer_dropout": _require_open_upper_probability(
                    parameters["transformer_dropout"],
                    "model.parameters.transformer_dropout",
                ),
                "attention_dropout": _require_open_upper_probability(
                    parameters["attention_dropout"],
                    "model.parameters.attention_dropout",
                ),
                "bottleneck_dropout": _require_open_upper_probability(
                    parameters["bottleneck_dropout"],
                    "model.parameters.bottleneck_dropout",
                ),
                "bilinear_decoder": require_boolean(
                    parameters["bilinear_decoder"],
                    "model.parameters.bilinear_decoder",
                ),
            }
        )

    elif model_name == "swin_unet":
        depths = _require_positive_integer_sequence(
            parameters["depths"],
            context="model.parameters.depths",
            expected_length=4,
        )

        heads = _require_positive_integer_sequence(
            parameters["number_of_heads"],
            context="model.parameters.number_of_heads",
            expected_length=4,
        )

        embedding_dimension = require_positive_integer(
            parameters["embedding_dimension"],
            "model.parameters.embedding_dimension",
        )

        stage_dimensions = tuple(
            embedding_dimension * (2 ** index)
            for index in range(4)
        )

        for stage_index, (dimension, head_count) in enumerate(
            zip(stage_dimensions, heads, strict=True),
            start=1,
        ):
            if dimension % head_count != 0:
                raise ValueError(
                    "Swin-UNet stage dimension must be divisible by its "
                    f"head count at stage {stage_index}."
                )

        result.update(
            {
                "patch_size": require_positive_integer(
                    parameters["patch_size"],
                    "model.parameters.patch_size",
                ),
                "embedding_dimension": embedding_dimension,
                "depths": list(depths),
                "number_of_heads": list(heads),
                "window_size": require_positive_integer(
                    parameters["window_size"],
                    "model.parameters.window_size",
                ),
                "mlp_ratio": _require_positive_float(
                    parameters["mlp_ratio"],
                    "model.parameters.mlp_ratio",
                ),
                "dropout_probability": _require_open_upper_probability(
                    parameters["dropout_probability"],
                    "model.parameters.dropout_probability",
                ),
                "attention_dropout": _require_open_upper_probability(
                    parameters["attention_dropout"],
                    "model.parameters.attention_dropout",
                ),
                "drop_path_rate": _require_open_upper_probability(
                    parameters["drop_path_rate"],
                    "model.parameters.drop_path_rate",
                ),
            }
        )

    else:
        raise AssertionError("Unreachable baseline model branch.")

    return result


def validate_baseline_model_and_loss(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Validate a mask-only baseline model and BCE-Dice criterion."""

    model = require_mapping(payload.get("model"), "model")
    require_keys(model, ["name", "parameters"], "model")

    if set(model) != {"name", "parameters"}:
        raise ValueError(
            "Baseline model section may contain only 'name' and "
            "'parameters'."
        )

    model_name = normalize_baseline_name(model["name"])
    expected_model = BASELINE_EXPERIMENT_MODEL_MAP[experiment_id]

    if model_name != expected_model:
        raise RuntimeError(
            f"{experiment_id} must use {expected_model!r}, "
            f"not {model_name!r}."
        )

    parameters = require_mapping(
        model["parameters"],
        "model.parameters",
    )

    allowed_parameters = MODEL_ALLOWED_PARAMETERS[model_name]
    observed_parameters = set(parameters)

    unknown_parameters = sorted(
        observed_parameters - allowed_parameters
    )

    if unknown_parameters:
        raise ValueError(
            f"Unsupported parameters for {model_name}: "
            f"{unknown_parameters}."
        )

    required_parameters = BASELINE_REQUIRED_MODEL_PARAMETERS[model_name]
    missing_parameters = sorted(
        required_parameters - observed_parameters
    )

    if missing_parameters:
        raise KeyError(
            f"Missing required parameters for {model_name}: "
            f"{missing_parameters}."
        )

    if observed_parameters != required_parameters:
        unexpected = sorted(
            observed_parameters - required_parameters
        )
        raise ValueError(
            "Baseline model configuration must use the complete canonical "
            f"parameter set. Unexpected parameters: {unexpected}."
        )

    common_parameters = _validate_common_model_parameters(parameters)
    specific_parameters = _validate_model_specific_parameters(
        model_name,
        parameters,
    )

    loss = require_mapping(payload.get("loss"), "loss")
    required_loss_keys = {
        "name",
        "weights",
        "pos_weight",
        "dice_smooth",
        "dice_epsilon",
        "reduction",
        "return_components",
    }

    require_keys(loss, sorted(required_loss_keys), "loss")

    if set(loss) != required_loss_keys:
        raise ValueError(
            "Baseline loss section contains unsupported fields: "
            f"{sorted(set(loss) - required_loss_keys)}."
        )

    loss_name = _require_nonempty_text(
        loss["name"],
        "loss.name",
    ).lower()

    if loss_name != "bce_dice_loss":
        raise RuntimeError(
            "Baseline experiments must use bce_dice_loss."
        )

    weights = require_mapping(loss["weights"], "loss.weights")
    require_keys(weights, ["bce", "dice"], "loss.weights")

    if set(weights) != {"bce", "dice"}:
        raise ValueError(
            "Baseline loss weights may contain only 'bce' and 'dice'."
        )

    bce_weight = _require_nonnegative_float(
        weights["bce"],
        "loss.weights.bce",
    )
    dice_weight = _require_nonnegative_float(
        weights["dice"],
        "loss.weights.dice",
    )

    if bce_weight != 1.0 or dice_weight != 1.0:
        raise RuntimeError(
            "The controlled baseline protocol requires BCE weight 1.0 "
            "and Dice weight 1.0."
        )

    if loss["pos_weight"] is not None:
        raise RuntimeError(
            "The primary controlled baseline protocol requires "
            "loss.pos_weight: null."
        )

    dice_smooth = _require_nonnegative_float(
        loss["dice_smooth"],
        "loss.dice_smooth",
    )
    dice_epsilon = _require_positive_float(
        loss["dice_epsilon"],
        "loss.dice_epsilon",
    )

    if dice_smooth != 1.0:
        raise RuntimeError("loss.dice_smooth must equal 1.0.")

    if dice_epsilon != 1e-7:
        raise RuntimeError("loss.dice_epsilon must equal 1e-7.")

    reduction = _require_nonempty_text(
        loss["reduction"],
        "loss.reduction",
    ).lower()

    if reduction != "mean":
        raise RuntimeError("Baseline loss reduction must be mean.")

    if (
        require_boolean(
            loss["return_components"],
            "loss.return_components",
        )
        is not True
    ):
        raise RuntimeError(
            "Baseline loss must return components for trainer logging."
        )

    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "model_parameters": {
            **common_parameters,
            **specific_parameters,
        },
        "output_keys": ["mask_logits"],
        "learning_type": "fully_supervised",
        "uses_boundary_conditioning": False,
        "uses_auxiliary_targets": False,
        "loss": {
            "name": loss_name,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            "pos_weight": None,
            "dice_smooth": dice_smooth,
            "dice_epsilon": dice_epsilon,
            "reduction": reduction,
            "return_components": True,
        },
    }


def validate_baseline_inference_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate mask-only baseline inference settings."""

    inference = require_mapping(payload.get("inference"), "inference")
    required_inference_keys = {
        "probability_threshold",
        "save_binary_predictions",
        "save_probability_maps",
        "save_contour_predictions",
        "save_sdm_predictions",
    }

    require_keys(inference, sorted(required_inference_keys), "inference")

    if set(inference) != required_inference_keys:
        raise ValueError(
            "Baseline inference section contains unsupported fields: "
            f"{sorted(set(inference) - required_inference_keys)}."
        )

    threshold = require_probability(
        inference["probability_threshold"],
        "inference.probability_threshold",
    )

    if threshold != 0.5:
        raise RuntimeError(
            "The controlled baseline probability threshold must be 0.5."
        )

    for key in (
        "save_binary_predictions",
        "save_probability_maps",
    ):
        if require_boolean(inference[key], f"inference.{key}") is not True:
            raise RuntimeError(f"inference.{key} must be enabled.")

    for key in (
        "save_contour_predictions",
        "save_sdm_predictions",
    ):
        if require_boolean(inference[key], f"inference.{key}") is not False:
            raise RuntimeError(
                f"Mask-only baseline inference must disable {key}."
            )

    return {
        "probability_threshold": threshold,
        "save_binary_predictions": True,
        "save_probability_maps": True,
        "save_contour_predictions": False,
        "save_sdm_predictions": False,
    }


def validate_baseline_output_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate experiment output root and relative directory names."""

    outputs = require_mapping(payload.get("outputs"), "outputs")
    require_keys(outputs, ["root", "directories"], "outputs")

    root = _require_nonempty_text(outputs["root"], "outputs.root")
    directories = require_mapping(
        outputs["directories"],
        "outputs.directories",
    )

    require_keys(
        directories,
        list(BASELINE_REQUIRED_OUTPUT_DIRECTORIES),
        "outputs.directories",
    )

    resolved_directories: dict[str, str] = {}

    for name in BASELINE_REQUIRED_OUTPUT_DIRECTORIES:
        relative_text = _require_nonempty_text(
            directories[name],
            f"outputs.directories.{name}",
        )
        relative_path = Path(relative_text)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"outputs.directories.{name} must be a safe relative path."
            )

        resolved_directories[name] = relative_text

    if len(set(resolved_directories.values())) != len(
        resolved_directories
    ):
        raise ValueError("Output directory names must be unique.")

    return {
        "root": root,
        "directories": resolved_directories,
    }


def validate_baseline_experiment_configuration(
    config_path: str | Path,
) -> ValidatedBaselineExperimentConfig:
    """Load and fully validate one E01-E05 baseline YAML file."""

    source_path = Path(config_path).expanduser().resolve()
    payload = load_yaml(source_path)

    require_keys(
        payload,
        list(BASELINE_REQUIRED_TOP_LEVEL_KEYS),
        "baseline experiment configuration",
    )

    if payload["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported configuration schema: "
            f"{payload['schema_version']!r}."
        )

    identity = validate_baseline_experiment_identity(payload)

    artifact_roots = resolve_artifact_roots(payload)

    (
        training_readiness_path,
        training_readiness,
    ) = validate_training_readiness(payload, artifact_roots)

    if training_readiness.get("status") != EXPECTED_TRAINING_STATUS:
        raise RuntimeError("Training-readiness status is not approved.")

    manifests = resolve_manifests(payload, artifact_roots)

    validation_summary = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "experiment": identity,
        "training_readiness": {
            "status": training_readiness["status"],
            "training_allowed": training_readiness["training_allowed"],
        },
        "data": validate_data_protocol(payload),
        "augmentation": validate_augmentation_protocol(payload),
        "model_and_loss": validate_baseline_model_and_loss(
            payload,
            experiment_id=identity["id"],
        ),
        "training": validate_training_protocol(payload),
        "inference": validate_baseline_inference_protocol(payload),
        "evaluation": validate_evaluation_protocol(payload),
        "reproducibility": validate_reproducibility_protocol(payload),
        "smoke_test": validate_smoke_test_protocol(payload),
        "outputs": validate_baseline_output_protocol(payload),
        "manifest_count": len(manifests),
        "all_checks_passed": True,
    }

    return ValidatedBaselineExperimentConfig(
        source_path=source_path,
        payload=payload,
        source_file_sha256=sha256_file(source_path),
        canonical_sha256=canonical_payload_sha256(payload),
        artifact_roots=artifact_roots,
        manifests=manifests,
        training_readiness_path=training_readiness_path,
        training_readiness=training_readiness,
        validation_summary=validation_summary,
    )


def save_baseline_validation_bundle(
    validated: ValidatedBaselineExperimentConfig,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save an exact configuration copy and baseline validation report."""

    if not isinstance(validated, ValidatedBaselineExperimentConfig):
        raise TypeError(
            "validated must be a ValidatedBaselineExperimentConfig."
        )

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configuration_copy_path = output_root / validated.source_path.name
    validation_report_path = output_root / "BASELINE_CONFIG_VALIDATION.json"

    shutil.copy2(validated.source_path, configuration_copy_path)

    if sha256_file(configuration_copy_path) != validated.source_file_sha256:
        raise RuntimeError(
            "Saved configuration copy does not match the source file."
        )

    temporary_report_path = validation_report_path.with_suffix(
        ".json.tmp"
    )

    temporary_report_path.write_text(
        json.dumps(validated.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_report_path.replace(validation_report_path)

    return {
        "configuration_copy": configuration_copy_path,
        "validation_report": validation_report_path,
    }


def summarize_validated_baseline_configuration(
    validated: ValidatedBaselineExperimentConfig,
) -> str:
    """Return a readable baseline configuration summary."""

    identity = validated.validation_summary["experiment"]
    model_and_loss = validated.validation_summary["model_and_loss"]

    return "\n".join(
        [
            f"Baseline configuration : {validated.source_path}",
            f"Experiment ID         : {identity['id']}",
            f"Experiment name       : {identity['name']}",
            f"Baseline model        : {model_and_loss['model_name']}",
            f"Configuration SHA-256 : {validated.canonical_sha256}",
            f"Training readiness    : {validated.training_readiness['status']}",
            f"Train rows            : {validated.manifests['train'].observed_rows}",
            (
                "Validation rows       : "
                f"{validated.manifests['validation'].observed_rows}"
            ),
            "Validation result      : all checks passed",
        ]
    )


def _self_test_model_parameters() -> dict[str, dict[str, Any]]:
    """Return reduced canonical parameters used only for validation tests."""

    return {
        "unet": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 16,
            "bilinear": False,
            "dropout_probability": 0.0,
        },
        "unetpp": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 16,
            "deep_supervision": False,
            "dropout_probability": 0.0,
        },
        "deeplabv3plus": {
            "input_channels": 3,
            "output_channels": 1,
            "backbone_name": "resnet50",
            "backbone_pretrained": False,
            "backbone_weights_path": None,
            "aspp_channels": 64,
            "decoder_channels": 64,
            "low_level_projection_channels": 24,
            "atrous_rates": [6, 12, 18],
            "dropout_probability": 0.1,
        },
        "transunet": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 16,
            "encoder_blocks": [1, 1, 1, 1],
            "transformer_dimension": 128,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "transformer_mlp_dimension": 256,
            "transformer_dropout": 0.1,
            "attention_dropout": 0.0,
            "bottleneck_dropout": 0.0,
            "bilinear_decoder": True,
        },
        "swin_unet": {
            "input_channels": 3,
            "output_channels": 1,
            "patch_size": 4,
            "embedding_dimension": 24,
            "depths": [1, 1, 1, 1],
            "number_of_heads": [3, 3, 6, 12],
            "window_size": 4,
            "mlp_ratio": 2.0,
            "dropout_probability": 0.0,
            "attention_dropout": 0.0,
            "drop_path_rate": 0.0,
        },
    }


def run_baseline_config_self_test() -> dict[str, Any]:
    """Validate all five model assignments without requiring artifacts."""

    parameters_by_model = _self_test_model_parameters()
    validated_models: dict[str, str] = {}

    for experiment_id, model_name in BASELINE_EXPERIMENT_MODEL_MAP.items():
        payload = {
            "experiment": {
                "id": experiment_id,
                "name": f"{experiment_id.lower()}_{model_name}",
                "purpose": "baseline_validation_self_test",
                "primary_seed": 42,
                "additional_reproducibility_seeds": [123, 2026],
                "deterministic": True,
            },
            "model": {
                "name": model_name,
                "parameters": parameters_by_model[model_name],
            },
            "loss": {
                "name": "bce_dice_loss",
                "weights": {
                    "bce": 1.0,
                    "dice": 1.0,
                },
                "pos_weight": None,
                "dice_smooth": 1.0,
                "dice_epsilon": 1e-7,
                "reduction": "mean",
                "return_components": True,
            },
        }

        identity = validate_baseline_experiment_identity(payload)
        result = validate_baseline_model_and_loss(
            payload,
            experiment_id=identity["id"],
        )
        validated_models[experiment_id] = result["model_name"]

    mismatched_model_rejected = False
    mismatch_payload = {
        "model": {
            "name": "transunet",
            "parameters": parameters_by_model["transunet"],
        },
        "loss": {
            "name": "bce_dice_loss",
            "weights": {"bce": 1.0, "dice": 1.0},
            "pos_weight": None,
            "dice_smooth": 1.0,
            "dice_epsilon": 1e-7,
            "reduction": "mean",
            "return_components": True,
        },
    }

    try:
        validate_baseline_model_and_loss(
            mismatch_payload,
            experiment_id="E01",
        )
    except RuntimeError:
        mismatched_model_rejected = True

    proposed_loss_rejected = False
    proposed_payload = {
        "model": {
            "name": "unet",
            "parameters": parameters_by_model["unet"],
        },
        "loss": {
            "name": "bcs_hctnet_composite_loss",
            "weights": {"bce": 1.0, "dice": 1.0},
            "pos_weight": None,
            "dice_smooth": 1.0,
            "dice_epsilon": 1e-7,
            "reduction": "mean",
            "return_components": True,
        },
    }

    try:
        validate_baseline_model_and_loss(
            proposed_payload,
            experiment_id="E01",
        )
    except RuntimeError:
        proposed_loss_rejected = True

    checks = {
        "all_five_experiments_validated": (
            validated_models == BASELINE_EXPERIMENT_MODEL_MAP
        ),
        "mismatched_model_rejected": mismatched_model_rejected,
        "proposed_loss_rejected": proposed_loss_rejected,
        "mask_only_protocol": all(
            model_name in MODEL_ALLOWED_PARAMETERS
            for model_name in validated_models.values()
        ),
    }

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol_version": BASELINE_CONFIG_VALIDATION_PROTOCOL_VERSION,
        "checks": checks,
        "experiment_model_map": validated_models,
    }
