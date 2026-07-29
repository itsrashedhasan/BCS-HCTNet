"""Shared supervised segmentation training engine.

This module provides the common training implementation for:

- baseline lesion-segmentation models;
- BCS-HCTNet;
- later ablation variants.

The trainer includes:

- device-safe batch transfer;
- standard or multi-output criteria;
- optional CUDA automatic mixed precision;
- gradient clipping;
- sample-weighted loss aggregation;
- common validation through ``src.train.validate``;
- scheduler stepping;
- early stopping;
- best and latest checkpoints;
- persistent epoch logging;
- per-epoch validation reports;
- Kaggle-session resume state.

No model-specific architecture logic belongs in this module.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.train.checkpoint import (
    save_checkpoint,
)
from src.train.scheduler import (
    SchedulerBundle,
    current_learning_rates,
    step_scheduler,
)
from src.train.validate import (
    CriterionMode,
    ValidationResult,
    compute_validation_loss,
    extract_mask_logits,
    extract_validation_image,
    extract_validation_target,
    validate_segmentation_model,
)
from src.utils.logger import (
    ExperimentLogger,
)


TRAINER_PROTOCOL_VERSION = (
    "BCS-HCTNet-shared-supervised-trainer-v1"
)

MonitorMode = Literal[
    "min",
    "max",
]

DEFAULT_LOGGER_FIELDS = (
    "epoch",
    "global_step",
    "learning_rate_start",
    "learning_rate_next",
    "train_loss",
    "validation_loss",
    "validation_dice",
    "validation_iou",
    "validation_thresholded_jaccard",
    "validation_boundary_f1",
    "validation_hd95",
    "validation_assd",
    "monitor_value",
    "is_best",
    "best_epoch",
    "epochs_without_improvement",
    "train_seconds",
    "validation_seconds",
)


@dataclass(frozen=True)
class TrainingEpochResult:
    """Result of one completed training epoch."""

    protocol_version: str
    epoch: int
    global_step_start: int
    global_step_end: int
    number_of_batches: int
    number_of_images: int
    mean_loss: float
    mean_loss_components: dict[str, float]
    learning_rates: list[float]
    elapsed_seconds: float
    images_per_second: float
    maximum_gradient_norm: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "protocol_version": (
                self.protocol_version
            ),
            "epoch": self.epoch,
            "global_step_start": (
                self.global_step_start
            ),
            "global_step_end": (
                self.global_step_end
            ),
            "number_of_batches": (
                self.number_of_batches
            ),
            "number_of_images": (
                self.number_of_images
            ),
            "mean_loss": (
                self.mean_loss
            ),
            "mean_loss_components": dict(
                self.mean_loss_components
            ),
            "learning_rates": list(
                self.learning_rates
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "images_per_second": (
                self.images_per_second
            ),
            "maximum_gradient_norm": (
                self.maximum_gradient_norm
            ),
        }


def resolve_training_device(
    device: str | torch.device | None,
) -> torch.device:
    """Resolve and validate the requested training device."""

    if device is None:
        resolved = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        resolved = torch.device(
            device
        )

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA training was requested, but "
            "CUDA is not available."
        )

    return resolved


def create_gradient_scaler(
    *,
    enabled: bool,
) -> object:
    """Create a GradScaler across supported PyTorch APIs."""

    if not isinstance(
        enabled,
        bool,
    ):
        raise TypeError(
            "enabled must be Boolean."
        )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def move_value_to_device(
    value: Any,
    *,
    device: torch.device,
    non_blocking: bool,
) -> Any:
    """Recursively move tensor values to the training device."""

    if isinstance(
        value,
        Tensor,
    ):
        return value.to(
            device=device,
            non_blocking=non_blocking,
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            key: move_value_to_device(
                nested_value,
                device=device,
                non_blocking=non_blocking,
            )
            for key, nested_value
            in value.items()
        }

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            move_value_to_device(
                nested_value,
                device=device,
                non_blocking=non_blocking,
            )
            for nested_value in value
        )

    if isinstance(
        value,
        list,
    ):
        return [
            move_value_to_device(
                nested_value,
                device=device,
                non_blocking=non_blocking,
            )
            for nested_value in value
        ]

    return value


def move_batch_to_device(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Move a mapping-based batch recursively to one device."""

    if not isinstance(
        batch,
        Mapping,
    ):
        raise TypeError(
            "Training batch must be a mapping."
        )

    non_blocking = (
        device.type == "cuda"
    )

    return {
        key: move_value_to_device(
            value,
            device=device,
            non_blocking=non_blocking,
        )
        for key, value
        in batch.items()
    }


def validate_gradient_clip_norm(
    value: float | None,
) -> float | None:
    """Validate an optional positive gradient-clipping norm."""

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            "gradient_clip_norm must be numeric."
        )

    resolved = float(
        value
    )

    if (
        not math.isfinite(resolved)
        or resolved <= 0.0
    ):
        raise ValueError(
            "gradient_clip_norm must be positive "
            "and finite."
        )

    return resolved


def validate_positive_optional_integer(
    value: int | None,
    context: str,
) -> int | None:
    """Validate an optional positive integer."""

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{context} must be None or a "
            "positive integer."
        )

    return value


def validate_criterion_mode(
    criterion_mode: object,
) -> CriterionMode:
    """Validate the supported criterion invocation mode."""

    normalized = str(
        criterion_mode
    ).strip().lower()

    if normalized not in {
        "logits_target",
        "output_batch",
    }:
        raise ValueError(
            "criterion_mode must be "
            "'logits_target' or 'output_batch'."
        )

    return normalized  # type: ignore[return-value]


def train_one_epoch(
    *,
    model: nn.Module,
    data_loader: Iterable[
        Mapping[str, Any]
    ],
    optimizer: Optimizer,
    criterion: Any,
    epoch: int,
    global_step_start: int,
    device: str | torch.device,
    criterion_mode: CriterionMode = (
        "logits_target"
    ),
    amp_enabled: bool = False,
    scaler: object | None = None,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    allow_logit_resize: bool = False,
) -> TrainingEpochResult:
    """Train a segmentation model for one epoch."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise TypeError(
            "optimizer must be a "
            "torch.optim.Optimizer."
        )

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(
            "epoch must be a non-negative integer."
        )

    if (
        isinstance(global_step_start, bool)
        or not isinstance(
            global_step_start,
            int,
        )
        or global_step_start < 0
    ):
        raise ValueError(
            "global_step_start must be a "
            "non-negative integer."
        )

    if not isinstance(
        amp_enabled,
        bool,
    ):
        raise TypeError(
            "amp_enabled must be Boolean."
        )

    if not isinstance(
        allow_logit_resize,
        bool,
    ):
        raise TypeError(
            "allow_logit_resize must be Boolean."
        )

    resolved_device = torch.device(
        device
    )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA training was requested, but "
            "CUDA is not available."
        )

    resolved_criterion_mode = (
        validate_criterion_mode(
            criterion_mode
        )
    )

    resolved_gradient_clip = (
        validate_gradient_clip_norm(
            gradient_clip_norm
        )
    )

    resolved_max_batches = (
        validate_positive_optional_integer(
            max_batches,
            "max_batches",
        )
    )

    effective_amp = bool(
        amp_enabled
        and resolved_device.type == "cuda"
    )

    active_scaler = (
        scaler
        if scaler is not None
        else create_gradient_scaler(
            enabled=effective_amp
        )
    )

    scaler_scale = getattr(
        active_scaler,
        "scale",
        None,
    )

    scaler_step = getattr(
        active_scaler,
        "step",
        None,
    )

    scaler_update = getattr(
        active_scaler,
        "update",
        None,
    )

    scaler_unscale = getattr(
        active_scaler,
        "unscale_",
        None,
    )

    if not all(
        callable(function)
        for function in (
            scaler_scale,
            scaler_step,
            scaler_update,
            scaler_unscale,
        )
    ):
        raise TypeError(
            "scaler must provide scale(), step(), "
            "update(), and unscale_()."
        )

    model.train()

    global_step = int(
        global_step_start
    )

    completed_batches = 0
    total_images = 0
    weighted_total_loss = 0.0

    weighted_components: dict[
        str,
        float,
    ] = {}

    maximum_gradient_norm: float | None = None

    start_time = time.perf_counter()

    for batch_index, raw_batch in enumerate(
        data_loader
    ):
        if (
            resolved_max_batches is not None
            and batch_index
            >= resolved_max_batches
        ):
            break

        if not isinstance(
            raw_batch,
            Mapping,
        ):
            raise TypeError(
                "Every training batch must be "
                "a mapping."
            )

        batch = move_batch_to_device(
            raw_batch,
            device=resolved_device,
        )

        image = extract_validation_image(
            batch
        ).to(
            dtype=torch.float32
        )

        target = extract_validation_target(
            batch
        ).to(
            dtype=torch.float32
        )

        if (
            image.shape[0]
            != target.shape[0]
        ):
            raise RuntimeError(
                "Training image and target batch "
                "sizes do not match."
            )

        batch_size = int(
            image.shape[0]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=(
                resolved_device.type
            ),
            dtype=torch.float16,
            enabled=effective_amp,
        ):
            model_output = model(
                image
            )

            mask_logits = extract_mask_logits(
                model_output
            )

            if (
                mask_logits.shape[0]
                != batch_size
            ):
                raise RuntimeError(
                    "Training mask-logit batch size "
                    "does not match input."
                )

            if (
                mask_logits.shape[-2:]
                != target.shape[-2:]
            ):
                if not allow_logit_resize:
                    raise RuntimeError(
                        "Training logits and target "
                        "spatial dimensions differ. "
                        f"Logits: "
                        f"{tuple(mask_logits.shape)}; "
                        f"target: "
                        f"{tuple(target.shape)}."
                    )

                mask_logits = (
                    torch.nn.functional.interpolate(
                        mask_logits,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )

                if isinstance(
                    model_output,
                    Mapping,
                ):
                    model_output = dict(
                        model_output
                    )

                    model_output[
                        "mask_logits"
                    ] = mask_logits

            (
                total_loss,
                loss_components,
            ) = compute_validation_loss(
                criterion=criterion,
                criterion_mode=(
                    resolved_criterion_mode
                ),
                model_output=model_output,
                mask_logits=mask_logits,
                target=target,
                batch=batch,
            )

        if not total_loss.requires_grad:
            raise RuntimeError(
                "Training loss does not require "
                "gradients."
            )

        scaled_loss = scaler_scale(
            total_loss
        )

        scaled_loss.backward()

        scaler_unscale(
            optimizer
        )

        if resolved_gradient_clip is not None:
            gradient_norm_tensor = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        resolved_gradient_clip
                    ),
                    # During CUDA AMP, GradScaler may intentionally
                    # encounter non-finite scaled gradients while it
                    # calibrates the loss scale. ``unscale_`` records
                    # that state, and ``step`` safely skips the optimizer
                    # update before ``update`` reduces the scale. Raising
                    # here would prevent that standard recovery path.
                    error_if_nonfinite=(
                        not effective_amp
                    ),
                )
            )

            gradient_norm = float(
                gradient_norm_tensor.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .item()
            )

            if math.isfinite(
                gradient_norm
            ):
                maximum_gradient_norm = (
                    gradient_norm
                    if maximum_gradient_norm is None
                    else max(
                        maximum_gradient_norm,
                        gradient_norm,
                    )
                )

            elif not effective_amp:
                raise RuntimeError(
                    "Gradient norm is non-finite."
                )

        scaler_step(
            optimizer
        )

        scaler_update()

        batch_loss = float(
            total_loss.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .item()
        )

        if not math.isfinite(
            batch_loss
        ):
            raise RuntimeError(
                "Training loss is non-finite."
            )

        weighted_total_loss += (
            batch_loss
            * batch_size
        )

        for (
            component_name,
            component_tensor,
        ) in loss_components.items():
            component_value = float(
                component_tensor.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .item()
            )

            if not math.isfinite(
                component_value
            ):
                raise RuntimeError(
                    "Training loss component "
                    f"{component_name!r} is "
                    "non-finite."
                )

            weighted_components[
                component_name
            ] = (
                weighted_components.get(
                    component_name,
                    0.0,
                )
                + component_value
                * batch_size
            )

        total_images += batch_size
        completed_batches += 1
        global_step += 1

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if completed_batches == 0:
        raise RuntimeError(
            "Training data loader produced "
            "no batches."
        )

    if total_images == 0:
        raise RuntimeError(
            "Training processed no images."
        )

    mean_loss = (
        weighted_total_loss
        / total_images
    )

    mean_components = {
        name: (
            weighted_value
            / total_images
        )
        for name, weighted_value
        in weighted_components.items()
    }

    images_per_second = (
        total_images
        / elapsed_seconds
        if elapsed_seconds > 0.0
        else 0.0
    )

    if not math.isfinite(
        images_per_second
    ):
        images_per_second = 0.0

    return TrainingEpochResult(
        protocol_version=(
            TRAINER_PROTOCOL_VERSION
        ),
        epoch=epoch,
        global_step_start=(
            global_step_start
        ),
        global_step_end=global_step,
        number_of_batches=(
            completed_batches
        ),
        number_of_images=(
            total_images
        ),
        mean_loss=float(
            mean_loss
        ),
        mean_loss_components=(
            mean_components
        ),
        learning_rates=(
            current_learning_rates(
                optimizer
            )
        ),
        elapsed_seconds=float(
            elapsed_seconds
        ),
        images_per_second=float(
            images_per_second
        ),
        maximum_gradient_norm=(
            maximum_gradient_norm
        ),
    )


def resolve_validation_monitor(
    result: ValidationResult,
    monitor_name: str,
) -> float:
    """Resolve one monitored validation value."""

    normalized = str(
        monitor_name
    ).strip().lower()

    if not normalized:
        raise ValueError(
            "monitor_name cannot be empty."
        )

    if normalized in {
        "validation_loss",
        "val_loss",
        "loss",
    }:
        if result.mean_loss is None:
            raise RuntimeError(
                "Validation loss is unavailable."
            )

        return float(
            result.mean_loss
        )

    metric_name = normalized

    for prefix in (
        "validation_",
        "val_",
    ):
        if metric_name.startswith(
            prefix
        ):
            metric_name = metric_name[
                len(prefix):
            ]

    metric_summary = result.metric_summary.get(
        "metrics"
    )

    if not isinstance(
        metric_summary,
        Mapping,
    ):
        raise RuntimeError(
            "Validation metric summary is invalid."
        )

    metric_report = metric_summary.get(
        metric_name
    )

    if not isinstance(
        metric_report,
        Mapping,
    ):
        raise KeyError(
            f"Unknown validation monitor "
            f"{monitor_name!r}."
        )

    mean_value = metric_report.get(
        "mean"
    )

    if mean_value is None:
        raise KeyError(
            f"Validation metric "
            f"{metric_name!r} has no mean."
        )

    value = float(
        mean_value
    )

    if not math.isfinite(
        value
    ):
        raise RuntimeError(
            "Validation monitor is non-finite."
        )

    return value


def metric_improved(
    *,
    current: float,
    best: float | None,
    mode: MonitorMode,
    minimum_delta: float,
) -> bool:
    """Determine whether the monitored metric improved."""

    if mode not in {
        "min",
        "max",
    }:
        raise ValueError(
            "mode must be 'min' or 'max'."
        )

    current_value = float(
        current
    )

    delta = float(
        minimum_delta
    )

    if (
        not math.isfinite(current_value)
        or not math.isfinite(delta)
        or delta < 0.0
    ):
        raise ValueError(
            "Metric and minimum_delta must "
            "be finite; minimum_delta must be "
            "non-negative."
        )

    if best is None:
        return True

    best_value = float(
        best
    )

    if not math.isfinite(
        best_value
    ):
        raise ValueError(
            "best metric must be finite."
        )

    if mode == "max":
        return (
            current_value
            > best_value + delta
        )

    return (
        current_value
        < best_value - delta
    )


def extract_metric_mean(
    result: ValidationResult,
    metric_name: str,
) -> float:
    """Extract a named validation metric mean."""

    return resolve_validation_monitor(
        result,
        metric_name,
    )


def write_validation_artifacts(
    *,
    result: ValidationResult,
    output_directory: Path,
    epoch: int,
) -> dict[str, str]:
    """Write per-image CSV and summary JSON for one epoch."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"epoch_{epoch:04d}"
    )

    csv_path = (
        output_directory
        / f"{prefix}_per_image_metrics.csv"
    )

    json_path = (
        output_directory
        / f"{prefix}_validation_summary.json"
    )

    rows = result.per_image_rows

    if not rows:
        raise RuntimeError(
            "Validation result contains no "
            "per-image rows."
        )

    fieldnames = list(
        rows[0]
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    json_path.write_text(
        json.dumps(
            result.to_dict(
                include_per_image_rows=False
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "per_image_csv": str(
            csv_path
        ),
        "summary_json": str(
            json_path
        ),
    }


class SegmentationTrainer:
    """Generic supervised segmentation experiment trainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Any,
        train_loader: Iterable[
            Mapping[str, Any]
        ],
        validation_loader: Iterable[
            Mapping[str, Any]
        ],
        output_directory: str | Path,
        experiment_id: str,
        total_epochs: int,
        scheduler_bundle: SchedulerBundle,
        criterion_mode: CriterionMode = (
            "logits_target"
        ),
        device: str | torch.device | None = None,
        amp_enabled: bool = True,
        gradient_clip_norm: float | None = None,
        monitor_name: str = "validation_dice",
        monitor_mode: MonitorMode = "max",
        minimum_delta: float = 0.0,
        early_stopping_patience: int | None = None,
        prediction_threshold: float = 0.5,
        target_threshold: float = 0.5,
        jaccard_quality_threshold: float = 0.65,
        boundary_tolerance_pixels: float = 2.0,
        dataset_name: str = "ISIC2018",
        validation_split_name: str = "validation",
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
        allow_logit_resize: bool = False,
        start_epoch: int = 0,
        global_step: int = 0,
        best_metric: float | None = None,
        best_epoch: int | None = None,
        epochs_without_improvement: int = 0,
        logger_resume: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the shared trainer."""

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        if not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TypeError(
                "optimizer must be a "
                "torch.optim.Optimizer."
            )

        if not isinstance(
            scheduler_bundle,
            SchedulerBundle,
        ):
            raise TypeError(
                "scheduler_bundle must be a "
                "SchedulerBundle."
            )

        if (
            isinstance(total_epochs, bool)
            or not isinstance(total_epochs, int)
            or total_epochs <= 0
        ):
            raise ValueError(
                "total_epochs must be a "
                "positive integer."
            )

        if (
            isinstance(start_epoch, bool)
            or not isinstance(start_epoch, int)
            or start_epoch < 0
            or start_epoch >= total_epochs
        ):
            raise ValueError(
                "start_epoch must be between zero "
                "and total_epochs - 1."
            )

        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError(
                "global_step must be non-negative."
            )

        if monitor_mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "monitor_mode must be "
                "'min' or 'max'."
            )

        if (
            early_stopping_patience is not None
            and (
                isinstance(
                    early_stopping_patience,
                    bool,
                )
                or not isinstance(
                    early_stopping_patience,
                    int,
                )
                or early_stopping_patience <= 0
            )
        ):
            raise ValueError(
                "early_stopping_patience must be "
                "None or a positive integer."
            )

        if (
            isinstance(
                epochs_without_improvement,
                bool,
            )
            or not isinstance(
                epochs_without_improvement,
                int,
            )
            or epochs_without_improvement < 0
        ):
            raise ValueError(
                "epochs_without_improvement must "
                "be non-negative."
            )

        normalized_experiment_id = str(
            experiment_id
        ).strip()

        if not normalized_experiment_id:
            raise ValueError(
                "experiment_id cannot be empty."
            )

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = (
            validation_loader
        )

        self.output_directory = (
            Path(
                output_directory
            )
            .expanduser()
            .resolve()
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_directory = (
            self.output_directory
            / "checkpoints"
        )

        self.validation_directory = (
            self.output_directory
            / "validation"
        )

        self.log_directory = (
            self.output_directory
            / "logs"
        )

        self.experiment_id = (
            normalized_experiment_id
        )

        self.total_epochs = (
            total_epochs
        )

        self.scheduler_bundle = (
            scheduler_bundle
        )

        self.criterion_mode = (
            validate_criterion_mode(
                criterion_mode
            )
        )

        self.device = (
            resolve_training_device(
                device
            )
        )

        self.amp_enabled = bool(
            amp_enabled
            and self.device.type == "cuda"
        )

        self.gradient_clip_norm = (
            validate_gradient_clip_norm(
                gradient_clip_norm
            )
        )

        self.monitor_name = str(
            monitor_name
        ).strip()

        if not self.monitor_name:
            raise ValueError(
                "monitor_name cannot be empty."
            )

        self.monitor_mode = (
            monitor_mode
        )

        self.minimum_delta = float(
            minimum_delta
        )

        if (
            not math.isfinite(
                self.minimum_delta
            )
            or self.minimum_delta < 0.0
        ):
            raise ValueError(
                "minimum_delta must be finite "
                "and non-negative."
            )

        self.early_stopping_patience = (
            early_stopping_patience
        )

        self.prediction_threshold = float(
            prediction_threshold
        )

        self.target_threshold = float(
            target_threshold
        )

        self.jaccard_quality_threshold = float(
            jaccard_quality_threshold
        )

        self.boundary_tolerance_pixels = float(
            boundary_tolerance_pixels
        )

        self.dataset_name = str(
            dataset_name
        ).strip()

        self.validation_split_name = str(
            validation_split_name
        ).strip()

        self.max_train_batches = (
            validate_positive_optional_integer(
                max_train_batches,
                "max_train_batches",
            )
        )

        self.max_validation_batches = (
            validate_positive_optional_integer(
                max_validation_batches,
                "max_validation_batches",
            )
        )

        self.allow_logit_resize = bool(
            allow_logit_resize
        )

        self.start_epoch = start_epoch
        self.global_step = global_step
        self.best_metric = best_metric
        self.best_epoch = best_epoch

        self.epochs_without_improvement = (
            epochs_without_improvement
        )

        self.scaler = create_gradient_scaler(
            enabled=self.amp_enabled
        )

        self.model.to(
            self.device
        )

        self.logger = ExperimentLogger(
            self.log_directory,
            experiment_id=(
                self.experiment_id
            ),
            fieldnames=(
                DEFAULT_LOGGER_FIELDS
            ),
            resume=logger_resume,
        )

        if not logger_resume:
            logger_metadata = {
                "trainer_protocol_version": (
                    TRAINER_PROTOCOL_VERSION
                ),
                "experiment_id": (
                    self.experiment_id
                ),
                "device": str(
                    self.device
                ),
                "amp_enabled": (
                    self.amp_enabled
                ),
                "total_epochs": (
                    self.total_epochs
                ),
                "criterion_mode": (
                    self.criterion_mode
                ),
                "monitor_name": (
                    self.monitor_name
                ),
                "monitor_mode": (
                    self.monitor_mode
                ),
                "minimum_delta": (
                    self.minimum_delta
                ),
                "early_stopping_patience": (
                    self.early_stopping_patience
                ),
                "scheduler_name": (
                    self.scheduler_bundle.name
                ),
            }

            if metadata is not None:
                for key, value in (
                    metadata.items()
                ):
                    if isinstance(
                        value,
                        (
                            str,
                            int,
                            float,
                            bool,
                            Path,
                        ),
                    ) or value is None:
                        logger_metadata[
                            str(key)
                        ] = value

            self.logger.set_metadata(
                logger_metadata
            )

    def fit(self) -> dict[str, Any]:
        """Run training, validation, persistence, and early stopping."""

        completed_epochs: list[
            dict[str, Any]
        ] = []

        stopped_early = False

        for epoch in range(
            self.start_epoch,
            self.total_epochs,
        ):
            learning_rate_start = (
                current_learning_rates(
                    self.optimizer
                )[0]
            )

            training_result = (
                train_one_epoch(
                    model=self.model,
                    data_loader=(
                        self.train_loader
                    ),
                    optimizer=(
                        self.optimizer
                    ),
                    criterion=self.criterion,
                    epoch=epoch,
                    global_step_start=(
                        self.global_step
                    ),
                    device=self.device,
                    criterion_mode=(
                        self.criterion_mode
                    ),
                    amp_enabled=(
                        self.amp_enabled
                    ),
                    scaler=self.scaler,
                    gradient_clip_norm=(
                        self.gradient_clip_norm
                    ),
                    max_batches=(
                        self.max_train_batches
                    ),
                    allow_logit_resize=(
                        self.allow_logit_resize
                    ),
                )
            )

            self.global_step = (
                training_result
                .global_step_end
            )

            validation_result = (
                validate_segmentation_model(
                    model=self.model,
                    data_loader=(
                        self.validation_loader
                    ),
                    criterion=(
                        self.criterion
                    ),
                    criterion_mode=(
                        self.criterion_mode
                    ),
                    device=self.device,
                    amp_enabled=(
                        self.amp_enabled
                    ),
                    prediction_threshold=(
                        self.prediction_threshold
                    ),
                    target_threshold=(
                        self.target_threshold
                    ),
                    jaccard_quality_threshold=(
                        self.jaccard_quality_threshold
                    ),
                    boundary_tolerance_pixels=(
                        self.boundary_tolerance_pixels
                    ),
                    dataset_name=(
                        self.dataset_name
                    ),
                    split_name=(
                        self.validation_split_name
                    ),
                    max_batches=(
                        self.max_validation_batches
                    ),
                    require_sample_ids=True,
                    allow_logit_resize=(
                        self.allow_logit_resize
                    ),
                )
            )

            monitor_value = (
                resolve_validation_monitor(
                    validation_result,
                    self.monitor_name,
                )
            )

            is_best = metric_improved(
                current=monitor_value,
                best=self.best_metric,
                mode=self.monitor_mode,
                minimum_delta=(
                    self.minimum_delta
                ),
            )

            if is_best:
                self.best_metric = (
                    monitor_value
                )

                self.best_epoch = epoch

                self.epochs_without_improvement = 0

            else:
                self.epochs_without_improvement += 1

            scheduler_metric = (
                monitor_value
                if (
                    self.scheduler_bundle
                    .step_mode
                    == "metric"
                )
                else None
            )

            step_scheduler(
                self.scheduler_bundle,
                metric=scheduler_metric,
            )

            learning_rate_next = (
                current_learning_rates(
                    self.optimizer
                )[0]
            )

            validation_artifacts = (
                write_validation_artifacts(
                    result=validation_result,
                    output_directory=(
                        self.validation_directory
                    ),
                    epoch=epoch,
                )
            )

            dice = extract_metric_mean(
                validation_result,
                "dice",
            )

            iou = extract_metric_mean(
                validation_result,
                "iou",
            )

            thresholded_jaccard = (
                extract_metric_mean(
                    validation_result,
                    "thresholded_jaccard",
                )
            )

            boundary_f1 = (
                extract_metric_mean(
                    validation_result,
                    "boundary_f1",
                )
            )

            hd95 = extract_metric_mean(
                validation_result,
                "hd95",
            )

            assd = extract_metric_mean(
                validation_result,
                "assd",
            )

            validation_loss = (
                validation_result.mean_loss
            )

            if validation_loss is None:
                raise RuntimeError(
                    "Trainer requires validation "
                    "loss when a criterion is used."
                )

            log_record = {
                "epoch": epoch,
                "global_step": (
                    self.global_step
                ),
                "learning_rate_start": (
                    learning_rate_start
                ),
                "learning_rate_next": (
                    learning_rate_next
                ),
                "train_loss": (
                    training_result.mean_loss
                ),
                "validation_loss": (
                    validation_loss
                ),
                "validation_dice": (
                    dice
                ),
                "validation_iou": iou,
                "validation_thresholded_jaccard": (
                    thresholded_jaccard
                ),
                "validation_boundary_f1": (
                    boundary_f1
                ),
                "validation_hd95": (
                    hd95
                ),
                "validation_assd": (
                    assd
                ),
                "monitor_value": (
                    monitor_value
                ),
                "is_best": is_best,
                "best_epoch": (
                    self.best_epoch
                    if self.best_epoch
                    is not None
                    else -1
                ),
                "epochs_without_improvement": (
                    self.epochs_without_improvement
                ),
                "train_seconds": (
                    training_result
                    .elapsed_seconds
                ),
                "validation_seconds": (
                    validation_result
                    .elapsed_seconds
                ),
            }

            self.logger.log_epoch(
                log_record
            )

            checkpoint_metrics = {
                "train_loss": (
                    training_result.mean_loss
                ),
                "validation_loss": (
                    validation_loss
                ),
                "validation_dice": (
                    dice
                ),
                "validation_iou": (
                    iou
                ),
                "validation_thresholded_jaccard": (
                    thresholded_jaccard
                ),
                "validation_boundary_f1": (
                    boundary_f1
                ),
                "validation_hd95": (
                    hd95
                ),
                "validation_assd": (
                    assd
                ),
                "monitor_value": (
                    monitor_value
                ),
            }

            checkpoint_metadata = {
                "experiment_id": (
                    self.experiment_id
                ),
                "monitor_name": (
                    self.monitor_name
                ),
                "monitor_mode": (
                    self.monitor_mode
                ),
                "epochs_without_improvement": (
                    self.epochs_without_improvement
                ),
                "validation_artifacts": (
                    validation_artifacts
                ),
                "trainer_protocol_version": (
                    TRAINER_PROTOCOL_VERSION
                ),
            }

            latest_checkpoint = (
                save_checkpoint(
                    self.checkpoint_directory
                    / "latest.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=(
                        self.scheduler_bundle
                        .scheduler
                    ),
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=(
                        self.global_step
                    ),
                    best_metric=(
                        self.best_metric
                    ),
                    best_epoch=(
                        self.best_epoch
                    ),
                    metrics=(
                        checkpoint_metrics
                    ),
                    metadata=(
                        checkpoint_metadata
                    ),
                    include_rng_state=True,
                    write_checksum=True,
                )
            )

            best_checkpoint = None

            if is_best:
                best_checkpoint = (
                    save_checkpoint(
                        self.checkpoint_directory
                        / "best.pt",
                        model=self.model,
                        optimizer=(
                            self.optimizer
                        ),
                        scheduler=(
                            self.scheduler_bundle
                            .scheduler
                        ),
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=(
                            self.global_step
                        ),
                        best_metric=(
                            self.best_metric
                        ),
                        best_epoch=(
                            self.best_epoch
                        ),
                        metrics=(
                            checkpoint_metrics
                        ),
                        metadata=(
                            checkpoint_metadata
                        ),
                        include_rng_state=True,
                        write_checksum=True,
                    )
                )

            completed_epochs.append(
                {
                    "epoch": epoch,
                    "training": (
                        training_result.to_dict()
                    ),
                    "validation": (
                        validation_result.to_dict(
                            include_per_image_rows=False
                        )
                    ),
                    "monitor_value": (
                        monitor_value
                    ),
                    "is_best": is_best,
                    "latest_checkpoint": (
                        latest_checkpoint
                    ),
                    "best_checkpoint": (
                        best_checkpoint
                    ),
                    "validation_artifacts": (
                        validation_artifacts
                    ),
                }
            )

            if (
                self.early_stopping_patience
                is not None
                and self.epochs_without_improvement
                >= self.early_stopping_patience
            ):
                stopped_early = True
                break

        summary = {
            "protocol_version": (
                TRAINER_PROTOCOL_VERSION
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "device": str(
                self.device
            ),
            "amp_enabled": (
                self.amp_enabled
            ),
            "configured_epochs": (
                self.total_epochs
            ),
            "completed_epochs": len(
                completed_epochs
            ),
            "start_epoch": (
                self.start_epoch
            ),
            "final_epoch": (
                completed_epochs[-1][
                    "epoch"
                ]
                if completed_epochs
                else None
            ),
            "final_global_step": (
                self.global_step
            ),
            "monitor_name": (
                self.monitor_name
            ),
            "monitor_mode": (
                self.monitor_mode
            ),
            "best_metric": (
                self.best_metric
            ),
            "best_epoch": (
                self.best_epoch
            ),
            "epochs_without_improvement": (
                self.epochs_without_improvement
            ),
            "stopped_early": (
                stopped_early
            ),
            "logger": (
                self.logger.summary()
            ),
            "epochs": (
                completed_epochs
            ),
        }

        summary_path = (
            self.output_directory
            / "TRAINING_SUMMARY.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        summary[
            "summary_path"
        ] = str(
            summary_path
        )

        return summary


Trainer = SegmentationTrainer


class TinySegmentationModel(nn.Module):
    """Small model used only by the trainer self-test."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                3,
                4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                4,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Predict one-channel mask logits."""

        return {
            "mask_logits": (
                self.network(
                    image
                )
            )
        }


def run_trainer_self_test() -> dict[str, Any]:
    """Run a complete CPU training/validation persistence test."""

    import tempfile

    from src.train.scheduler import (
        build_scheduler,
    )

    torch.manual_seed(
        42
    )

    def make_batch(
        start_index: int,
        batch_size: int,
    ) -> dict[str, Any]:
        image = torch.randn(
            batch_size,
            3,
            8,
            8,
        )

        mask = (
            image[:, :1]
            > 0.0
        ).to(
            dtype=torch.float32
        )

        return {
            "image": image,
            "mask": mask,
            "image_id": [
                (
                    "synthetic_"
                    f"{start_index + index:03d}"
                )
                for index in range(
                    batch_size
                )
            ],
        }

    training_batches = [
        make_batch(
            0,
            2,
        ),
        make_batch(
            2,
            2,
        ),
    ]

    validation_batches = [
        make_batch(
            100,
            2,
        ),
        make_batch(
            102,
            2,
        ),
    ]

    model = TinySegmentationModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )

    scheduler_bundle = (
        build_scheduler(
            optimizer,
            {
                "name": "none",
            },
            total_epochs=2,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    initial_parameters = {
        name: parameter.detach()
        .clone()
        for name, parameter
        in model.named_parameters()
    }

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        trainer = SegmentationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=(
                training_batches
            ),
            validation_loader=(
                validation_batches
            ),
            output_directory=(
                temporary_directory
            ),
            experiment_id=(
                "TRAINER_SELF_TEST"
            ),
            total_epochs=2,
            scheduler_bundle=(
                scheduler_bundle
            ),
            criterion_mode=(
                "logits_target"
            ),
            device="cpu",
            amp_enabled=False,
            gradient_clip_norm=1.0,
            monitor_name=(
                "validation_dice"
            ),
            monitor_mode="max",
            minimum_delta=0.0,
            early_stopping_patience=5,
            max_train_batches=2,
            max_validation_batches=2,
            metadata={
                "purpose": (
                    "trainer_self_test"
                )
            },
        )

        summary = trainer.fit()

        root = Path(
            temporary_directory
        )

        expected_paths = {
            "latest_checkpoint": (
                root
                / "checkpoints"
                / "latest.pt"
            ),
            "latest_checksum": (
                root
                / "checkpoints"
                / "latest.pt.sha256"
            ),
            "best_checkpoint": (
                root
                / "checkpoints"
                / "best.pt"
            ),
            "best_checksum": (
                root
                / "checkpoints"
                / "best.pt.sha256"
            ),
            "history": (
                root
                / "logs"
                / "epoch_history.csv"
            ),
            "latest_epoch": (
                root
                / "logs"
                / "latest_epoch.json"
            ),
            "logger_state": (
                root
                / "logs"
                / "logger_state.json"
            ),
            "summary": (
                root
                / "TRAINING_SUMMARY.json"
            ),
        }

        artifact_existence = {
            name: path.is_file()
            for name, path
            in expected_paths.items()
        }

        validation_csv_files = sorted(
            (
                root
                / "validation"
            ).glob(
                "*_per_image_metrics.csv"
            )
        )

        validation_json_files = sorted(
            (
                root
                / "validation"
            ).glob(
                "*_validation_summary.json"
            )
        )

        with expected_paths[
            "history"
        ].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            history_rows = list(
                csv.DictReader(
                    input_file
                )
            )

    parameters_changed = any(
        not torch.equal(
            parameter.detach(),
            initial_parameters[name],
        )
        for name, parameter
        in model.named_parameters()
    )

    checks = {
        "protocol_version": (
            summary[
                "protocol_version"
            ]
            == TRAINER_PROTOCOL_VERSION
        ),
        "cpu_device": (
            summary[
                "device"
            ]
            == "cpu"
        ),
        "amp_disabled": (
            summary[
                "amp_enabled"
            ]
            is False
        ),
        "two_epochs_completed": (
            summary[
                "completed_epochs"
            ]
            == 2
        ),
        "final_epoch": (
            summary[
                "final_epoch"
            ]
            == 1
        ),
        "global_step_count": (
            summary[
                "final_global_step"
            ]
            == 4
        ),
        "model_parameters_updated": (
            parameters_changed
        ),
        "best_epoch_exists": (
            summary[
                "best_epoch"
            ]
            is not None
        ),
        "best_metric_finite": (
            summary[
                "best_metric"
            ]
            is not None
            and math.isfinite(
                float(
                    summary[
                        "best_metric"
                    ]
                )
            )
        ),
        "not_stopped_early": (
            summary[
                "stopped_early"
            ]
            is False
        ),
        "all_persistent_artifacts_exist": (
            all(
                artifact_existence.values()
            )
        ),
        "history_has_two_rows": (
            len(
                history_rows
            )
            == 2
        ),
        "validation_csv_count": (
            len(
                validation_csv_files
            )
            == 2
        ),
        "validation_json_count": (
            len(
                validation_json_files
            )
            == 2
        ),
        "training_losses_finite": all(
            math.isfinite(
                float(
                    epoch_report[
                        "training"
                    ][
                        "mean_loss"
                    ]
                )
            )
            for epoch_report
            in summary[
                "epochs"
            ]
        ),
        "validation_losses_finite": all(
            math.isfinite(
                float(
                    epoch_report[
                        "validation"
                    ][
                        "mean_loss"
                    ]
                )
            )
            for epoch_report
            in summary[
                "epochs"
            ]
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
            TRAINER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "completed_epochs": (
            summary[
                "completed_epochs"
            ]
        ),
        "final_global_step": (
            summary[
                "final_global_step"
            ]
        ),
        "best_epoch": (
            summary[
                "best_epoch"
            ]
        ),
        "best_metric": (
            summary[
                "best_metric"
            ]
        ),
    }