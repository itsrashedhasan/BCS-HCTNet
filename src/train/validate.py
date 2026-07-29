"""Shared validation loop for segmentation experiments.

This module evaluates baseline models and BCS-HCTNet using one common
protocol:

- deterministic evaluation mode;
- no gradient calculation;
- optional CUDA mixed precision;
- strict output and target shape validation;
- sample-weighted validation loss;
- per-image overlap and boundary metrics;
- duplicate-image protection;
- optional smoke-test batch limit;
- preservation of the model's original training mode.

Model outputs
-------------
The model may return:

- a tensor containing mask logits; or
- a mapping containing one of:
  ``mask_logits``, ``logits``, ``out``, ``prediction``, or ``mask``.

The preferred output key is ``mask_logits``.

Batch structure
---------------
The validation batch must contain:

- ``image`` or ``images``;
- ``mask`` or ``target``;
- sample identifiers through ``image_id``, ``sample_id``, or ``id``.

Identifiers may also be stored inside a ``metadata`` mapping.

Loss protocol
-------------
The default criterion protocol is:

    criterion(mask_logits, target)

The criterion may return:

- one scalar tensor; or
- a mapping containing ``total_loss``, ``loss``, or ``total``, plus optional
  scalar component losses.

No prediction resizing occurs by default. A model must normally return logits
at the same spatial resolution as the target.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.metrics.metric_tracker import (
    SegmentationMetricTracker,
)


VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-shared-validation-loop-v1"
)

CriterionMode = Literal[
    "logits_target",
    "output_batch",
]

MASK_OUTPUT_KEYS = (
    "mask_logits",
    "logits",
    "out",
    "prediction",
    "mask",
)

IMAGE_BATCH_KEYS = (
    "image",
    "images",
)

TARGET_BATCH_KEYS = (
    "mask",
    "target",
    "targets",
)

SAMPLE_ID_KEYS = (
    "image_id",
    "sample_id",
    "id",
)


@dataclass(frozen=True)
class ValidationResult:
    """Complete output of one validation pass."""

    protocol_version: str
    dataset_name: str
    split_name: str
    device: str
    amp_enabled: bool
    number_of_batches: int
    number_of_images: int
    mean_loss: float | None
    mean_loss_components: dict[str, float]
    elapsed_seconds: float
    images_per_second: float
    metric_summary: dict[str, Any]
    per_image_rows: list[dict[str, Any]]

    def to_dict(
        self,
        *,
        include_per_image_rows: bool = True,
    ) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        report: dict[str, Any] = {
            "protocol_version": (
                self.protocol_version
            ),
            "dataset_name": (
                self.dataset_name
            ),
            "split_name": (
                self.split_name
            ),
            "device": self.device,
            "amp_enabled": (
                self.amp_enabled
            ),
            "number_of_batches": (
                self.number_of_batches
            ),
            "number_of_images": (
                self.number_of_images
            ),
            "mean_loss": self.mean_loss,
            "mean_loss_components": dict(
                self.mean_loss_components
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "images_per_second": (
                self.images_per_second
            ),
            "metric_summary": (
                self.metric_summary
            ),
        }

        if include_per_image_rows:
            report[
                "per_image_rows"
            ] = [
                dict(row)
                for row
                in self.per_image_rows
            ]

        return report


def _resolve_device(
    model: nn.Module,
    device: str | torch.device | None,
) -> torch.device:
    """Resolve and validate the evaluation device."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if device is not None:
        resolved_device = torch.device(
            device
        )

    else:
        first_parameter = next(
            model.parameters(),
            None,
        )

        if first_parameter is not None:
            resolved_device = (
                first_parameter.device
            )

        else:
            first_buffer = next(
                model.buffers(),
                None,
            )

            resolved_device = (
                first_buffer.device
                if first_buffer is not None
                else torch.device("cpu")
            )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA validation was requested, "
            "but CUDA is not available."
        )

    return resolved_device


def _require_batch_mapping(
    batch: object,
) -> Mapping[str, Any]:
    """Require one mapping-based validation batch."""

    if not isinstance(
        batch,
        Mapping,
    ):
        raise TypeError(
            "Each validation batch must be "
            "a mapping."
        )

    return batch


def _find_batch_value(
    batch: Mapping[str, Any],
    keys: Sequence[str],
    context: str,
) -> Any:
    """Find the first available approved batch key."""

    for key in keys:
        if key in batch:
            return batch[key]

    raise KeyError(
        f"Validation batch does not contain "
        f"{context}. Accepted keys: "
        f"{list(keys)}."
    )


def extract_validation_image(
    batch: Mapping[str, Any],
) -> Tensor:
    """Extract and validate the image tensor."""

    image = _find_batch_value(
        batch,
        IMAGE_BATCH_KEYS,
        "an image tensor",
    )

    if not isinstance(
        image,
        Tensor,
    ):
        raise TypeError(
            "Validation image must be "
            "a torch.Tensor."
        )

    if image.ndim != 4:
        raise ValueError(
            "Validation image must have shape "
            "[B, C, H, W], received "
            f"{tuple(image.shape)}."
        )

    if image.shape[0] <= 0:
        raise ValueError(
            "Validation image batch cannot "
            "be empty."
        )

    if image.shape[1] != 3:
        raise ValueError(
            "Validation images must contain "
            "three RGB channels."
        )

    if not torch.isfinite(
        image
    ).all():
        raise ValueError(
            "Validation image contains "
            "non-finite values."
        )

    return image


def extract_validation_target(
    batch: Mapping[str, Any],
) -> Tensor:
    """Extract and validate the binary target tensor."""

    target = _find_batch_value(
        batch,
        TARGET_BATCH_KEYS,
        "a target mask",
    )

    if not isinstance(
        target,
        Tensor,
    ):
        raise TypeError(
            "Validation target must be "
            "a torch.Tensor."
        )

    if target.ndim == 3:
        target = target.unsqueeze(
            1
        )

    if target.ndim != 4:
        raise ValueError(
            "Validation target must have shape "
            "[B, 1, H, W], received "
            f"{tuple(target.shape)}."
        )

    if target.shape[1] != 1:
        raise ValueError(
            "Validation target must contain "
            "one binary channel."
        )

    if not torch.isfinite(
        target
    ).all():
        raise ValueError(
            "Validation target contains "
            "non-finite values."
        )

    minimum = float(
        target.min().item()
    )

    maximum = float(
        target.max().item()
    )

    tolerance = 1e-6

    if (
        minimum < -tolerance
        or maximum > 1.0 + tolerance
    ):
        raise ValueError(
            "Validation target values must be "
            "within [0, 1]. Observed range: "
            f"[{minimum}, {maximum}]."
        )

    return target


def extract_mask_logits(
    model_output: object,
) -> Tensor:
    """Extract mask logits from a model output."""

    if isinstance(
        model_output,
        Tensor,
    ):
        logits = model_output

    elif isinstance(
        model_output,
        Mapping,
    ):
        logits = None

        for key in MASK_OUTPUT_KEYS:
            candidate = model_output.get(
                key
            )

            if isinstance(
                candidate,
                Tensor,
            ):
                logits = candidate
                break

        if logits is None:
            tensor_keys = [
                str(key)
                for key, value
                in model_output.items()
                if isinstance(
                    value,
                    Tensor,
                )
            ]

            raise KeyError(
                "Could not locate mask logits "
                "inside model output. Accepted "
                f"keys: {list(MASK_OUTPUT_KEYS)}. "
                f"Tensor-valued keys found: "
                f"{tensor_keys}."
            )

    else:
        raise TypeError(
            "Model output must be a tensor "
            "or mapping."
        )

    if logits.ndim == 3:
        logits = logits.unsqueeze(
            1
        )

    if logits.ndim != 4:
        raise ValueError(
            "Mask logits must have shape "
            "[B, 1, H, W], received "
            f"{tuple(logits.shape)}."
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "Mask logits must contain exactly "
            "one segmentation channel."
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "Mask logits contain non-finite values."
        )

    return logits


def _normalize_identifier_values(
    value: object,
    *,
    batch_size: int,
) -> list[str]:
    """Convert collated identifier values to strings."""

    if isinstance(
        value,
        Tensor,
    ):
        if value.ndim == 0:
            raw_values: list[Any] = [
                value.detach()
                .cpu()
                .item()
            ]

        else:
            raw_values = (
                value.detach()
                .cpu()
                .tolist()
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        raw_values = list(
            value
        )

    else:
        raw_values = [
            value
        ]

    if len(
        raw_values
    ) != batch_size:
        raise ValueError(
            "Sample identifier count does not "
            f"match batch size: "
            f"{len(raw_values)} versus "
            f"{batch_size}."
        )

    normalized = [
        str(identifier).strip()
        for identifier
        in raw_values
    ]

    if any(
        not identifier
        for identifier
        in normalized
    ):
        raise ValueError(
            "Sample identifiers cannot be empty."
        )

    return normalized


def extract_sample_ids(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    generated_start_index: int,
    require_sample_ids: bool = True,
) -> list[str]:
    """Extract stable image identifiers from a batch."""

    if not isinstance(
        require_sample_ids,
        bool,
    ):
        raise TypeError(
            "require_sample_ids must be Boolean."
        )

    for key in SAMPLE_ID_KEYS:
        if key in batch:
            return _normalize_identifier_values(
                batch[key],
                batch_size=batch_size,
            )

    metadata = batch.get(
        "metadata"
    )

    if isinstance(
        metadata,
        Mapping,
    ):
        for key in SAMPLE_ID_KEYS:
            if key in metadata:
                return (
                    _normalize_identifier_values(
                        metadata[key],
                        batch_size=batch_size,
                    )
                )

    provenance = batch.get(
        "provenance"
    )

    if isinstance(
        provenance,
        Mapping,
    ):
        for key in SAMPLE_ID_KEYS:
            if key in provenance:
                return (
                    _normalize_identifier_values(
                        provenance[key],
                        batch_size=batch_size,
                    )
                )

    if require_sample_ids:
        raise KeyError(
            "Validation batch contains no stable "
            "sample identifiers. Expected one of "
            f"{list(SAMPLE_ID_KEYS)} directly or "
            "inside metadata/provenance."
        )

    return [
        (
            "generated_validation_"
            f"{generated_start_index + index:08d}"
        )
        for index in range(
            batch_size
        )
    ]


def _validate_criterion_mode(
    criterion_mode: object,
) -> CriterionMode:
    """Validate the approved criterion invocation mode."""

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


def _extract_scalar_loss(
    value: object,
    context: str,
) -> Tensor:
    """Require a finite scalar loss tensor."""

    if not isinstance(
        value,
        Tensor,
    ):
        raise TypeError(
            f"{context} must be a torch.Tensor."
        )

    if value.numel() != 1:
        raise ValueError(
            f"{context} must contain one scalar "
            f"value, received shape "
            f"{tuple(value.shape)}."
        )

    if not torch.isfinite(
        value
    ).all():
        raise RuntimeError(
            f"{context} is non-finite."
        )

    return value.reshape(
        ()
    )


def normalize_loss_output(
    loss_output: object,
) -> tuple[
    Tensor,
    dict[str, Tensor],
]:
    """Normalize scalar or mapping-based criterion output."""

    if isinstance(
        loss_output,
        Tensor,
    ):
        total_loss = _extract_scalar_loss(
            loss_output,
            "criterion loss",
        )

        return (
            total_loss,
            {
                "total_loss": total_loss,
            },
        )

    if not isinstance(
        loss_output,
        Mapping,
    ):
        raise TypeError(
            "Criterion must return a scalar tensor "
            "or a mapping of scalar tensors."
        )

    total_loss = None

    for key in (
        "total_loss",
        "loss",
        "total",
    ):
        candidate = loss_output.get(
            key
        )

        if candidate is not None:
            total_loss = _extract_scalar_loss(
                candidate,
                f"criterion[{key!r}]",
            )
            break

    if total_loss is None:
        raise KeyError(
            "Criterion mapping must contain one "
            "of 'total_loss', 'loss', or 'total'."
        )

    components: dict[
        str,
        Tensor,
    ] = {}

    for key, value in loss_output.items():
        if isinstance(
            value,
            Tensor,
        ) and value.numel() == 1:
            components[
                str(key)
            ] = _extract_scalar_loss(
                value,
                f"criterion[{key!r}]",
            )

    components[
        "total_loss"
    ] = total_loss

    return (
        total_loss,
        components,
    )


def compute_validation_loss(
    *,
    criterion: Any,
    criterion_mode: CriterionMode,
    model_output: object,
    mask_logits: Tensor,
    target: Tensor,
    batch: Mapping[str, Any],
) -> tuple[
    Tensor,
    dict[str, Tensor],
]:
    """Invoke and normalize the validation criterion."""

    if criterion_mode == "logits_target":
        loss_output = criterion(
            mask_logits,
            target,
        )

    elif criterion_mode == "output_batch":
        loss_output = criterion(
            model_output,
            batch,
        )

    else:
        raise AssertionError(
            "Unreachable criterion mode."
        )

    return normalize_loss_output(
        loss_output
    )


def _validate_max_batches(
    max_batches: int | None,
) -> int | None:
    """Validate an optional smoke-test batch limit."""

    if max_batches is None:
        return None

    if (
        isinstance(max_batches, bool)
        or not isinstance(
            max_batches,
            int,
        )
        or max_batches <= 0
    ):
        raise ValueError(
            "max_batches must be None or a "
            "positive integer."
        )

    return max_batches


def validate_segmentation_model(
    *,
    model: nn.Module,
    data_loader: Iterable[
        Mapping[str, Any]
    ],
    criterion: Any | None = None,
    criterion_mode: CriterionMode = (
        "logits_target"
    ),
    device: str | torch.device | None = None,
    amp_enabled: bool = False,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
    jaccard_quality_threshold: float = 0.65,
    boundary_tolerance_pixels: float = 2.0,
    dataset_name: str = "ISIC2018",
    split_name: str = "validation",
    max_batches: int | None = None,
    require_sample_ids: bool = True,
    allow_logit_resize: bool = False,
) -> ValidationResult:
    """Run one complete validation pass."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
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

    resolved_criterion_mode = (
        _validate_criterion_mode(
            criterion_mode
        )
    )

    resolved_max_batches = (
        _validate_max_batches(
            max_batches
        )
    )

    resolved_device = _resolve_device(
        model,
        device,
    )

    effective_amp_enabled = bool(
        amp_enabled
        and resolved_device.type == "cuda"
    )

    normalized_dataset_name = str(
        dataset_name
    ).strip()

    normalized_split_name = str(
        split_name
    ).strip()

    if not normalized_dataset_name:
        raise ValueError(
            "dataset_name cannot be empty."
        )

    if not normalized_split_name:
        raise ValueError(
            "split_name cannot be empty."
        )

    metric_tracker = (
        SegmentationMetricTracker(
            prediction_threshold=(
                prediction_threshold
            ),
            target_threshold=(
                target_threshold
            ),
            jaccard_quality_threshold=(
                jaccard_quality_threshold
            ),
            boundary_tolerance_pixels=(
                boundary_tolerance_pixels
            ),
            from_logits=True,
            split_name=(
                normalized_split_name
            ),
            dataset_name=(
                normalized_dataset_name
            ),
        )
    )

    original_training_mode = bool(
        model.training
    )

    model.eval()

    total_images = 0
    completed_batches = 0
    weighted_total_loss = 0.0

    weighted_loss_components: dict[
        str,
        float,
    ] = {}

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(
                data_loader
            ):
                if (
                    resolved_max_batches
                    is not None
                    and batch_index
                    >= resolved_max_batches
                ):
                    break

                batch = _require_batch_mapping(
                    raw_batch
                )

                image = extract_validation_image(
                    batch
                )

                target = extract_validation_target(
                    batch
                )

                if (
                    image.shape[0]
                    != target.shape[0]
                ):
                    raise RuntimeError(
                        "Validation image and target "
                        "batch sizes do not match."
                    )

                batch_size = int(
                    image.shape[0]
                )

                sample_ids = extract_sample_ids(
                    batch,
                    batch_size=batch_size,
                    generated_start_index=(
                        total_images
                    ),
                    require_sample_ids=(
                        require_sample_ids
                    ),
                )

                image = image.to(
                    device=resolved_device,
                    dtype=torch.float32,
                    non_blocking=(
                        resolved_device.type
                        == "cuda"
                    ),
                )

                target = target.to(
                    device=resolved_device,
                    dtype=torch.float32,
                    non_blocking=(
                        resolved_device.type
                        == "cuda"
                    ),
                )

                with torch.autocast(
                    device_type=(
                        resolved_device.type
                    ),
                    dtype=torch.float16,
                    enabled=(
                        effective_amp_enabled
                    ),
                ):
                    model_output = model(
                        image
                    )

                    mask_logits = (
                        extract_mask_logits(
                            model_output
                        )
                    )

                    if (
                        mask_logits.shape[0]
                        != batch_size
                    ):
                        raise RuntimeError(
                            "Mask-logit batch size "
                            "does not match input."
                        )

                    if (
                        mask_logits.shape[-2:]
                        != target.shape[-2:]
                    ):
                        if not allow_logit_resize:
                            raise RuntimeError(
                                "Mask logits and target "
                                "spatial dimensions do "
                                "not match. Logits: "
                                f"{tuple(mask_logits.shape)}; "
                                "target: "
                                f"{tuple(target.shape)}. "
                                "Automatic resizing is "
                                "disabled."
                            )

                        mask_logits = F.interpolate(
                            mask_logits,
                            size=(
                                target.shape[-2:]
                            ),
                            mode="bilinear",
                            align_corners=False,
                        )

                    if criterion is not None:
                        (
                            total_loss,
                            loss_components,
                        ) = compute_validation_loss(
                            criterion=criterion,
                            criterion_mode=(
                                resolved_criterion_mode
                            ),
                            model_output=(
                                model_output
                            ),
                            mask_logits=(
                                mask_logits
                            ),
                            target=target,
                            batch=batch,
                        )

                    else:
                        total_loss = None
                        loss_components = {}

                if total_loss is not None:
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
                            "Validation loss is "
                            "non-finite."
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
                                "Validation loss "
                                f"component "
                                f"{component_name!r} "
                                "is non-finite."
                            )

                        weighted_loss_components[
                            component_name
                        ] = (
                            weighted_loss_components.get(
                                component_name,
                                0.0,
                            )
                            + component_value
                            * batch_size
                        )

                metric_tracker.update(
                    prediction=(
                        mask_logits.detach()
                    ),
                    target=target.detach(),
                    sample_ids=sample_ids,
                )

                total_images += batch_size
                completed_batches += 1

    finally:
        model.train(
            original_training_mode
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if completed_batches == 0:
        raise RuntimeError(
            "Validation data loader produced "
            "no batches."
        )

    if total_images == 0:
        raise RuntimeError(
            "Validation evaluated no images."
        )

    if len(
        metric_tracker
    ) != total_images:
        raise RuntimeError(
            "Metric-tracker image count does "
            "not match the validation count."
        )

    mean_loss = (
        weighted_total_loss
        / total_images
        if criterion is not None
        else None
    )

    mean_loss_components = {
        name: (
            weighted_value
            / total_images
        )
        for name, weighted_value
        in weighted_loss_components.items()
    }

    for name, value in (
        mean_loss_components.items()
    ):
        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                f"Mean validation loss component "
                f"{name!r} is non-finite."
            )

    images_per_second = (
        total_images
        / elapsed_seconds
        if elapsed_seconds > 0.0
        else float("inf")
    )

    if not math.isfinite(
        images_per_second
    ):
        images_per_second = 0.0

    return ValidationResult(
        protocol_version=(
            VALIDATION_PROTOCOL_VERSION
        ),
        dataset_name=(
            normalized_dataset_name
        ),
        split_name=(
            normalized_split_name
        ),
        device=str(
            resolved_device
        ),
        amp_enabled=(
            effective_amp_enabled
        ),
        number_of_batches=(
            completed_batches
        ),
        number_of_images=(
            total_images
        ),
        mean_loss=(
            mean_loss
        ),
        mean_loss_components=(
            mean_loss_components
        ),
        elapsed_seconds=float(
            elapsed_seconds
        ),
        images_per_second=float(
            images_per_second
        ),
        metric_summary=(
            metric_tracker.summary()
        ),
        per_image_rows=(
            metric_tracker.rows()
        ),
    )


validate = validate_segmentation_model


class SyntheticValidationModel(nn.Module):
    """Small deterministic model used by the self-test."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return channel zero as segmentation logits."""

        return {
            "mask_logits": (
                image[:, :1]
                * self.scale
            )
        }


def run_validation_self_test() -> dict[str, Any]:
    """Run a deterministic CPU validation self-test."""

    model = SyntheticValidationModel()

    model.train()

    first_target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    second_target = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    def target_to_image(
        target: Tensor,
    ) -> Tensor:
        logits = torch.where(
            target > 0.5,
            torch.full_like(
                target,
                8.0,
            ),
            torch.full_like(
                target,
                -8.0,
            ),
        )

        return torch.cat(
            [
                logits,
                torch.zeros_like(
                    logits
                ),
                torch.zeros_like(
                    logits
                ),
            ],
            dim=1,
        )

    batches = [
        {
            "image": target_to_image(
                first_target
            ),
            "mask": first_target,
            "image_id": [
                "sample_001",
                "sample_002",
            ],
        },
        {
            "image": target_to_image(
                second_target
            ),
            "mask": second_target,
            "metadata": {
                "image_id": [
                    "sample_003",
                ]
            },
        },
    ]

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    result = validate_segmentation_model(
        model=model,
        data_loader=batches,
        criterion=criterion,
        criterion_mode=(
            "logits_target"
        ),
        device="cpu",
        amp_enabled=False,
        prediction_threshold=0.5,
        target_threshold=0.5,
        jaccard_quality_threshold=0.65,
        boundary_tolerance_pixels=1.0,
        dataset_name="synthetic",
        split_name="validation",
        require_sample_ids=True,
        allow_logit_resize=False,
    )

    summary = result.metric_summary

    metric_means = summary[
        "metrics"
    ]

    rows = result.per_image_rows

    model_mode_restored = (
        model.training is True
    )

    expected_ids = [
        "sample_001",
        "sample_002",
        "sample_003",
    ]

    observed_ids = [
        str(
            row[
                "sample_id"
            ]
        )
        for row in rows
    ]

    checks = {
        "protocol_version": (
            result.protocol_version
            == VALIDATION_PROTOCOL_VERSION
        ),
        "cpu_device": (
            result.device == "cpu"
        ),
        "amp_disabled": (
            result.amp_enabled is False
        ),
        "batch_count": (
            result.number_of_batches
            == 2
        ),
        "image_count": (
            result.number_of_images
            == 3
        ),
        "mean_loss_exists": (
            result.mean_loss is not None
        ),
        "mean_loss_finite": (
            result.mean_loss is not None
            and math.isfinite(
                result.mean_loss
            )
        ),
        "mean_loss_positive": (
            result.mean_loss is not None
            and result.mean_loss > 0.0
        ),
        "total_loss_component": (
            "total_loss"
            in result.mean_loss_components
        ),
        "sample_ids_preserved": (
            observed_ids
            == expected_ids
        ),
        "per_image_row_count": (
            len(
                rows
            )
            == 3
        ),
        "summary_image_count": (
            summary[
                "number_of_images"
            ]
            == 3
        ),
        "perfect_mean_dice": (
            math.isclose(
                metric_means[
                    "dice"
                ][
                    "mean"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "perfect_mean_iou": (
            math.isclose(
                metric_means[
                    "iou"
                ][
                    "mean"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "perfect_thresholded_jaccard": (
            math.isclose(
                metric_means[
                    "thresholded_jaccard"
                ][
                    "mean"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "perfect_boundary_f1": (
            math.isclose(
                metric_means[
                    "boundary_f1"
                ][
                    "mean"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "zero_hd95": (
            math.isclose(
                metric_means[
                    "hd95"
                ][
                    "mean"
                ],
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "zero_assd": (
            math.isclose(
                metric_means[
                    "assd"
                ][
                    "mean"
                ],
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "model_mode_restored": (
            model_mode_restored
        ),
        "throughput_nonnegative": (
            result.images_per_second
            >= 0.0
        ),
        "elapsed_nonnegative": (
            result.elapsed_seconds
            >= 0.0
        ),
        "serializable_report": (
            result.to_dict()[
                "number_of_images"
            ]
            == 3
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
            VALIDATION_PROTOCOL_VERSION
        ),
        "checks": checks,
        "number_of_batches": (
            result.number_of_batches
        ),
        "number_of_images": (
            result.number_of_images
        ),
        "mean_loss": (
            result.mean_loss
        ),
        "sample_ids": (
            observed_ids
        ),
        "metric_means": {
            name: statistics[
                "mean"
            ]
            for name, statistics
            in metric_means.items()
        },
    }