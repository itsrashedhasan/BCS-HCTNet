"""Dependency-free model complexity and inference profiling.

This module measures quantities needed for controlled model comparison:

- total and trainable parameter counts;
- persistent parameter and buffer memory;
- output tensor shape;
- tracked multiply-accumulate operations for Conv2d,
  ConvTranspose2d, and Linear layers;
- mean and median inference latency;
- inference throughput;
- optional CUDA peak-memory usage.

Important
---------
``tracked_macs`` is not presented as an exact total-model FLOP count.
It covers supported convolutional and linear layers but does not include
every normalization, activation, interpolation, softmax, or attention-matrix
operation. Therefore, research tables must label it as ``tracked MACs`` unless
a separately validated full FLOP profiler is used.

The profiler supports models returning:

- a tensor;
- a mapping containing ``mask_logits``;
- nested lists or tuples containing tensors.

The model's original training mode and device are restored by default.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


COMPLEXITY_PROTOCOL_VERSION = (
    "BCS-HCTNet-model-complexity-v1"
)

MEBIBYTE = 1024 ** 2
GIGA_MAC = 1_000_000_000


@dataclass(frozen=True)
class ComplexityProfile:
    """Serializable model-complexity and inference profile."""

    protocol_version: str
    model_class: str
    device: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    parameter_count: int
    trainable_parameter_count: int
    nontrainable_parameter_count: int
    buffer_count: int
    parameter_bytes: int
    buffer_bytes: int
    persistent_state_bytes: int
    persistent_state_mebibytes: float
    tracked_macs: int | None
    tracked_gmacs: float | None
    warmup_steps: int
    measurement_steps: int
    mean_latency_milliseconds: float
    median_latency_milliseconds: float
    minimum_latency_milliseconds: float
    maximum_latency_milliseconds: float
    throughput_images_per_second: float
    peak_device_memory_mebibytes: float | None
    automatic_mixed_precision: bool
    mac_coverage: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-compatible profile mapping."""

        result = asdict(
            self
        )

        result[
            "input_shape"
        ] = list(
            self.input_shape
        )

        result[
            "output_shape"
        ] = list(
            self.output_shape
        )

        return result


def _require_positive_integer(
    value: object,
    context: str,
) -> int:
    """Require a positive integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{context} must be a positive integer."
        )

    return value


def _require_nonnegative_integer(
    value: object,
    context: str,
) -> int:
    """Require a non-negative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{context} must be a non-negative integer."
        )

    return value


def _validate_boolean(
    value: object,
    context: str,
) -> bool:
    """Require a Boolean value."""

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be Boolean."
        )

    return value


def _normalize_input_shape(
    input_shape: Sequence[int],
) -> tuple[int, ...]:
    """Validate a model input shape."""

    if (
        isinstance(
            input_shape,
            (
                str,
                bytes,
            ),
        )
        or not isinstance(
            input_shape,
            Sequence,
        )
    ):
        raise TypeError(
            "input_shape must be a sequence "
            "of positive integers."
        )

    resolved = tuple(
        _require_positive_integer(
            value,
            "input_shape value",
        )
        for value in input_shape
    )

    if len(
        resolved
    ) != 4:
        raise ValueError(
            "Segmentation profiling expects "
            "input_shape [B, C, H, W]."
        )

    return resolved


def _normalize_device(
    device: str | torch.device,
) -> torch.device:
    """Normalize and validate a requested device."""

    resolved = torch.device(
        device
    )

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA profiling was requested, but "
            "CUDA is not available."
        )

    if (
        resolved.type == "mps"
        and (
            not hasattr(
                torch.backends,
                "mps",
            )
            or not torch.backends.mps.is_available()
        )
    ):
        raise RuntimeError(
            "MPS profiling was requested, but "
            "MPS is not available."
        )

    return resolved


def _first_model_device(
    model: nn.Module,
) -> torch.device:
    """Return the device of the first parameter or buffer."""

    for parameter in model.parameters():
        return parameter.device

    for buffer in model.buffers():
        return buffer.device

    return torch.device(
        "cpu"
    )


def _tensor_bytes(
    tensor: Tensor,
) -> int:
    """Return storage required by one tensor."""

    return int(
        tensor.numel()
        * tensor.element_size()
    )


def count_model_parameters(
    model: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    """Count model parameters."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    resolved_trainable_only = (
        _validate_boolean(
            trainable_only,
            "trainable_only",
        )
    )

    return int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if (
                parameter.requires_grad
                or not resolved_trainable_only
            )
        )
    )


def count_model_buffers(
    model: nn.Module,
) -> int:
    """Count persistent and non-persistent buffer elements."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    return int(
        sum(
            buffer.numel()
            for buffer
            in model.buffers()
        )
    )


def model_persistent_state_bytes(
    model: nn.Module,
) -> dict[str, int]:
    """Return parameter and buffer memory usage."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    parameter_bytes = int(
        sum(
            _tensor_bytes(
                parameter
            )
            for parameter
            in model.parameters()
        )
    )

    buffer_bytes = int(
        sum(
            _tensor_bytes(
                buffer
            )
            for buffer
            in model.buffers()
        )
    )

    return {
        "parameter_bytes": (
            parameter_bytes
        ),
        "buffer_bytes": (
            buffer_bytes
        ),
        "persistent_state_bytes": (
            parameter_bytes
            + buffer_bytes
        ),
    }


def _extract_output_tensor(
    output: object,
) -> Tensor:
    """Extract the primary prediction tensor from a model output."""

    if isinstance(
        output,
        Tensor,
    ):
        return output

    if isinstance(
        output,
        Mapping,
    ):
        preferred_keys = (
            "mask_logits",
            "logits",
            "prediction",
            "predictions",
            "output",
        )

        for key in preferred_keys:
            value = output.get(
                key
            )

            if isinstance(
                value,
                Tensor,
            ):
                return value

        for value in output.values():
            try:
                return _extract_output_tensor(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        raise ValueError(
            "Model output mapping contains no "
            "tensor prediction."
        )

    if isinstance(
        output,
        (
            list,
            tuple,
        ),
    ):
        for value in output:
            try:
                return _extract_output_tensor(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        raise ValueError(
            "Model output sequence contains no "
            "tensor prediction."
        )

    raise TypeError(
        "Unsupported model output type: "
        f"{type(output).__name__}."
    )


def _synchronize_device(
    device: torch.device,
) -> None:
    """Synchronize asynchronous device operations."""

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    elif (
        device.type == "mps"
        and hasattr(
            torch,
            "mps",
        )
    ):
        torch.mps.synchronize()


def _autocast_context(
    *,
    device: torch.device,
    enabled: bool,
):
    """Return an appropriate inference autocast context."""

    if not enabled:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    if device.type == "cpu":
        return torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=True,
        )

    raise ValueError(
        "Automatic mixed precision profiling "
        f"is unsupported for device "
        f"{device.type!r}."
    )


class _TrackedMACCounter:
    """Forward-hook based Conv2d, ConvTranspose2d, and Linear MAC counter."""

    def __init__(
        self,
    ) -> None:
        """Initialize an empty counter."""

        self.total_macs = 0
        self.handles: list[
            torch.utils.hooks.RemovableHandle
        ] = []

    @staticmethod
    def _first_tensor(
        value: object,
    ) -> Tensor:
        """Extract the first tensor from a hook value."""

        if isinstance(
            value,
            Tensor,
        ):
            return value

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            for item in value:
                if isinstance(
                    item,
                    Tensor,
                ):
                    return item

        raise TypeError(
            "Hook value contains no tensor."
        )

    def _convolution_hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        """Count convolution MACs."""

        output_tensor = (
            self._first_tensor(
                output
            )
        )

        if output_tensor.ndim != 4:
            return

        if not isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
            ),
        ):
            return

        batch_size = int(
            output_tensor.shape[0]
        )

        output_channels = int(
            output_tensor.shape[1]
        )

        output_height = int(
            output_tensor.shape[2]
        )

        output_width = int(
            output_tensor.shape[3]
        )

        kernel_height = int(
            module.kernel_size[0]
        )

        kernel_width = int(
            module.kernel_size[1]
        )

        channels_per_group = int(
            module.in_channels
            // module.groups
        )

        macs = (
            batch_size
            * output_channels
            * output_height
            * output_width
            * channels_per_group
            * kernel_height
            * kernel_width
        )

        self.total_macs += int(
            macs
        )

    def _linear_hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        """Count linear-layer MACs."""

        output_tensor = (
            self._first_tensor(
                output
            )
        )

        if not isinstance(
            module,
            nn.Linear,
        ):
            return

        output_elements = int(
            output_tensor.numel()
        )

        macs = (
            output_elements
            * int(
                module.in_features
            )
        )

        self.total_macs += int(
            macs
        )

    def attach(
        self,
        model: nn.Module,
    ) -> None:
        """Attach supported-layer forward hooks."""

        for module in model.modules():
            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                ),
            ):
                self.handles.append(
                    module.register_forward_hook(
                        self._convolution_hook
                    )
                )

            elif isinstance(
                module,
                nn.Linear,
            ):
                self.handles.append(
                    module.register_forward_hook(
                        self._linear_hook
                    )
                )

    def remove(
        self,
    ) -> None:
        """Remove all active hooks."""

        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def calculate_tracked_macs(
    model: nn.Module,
    input_tensor: Tensor,
    *,
    use_amp: bool = False,
) -> tuple[int, tuple[int, ...]]:
    """Count supported-layer MACs for one forward pass."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        input_tensor,
        Tensor,
    ):
        raise TypeError(
            "input_tensor must be a torch.Tensor."
        )

    resolved_use_amp = (
        _validate_boolean(
            use_amp,
            "use_amp",
        )
    )

    device = input_tensor.device

    counter = _TrackedMACCounter()

    counter.attach(
        model
    )

    try:
        with torch.inference_mode():
            with _autocast_context(
                device=device,
                enabled=resolved_use_amp,
            ):
                output = model(
                    input_tensor
                )

        output_tensor = (
            _extract_output_tensor(
                output
            )
        )

    finally:
        counter.remove()

    if not torch.isfinite(
        output_tensor
    ).all():
        raise RuntimeError(
            "Model output contains non-finite "
            "values during MAC profiling."
        )

    return (
        int(
            counter.total_macs
        ),
        tuple(
            int(value)
            for value
            in output_tensor.shape
        ),
    )


def measure_inference_latency(
    model: nn.Module,
    input_tensor: Tensor,
    *,
    warmup_steps: int = 2,
    measurement_steps: int = 10,
    use_amp: bool = False,
) -> dict[str, float | tuple[int, ...]]:
    """Measure model inference latency and throughput."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        input_tensor,
        Tensor,
    ):
        raise TypeError(
            "input_tensor must be a torch.Tensor."
        )

    resolved_warmup_steps = (
        _require_nonnegative_integer(
            warmup_steps,
            "warmup_steps",
        )
    )

    resolved_measurement_steps = (
        _require_positive_integer(
            measurement_steps,
            "measurement_steps",
        )
    )

    resolved_use_amp = (
        _validate_boolean(
            use_amp,
            "use_amp",
        )
    )

    device = input_tensor.device

    output_tensor: Tensor | None = None

    with torch.inference_mode():
        for _ in range(
            resolved_warmup_steps
        ):
            with _autocast_context(
                device=device,
                enabled=resolved_use_amp,
            ):
                output = model(
                    input_tensor
                )

                output_tensor = (
                    _extract_output_tensor(
                        output
                    )
                )

        _synchronize_device(
            device
        )

        elapsed_seconds: list[
            float
        ] = []

        for _ in range(
            resolved_measurement_steps
        ):
            _synchronize_device(
                device
            )

            start_time = (
                time.perf_counter()
            )

            with _autocast_context(
                device=device,
                enabled=resolved_use_amp,
            ):
                output = model(
                    input_tensor
                )

                output_tensor = (
                    _extract_output_tensor(
                        output
                    )
                )

            _synchronize_device(
                device
            )

            duration = (
                time.perf_counter()
                - start_time
            )

            if (
                not math.isfinite(
                    duration
                )
                or duration <= 0.0
            ):
                raise RuntimeError(
                    "Measured inference duration "
                    "is invalid."
                )

            elapsed_seconds.append(
                duration
            )

    if output_tensor is None:
        raise RuntimeError(
            "Inference profiling produced no "
            "output tensor."
        )

    if not torch.isfinite(
        output_tensor
    ).all():
        raise RuntimeError(
            "Model output contains non-finite "
            "values during latency profiling."
        )

    latency_milliseconds = [
        duration * 1000.0
        for duration
        in elapsed_seconds
    ]

    mean_latency = float(
        statistics.fmean(
            latency_milliseconds
        )
    )

    median_latency = float(
        statistics.median(
            latency_milliseconds
        )
    )

    batch_size = int(
        input_tensor.shape[0]
    )

    throughput = float(
        batch_size
        / (
            mean_latency
            / 1000.0
        )
    )

    return {
        "mean_latency_milliseconds": (
            mean_latency
        ),
        "median_latency_milliseconds": (
            median_latency
        ),
        "minimum_latency_milliseconds": (
            float(
                min(
                    latency_milliseconds
                )
            )
        ),
        "maximum_latency_milliseconds": (
            float(
                max(
                    latency_milliseconds
                )
            )
        ),
        "throughput_images_per_second": (
            throughput
        ),
        "output_shape": tuple(
            int(value)
            for value
            in output_tensor.shape
        ),
    }


def profile_model_complexity(
    model: nn.Module,
    *,
    input_shape: Sequence[int] = (
        1,
        3,
        352,
        352,
    ),
    device: str | torch.device = "cpu",
    input_dtype: torch.dtype = (
        torch.float32
    ),
    warmup_steps: int = 2,
    measurement_steps: int = 10,
    use_amp: bool = False,
    count_tracked_macs: bool = True,
    restore_original_device: bool = True,
) -> ComplexityProfile:
    """Create a complete model-complexity profile."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    resolved_input_shape = (
        _normalize_input_shape(
            input_shape
        )
    )

    resolved_device = (
        _normalize_device(
            device
        )
    )

    if not isinstance(
        input_dtype,
        torch.dtype,
    ):
        raise TypeError(
            "input_dtype must be a torch.dtype."
        )

    resolved_warmup_steps = (
        _require_nonnegative_integer(
            warmup_steps,
            "warmup_steps",
        )
    )

    resolved_measurement_steps = (
        _require_positive_integer(
            measurement_steps,
            "measurement_steps",
        )
    )

    resolved_use_amp = (
        _validate_boolean(
            use_amp,
            "use_amp",
        )
    )

    resolved_count_macs = (
        _validate_boolean(
            count_tracked_macs,
            "count_tracked_macs",
        )
    )

    resolved_restore_device = (
        _validate_boolean(
            restore_original_device,
            "restore_original_device",
        )
    )

    original_device = (
        _first_model_device(
            model
        )
    )

    original_training_mode = (
        model.training
    )

    total_parameters = (
        count_model_parameters(
            model,
            trainable_only=False,
        )
    )

    trainable_parameters = (
        count_model_parameters(
            model,
            trainable_only=True,
        )
    )

    nontrainable_parameters = (
        total_parameters
        - trainable_parameters
    )

    buffer_count = count_model_buffers(
        model
    )

    memory = model_persistent_state_bytes(
        model
    )

    model.to(
        resolved_device
    )

    model.eval()

    input_tensor = torch.randn(
        resolved_input_shape,
        device=resolved_device,
        dtype=input_dtype,
    )

    tracked_macs: int | None = None
    output_shape: tuple[
        int,
        ...
    ] | None = None
    latency: dict[
        str,
        float | tuple[int, ...],
    ]

    peak_memory_mebibytes: (
        float | None
    ) = None

    try:
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats(
                resolved_device
            )

        if resolved_count_macs:
            (
                tracked_macs,
                output_shape,
            ) = calculate_tracked_macs(
                model,
                input_tensor,
                use_amp=resolved_use_amp,
            )

        latency = measure_inference_latency(
            model,
            input_tensor,
            warmup_steps=(
                resolved_warmup_steps
            ),
            measurement_steps=(
                resolved_measurement_steps
            ),
            use_amp=resolved_use_amp,
        )

        measured_output_shape = (
            latency[
                "output_shape"
            ]
        )

        if not isinstance(
            measured_output_shape,
            tuple,
        ):
            raise RuntimeError(
                "Latency profiler returned an "
                "invalid output shape."
            )

        if output_shape is None:
            output_shape = (
                measured_output_shape
            )

        elif (
            output_shape
            != measured_output_shape
        ):
            raise RuntimeError(
                "MAC and latency profiling "
                "produced different output shapes."
            )

        if resolved_device.type == "cuda":
            peak_memory_mebibytes = float(
                torch.cuda.max_memory_allocated(
                    resolved_device
                )
                / MEBIBYTE
            )

    finally:
        del input_tensor

        if original_training_mode:
            model.train()

        else:
            model.eval()

        if (
            resolved_restore_device
            and original_device
            != resolved_device
        ):
            model.to(
                original_device
            )

        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    if output_shape is None:
        raise RuntimeError(
            "Complexity profiling did not "
            "produce an output shape."
        )

    tracked_gmacs = (
        None
        if tracked_macs is None
        else float(
            tracked_macs
            / GIGA_MAC
        )
    )

    profile = ComplexityProfile(
        protocol_version=(
            COMPLEXITY_PROTOCOL_VERSION
        ),
        model_class=(
            model.__class__.__name__
        ),
        device=str(
            resolved_device
        ),
        input_shape=(
            resolved_input_shape
        ),
        output_shape=(
            output_shape
        ),
        input_dtype=str(
            input_dtype
        ).replace(
            "torch.",
            "",
        ),
        parameter_count=(
            total_parameters
        ),
        trainable_parameter_count=(
            trainable_parameters
        ),
        nontrainable_parameter_count=(
            nontrainable_parameters
        ),
        buffer_count=(
            buffer_count
        ),
        parameter_bytes=(
            memory[
                "parameter_bytes"
            ]
        ),
        buffer_bytes=(
            memory[
                "buffer_bytes"
            ]
        ),
        persistent_state_bytes=(
            memory[
                "persistent_state_bytes"
            ]
        ),
        persistent_state_mebibytes=float(
            memory[
                "persistent_state_bytes"
            ]
            / MEBIBYTE
        ),
        tracked_macs=(
            tracked_macs
        ),
        tracked_gmacs=(
            tracked_gmacs
        ),
        warmup_steps=(
            resolved_warmup_steps
        ),
        measurement_steps=(
            resolved_measurement_steps
        ),
        mean_latency_milliseconds=float(
            latency[
                "mean_latency_milliseconds"
            ]
        ),
        median_latency_milliseconds=float(
            latency[
                "median_latency_milliseconds"
            ]
        ),
        minimum_latency_milliseconds=float(
            latency[
                "minimum_latency_milliseconds"
            ]
        ),
        maximum_latency_milliseconds=float(
            latency[
                "maximum_latency_milliseconds"
            ]
        ),
        throughput_images_per_second=float(
            latency[
                "throughput_images_per_second"
            ]
        ),
        peak_device_memory_mebibytes=(
            peak_memory_mebibytes
        ),
        automatic_mixed_precision=(
            resolved_use_amp
        ),
        mac_coverage=(
            "Conv2d, ConvTranspose2d, and "
            "Linear layers only"
        ),
    )

    numeric_values = (
        profile.persistent_state_mebibytes,
        profile.mean_latency_milliseconds,
        profile.median_latency_milliseconds,
        profile.minimum_latency_milliseconds,
        profile.maximum_latency_milliseconds,
        profile.throughput_images_per_second,
    )

    if not all(
        math.isfinite(
            value
        )
        and value >= 0.0
        for value
        in numeric_values
    ):
        raise RuntimeError(
            "Complexity profile contains "
            "invalid numeric values."
        )

    return profile


def save_complexity_profile(
    profile: ComplexityProfile,
    output_path: str | Path,
) -> Path:
    """Save a complexity profile as formatted JSON."""

    if not isinstance(
        profile,
        ComplexityProfile,
    ):
        raise TypeError(
            "profile must be a ComplexityProfile."
        )

    resolved_path = Path(
        output_path
    ).expanduser().resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        resolved_path.with_suffix(
            resolved_path.suffix
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile.to_dict(),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    temporary_path.replace(
        resolved_path
    )

    return resolved_path


class _ComplexitySelfTestModel(
    nn.Module
):
    """Small deterministic segmentation model used by the self-test."""

    def __init__(
        self,
    ) -> None:
        """Initialize the synthetic model."""

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                4
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.head = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Return one-channel mask logits."""

        return {
            "mask_logits": self.head(
                self.features(
                    image
                )
            )
        }


def run_complexity_self_test() -> dict[str, Any]:
    """Run deterministic CPU parameter, memory, MAC, and latency tests."""

    import tempfile

    torch.manual_seed(
        42
    )

    model = (
        _ComplexitySelfTestModel()
    )

    model.train()

    total_parameters = (
        count_model_parameters(
            model
        )
    )

    trainable_parameters = (
        count_model_parameters(
            model,
            trainable_only=True,
        )
    )

    buffer_count = (
        count_model_buffers(
            model
        )
    )

    state_memory = (
        model_persistent_state_bytes(
            model
        )
    )

    profile = (
        profile_model_complexity(
            model,
            input_shape=(
                2,
                3,
                16,
                16,
            ),
            device="cpu",
            input_dtype=(
                torch.float32
            ),
            warmup_steps=1,
            measurement_steps=3,
            use_amp=False,
            count_tracked_macs=True,
            restore_original_device=True,
        )
    )

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        output_path = (
            Path(
                temporary_directory
            )
            / "complexity.json"
        )

        saved_path = (
            save_complexity_profile(
                profile,
                output_path,
            )
        )

        with saved_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved_payload = json.load(
                file
            )

        json_saved_correctly = (
            saved_payload[
                "protocol_version"
            ]
            == COMPLEXITY_PROTOCOL_VERSION
            and saved_payload[
                "parameter_count"
            ]
            == 121
            and saved_payload[
                "tracked_macs"
            ]
            == 57_344
        )

    expected_parameter_bytes = (
        121 * 4
    )

    expected_buffer_bytes = (
        4 * 4
        + 4 * 4
        + 1 * 8
    )

    checks = {
        "parameter_count_exact": (
            total_parameters
            == 121
        ),
        "all_parameters_trainable": (
            trainable_parameters
            == 121
        ),
        "buffer_count_exact": (
            buffer_count == 9
        ),
        "parameter_bytes_exact": (
            state_memory[
                "parameter_bytes"
            ]
            == expected_parameter_bytes
        ),
        "buffer_bytes_exact": (
            state_memory[
                "buffer_bytes"
            ]
            == expected_buffer_bytes
        ),
        "persistent_bytes_exact": (
            state_memory[
                "persistent_state_bytes"
            ]
            == (
                expected_parameter_bytes
                + expected_buffer_bytes
            )
        ),
        "profile_protocol_correct": (
            profile.protocol_version
            == COMPLEXITY_PROTOCOL_VERSION
        ),
        "profile_model_class_correct": (
            profile.model_class
            == (
                "_ComplexitySelfTestModel"
            )
        ),
        "profile_input_shape_correct": (
            profile.input_shape
            == (
                2,
                3,
                16,
                16,
            )
        ),
        "profile_output_shape_correct": (
            profile.output_shape
            == (
                2,
                1,
                16,
                16,
            )
        ),
        "profile_parameter_count_exact": (
            profile.parameter_count
            == 121
        ),
        "profile_buffer_count_exact": (
            profile.buffer_count
            == 9
        ),
        "tracked_macs_exact": (
            profile.tracked_macs
            == 57_344
        ),
        "tracked_gmacs_correct": (
            profile.tracked_gmacs
            is not None
            and math.isclose(
                profile.tracked_gmacs,
                57_344
                / GIGA_MAC,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ),
        "mean_latency_positive": (
            profile
            .mean_latency_milliseconds
            > 0.0
        ),
        "median_latency_positive": (
            profile
            .median_latency_milliseconds
            > 0.0
        ),
        "throughput_positive": (
            profile
            .throughput_images_per_second
            > 0.0
        ),
        "cpu_peak_memory_is_none": (
            profile
            .peak_device_memory_mebibytes
            is None
        ),
        "model_training_mode_restored": (
            model.training is True
        ),
        "model_device_restored": (
            _first_model_device(
                model
            ).type
            == "cpu"
        ),
        "json_saved_correctly": (
            json_saved_correctly
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
            COMPLEXITY_PROTOCOL_VERSION
        ),
        "checks": checks,
        "profile": (
            profile.to_dict()
        ),
    }