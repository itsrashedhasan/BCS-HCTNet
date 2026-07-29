"""Reliable training checkpoint management.

This module saves and restores all state required to resume a baseline or
BCS-HCTNet experiment:

- model parameters;
- optimizer state;
- learning-rate scheduler state;
- mixed-precision gradient-scaler state;
- epoch and global step;
- best validation metric;
- experiment metadata;
- Python, NumPy, PyTorch, and CUDA RNG states.

Checkpoint files are written atomically. A SHA-256 sidecar is also created so
corrupted or partially copied checkpoints can be detected before loading.

Only load checkpoints created by this project or another trusted source.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


CHECKPOINT_PROTOCOL_VERSION = (
    "BCS-HCTNet-training-checkpoint-v1"
)

CHECKPOINT_SCHEMA_VERSION = 1


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a file SHA-256 checksum."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot hash missing file: {path}"
        )

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


def checkpoint_checksum_path(
    checkpoint_path: Path,
) -> Path:
    """Return the SHA-256 sidecar path."""

    return checkpoint_path.with_name(
        checkpoint_path.name + ".sha256"
    )


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write text atomically in the destination directory."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                f".{path.name}."
            ),
            suffix=".tmp",
            dir=str(
                path.parent
            ),
        )
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(
                content
            )

            output_file.flush()
            os.fsync(
                output_file.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:
        temporary_path.unlink(
            missing_ok=True
        )

        raise


def _atomic_torch_save(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """Save a PyTorch object atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=(
                f".{path.name}."
            ),
            suffix=".tmp",
            dir=str(
                path.parent
            ),
        )
    )

    os.close(
        file_descriptor
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        torch.save(
            dict(payload),
            temporary_path,
        )

        with temporary_path.open(
            "rb"
        ) as input_file:
            os.fsync(
                input_file.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:
        temporary_path.unlink(
            missing_ok=True
        )

        raise


def unwrap_model(
    model: nn.Module,
) -> nn.Module:
    """Return the underlying module for wrapped models."""

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    wrapped_module = getattr(
        model,
        "module",
        None,
    )

    if isinstance(
        wrapped_module,
        nn.Module,
    ):
        return wrapped_module

    return model


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch, and CUDA RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "cuda_device_count": int(
            torch.cuda.device_count()
        )
        if torch.cuda.is_available()
        else 0,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: Mapping[str, Any],
) -> None:
    """Restore captured RNG state."""

    if not isinstance(
        state,
        Mapping,
    ):
        raise TypeError(
            "RNG state must be a mapping."
        )

    required_keys = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }

    missing = sorted(
        required_keys
        - set(
            state
        )
    )

    if missing:
        raise KeyError(
            f"RNG state is missing keys: {missing}."
        )

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch_cpu_state = state[
        "torch_cpu"
    ]

    if not isinstance(
        torch_cpu_state,
        Tensor,
    ):
        raise TypeError(
            "torch_cpu RNG state must be a tensor."
        )

    torch.set_rng_state(
        torch_cpu_state
    )

    cuda_states = state.get(
        "torch_cuda"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        if not isinstance(
            cuda_states,
            list,
        ):
            raise TypeError(
                "torch_cuda RNG state must be "
                "a list of tensors."
            )

        available_device_count = int(
            torch.cuda.device_count()
        )

        if len(
            cuda_states
        ) != available_device_count:
            raise RuntimeError(
                "Checkpoint CUDA RNG state count "
                "does not match the current CUDA "
                "device count."
            )

        torch.cuda.set_rng_state_all(
            cuda_states
        )


def _state_dict_or_none(
    component: object | None,
    context: str,
) -> dict[str, Any] | None:
    """Return a component state dictionary when supplied."""

    if component is None:
        return None

    state_dict_function = getattr(
        component,
        "state_dict",
        None,
    )

    if not callable(
        state_dict_function
    ):
        raise TypeError(
            f"{context} must provide state_dict()."
        )

    state = state_dict_function()

    if not isinstance(
        state,
        Mapping,
    ):
        raise TypeError(
            f"{context}.state_dict() must "
            "return a mapping."
        )

    return dict(
        state
    )


def build_checkpoint(
    *,
    model: nn.Module,
    epoch: int,
    global_step: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_rng_state: bool = True,
) -> dict[str, Any]:
    """Build a complete serializable checkpoint payload."""

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(
            "epoch must be a non-negative integer."
        )

    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            "global_step must be a "
            "non-negative integer."
        )

    if (
        best_epoch is not None
        and (
            isinstance(best_epoch, bool)
            or not isinstance(
                best_epoch,
                int,
            )
            or best_epoch < 0
        )
    ):
        raise ValueError(
            "best_epoch must be None or a "
            "non-negative integer."
        )

    if best_metric is not None:
        best_metric = float(
            best_metric
        )

        if not np.isfinite(
            best_metric
        ):
            raise ValueError(
                "best_metric must be finite."
            )

    if not isinstance(
        include_rng_state,
        bool,
    ):
        raise TypeError(
            "include_rng_state must be Boolean."
        )

    resolved_model = unwrap_model(
        model
    )

    model_state = (
        resolved_model.state_dict()
    )

    payload: dict[str, Any] = {
        "protocol_version": (
            CHECKPOINT_PROTOCOL_VERSION
        ),
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_class": (
            resolved_model.__class__.__name__
        ),
        "model_state_dict": (
            model_state
        ),
        "optimizer_state_dict": (
            _state_dict_or_none(
                optimizer,
                "optimizer",
            )
        ),
        "scheduler_state_dict": (
            _state_dict_or_none(
                scheduler,
                "scheduler",
            )
        ),
        "scaler_state_dict": (
            _state_dict_or_none(
                scaler,
                "scaler",
            )
        ),
        "metrics": (
            dict(metrics)
            if metrics is not None
            else {}
        ),
        "metadata": (
            dict(metadata)
            if metadata is not None
            else {}
        ),
        "rng_state": (
            capture_rng_state()
            if include_rng_state
            else None
        ),
        "environment": {
            "torch_version": (
                torch.__version__
            ),
            "numpy_version": (
                np.__version__
            ),
            "cuda_available": bool(
                torch.cuda.is_available()
            ),
            "cuda_device_count": int(
                torch.cuda.device_count()
            )
            if torch.cuda.is_available()
            else 0,
        },
    }

    return payload


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    global_step: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_rng_state: bool = True,
    write_checksum: bool = True,
) -> dict[str, Any]:
    """Atomically save a complete training checkpoint."""

    if not isinstance(
        write_checksum,
        bool,
    ):
        raise TypeError(
            "write_checksum must be Boolean."
        )

    resolved_path = (
        Path(
            checkpoint_path
        )
        .expanduser()
        .resolve()
    )

    if resolved_path.exists() and (
        not resolved_path.is_file()
    ):
        raise RuntimeError(
            "Checkpoint path exists but is "
            f"not a file: {resolved_path}"
        )

    payload = build_checkpoint(
        model=model,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_metric,
        best_epoch=best_epoch,
        metrics=metrics,
        metadata=metadata,
        include_rng_state=(
            include_rng_state
        ),
    )

    _atomic_torch_save(
        payload,
        resolved_path,
    )

    checksum = sha256_file(
        resolved_path
    )

    checksum_path = (
        checkpoint_checksum_path(
            resolved_path
        )
    )

    if write_checksum:
        _atomic_write_text(
            checksum_path,
            checksum + "\n",
        )

    elif checksum_path.exists():
        checksum_path.unlink()

    return {
        "checkpoint_path": str(
            resolved_path
        ),
        "checksum_path": (
            str(
                checksum_path
            )
            if write_checksum
            else None
        ),
        "sha256": checksum,
        "bytes": (
            resolved_path.stat().st_size
        ),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "protocol_version": (
            CHECKPOINT_PROTOCOL_VERSION
        ),
    }


def verify_checkpoint_checksum(
    checkpoint_path: str | Path,
    *,
    require_checksum: bool = True,
) -> dict[str, Any]:
    """Verify a checkpoint against its SHA-256 sidecar."""

    if not isinstance(
        require_checksum,
        bool,
    ):
        raise TypeError(
            "require_checksum must be Boolean."
        )

    resolved_path = (
        Path(
            checkpoint_path
        )
        .expanduser()
        .resolve()
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_path}"
        )

    checksum_path = (
        checkpoint_checksum_path(
            resolved_path
        )
    )

    observed_checksum = sha256_file(
        resolved_path
    )

    if not checksum_path.is_file():
        if require_checksum:
            raise FileNotFoundError(
                "Checkpoint checksum sidecar "
                f"not found: {checksum_path}"
            )

        return {
            "checkpoint_path": str(
                resolved_path
            ),
            "checksum_path": None,
            "observed_sha256": (
                observed_checksum
            ),
            "expected_sha256": None,
            "checksum_present": False,
            "checksum_matches": None,
        }

    expected_checksum = (
        checksum_path.read_text(
            encoding="utf-8"
        )
        .strip()
        .lower()
    )

    if len(
        expected_checksum
    ) != 64:
        raise RuntimeError(
            "Checkpoint checksum sidecar "
            "does not contain a valid SHA-256 "
            f"value: {checksum_path}"
        )

    checksum_matches = (
        observed_checksum.lower()
        == expected_checksum
    )

    if not checksum_matches:
        raise RuntimeError(
            "Checkpoint SHA-256 verification "
            f"failed: {resolved_path}"
        )

    return {
        "checkpoint_path": str(
            resolved_path
        ),
        "checksum_path": str(
            checksum_path
        ),
        "observed_sha256": (
            observed_checksum
        ),
        "expected_sha256": (
            expected_checksum
        ),
        "checksum_present": True,
        "checksum_matches": True,
    }


def _trusted_torch_load(
    path: Path,
    map_location: str | torch.device,
) -> object:
    """Load a trusted project checkpoint across PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def _require_checkpoint_payload(
    payload: object,
) -> dict[str, Any]:
    """Validate the top-level checkpoint schema."""

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Checkpoint must contain a mapping."
        )

    checkpoint = dict(
        payload
    )

    required_keys = {
        "protocol_version",
        "schema_version",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "metrics",
        "metadata",
        "rng_state",
    }

    missing = sorted(
        required_keys
        - set(
            checkpoint
        )
    )

    if missing:
        raise KeyError(
            f"Checkpoint is missing keys: {missing}."
        )

    if (
        checkpoint[
            "protocol_version"
        ]
        != CHECKPOINT_PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "Unsupported checkpoint protocol: "
            f"{checkpoint['protocol_version']!r}."
        )

    if (
        checkpoint[
            "schema_version"
        ]
        != CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Unsupported checkpoint schema "
            f"version: "
            f"{checkpoint['schema_version']!r}."
        )

    if not isinstance(
        checkpoint[
            "model_state_dict"
        ],
        Mapping,
    ):
        raise TypeError(
            "model_state_dict must be a mapping."
        )

    return checkpoint


def _load_component_state(
    *,
    component: object | None,
    saved_state: object,
    context: str,
    required: bool,
) -> bool:
    """Restore one optional optimizer-like component."""

    if component is None:
        if required and saved_state is not None:
            raise RuntimeError(
                f"{context} state exists in the "
                "checkpoint, but no current "
                f"{context} was supplied."
            )

        return False

    if saved_state is None:
        if required:
            raise RuntimeError(
                f"Checkpoint contains no "
                f"{context} state."
            )

        return False

    if not isinstance(
        saved_state,
        Mapping,
    ):
        raise TypeError(
            f"Checkpoint {context} state must "
            "be a mapping."
        )

    load_state_dict_function = getattr(
        component,
        "load_state_dict",
        None,
    )

    if not callable(
        load_state_dict_function
    ):
        raise TypeError(
            f"{context} must provide "
            "load_state_dict()."
        )

    load_state_dict_function(
        dict(
            saved_state
        )
    )

    return True


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    map_location: str | torch.device = "cpu",
    strict_model: bool = True,
    restore_rng: bool = False,
    verify_checksum: bool = True,
    require_optimizer_state: bool = False,
    require_scheduler_state: bool = False,
    require_scaler_state: bool = False,
) -> dict[str, Any]:
    """Load a trusted project checkpoint and restore training state."""

    for name, value in {
        "strict_model": strict_model,
        "restore_rng": restore_rng,
        "verify_checksum": verify_checksum,
        "require_optimizer_state": (
            require_optimizer_state
        ),
        "require_scheduler_state": (
            require_scheduler_state
        ),
        "require_scaler_state": (
            require_scaler_state
        ),
    }.items():
        if not isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be Boolean."
            )

    resolved_path = (
        Path(
            checkpoint_path
        )
        .expanduser()
        .resolve()
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_path}"
        )

    checksum_report = None

    if verify_checksum:
        checksum_report = (
            verify_checkpoint_checksum(
                resolved_path,
                require_checksum=True,
            )
        )

    payload = _trusted_torch_load(
        resolved_path,
        map_location=map_location,
    )

    checkpoint = (
        _require_checkpoint_payload(
            payload
        )
    )

    resolved_model = unwrap_model(
        model
    )

    incompatible = (
        resolved_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=strict_model,
        )
    )

    optimizer_restored = (
        _load_component_state(
            component=optimizer,
            saved_state=checkpoint[
                "optimizer_state_dict"
            ],
            context="optimizer",
            required=(
                require_optimizer_state
            ),
        )
    )

    scheduler_restored = (
        _load_component_state(
            component=scheduler,
            saved_state=checkpoint[
                "scheduler_state_dict"
            ],
            context="scheduler",
            required=(
                require_scheduler_state
            ),
        )
    )

    scaler_restored = (
        _load_component_state(
            component=scaler,
            saved_state=checkpoint[
                "scaler_state_dict"
            ],
            context="scaler",
            required=(
                require_scaler_state
            ),
        )
    )

    rng_restored = False

    if restore_rng:
        rng_state = checkpoint[
            "rng_state"
        ]

        if rng_state is None:
            raise RuntimeError(
                "Checkpoint does not contain "
                "an RNG state."
            )

        if not isinstance(
            rng_state,
            Mapping,
        ):
            raise TypeError(
                "Checkpoint RNG state must "
                "be a mapping."
            )

        restore_rng_state(
            rng_state
        )

        rng_restored = True

    missing_keys = list(
        incompatible.missing_keys
    )

    unexpected_keys = list(
        incompatible.unexpected_keys
    )

    return {
        "checkpoint_path": str(
            resolved_path
        ),
        "protocol_version": (
            checkpoint[
                "protocol_version"
            ]
        ),
        "schema_version": (
            checkpoint[
                "schema_version"
            ]
        ),
        "epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "global_step": int(
            checkpoint[
                "global_step"
            ]
        ),
        "best_metric": (
            checkpoint.get(
                "best_metric"
            )
        ),
        "best_epoch": (
            checkpoint.get(
                "best_epoch"
            )
        ),
        "metrics": dict(
            checkpoint[
                "metrics"
            ]
        ),
        "metadata": dict(
            checkpoint[
                "metadata"
            ]
        ),
        "model_missing_keys": (
            missing_keys
        ),
        "model_unexpected_keys": (
            unexpected_keys
        ),
        "optimizer_restored": (
            optimizer_restored
        ),
        "scheduler_restored": (
            scheduler_restored
        ),
        "scaler_restored": (
            scaler_restored
        ),
        "rng_restored": (
            rng_restored
        ),
        "checksum_verification": (
            checksum_report
        ),
    }


def save_checkpoint_manifest(
    output_path: str | Path,
    checkpoint_records: list[
        Mapping[str, Any]
    ],
) -> Path:
    """Write a JSON manifest of experiment checkpoints."""

    resolved_path = (
        Path(
            output_path
        )
        .expanduser()
        .resolve()
    )

    records = [
        dict(record)
        for record in checkpoint_records
    ]

    report = {
        "protocol_version": (
            CHECKPOINT_PROTOCOL_VERSION
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "number_of_checkpoints": len(
            records
        ),
        "checkpoints": records,
    }

    _atomic_write_text(
        resolved_path,
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )

    return resolved_path


def run_checkpoint_self_test() -> dict[str, Any]:
    """Run a complete CPU save, verify, mutate, and restore test."""

    import tempfile

    random.seed(
        42
    )

    np.random.seed(
        42
    )

    torch.manual_seed(
        42
    )

    model = nn.Sequential(
        nn.Linear(
            4,
            8,
        ),
        nn.ReLU(),
        nn.Linear(
            8,
            2,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.001,
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.1,
        )
    )

    input_tensor = torch.randn(
        3,
        4,
    )

    target_tensor = torch.randn(
        3,
        2,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    prediction = model(
        input_tensor
    )

    loss = torch.nn.functional.mse_loss(
        prediction,
        target_tensor,
    )

    loss.backward()

    optimizer.step()
    scheduler.step()

    expected_model_state = {
        name: tensor.detach()
        .clone()
        for name, tensor
        in model.state_dict().items()
    }

    expected_learning_rate = float(
        optimizer.param_groups[0][
            "lr"
        ]
    )

    expected_scheduler_epoch = int(
        scheduler.last_epoch
    )

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        checkpoint_path = (
            Path(
                temporary_directory
            )
            / "checkpoint.pt"
        )

        save_report = save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            global_step=17,
            best_metric=0.8125,
            best_epoch=2,
            metrics={
                "validation_dice": 0.8125,
                "validation_loss": 0.345,
            },
            metadata={
                "experiment_id": "SELF_TEST",
                "seed": 42,
            },
            include_rng_state=True,
            write_checksum=True,
        )

        checksum_report = (
            verify_checkpoint_checksum(
                checkpoint_path
            )
        )

        with torch.no_grad():
            for parameter in (
                model.parameters()
            ):
                parameter.add_(
                    100.0
                )

        optimizer.param_groups[0][
            "lr"
        ] = 0.123

        scheduler.last_epoch = 999

        load_report = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict_model=True,
            restore_rng=False,
            verify_checksum=True,
            require_optimizer_state=True,
            require_scheduler_state=True,
        )

        restored_model_state = (
            model.state_dict()
        )

        model_restored = all(
            torch.equal(
                restored_model_state[name],
                expected_tensor,
            )
            for name, expected_tensor
            in expected_model_state.items()
        )

        checksum_path = (
            checkpoint_checksum_path(
                checkpoint_path
            )
        )

        checkpoint_exists = (
            checkpoint_path.is_file()
        )

        checksum_exists = (
            checksum_path.is_file()
        )

        checkpoint_bytes = (
            checkpoint_path.stat().st_size
        )

    checks = {
        "checkpoint_exists": (
            checkpoint_exists
        ),
        "checksum_exists": (
            checksum_exists
        ),
        "checkpoint_nonempty": (
            checkpoint_bytes > 0
        ),
        "checksum_matches": (
            checksum_report[
                "checksum_matches"
            ]
            is True
        ),
        "model_restored": (
            model_restored
        ),
        "optimizer_restored": (
            load_report[
                "optimizer_restored"
            ]
            is True
        ),
        "scheduler_restored": (
            load_report[
                "scheduler_restored"
            ]
            is True
        ),
        "learning_rate_restored": (
            float(
                optimizer.param_groups[0][
                    "lr"
                ]
            )
            == expected_learning_rate
        ),
        "scheduler_epoch_restored": (
            int(
                scheduler.last_epoch
            )
            == expected_scheduler_epoch
        ),
        "epoch_restored": (
            load_report[
                "epoch"
            ]
            == 3
        ),
        "global_step_restored": (
            load_report[
                "global_step"
            ]
            == 17
        ),
        "best_metric_restored": (
            load_report[
                "best_metric"
            ]
            == 0.8125
        ),
        "best_epoch_restored": (
            load_report[
                "best_epoch"
            ]
            == 2
        ),
        "metadata_restored": (
            load_report[
                "metadata"
            ][
                "experiment_id"
            ]
            == "SELF_TEST"
        ),
        "metrics_restored": (
            load_report[
                "metrics"
            ][
                "validation_dice"
            ]
            == 0.8125
        ),
        "strict_model_clean": (
            load_report[
                "model_missing_keys"
            ]
            == []
            and load_report[
                "model_unexpected_keys"
            ]
            == []
        ),
        "rng_saved": (
            save_report[
                "protocol_version"
            ]
            == (
                CHECKPOINT_PROTOCOL_VERSION
            )
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
            CHECKPOINT_PROTOCOL_VERSION
        ),
        "checks": checks,
        "saved_epoch": (
            save_report[
                "epoch"
            ]
        ),
        "saved_global_step": (
            save_report[
                "global_step"
            ]
        ),
        "saved_bytes": (
            save_report[
                "bytes"
            ]
        ),
        "saved_sha256": (
            save_report[
                "sha256"
            ]
        ),
        "restored_learning_rate": float(
            optimizer.param_groups[0][
                "lr"
            ]
        ),
        "restored_scheduler_epoch": int(
            scheduler.last_epoch
        ),
    }