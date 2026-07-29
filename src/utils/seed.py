"""Reproducibility utilities for BCS-HCTNet.

This module controls random-number generation across:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- PyTorch DataLoader workers.

It also supports saving and restoring RNG states inside checkpoints so a
training run resumed in a later Kaggle session can continue reproducibly.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


DEFAULT_SEED = 42

MAX_NUMPY_SEED = 2**32 - 1

REPRODUCIBILITY_PROTOCOL_VERSION = (
    "BCS-HCTNet-reproducibility-v1"
)


@dataclass(frozen=True)
class ReproducibilitySettings:
    """Validated deterministic-execution settings."""

    seed: int = DEFAULT_SEED
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    deterministic_warn_only: bool = False

    def validate(self) -> None:
        """Validate all reproducibility settings."""

        if isinstance(
            self.seed,
            bool,
        ) or not isinstance(
            self.seed,
            int,
        ):
            raise TypeError(
                "seed must be an integer, "
                f"received {self.seed!r}."
            )

        if not (
            0
            <= self.seed
            <= MAX_NUMPY_SEED
        ):
            raise ValueError(
                "seed must be in the range "
                f"[0, {MAX_NUMPY_SEED}], "
                f"received {self.seed}."
            )

        for name, value in {
            "deterministic_algorithms": (
                self.deterministic_algorithms
            ),
            "cudnn_deterministic": (
                self.cudnn_deterministic
            ),
            "cudnn_benchmark": (
                self.cudnn_benchmark
            ),
            "deterministic_warn_only": (
                self.deterministic_warn_only
            ),
        }.items():
            if not isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{name} must be Boolean, "
                    f"received {value!r}."
                )

        if (
            self.deterministic_algorithms
            and self.cudnn_benchmark
        ):
            raise ValueError(
                "cudnn_benchmark must be false "
                "when deterministic algorithms "
                "are enabled."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        self.validate()

        return {
            "protocol_version": (
                REPRODUCIBILITY_PROTOCOL_VERSION
            ),
            "seed": self.seed,
            "deterministic_algorithms": (
                self.deterministic_algorithms
            ),
            "cudnn_deterministic": (
                self.cudnn_deterministic
            ),
            "cudnn_benchmark": (
                self.cudnn_benchmark
            ),
            "deterministic_warn_only": (
                self.deterministic_warn_only
            ),
        }


def validate_seed(
    seed: int,
) -> int:
    """Validate and return an RNG seed."""

    settings = ReproducibilitySettings(
        seed=seed
    )

    settings.validate()

    return settings.seed


def seed_everything(
    seed: int = DEFAULT_SEED,
    deterministic_algorithms: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    deterministic_warn_only: bool = False,
) -> ReproducibilitySettings:
    """Seed Python, NumPy, PyTorch CPU, and CUDA.

    Call this once at the beginning of every training or evaluation
    process before constructing datasets, data loaders, models, or
    optimizers.

    Setting ``PYTHONHASHSEED`` during a running interpreter cannot
    retroactively change hashes already created by that interpreter,
    but recording the value ensures child processes inherit it.
    """

    settings = ReproducibilitySettings(
        seed=seed,
        deterministic_algorithms=(
            deterministic_algorithms
        ),
        cudnn_deterministic=(
            cudnn_deterministic
        ),
        cudnn_benchmark=(
            cudnn_benchmark
        ),
        deterministic_warn_only=(
            deterministic_warn_only
        ),
    )

    settings.validate()

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        settings.seed
    )

    # Required by deterministic CUDA matrix operations on supported
    # CUDA and cuBLAS versions. It is harmless during CPU-only runs.
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(
        settings.seed
    )

    np.random.seed(
        settings.seed
    )

    torch.manual_seed(
        settings.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            settings.seed
        )

        torch.cuda.manual_seed_all(
            settings.seed
        )

    torch.use_deterministic_algorithms(
        settings.deterministic_algorithms,
        warn_only=(
            settings.deterministic_warn_only
        ),
    )

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = (
            settings.cudnn_deterministic
        )

        torch.backends.cudnn.benchmark = (
            settings.cudnn_benchmark
        )

    return settings


def seed_worker(
    worker_id: int,
) -> None:
    """Seed one PyTorch DataLoader worker deterministically.

    PyTorch assigns each worker a deterministic initial seed derived
    from the DataLoader generator. NumPy and Python RNGs are seeded
    from that value so all augmentation libraries remain aligned.
    """

    if isinstance(
        worker_id,
        bool,
    ) or not isinstance(
        worker_id,
        int,
    ):
        raise TypeError(
            "worker_id must be an integer, "
            f"received {worker_id!r}."
        )

    if worker_id < 0:
        raise ValueError(
            "worker_id must be non-negative."
        )

    worker_seed = int(
        torch.initial_seed()
        % (
            MAX_NUMPY_SEED
            + 1
        )
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def create_torch_generator(
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> torch.Generator:
    """Create a deterministically seeded PyTorch generator."""

    validated_seed = validate_seed(
        seed
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        validated_seed
    )

    return generator


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG states for resumable checkpoints."""

    state: dict[str, Any] = {
        "protocol_version": (
            REPRODUCIBILITY_PROTOCOL_VERSION
        ),
        "python_random_state": (
            random.getstate()
        ),
        "numpy_random_state": (
            np.random.get_state()
        ),
        "torch_cpu_rng_state": (
            torch.get_rng_state()
        ),
        "torch_cuda_rng_states": None,
    }

    if torch.cuda.is_available():
        state[
            "torch_cuda_rng_states"
        ] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
    state: Mapping[str, Any],
) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""

    if not isinstance(
        state,
        Mapping,
    ):
        raise TypeError(
            "RNG state must be a mapping."
        )

    required_keys = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }

    missing = sorted(
        required_keys
        - set(state)
    )

    if missing:
        raise KeyError(
            "RNG state is missing keys: "
            f"{missing}."
        )

    random.setstate(
        state[
            "python_random_state"
        ]
    )

    np.random.set_state(
        state[
            "numpy_random_state"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_cpu_rng_state"
        ]
    )

    cuda_states = state[
        "torch_cuda_rng_states"
    ]

    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG "
                "states, but CUDA is unavailable."
            )

        torch.cuda.set_rng_state_all(
            list(
                cuda_states
            )
        )


def reproducibility_environment() -> dict[str, Any]:
    """Return the active reproducibility environment."""

    cudnn_available = bool(
        hasattr(
            torch.backends,
            "cudnn",
        )
    )

    return {
        "protocol_version": (
            REPRODUCIBILITY_PROTOCOL_VERSION
        ),
        "python_hash_seed": (
            os.environ.get(
                "PYTHONHASHSEED"
            )
        ),
        "cublas_workspace_config": (
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            )
        ),
        "torch_version": (
            torch.__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_available": (
            cudnn_available
        ),
        "cudnn_deterministic": (
            bool(
                torch.backends.cudnn.deterministic
            )
            if cudnn_available
            else None
        ),
        "cudnn_benchmark": (
            bool(
                torch.backends.cudnn.benchmark
            )
            if cudnn_available
            else None
        ),
    }


def run_reproducibility_self_test(
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Verify deterministic Python, NumPy, and PyTorch sequences."""

    validated_seed = validate_seed(
        seed
    )

    seed_everything(
        seed=validated_seed,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
    )

    first_python = [
        random.random()
        for _ in range(5)
    ]

    first_numpy = (
        np.random.random(5)
        .astype(
            np.float64
        )
        .tolist()
    )

    first_torch = (
        torch.rand(5)
        .to(
            dtype=torch.float64
        )
        .tolist()
    )

    captured_state = (
        capture_rng_state()
    )

    continuation_python = [
        random.random()
        for _ in range(5)
    ]

    continuation_numpy = (
        np.random.random(5)
        .astype(
            np.float64
        )
        .tolist()
    )

    continuation_torch = (
        torch.rand(5)
        .to(
            dtype=torch.float64
        )
        .tolist()
    )

    restore_rng_state(
        captured_state
    )

    restored_python = [
        random.random()
        for _ in range(5)
    ]

    restored_numpy = (
        np.random.random(5)
        .astype(
            np.float64
        )
        .tolist()
    )

    restored_torch = (
        torch.rand(5)
        .to(
            dtype=torch.float64
        )
        .tolist()
    )

    seed_everything(
        seed=validated_seed,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
    )

    repeated_python = [
        random.random()
        for _ in range(5)
    ]

    repeated_numpy = (
        np.random.random(5)
        .astype(
            np.float64
        )
        .tolist()
    )

    repeated_torch = (
        torch.rand(5)
        .to(
            dtype=torch.float64
        )
        .tolist()
    )

    checks = {
        "python_seed_repeats": (
            first_python
            == repeated_python
        ),
        "numpy_seed_repeats": (
            first_numpy
            == repeated_numpy
        ),
        "torch_seed_repeats": (
            first_torch
            == repeated_torch
        ),
        "python_state_restores": (
            continuation_python
            == restored_python
        ),
        "numpy_state_restores": (
            continuation_numpy
            == restored_numpy
        ),
        "torch_state_restores": (
            continuation_torch
            == restored_torch
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
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
        "seed": validated_seed,
        "checks": checks,
        "environment": (
            reproducibility_environment()
        ),
    }