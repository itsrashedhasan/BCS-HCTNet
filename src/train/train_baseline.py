"""Validated command-line entry point for E01-E05 baseline training.

This module connects the already validated baseline configuration, dataset,
model, loss, scheduler, trainer, checkpoint, logging, and reproducibility
interfaces without duplicating their implementation.

Supported run modes
-------------------
``--smoke-test``
    Uses the deterministic smoke subsets and smoke epoch count from the YAML.
    The learning-rate scheduler still uses the full configured training horizon
    so a one-epoch smoke run exercises the real first-epoch learning rate rather
    than collapsing cosine annealing to zero.

Full training
    Uses the complete approved train and validation manifests and the configured
    maximum epoch count. Full training requires CUDA unless the operator gives
    the explicit ``--allow-cpu-full-training`` override.

Resume behavior
---------------
A run can resume from a trusted project checkpoint using
``--resume-checkpoint``. When ``training.resume.strict_configuration_match`` is
true, the checkpoint must be accompanied by the ``RUN_MANIFEST.json`` written
by this entry point, and its canonical configuration SHA-256 must match the
current validated configuration. Logger, validation, and checkpoint artifacts
are copied into the new writable output directory when the source checkpoint is
mounted from a read-only Kaggle Dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from src.data.dataloader import DataLoaderBundle, build_loader_bundle
from src.losses import BCEDiceLoss, build_baseline_mask_criterion
from src.models.baselines import (
    build_baseline_model,
    get_baseline_architecture_summary,
)
from src.train.baseline_experiment_config import (
    ValidatedBaselineExperimentConfig,
    save_baseline_validation_bundle,
    validate_baseline_experiment_configuration,
)
from src.train.checkpoint import load_checkpoint
from src.train.scheduler import SchedulerBundle, build_scheduler
from src.train.trainer import SegmentationTrainer, resolve_training_device
from src.utils.seed import reproducibility_environment, seed_everything


BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION = (
    "BCS-HCTNet-baseline-training-entrypoint-v1"
)

DEFAULT_ISIC2018_SOURCE_ROOT = Path(
    "/kaggle/input/datasets/asrafulislam7/isic-2018/ISIC_2018"
)

RUN_MANIFEST_FILENAME = "RUN_MANIFEST.json"
RUN_FAILURE_FILENAME = "RUN_FAILURE.json"

TARGET_THRESHOLD = 0.5
JACCARD_QUALITY_THRESHOLD = 0.65
BOUNDARY_TOLERANCE_PIXELS = 2.0


class BaselineTrainingError(RuntimeError):
    """Raised when an approved baseline run cannot be started safely."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write a JSON mapping atomically."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = resolved_path.with_suffix(
        resolved_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(resolved_path)
    return resolved_path


def _path_is_nonempty(path: Path) -> bool:
    """Return whether a directory exists and contains at least one entry."""

    return path.is_dir() and next(path.iterdir(), None) is not None


def _run_git_command(
    repository_root: Path,
    arguments: Sequence[str],
) -> str | None:
    """Run one read-only Git command and return stripped output."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    output = completed.stdout.strip()
    return output or None


def _repository_environment() -> dict[str, Any]:
    """Collect non-destructive repository provenance."""

    repository_root = Path.cwd().resolve()
    commit = _run_git_command(repository_root, ["rev-parse", "HEAD"])
    branch = _run_git_command(
        repository_root,
        ["rev-parse", "--abbrev-ref", "HEAD"],
    )
    status = _run_git_command(repository_root, ["status", "--porcelain"])

    return {
        "repository_root": str(repository_root),
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_dirty": bool(status),
        "git_status_porcelain": status or "",
    }


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    """Collect Python, PyTorch, CUDA, and reproducibility metadata."""

    cuda_devices = []

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
            )

    return {
        "entrypoint_protocol_version": (
            BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
        ),
        "captured_at_utc": _utc_now(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "torch_version": torch.__version__,
        "requested_runtime_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": cuda_devices,
        "reproducibility_environment": reproducibility_environment(),
        "repository": _repository_environment(),
    }


def _normalize_source_roots(
    values: Sequence[str | Path] | None,
) -> list[Path]:
    """Resolve and validate source-image roots."""

    raw_values: Sequence[str | Path]

    if values:
        raw_values = values
    else:
        raw_values = [DEFAULT_ISIC2018_SOURCE_ROOT]

    resolved: list[Path] = []
    seen: set[Path] = set()

    for value in raw_values:
        path = Path(value).expanduser().resolve()

        if path in seen:
            continue

        if not path.is_dir():
            raise FileNotFoundError(
                f"Source-image root does not exist: {path}"
            )

        seen.add(path)
        resolved.append(path)

    if not resolved:
        raise BaselineTrainingError(
            "At least one source-image root is required."
        )

    return resolved


def _resolve_output_root(
    validated: ValidatedBaselineExperimentConfig,
    *,
    smoke_test: bool,
    output_root_override: str | Path | None,
) -> Path:
    """Resolve the writable run output root."""

    if output_root_override is not None:
        return Path(output_root_override).expanduser().resolve()

    configured_root = Path(
        validated.payload["outputs"]["root"]
    ).expanduser().resolve()

    if smoke_test:
        return configured_root / "smoke_test"

    return configured_root


def _resolve_training_epochs(
    payload: Mapping[str, Any],
    *,
    smoke_test: bool,
) -> tuple[int, int]:
    """Return trainer epochs and scheduler horizon.

    The scheduler horizon always remains the configured full-training horizon.
    This prevents a one-epoch cosine smoke test from starting at zero learning
    rate.
    """

    maximum_epochs = int(payload["training"]["maximum_epochs"])

    if maximum_epochs <= 0:
        raise BaselineTrainingError(
            "training.maximum_epochs must be positive."
        )

    trainer_epochs = (
        int(payload["smoke_test"]["epochs"])
        if smoke_test
        else maximum_epochs
    )

    if trainer_epochs <= 0:
        raise BaselineTrainingError(
            "The resolved trainer epoch count must be positive."
        )

    return trainer_epochs, maximum_epochs


def _normalize_monitor_mode(value: object) -> str:
    """Map configuration monitor modes to the trainer contract."""

    normalized = str(value).strip().lower()

    aliases = {
        "maximum": "max",
        "max": "max",
        "minimum": "min",
        "min": "min",
    }

    if normalized not in aliases:
        raise BaselineTrainingError(
            "training.checkpoint.mode must be maximum/max or minimum/min."
        )

    return aliases[normalized]


def build_baseline_optimizer(
    model: nn.Module,
    payload: Mapping[str, Any],
) -> Optimizer:
    """Build the validated AdamW baseline optimizer."""

    optimizer_configuration = payload["training"]["optimizer"]
    optimizer_name = str(
        optimizer_configuration["name"]
    ).strip().lower()

    if optimizer_name != "adamw":
        raise BaselineTrainingError(
            "The controlled baseline entry point supports only AdamW."
        )

    learning_rate = float(
        optimizer_configuration["learning_rate"]
    )
    weight_decay = float(
        optimizer_configuration["weight_decay"]
    )

    if learning_rate <= 0.0:
        raise BaselineTrainingError(
            "Optimizer learning rate must be positive."
        )

    if weight_decay < 0.0:
        raise BaselineTrainingError(
            "Optimizer weight decay must be non-negative."
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise BaselineTrainingError(
            "The baseline model has no trainable parameters."
        )

    return AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_baseline_criterion(
    payload: Mapping[str, Any],
) -> BCEDiceLoss:
    """Build the approved shared BCE-Dice mask criterion."""

    loss = payload["loss"]
    weights = loss["weights"]

    criterion = build_baseline_mask_criterion(
        bce_weight=float(weights["bce"]),
        dice_weight=float(weights["dice"]),
        pos_weight=(
            None
            if loss["pos_weight"] is None
            else float(loss["pos_weight"])
        ),
        dice_smooth=float(loss["dice_smooth"]),
        dice_epsilon=float(loss["dice_epsilon"]),
        return_components=bool(loss["return_components"]),
    )

    if criterion.reduction != str(loss["reduction"]).strip().lower():
        raise BaselineTrainingError(
            "Built criterion reduction does not match the configuration."
        )

    return criterion


def _optimizer_summary(optimizer: Optimizer) -> dict[str, Any]:
    """Return effective optimizer hyperparameters."""

    groups = []

    for index, group in enumerate(optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": list(group.get("betas", ())),
                "epsilon": float(group.get("eps", 0.0)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "maximize": bool(group.get("maximize", False)),
                "parameter_tensor_count": len(group["params"]),
                "parameter_element_count": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
        )

    return {
        "class": optimizer.__class__.__name__,
        "parameter_groups": groups,
    }


def _create_output_directories(
    output_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Create every configured output directory plus trainer validation."""

    output_root.mkdir(parents=True, exist_ok=True)

    configured = payload["outputs"]["directories"]
    result: dict[str, Path] = {}

    for name, relative_name in configured.items():
        path = output_root / str(relative_name)
        path.mkdir(parents=True, exist_ok=True)
        result[str(name)] = path

    validation_path = output_root / "validation"
    validation_path.mkdir(parents=True, exist_ok=True)
    result["validation"] = validation_path

    return result


def _find_run_manifest(checkpoint_path: Path) -> Path | None:
    """Locate the run manifest accompanying a checkpoint."""

    candidates = [
        checkpoint_path.parent.parent / RUN_MANIFEST_FILENAME,
        checkpoint_path.parent / RUN_MANIFEST_FILENAME,
    ]

    for parent in checkpoint_path.parents:
        candidates.append(parent / RUN_MANIFEST_FILENAME)

        if len(candidates) >= 8:
            break

    seen: set[Path] = set()

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        if resolved.is_file():
            return resolved

    return None


def _load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a JSON object in {path}.")

    return dict(payload)


def _verify_resume_configuration(
    *,
    validated: ValidatedBaselineExperimentConfig,
    checkpoint_path: Path,
    strict_configuration_match: bool,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Verify resume provenance against the current configuration."""

    manifest_path = _find_run_manifest(checkpoint_path)

    if manifest_path is None:
        if strict_configuration_match:
            raise BaselineTrainingError(
                "Strict resume requires the RUN_MANIFEST.json that "
                "accompanied the checkpoint."
            )

        return None, None

    manifest = _load_json_mapping(manifest_path)

    configuration = manifest.get("configuration")
    experiment = manifest.get("experiment")

    if not isinstance(configuration, Mapping):
        raise BaselineTrainingError(
            f"Resume manifest has no valid configuration section: {manifest_path}"
        )

    if not isinstance(experiment, Mapping):
        raise BaselineTrainingError(
            f"Resume manifest has no valid experiment section: {manifest_path}"
        )

    current_identity = validated.validation_summary["experiment"]

    comparisons = {
        "canonical_sha256": (
            str(configuration.get("canonical_sha256", ""))
            == validated.canonical_sha256
        ),
        "experiment_id": (
            str(experiment.get("id", ""))
            == str(current_identity["id"])
        ),
        "model_name": (
            str(experiment.get("model_name", ""))
            == str(current_identity["expected_model"])
        ),
    }

    if strict_configuration_match and not all(comparisons.values()):
        failed = [name for name, passed in comparisons.items() if not passed]
        raise BaselineTrainingError(
            "Resume configuration mismatch for: " + ", ".join(failed)
        )

    return manifest_path, manifest


def _copy_resume_artifacts(
    *,
    checkpoint_path: Path,
    manifest_path: Path | None,
    destination_root: Path,
) -> dict[str, Any]:
    """Copy prior writable run state from a mounted resume dataset."""

    source_root = (
        manifest_path.parent
        if manifest_path is not None
        else checkpoint_path.parent.parent
    ).resolve()

    destination_root = destination_root.resolve()

    copied: list[str] = []

    if source_root != destination_root:
        for directory_name in (
            "logs",
            "validation",
            "checkpoints",
        ):
            source = source_root / directory_name
            destination = destination_root / directory_name

            if source.is_dir():
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                )
                copied.append(directory_name)

        previous_summary = source_root / "TRAINING_SUMMARY.json"

        if previous_summary.is_file():
            shutil.copy2(
                previous_summary,
                destination_root / "RESUMED_FROM_TRAINING_SUMMARY.json",
            )
            copied.append("TRAINING_SUMMARY.json")

        if manifest_path is not None:
            shutil.copy2(
                manifest_path,
                destination_root / "RESUMED_FROM_RUN_MANIFEST.json",
            )
            copied.append(RUN_MANIFEST_FILENAME)

    return {
        "source_run_root": str(source_root),
        "destination_run_root": str(destination_root),
        "copied_artifacts": copied,
    }


def _resolve_resume_checkpoint(
    *,
    explicit_checkpoint: str | Path | None,
    output_root: Path,
    payload: Mapping[str, Any],
    smoke_test: bool,
) -> Path | None:
    """Resolve explicit or same-output automatic resume."""

    if explicit_checkpoint is not None:
        checkpoint = Path(explicit_checkpoint).expanduser().resolve()

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint}"
            )

        return checkpoint

    resume_configuration = payload["training"]["resume"]

    if smoke_test or not bool(resume_configuration["enabled"]):
        return None

    candidate = output_root / "checkpoints" / "latest.pt"

    if candidate.is_file():
        return candidate.resolve()

    return None


def _prepare_fresh_output_root(
    *,
    output_root: Path,
    resume_checkpoint: Path | None,
    overwrite_existing: bool,
) -> None:
    """Protect existing experiment artifacts from accidental replacement."""

    if not output_root.exists():
        return

    if not output_root.is_dir():
        raise BaselineTrainingError(
            f"Output root exists but is not a directory: {output_root}"
        )

    if resume_checkpoint is not None:
        checkpoint_is_inside_output = False

        try:
            resume_checkpoint.resolve().relative_to(output_root.resolve())
            checkpoint_is_inside_output = True
        except ValueError:
            checkpoint_is_inside_output = False

        if _path_is_nonempty(output_root) and not checkpoint_is_inside_output:
            if not overwrite_existing:
                raise FileExistsError(
                    "The writable output root already contains artifacts, "
                    "but the resume checkpoint comes from another location. "
                    "Pass --overwrite-existing to replace the writable copy: "
                    f"{output_root}"
                )

            shutil.rmtree(output_root)

        return

    if not _path_is_nonempty(output_root):
        return

    if not overwrite_existing:
        raise FileExistsError(
            "The output root already contains artifacts. Use an approved "
            "resume checkpoint or pass --overwrite-existing explicitly: "
            f"{output_root}"
        )

    shutil.rmtree(output_root)


def _validate_runtime_training_contract(payload: Mapping[str, Any]) -> None:
    """Validate trainer features that are intentionally fixed in this stage."""

    training = payload["training"]

    if int(training["gradient_accumulation_steps"]) != 1:
        raise BaselineTrainingError(
            "The current shared trainer requires "
            "training.gradient_accumulation_steps: 1."
        )

    if int(training["validation_every_epochs"]) != 1:
        raise BaselineTrainingError(
            "The current shared trainer validates every epoch; "
            "training.validation_every_epochs must equal 1."
        )

    checkpoint = training["checkpoint"]

    output_directories = payload["outputs"]["directories"]
    trainer_fixed_directories = {
        "checkpoints": "checkpoints",
        "logs": "logs",
    }

    for name, expected_relative_path in trainer_fixed_directories.items():
        if str(output_directories[name]) != expected_relative_path:
            raise BaselineTrainingError(
                f"outputs.directories.{name} must equal "
                f"{expected_relative_path!r} for the current shared trainer."
            )

    if not bool(checkpoint["save_best_only"]):
        raise BaselineTrainingError(
            "The approved baseline protocol requires save_best_only: true."
        )

    if not bool(checkpoint["save_last"]):
        raise BaselineTrainingError(
            "The approved baseline protocol requires save_last: true."
        )


def _build_initial_run_manifest(
    *,
    validated: ValidatedBaselineExperimentConfig,
    output_root: Path,
    source_roots: Sequence[Path],
    smoke_test: bool,
    trainer_epochs: int,
    scheduler_horizon_epochs: int,
    device: torch.device,
    num_workers_override: int | None,
    architecture_summary: Mapping[str, Any],
    optimizer_summary: Mapping[str, Any],
    criterion_summary: Mapping[str, Any],
    scheduler_summary: Mapping[str, Any],
    loader_summary: Mapping[str, Any],
    validation_bundle: Mapping[str, Path],
    resume: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the authoritative run manifest."""

    identity = validated.validation_summary["experiment"]

    return {
        "entrypoint_protocol_version": (
            BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
        ),
        "status": "initialized",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "experiment": {
            "id": identity["id"],
            "name": identity["name"],
            "model_name": identity["expected_model"],
            "purpose": identity["purpose"],
            "smoke_test": smoke_test,
        },
        "configuration": {
            "source_path": str(validated.source_path),
            "source_file_sha256": validated.source_file_sha256,
            "canonical_sha256": validated.canonical_sha256,
            "saved_configuration_copy": str(
                validation_bundle["configuration_copy"]
            ),
            "saved_validation_report": str(
                validation_bundle["validation_report"]
            ),
        },
        "runtime": {
            "output_root": str(output_root),
            "device": str(device),
            "amp_requested": bool(
                validated.payload["training"]["automatic_mixed_precision"]
            ),
            "trainer_epochs": trainer_epochs,
            "scheduler_horizon_epochs": scheduler_horizon_epochs,
            "num_workers_override": num_workers_override,
            "source_roots": [str(path) for path in source_roots],
            "evaluation_defaults": {
                "target_threshold": TARGET_THRESHOLD,
                "jaccard_quality_threshold": JACCARD_QUALITY_THRESHOLD,
                "boundary_tolerance_pixels": BOUNDARY_TOLERANCE_PIXELS,
            },
        },
        "model": dict(architecture_summary),
        "optimizer": dict(optimizer_summary),
        "criterion": dict(criterion_summary),
        "scheduler": dict(scheduler_summary),
        "data_loader": dict(loader_summary),
        "resume": dict(resume) if resume is not None else None,
        "training_summary_path": None,
        "failure_path": None,
    }


def _load_resume_state_before_trainer(
    *,
    checkpoint_path: Path,
    model: nn.Module,
) -> dict[str, Any]:
    """Read trusted checkpoint metadata before constructing the trainer."""

    return load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
        strict_model=True,
        restore_rng=False,
        verify_checksum=True,
        require_optimizer_state=False,
        require_scheduler_state=False,
        require_scaler_state=False,
    )


def _restore_resume_state_after_trainer(
    *,
    checkpoint_path: Path,
    trainer: SegmentationTrainer,
) -> dict[str, Any]:
    """Restore model, optimizer, scheduler, scaler, and RNG state exactly."""

    scheduler = trainer.scheduler_bundle.scheduler

    restored = load_checkpoint(
        checkpoint_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=scheduler,
        scaler=trainer.scaler,
        map_location=trainer.device,
        strict_model=True,
        restore_rng=True,
        verify_checksum=True,
        require_optimizer_state=True,
        require_scheduler_state=(scheduler is not None),
        require_scaler_state=trainer.amp_enabled,
    )

    return restored


def run_baseline_training(
    *,
    config_path: str | Path,
    source_roots: Sequence[str | Path] | None = None,
    smoke_test: bool = False,
    device: str | torch.device | None = None,
    num_workers_override: int | None = None,
    output_root_override: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    overwrite_existing: bool = False,
    allow_cpu_full_training: bool = False,
) -> dict[str, Any]:
    """Validate, construct, run, and persist one baseline experiment."""

    validated = validate_baseline_experiment_configuration(config_path)
    payload = validated.payload

    _validate_runtime_training_contract(payload)

    resolved_device = resolve_training_device(device)

    if (
        not smoke_test
        and resolved_device.type != "cuda"
        and not allow_cpu_full_training
    ):
        raise BaselineTrainingError(
            "Full baseline training requires CUDA. Use --smoke-test for a "
            "bounded integration run or explicitly pass "
            "--allow-cpu-full-training."
        )

    resolved_source_roots = _normalize_source_roots(source_roots)
    output_root = _resolve_output_root(
        validated,
        smoke_test=smoke_test,
        output_root_override=output_root_override,
    )

    resolved_resume_checkpoint = _resolve_resume_checkpoint(
        explicit_checkpoint=resume_checkpoint,
        output_root=output_root,
        payload=payload,
        smoke_test=smoke_test,
    )

    _prepare_fresh_output_root(
        output_root=output_root,
        resume_checkpoint=resolved_resume_checkpoint,
        overwrite_existing=overwrite_existing,
    )

    resume_manifest_path: Path | None = None
    resume_manifest: dict[str, Any] | None = None
    resume_copy_report: dict[str, Any] | None = None

    if resolved_resume_checkpoint is not None:
        strict_match = bool(
            payload["training"]["resume"]["strict_configuration_match"]
        )
        resume_manifest_path, resume_manifest = _verify_resume_configuration(
            validated=validated,
            checkpoint_path=resolved_resume_checkpoint,
            strict_configuration_match=strict_match,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        resume_copy_report = _copy_resume_artifacts(
            checkpoint_path=resolved_resume_checkpoint,
            manifest_path=resume_manifest_path,
            destination_root=output_root,
        )

    directories = _create_output_directories(output_root, payload)

    validation_bundle = save_baseline_validation_bundle(
        validated,
        directories["configurations"],
    )

    reproducibility = payload["reproducibility"]
    seed_settings = seed_everything(
        seed=int(reproducibility["seed_python"]),
        deterministic_algorithms=bool(
            reproducibility["deterministic_algorithms"]
        ),
        cudnn_deterministic=bool(
            reproducibility["cudnn_deterministic"]
        ),
        cudnn_benchmark=bool(reproducibility["cudnn_benchmark"]),
    )

    model = build_baseline_model(configuration=payload)
    architecture_summary = get_baseline_architecture_summary(model)
    optimizer = build_baseline_optimizer(model, payload)
    criterion = build_baseline_criterion(payload)

    trainer_epochs, scheduler_horizon_epochs = _resolve_training_epochs(
        payload,
        smoke_test=smoke_test,
    )

    scheduler_bundle = build_scheduler(
        optimizer,
        payload["training"]["scheduler"],
        total_epochs=scheduler_horizon_epochs,
    )

    loader_bundle: DataLoaderBundle = build_loader_bundle(
        validated=validated,
        source_roots=resolved_source_roots,
        smoke_test=smoke_test,
        num_workers_override=num_workers_override,
    )

    preload: dict[str, Any] | None = None
    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    logger_resume = False

    if resolved_resume_checkpoint is not None:
        preload = _load_resume_state_before_trainer(
            checkpoint_path=resolved_resume_checkpoint,
            model=model,
        )

        checkpoint_metadata = preload.get("metadata", {})
        checkpoint_experiment = str(
            checkpoint_metadata.get("experiment_id", "")
        )
        expected_experiment = str(
            validated.validation_summary["experiment"]["id"]
        )

        if checkpoint_experiment != expected_experiment:
            raise BaselineTrainingError(
                "Checkpoint experiment ID does not match the current run: "
                f"{checkpoint_experiment!r} != {expected_experiment!r}."
            )

        start_epoch = int(preload["epoch"]) + 1
        global_step = int(preload["global_step"])
        best_metric = (
            None
            if preload["best_metric"] is None
            else float(preload["best_metric"])
        )
        best_epoch = (
            None
            if preload["best_epoch"] is None
            else int(preload["best_epoch"])
        )
        epochs_without_improvement = int(
            checkpoint_metadata.get("epochs_without_improvement", 0)
        )
        logger_resume = True

        if start_epoch >= trainer_epochs:
            raise BaselineTrainingError(
                "The resume checkpoint has already reached or exceeded the "
                f"configured trainer horizon ({trainer_epochs} epochs)."
            )

    monitor_configuration = payload["training"]["checkpoint"]
    monitor_name = str(monitor_configuration["monitor"])
    monitor_mode = _normalize_monitor_mode(monitor_configuration["mode"])

    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=loader_bundle.train_loader,
        validation_loader=loader_bundle.validation_loader,
        output_directory=output_root,
        experiment_id=str(
            validated.validation_summary["experiment"]["id"]
        ),
        total_epochs=trainer_epochs,
        scheduler_bundle=scheduler_bundle,
        criterion_mode="logits_target",
        device=resolved_device,
        amp_enabled=bool(
            payload["training"]["automatic_mixed_precision"]
        ),
        gradient_clip_norm=(
            None
            if payload["training"]["gradient_clip_norm"] is None
            else float(payload["training"]["gradient_clip_norm"])
        ),
        monitor_name=monitor_name,
        monitor_mode=monitor_mode,
        minimum_delta=0.0,
        early_stopping_patience=int(
            payload["training"]["early_stopping_patience"]
        ),
        prediction_threshold=float(
            payload["inference"]["probability_threshold"]
        ),
        target_threshold=TARGET_THRESHOLD,
        jaccard_quality_threshold=JACCARD_QUALITY_THRESHOLD,
        boundary_tolerance_pixels=BOUNDARY_TOLERANCE_PIXELS,
        dataset_name=str(payload["data"]["dataset_name"]),
        validation_split_name="official_isic2018_validation",
        allow_logit_resize=False,
        start_epoch=start_epoch,
        global_step=global_step,
        best_metric=best_metric,
        best_epoch=best_epoch,
        epochs_without_improvement=epochs_without_improvement,
        logger_resume=logger_resume,
        metadata={
            "entrypoint_protocol_version": (
                BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
            ),
            "configuration_canonical_sha256": validated.canonical_sha256,
            "configuration_source_sha256": validated.source_file_sha256,
            "model_name": architecture_summary["architecture"],
            "smoke_test": smoke_test,
            "output_root": str(output_root),
        },
    )

    restored: dict[str, Any] | None = None

    if resolved_resume_checkpoint is not None:
        restored = _restore_resume_state_after_trainer(
            checkpoint_path=resolved_resume_checkpoint,
            trainer=trainer,
        )

        if trainer.logger.last_epoch != int(restored["epoch"]):
            raise BaselineTrainingError(
                "Resumed logger epoch does not match the checkpoint epoch."
            )

        if trainer.logger.last_global_step != int(restored["global_step"]):
            raise BaselineTrainingError(
                "Resumed logger global step does not match the checkpoint."
            )

    criterion_configuration = criterion.configuration()
    optimizer_configuration = _optimizer_summary(optimizer)
    scheduler_configuration = scheduler_bundle.architecture_summary()
    loader_summary = loader_bundle.summary()

    resume_report = None

    if resolved_resume_checkpoint is not None:
        resume_report = {
            "checkpoint_path": str(resolved_resume_checkpoint),
            "manifest_path": (
                str(resume_manifest_path)
                if resume_manifest_path is not None
                else None
            ),
            "copy_report": resume_copy_report,
            "preload": preload,
            "restored": restored,
            "source_manifest_status": (
                resume_manifest.get("status")
                if resume_manifest is not None
                else None
            ),
        }

    run_manifest = _build_initial_run_manifest(
        validated=validated,
        output_root=output_root,
        source_roots=resolved_source_roots,
        smoke_test=smoke_test,
        trainer_epochs=trainer_epochs,
        scheduler_horizon_epochs=scheduler_horizon_epochs,
        device=trainer.device,
        num_workers_override=num_workers_override,
        architecture_summary=architecture_summary,
        optimizer_summary=optimizer_configuration,
        criterion_summary=criterion_configuration,
        scheduler_summary=scheduler_configuration,
        loader_summary=loader_summary,
        validation_bundle=validation_bundle,
        resume=resume_report,
    )

    run_manifest["reproducibility_settings"] = seed_settings.to_dict()

    manifest_path = _atomic_write_json(
        output_root / RUN_MANIFEST_FILENAME,
        run_manifest,
    )

    _atomic_write_json(
        directories["environment"] / "RUN_ENVIRONMENT.json",
        _runtime_environment(trainer.device),
    )
    _atomic_write_json(
        directories["environment"] / "MODEL_ARCHITECTURE.json",
        architecture_summary,
    )
    _atomic_write_json(
        directories["environment"] / "DATALOADER_SUMMARY.json",
        loader_summary,
    )
    _atomic_write_json(
        directories["environment"] / "OPTIMIZER_SCHEDULER_LOSS.json",
        {
            "optimizer": optimizer_configuration,
            "scheduler": scheduler_configuration,
            "criterion": criterion_configuration,
        },
    )

    try:
        training_summary = trainer.fit()
    except BaseException as error:
        failure = {
            "entrypoint_protocol_version": (
                BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
            ),
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "experiment_id": (
                validated.validation_summary["experiment"]["id"]
            ),
            "model_name": (
                validated.validation_summary["experiment"]["expected_model"]
            ),
            "smoke_test": smoke_test,
            "output_root": str(output_root),
            "run_manifest": str(manifest_path),
        }
        failure_path = _atomic_write_json(
            output_root / RUN_FAILURE_FILENAME,
            failure,
        )
        run_manifest["status"] = "failed"
        run_manifest["updated_at_utc"] = _utc_now()
        run_manifest["failure_path"] = str(failure_path)
        _atomic_write_json(manifest_path, run_manifest)
        raise

    run_manifest["status"] = "completed"
    run_manifest["updated_at_utc"] = _utc_now()
    run_manifest["training_summary_path"] = training_summary["summary_path"]
    run_manifest["training_result"] = {
        "completed_epochs": training_summary["completed_epochs"],
        "final_epoch": training_summary["final_epoch"],
        "final_global_step": training_summary["final_global_step"],
        "best_metric": training_summary["best_metric"],
        "best_epoch": training_summary["best_epoch"],
        "stopped_early": training_summary["stopped_early"],
    }
    _atomic_write_json(manifest_path, run_manifest)

    result = {
        "status": "passed",
        "entrypoint_protocol_version": (
            BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
        ),
        "experiment_id": (
            validated.validation_summary["experiment"]["id"]
        ),
        "model_name": (
            validated.validation_summary["experiment"]["expected_model"]
        ),
        "smoke_test": smoke_test,
        "device": str(trainer.device),
        "amp_enabled": trainer.amp_enabled,
        "output_root": str(output_root),
        "run_manifest": str(manifest_path),
        "training_summary": training_summary,
    }

    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate and train one E01-E05 BCS-HCTNet baseline experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to one validated E01-E05 YAML configuration.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=None,
        help=(
            "Image source root. Repeat for multiple roots. When omitted, "
            f"the Kaggle ISIC 2018 root {DEFAULT_ISIC2018_SOURCE_ROOT} is used."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device such as cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the deterministic bounded smoke subset and epoch count.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional writable output-root override.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Trusted latest.pt checkpoint to resume, including .sha256 sidecar.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly delete a non-resume output root before starting.",
    )
    parser.add_argument(
        "--allow-cpu-full-training",
        action="store_true",
        help="Explicitly permit a non-smoke full run without CUDA.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.num_workers is not None and arguments.num_workers < 0:
        parser.error("--num-workers must be non-negative.")

    result = run_baseline_training(
        config_path=arguments.config,
        source_roots=arguments.source_root,
        smoke_test=bool(arguments.smoke_test),
        device=arguments.device,
        num_workers_override=arguments.num_workers,
        output_root_override=arguments.output_root,
        resume_checkpoint=arguments.resume_checkpoint,
        overwrite_existing=bool(arguments.overwrite_existing),
        allow_cpu_full_training=bool(arguments.allow_cpu_full_training),
    )

    printable = {
        "status": result["status"],
        "entrypoint_protocol_version": result[
            "entrypoint_protocol_version"
        ],
        "experiment_id": result["experiment_id"],
        "model_name": result["model_name"],
        "smoke_test": result["smoke_test"],
        "device": result["device"],
        "amp_enabled": result["amp_enabled"],
        "output_root": result["output_root"],
        "run_manifest": result["run_manifest"],
        "training_summary_path": result["training_summary"]["summary_path"],
        "completed_epochs": result["training_summary"]["completed_epochs"],
        "best_metric": result["training_summary"]["best_metric"],
        "best_epoch": result["training_summary"]["best_epoch"],
    }

    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


def run_baseline_training_entrypoint_self_test() -> dict[str, Any]:
    """Run artifact-independent CPU checks for entry-point helpers."""

    import tempfile

    torch.manual_seed(42)

    payload = {
        "training": {
            "maximum_epochs": 200,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "name": "cosine_annealing",
                "minimum_learning_rate": 0.0,
                "interval": "epoch",
            },
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
        "smoke_test": {"epochs": 1},
    }

    model = nn.Conv2d(3, 1, kernel_size=1)
    optimizer = build_baseline_optimizer(model, payload)
    criterion = build_baseline_criterion(payload)
    trainer_epochs, scheduler_horizon = _resolve_training_epochs(
        payload,
        smoke_test=True,
    )
    scheduler = build_scheduler(
        optimizer,
        payload["training"]["scheduler"],
        total_epochs=scheduler_horizon,
    )

    image = torch.randn(2, 3, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits: Tensor = model(image)
    components = criterion(logits, target)

    if not isinstance(components, Mapping):
        raise RuntimeError("Self-test criterion did not return components.")

    total_loss = components["total_loss"]
    total_loss.backward()

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = _atomic_write_json(
            Path(temporary_directory) / "test.json",
            {"status": "passed", "value": 1},
        )
        saved = _load_json_mapping(output_path)

    initial_learning_rate = float(optimizer.param_groups[0]["lr"])

    checks = {
        "optimizer_is_adamw": isinstance(optimizer, AdamW),
        "criterion_is_bce_dice": isinstance(criterion, BCEDiceLoss),
        "criterion_components_present": set(components) == {
            "total_loss",
            "bce_loss",
            "dice_loss",
        },
        "loss_is_finite": bool(torch.isfinite(total_loss).item()),
        "gradient_is_finite": bool(
            model.weight.grad is not None
            and torch.isfinite(model.weight.grad).all().item()
        ),
        "smoke_trainer_epochs_is_one": trainer_epochs == 1,
        "scheduler_horizon_is_full": scheduler_horizon == 200,
        "smoke_initial_learning_rate_nonzero": initial_learning_rate == 1e-4,
        "scheduler_is_cosine": scheduler.name == "cosine",
        "maximum_mode_normalized": _normalize_monitor_mode("maximum") == "max",
        "minimum_mode_normalized": _normalize_monitor_mode("minimum") == "min",
        "atomic_json_roundtrip": saved == {"status": "passed", "value": 1},
    }

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol_version": BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION,
        "checks": checks,
        "trainer_epochs": trainer_epochs,
        "scheduler_horizon_epochs": scheduler_horizon,
        "initial_learning_rate": initial_learning_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
