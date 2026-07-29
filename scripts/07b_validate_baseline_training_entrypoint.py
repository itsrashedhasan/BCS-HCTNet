"""Artifact-aware CPU integration validation for the baseline trainer.

This script exercises the complete shared baseline training path with the
approved E01 U-Net smoke-test protocol and real mounted ISIC 2018 artifacts.
It intentionally uses CPU and zero DataLoader workers so it can run before a
Kaggle GPU session is enabled.

Validated path
--------------
configuration -> artifact readiness -> dataset joins -> synchronized
transforms -> DataLoader -> U-Net -> BCE-Dice loss -> backward pass ->
validation metrics -> scheduler -> logger -> best/latest checkpoints ->
checksums -> run manifest -> training summary

Only E01 is trained here because all five baseline architectures already have
model-level forward/backward tests, while this stage validates the single
shared training entry point against real data. E01-E05 GPU smoke tests remain a
separate later stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.train.baseline_experiment_config import (  # noqa: E402
    validate_baseline_experiment_configuration,
)
from src.train.checkpoint import (  # noqa: E402
    verify_checkpoint_checksum,
)
from src.train.train_baseline import (  # noqa: E402
    BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION,
    RUN_FAILURE_FILENAME,
    RUN_MANIFEST_FILENAME,
    run_baseline_training,
)


STEP07B_PROTOCOL_VERSION = (
    "BCS-HCTNet-step07b-baseline-entrypoint-validation-v1"
)

DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "E01_unet.yaml"
)

DEFAULT_SOURCE_ROOT = Path(
    "/kaggle/input/datasets/asrafulislam7/isic-2018/ISIC_2018"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/kaggle/working/outputs/"
    "step07b_baseline_training_entrypoint_validation/"
    "E01_unet_cpu_smoke"
)

FINAL_REPORT_FILENAME = (
    "STEP07B_BASELINE_TRAINING_ENTRYPOINT_VALIDATION.json"
)

EXPECTED_EXPERIMENT_ID = "E01"
EXPECTED_MODEL_NAME = "unet"
EXPECTED_TRAIN_SAMPLES = 8
EXPECTED_VALIDATION_SAMPLES = 4
EXPECTED_BATCH_SIZE = 2
EXPECTED_EPOCHS = 1
EXPECTED_TRAIN_BATCHES = 4
EXPECTED_VALIDATION_BATCHES = 2
EXPECTED_FINAL_GLOBAL_STEP = 4


class Step07BValidationError(RuntimeError):
    """Raised when the real-data baseline integration check fails."""


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


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    """Load and require one JSON object."""

    resolved_path = Path(path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved_path}")

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a JSON object in {resolved_path}.")

    return dict(payload)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read all rows from one CSV file."""

    resolved_path = Path(path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {resolved_path}")

    with resolved_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _require_finite_number(value: object, context: str) -> float:
    """Require a finite numeric value."""

    if isinstance(value, bool):
        raise Step07BValidationError(f"{context} must be numeric.")

    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise Step07BValidationError(
            f"{context} must be numeric; received {value!r}."
        ) from error

    if not math.isfinite(resolved):
        raise Step07BValidationError(
            f"{context} must be finite; received {resolved}."
        )

    return resolved


def _record_check(
    checks: dict[str, bool],
    name: str,
    condition: object,
) -> None:
    """Record one Boolean validation check."""

    checks[name] = bool(condition)


def _require_all_checks(checks: Mapping[str, bool]) -> None:
    """Raise with the complete failed-check list."""

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise Step07BValidationError(
            "Step07B validation failed checks: " + ", ".join(failed)
        )


def _validate_approved_e01_protocol(
    config_path: Path,
) -> dict[str, Any]:
    """Validate and lock the exact E01 smoke-test contract."""

    validated = validate_baseline_experiment_configuration(config_path)
    payload = validated.payload
    identity = validated.validation_summary["experiment"]
    smoke = payload["smoke_test"]

    checks: dict[str, bool] = {}

    _record_check(
        checks,
        "experiment_is_E01",
        identity["id"] == EXPECTED_EXPERIMENT_ID,
    )
    _record_check(
        checks,
        "model_is_unet",
        identity["expected_model"] == EXPECTED_MODEL_NAME,
    )
    _record_check(
        checks,
        "training_is_unblocked",
        validated.training_readiness["training_allowed"] is True
        and validated.training_readiness["status"] == "training_unblocked",
    )
    _record_check(
        checks,
        "smoke_test_enabled",
        smoke["enabled"] is True,
    )
    _record_check(
        checks,
        "smoke_train_samples_exact",
        int(smoke["train_samples"]) == EXPECTED_TRAIN_SAMPLES,
    )
    _record_check(
        checks,
        "smoke_validation_samples_exact",
        int(smoke["validation_samples"]) == EXPECTED_VALIDATION_SAMPLES,
    )
    _record_check(
        checks,
        "smoke_batch_size_exact",
        int(smoke["batch_size"]) == EXPECTED_BATCH_SIZE,
    )
    _record_check(
        checks,
        "smoke_epoch_count_exact",
        int(smoke["epochs"]) == EXPECTED_EPOCHS,
    )
    _record_check(
        checks,
        "smoke_requires_backward",
        smoke["require_backward_pass"] is True,
    )
    _record_check(
        checks,
        "smoke_requires_finite_loss",
        smoke["require_loss_finite"] is True,
    )
    _record_check(
        checks,
        "smoke_requires_nonzero_gradients",
        smoke["require_nonzero_gradients"] is True,
    )

    _require_all_checks(checks)

    return {
        "checks": checks,
        "canonical_configuration_sha256": validated.canonical_sha256,
        "source_file_sha256": validated.source_file_sha256,
        "validation_protocol_version": validated.to_dict()[
            "validation_protocol_version"
        ],
    }


def _required_artifact_paths(output_root: Path) -> dict[str, Path]:
    """Return the exact expected artifact paths for one smoke run."""

    return {
        "run_manifest": output_root / RUN_MANIFEST_FILENAME,
        "training_summary": output_root / "TRAINING_SUMMARY.json",
        "latest_checkpoint": output_root / "checkpoints" / "latest.pt",
        "latest_checkpoint_checksum": (
            output_root / "checkpoints" / "latest.pt.sha256"
        ),
        "best_checkpoint": output_root / "checkpoints" / "best.pt",
        "best_checkpoint_checksum": (
            output_root / "checkpoints" / "best.pt.sha256"
        ),
        "epoch_history": output_root / "logs" / "epoch_history.csv",
        "latest_epoch_log": output_root / "logs" / "latest_epoch.json",
        "logger_metadata": (
            output_root / "logs" / "experiment_metadata.json"
        ),
        "logger_state": output_root / "logs" / "logger_state.json",
        "validation_csv": (
            output_root
            / "validation"
            / "epoch_0000_per_image_metrics.csv"
        ),
        "validation_summary": (
            output_root
            / "validation"
            / "epoch_0000_validation_summary.json"
        ),
        "runtime_environment": (
            output_root / "environment" / "RUN_ENVIRONMENT.json"
        ),
        "model_architecture": (
            output_root / "environment" / "MODEL_ARCHITECTURE.json"
        ),
        "dataloader_summary": (
            output_root / "environment" / "DATALOADER_SUMMARY.json"
        ),
        "optimizer_scheduler_loss": (
            output_root
            / "environment"
            / "OPTIMIZER_SCHEDULER_LOSS.json"
        ),
        "configuration_copy": (
            output_root / "configurations" / "E01_unet.yaml"
        ),
        "configuration_validation": (
            output_root
            / "configurations"
            / "BASELINE_CONFIG_VALIDATION.json"
        ),
    }


def _validate_completed_run(
    *,
    output_root: Path,
    result: Mapping[str, Any],
    approved_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every critical output of the completed CPU smoke run."""

    paths = _required_artifact_paths(output_root)
    checks: dict[str, bool] = {}

    for name, path in paths.items():
        _record_check(checks, f"artifact_exists__{name}", path.is_file())

    _record_check(
        checks,
        "failure_report_absent",
        not (output_root / RUN_FAILURE_FILENAME).exists(),
    )

    _require_all_checks(checks)

    manifest = _load_json_mapping(paths["run_manifest"])
    summary = _load_json_mapping(paths["training_summary"])
    validation = _load_json_mapping(paths["validation_summary"])
    architecture = _load_json_mapping(paths["model_architecture"])
    dataloader = _load_json_mapping(paths["dataloader_summary"])
    logger_state = _load_json_mapping(paths["logger_state"])
    environment = _load_json_mapping(paths["runtime_environment"])
    optimization = _load_json_mapping(paths["optimizer_scheduler_loss"])

    history_rows = _read_csv_rows(paths["epoch_history"])
    validation_rows = _read_csv_rows(paths["validation_csv"])

    latest_checksum = verify_checkpoint_checksum(
        paths["latest_checkpoint"],
        require_checksum=True,
    )
    best_checksum = verify_checkpoint_checksum(
        paths["best_checkpoint"],
        require_checksum=True,
    )

    _record_check(checks, "result_status_passed", result["status"] == "passed")
    _record_check(
        checks,
        "entrypoint_protocol_exact",
        result["entrypoint_protocol_version"]
        == BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION,
    )
    _record_check(
        checks,
        "result_experiment_exact",
        result["experiment_id"] == EXPECTED_EXPERIMENT_ID,
    )
    _record_check(
        checks,
        "result_model_exact",
        result["model_name"] == EXPECTED_MODEL_NAME,
    )
    _record_check(checks, "result_is_smoke_test", result["smoke_test"] is True)
    _record_check(checks, "result_device_is_cpu", result["device"] == "cpu")
    _record_check(checks, "amp_disabled_on_cpu", result["amp_enabled"] is False)

    _record_check(checks, "manifest_completed", manifest["status"] == "completed")
    _record_check(
        checks,
        "manifest_experiment_exact",
        manifest["experiment"]["id"] == EXPECTED_EXPERIMENT_ID,
    )
    _record_check(
        checks,
        "manifest_model_exact",
        manifest["experiment"]["model_name"] == EXPECTED_MODEL_NAME,
    )
    _record_check(
        checks,
        "manifest_configuration_hash_exact",
        manifest["configuration"]["canonical_sha256"]
        == approved_configuration["canonical_configuration_sha256"],
    )
    _record_check(
        checks,
        "manifest_trainer_epochs_exact",
        int(manifest["runtime"]["trainer_epochs"]) == EXPECTED_EPOCHS,
    )
    _record_check(
        checks,
        "manifest_scheduler_horizon_preserved",
        int(manifest["runtime"]["scheduler_horizon_epochs"]) == 200,
    )
    _record_check(
        checks,
        "manifest_num_workers_zero",
        int(manifest["runtime"]["num_workers_override"]) == 0,
    )

    _record_check(
        checks,
        "training_summary_completed_one_epoch",
        int(summary["completed_epochs"]) == EXPECTED_EPOCHS,
    )
    _record_check(
        checks,
        "training_summary_configured_one_epoch",
        int(summary["configured_epochs"]) == EXPECTED_EPOCHS,
    )
    _record_check(checks, "training_started_epoch_zero", summary["start_epoch"] == 0)
    _record_check(checks, "training_finished_epoch_zero", summary["final_epoch"] == 0)
    _record_check(
        checks,
        "final_global_step_exact",
        int(summary["final_global_step"]) == EXPECTED_FINAL_GLOBAL_STEP,
    )
    _record_check(checks, "best_epoch_is_zero", summary["best_epoch"] == 0)
    _record_check(checks, "not_stopped_early", summary["stopped_early"] is False)

    best_metric = _require_finite_number(summary["best_metric"], "best_metric")
    _record_check(checks, "best_metric_in_unit_interval", 0.0 <= best_metric <= 1.0)

    epochs = summary.get("epochs")
    _record_check(
        checks,
        "one_epoch_record_present",
        isinstance(epochs, list) and len(epochs) == 1,
    )
    _require_all_checks(checks)

    epoch = epochs[0]
    training_epoch = epoch["training"]
    validation_epoch = epoch["validation"]

    train_loss = _require_finite_number(training_epoch["mean_loss"], "train_loss")
    validation_loss = _require_finite_number(
        validation_epoch["mean_loss"],
        "validation_loss",
    )
    maximum_gradient_norm = _require_finite_number(
        training_epoch["maximum_gradient_norm"],
        "maximum_gradient_norm",
    )

    training_components = training_epoch["mean_loss_components"]
    validation_components = validation_epoch["mean_loss_components"]

    if not isinstance(training_components, Mapping):
        raise Step07BValidationError(
            "Training loss components must be a mapping."
        )

    if not isinstance(validation_components, Mapping):
        raise Step07BValidationError(
            "Validation loss components must be a mapping."
        )

    for component_name, component_value in training_components.items():
        _require_finite_number(
            component_value,
            f"training_component.{component_name}",
        )

    for component_name, component_value in validation_components.items():
        _require_finite_number(
            component_value,
            f"validation_component.{component_name}",
        )

    _record_check(checks, "train_loss_nonnegative", train_loss >= 0.0)
    _record_check(checks, "validation_loss_nonnegative", validation_loss >= 0.0)
    _record_check(
        checks,
        "nonzero_finite_gradient_observed",
        maximum_gradient_norm > 0.0,
    )
    _record_check(
        checks,
        "training_loss_components_present",
        {"bce_loss", "dice_loss"}.issubset(training_components),
    )
    _record_check(
        checks,
        "validation_loss_components_present",
        {"bce_loss", "dice_loss"}.issubset(validation_components),
    )
    _record_check(
        checks,
        "train_batch_count_exact",
        int(training_epoch["number_of_batches"]) == EXPECTED_TRAIN_BATCHES,
    )
    _record_check(
        checks,
        "train_image_count_exact",
        int(training_epoch["number_of_images"]) == EXPECTED_TRAIN_SAMPLES,
    )
    _record_check(
        checks,
        "validation_batch_count_exact",
        int(validation_epoch["number_of_batches"])
        == EXPECTED_VALIDATION_BATCHES,
    )
    _record_check(
        checks,
        "validation_image_count_exact",
        int(validation_epoch["number_of_images"])
        == EXPECTED_VALIDATION_SAMPLES,
    )
    _record_check(
        checks,
        "training_global_step_range_exact",
        int(training_epoch["global_step_start"]) == 0
        and int(training_epoch["global_step_end"])
        == EXPECTED_FINAL_GLOBAL_STEP,
    )
    _record_check(checks, "epoch_marked_best", epoch["is_best"] is True)

    _record_check(
        checks,
        "history_has_one_row",
        len(history_rows) == 1,
    )
    _record_check(
        checks,
        "validation_csv_has_four_rows",
        len(validation_rows) == EXPECTED_VALIDATION_SAMPLES,
    )
    _record_check(
        checks,
        "logger_state_has_one_record",
        int(logger_state["record_count"]) == 1,
    )
    _record_check(
        checks,
        "logger_state_epoch_zero",
        int(logger_state["last_epoch"]) == 0,
    )
    _record_check(
        checks,
        "logger_state_global_step_exact",
        int(logger_state["last_global_step"])
        == EXPECTED_FINAL_GLOBAL_STEP,
    )

    _record_check(
        checks,
        "dataloader_is_smoke_mode",
        dataloader["smoke_test"] is True,
    )
    _record_check(
        checks,
        "dataloader_active_train_rows_exact",
        int(dataloader["active_train_rows"]) == EXPECTED_TRAIN_SAMPLES,
    )
    _record_check(
        checks,
        "dataloader_active_validation_rows_exact",
        int(dataloader["active_validation_rows"])
        == EXPECTED_VALIDATION_SAMPLES,
    )
    _record_check(
        checks,
        "dataloader_train_batches_exact",
        int(dataloader["train_batches"]) == EXPECTED_TRAIN_BATCHES,
    )
    _record_check(
        checks,
        "dataloader_validation_batches_exact",
        int(dataloader["validation_batches"]) == EXPECTED_VALIDATION_BATCHES,
    )

    _record_check(
        checks,
        "architecture_is_unet",
        architecture["architecture"] == "standard_unet",
    )
    _record_check(
        checks,
        "architecture_mask_only",
        architecture["output_keys"] == ["mask_logits"]
        and architecture["uses_boundary_conditioning"] is False
        and architecture["uses_auxiliary_targets"] is False,
    )
    _record_check(
        checks,
        "architecture_has_parameters",
        int(architecture["parameter_count"]) > 0,
    )

    _record_check(
        checks,
        "criterion_is_bce_dice",
        optimization["criterion"]["name"] == "bce_dice_loss",
    )
    _record_check(
        checks,
        "optimizer_is_adamw",
        optimization["optimizer"]["class"] == "AdamW",
    )
    _record_check(
        checks,
        "scheduler_is_cosine",
        optimization["scheduler"]["name"] == "cosine",
    )

    _record_check(
        checks,
        "runtime_recorded_cpu",
        environment["requested_runtime_device"] == "cpu",
    )
    _record_check(
        checks,
        "latest_checkpoint_checksum_valid",
        latest_checksum["checksum_matches"] is True,
    )
    _record_check(
        checks,
        "best_checkpoint_checksum_valid",
        best_checksum["checksum_matches"] is True,
    )

    _require_all_checks(checks)

    return {
        "checks": checks,
        "run_manifest": manifest,
        "training_summary": summary,
        "checkpoint_verification": {
            "latest": latest_checksum,
            "best": best_checksum,
        },
        "artifact_paths": {
            name: str(path)
            for name, path in paths.items()
        },
    }


def run_step07b_validation(
    *,
    config_path: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    overwrite_existing: bool,
) -> dict[str, Any]:
    """Run the complete approved real-data CPU integration validation."""

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_source = Path(source_root).expanduser().resolve()
    resolved_output = Path(output_root).expanduser().resolve()

    if not resolved_config.is_file():
        raise FileNotFoundError(f"Configuration not found: {resolved_config}")

    if not resolved_source.is_dir():
        raise FileNotFoundError(f"ISIC 2018 source root not found: {resolved_source}")

    if resolved_output.exists() and overwrite_existing:
        shutil.rmtree(resolved_output)

    approved_configuration = _validate_approved_e01_protocol(resolved_config)

    started_at = _utc_now()

    result = run_baseline_training(
        config_path=resolved_config,
        source_roots=[resolved_source],
        smoke_test=True,
        device="cpu",
        num_workers_override=0,
        output_root_override=resolved_output,
        resume_checkpoint=None,
        overwrite_existing=overwrite_existing,
        allow_cpu_full_training=False,
    )

    completed_validation = _validate_completed_run(
        output_root=resolved_output,
        result=result,
        approved_configuration=approved_configuration,
    )

    report = {
        "status": "passed",
        "all_checks_passed": True,
        "protocol_version": STEP07B_PROTOCOL_VERSION,
        "training_entrypoint_protocol_version": (
            BASELINE_TRAINING_ENTRYPOINT_PROTOCOL_VERSION
        ),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "device": "cpu",
        "gpu_required": False,
        "purpose": (
            "real_artifact_end_to_end_validation_of_shared_baseline_"
            "training_entrypoint_before_gpu_smoke_tests"
        ),
        "configuration": approved_configuration,
        "output_root": str(resolved_output),
        "validation": completed_validation,
    }

    report_path = _atomic_write_json(
        resolved_output / FINAL_REPORT_FILENAME,
        report,
    )

    printable = {
        "status": report["status"],
        "all_checks_passed": report["all_checks_passed"],
        "protocol_version": report["protocol_version"],
        "experiment_id": report["experiment_id"],
        "model_name": report["model_name"],
        "device": report["device"],
        "completed_epochs": completed_validation["training_summary"][
            "completed_epochs"
        ],
        "final_global_step": completed_validation["training_summary"][
            "final_global_step"
        ],
        "best_metric": completed_validation["training_summary"]["best_metric"],
        "report_path": str(report_path),
    }

    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the real-data CPU integration validation for the E01 "
            "baseline training entry point."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Approved E01 U-Net configuration path.",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Mounted ISIC 2018 source-image root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Writable Step07B validation output root.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Delete and recreate an existing Step07B output directory.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    run_step07b_validation(
        config_path=arguments.config,
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        overwrite_existing=bool(arguments.overwrite_existing),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
