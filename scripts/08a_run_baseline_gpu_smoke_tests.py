"""Run sequential one-epoch GPU smoke tests for E01-E05 baselines.

This script is the first GPU-required stage of the BCS-HCTNet baseline
pipeline. It exercises the already validated shared training entry point with
all five baseline architectures using the bounded smoke protocol declared in
each experiment YAML file.

The smoke runs are integration tests, not scientific training results. Their
metrics must not be used in the thesis comparison table.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch

if TYPE_CHECKING:
    from src.train.baseline_experiment_config import (
        ValidatedBaselineExperimentConfig,
    )


STEP08A_PROTOCOL_VERSION = "BCS-HCTNet-step08a-baseline-gpu-smoke-v1"

DEFAULT_CONFIG_DIRECTORY = Path("configs/experiments")
DEFAULT_SOURCE_ROOT = Path(
    "/kaggle/input/datasets/asrafulislam7/isic-2018/ISIC_2018"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/kaggle/working/outputs/step08a_baseline_gpu_smoke_tests"
)

AGGREGATE_REPORT_FILENAME = "STEP08A_BASELINE_GPU_SMOKE_REPORT.json"
SUMMARY_CSV_FILENAME = "STEP08A_BASELINE_GPU_SMOKE_SUMMARY.csv"
FAILURE_REPORT_FILENAME = "STEP08A_BASELINE_GPU_SMOKE_FAILURE.json"


@dataclass(frozen=True)
class BaselineSpecification:
    """One approved baseline experiment assignment."""

    experiment_id: str
    model_name: str
    config_filename: str

    @property
    def output_directory_name(self) -> str:
        """Return the deterministic output-directory name."""

        return f"{self.experiment_id}_{self.model_name}_gpu_smoke"


APPROVED_BASELINES: tuple[BaselineSpecification, ...] = (
    BaselineSpecification("E01", "unet", "E01_unet.yaml"),
    BaselineSpecification("E02", "unetpp", "E02_unetpp.yaml"),
    BaselineSpecification(
        "E03",
        "deeplabv3plus",
        "E03_deeplabv3plus.yaml",
    ),
    BaselineSpecification("E04", "transunet", "E04_transunet.yaml"),
    BaselineSpecification("E05", "swin_unet", "E05_swin_unet.yaml"),
)


class BaselineGPUSmokeError(RuntimeError):
    """Raised when the Step08A smoke protocol is violated."""


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return path


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write the compact aggregate CSV without third-party dependencies."""

    import csv

    fieldnames = [
        "experiment_id",
        "model_name",
        "status",
        "device",
        "amp_enabled",
        "completed_epochs",
        "final_global_step",
        "best_metric",
        "best_epoch",
        "elapsed_seconds",
        "peak_allocated_mebibytes",
        "peak_reserved_mebibytes",
        "output_root",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    os.replace(temporary_path, path)
    return path


def _resolve_cuda_device(device: str) -> torch.device:
    """Validate and resolve the requested CUDA device."""

    resolved = torch.device(device)

    if resolved.type != "cuda":
        raise BaselineGPUSmokeError(
            "Step08A requires a CUDA device. Set the Kaggle accelerator to GPU."
        )

    if not torch.cuda.is_available():
        raise BaselineGPUSmokeError(
            "CUDA is not available. In Kaggle, open Notebook options and set "
            "Accelerator to GPU before running Step08A."
        )

    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
        resolved = torch.device(f"cuda:{index}")

    if index < 0 or index >= torch.cuda.device_count():
        raise BaselineGPUSmokeError(
            f"CUDA device index {index} is invalid; available count is "
            f"{torch.cuda.device_count()}."
        )

    torch.cuda.set_device(resolved)
    return resolved


def _device_summary(device: torch.device) -> dict[str, Any]:
    """Collect the CUDA environment used by the smoke tests."""

    properties = torch.cuda.get_device_properties(device)

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_count": torch.cuda.device_count(),
        "cuda_runtime_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "total_memory_bytes": int(properties.total_memory),
        "total_memory_gibibytes": float(
            properties.total_memory / (1024**3)
        ),
        "compute_capability": [properties.major, properties.minor],
    }


def _select_specifications(
    requested_experiments: Sequence[str] | None,
) -> tuple[BaselineSpecification, ...]:
    """Resolve an optional approved experiment subset in canonical order."""

    if not requested_experiments:
        return APPROVED_BASELINES

    normalized = {value.strip().upper() for value in requested_experiments}
    approved_ids = {spec.experiment_id for spec in APPROVED_BASELINES}
    unknown = sorted(normalized - approved_ids)

    if unknown:
        raise BaselineGPUSmokeError(
            "Unknown experiment IDs: " + ", ".join(unknown)
        )

    selected = tuple(
        spec for spec in APPROVED_BASELINES if spec.experiment_id in normalized
    )

    if not selected:
        raise BaselineGPUSmokeError("No baseline experiments were selected.")

    return selected


def _validate_configuration_assignment(
    specification: BaselineSpecification,
    config_path: Path,
) -> "ValidatedBaselineExperimentConfig":
    """Validate one YAML and its fixed E01-E05 model assignment."""

    from src.train.baseline_experiment_config import (
        validate_baseline_experiment_configuration,
    )

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing baseline configuration: {config_path}")

    validated = validate_baseline_experiment_configuration(config_path)
    experiment = validated.validation_summary["experiment"]

    observed_id = str(experiment["id"])
    observed_model = str(experiment["expected_model"])

    if observed_id != specification.experiment_id:
        raise BaselineGPUSmokeError(
            f"Configuration ID mismatch for {config_path}: "
            f"{observed_id!r} != {specification.experiment_id!r}."
        )

    if observed_model != specification.model_name:
        raise BaselineGPUSmokeError(
            f"Model assignment mismatch for {observed_id}: "
            f"{observed_model!r} != {specification.model_name!r}."
        )

    smoke = validated.payload["smoke_test"]
    expected_steps = math.ceil(
        int(smoke["train_samples"]) / int(smoke["batch_size"])
    ) * int(smoke["epochs"])

    if int(smoke["epochs"]) != 1:
        raise BaselineGPUSmokeError(
            f"{observed_id} smoke_test.epochs must be 1 for Step08A."
        )

    if expected_steps <= 0:
        raise BaselineGPUSmokeError(
            f"{observed_id} has an invalid expected smoke-step count."
        )

    return validated


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaselineGPUSmokeError(f"Expected JSON object in {path}.")
    return payload


def _validate_completed_smoke_run(
    *,
    specification: BaselineSpecification,
    validated: "ValidatedBaselineExperimentConfig",
    output_root: Path,
    result: dict[str, Any],
    device: torch.device,
    elapsed_seconds: float,
    peak_allocated_mebibytes: float,
    peak_reserved_mebibytes: float,
) -> dict[str, Any]:
    """Validate the critical outputs from one completed GPU smoke run."""

    from src.train.checkpoint import verify_checkpoint_checksum

    manifest_path = output_root / "RUN_MANIFEST.json"
    summary_path = output_root / "TRAINING_SUMMARY.json"
    latest_checkpoint = output_root / "checkpoints" / "latest.pt"
    best_checkpoint = output_root / "checkpoints" / "best.pt"
    failure_path = output_root / "RUN_FAILURE.json"

    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)

    latest_verification = verify_checkpoint_checksum(latest_checkpoint)
    best_verification = verify_checkpoint_checksum(best_checkpoint)

    smoke = validated.payload["smoke_test"]
    expected_epochs = int(smoke["epochs"])
    expected_steps = math.ceil(
        int(smoke["train_samples"]) / int(smoke["batch_size"])
    ) * expected_epochs

    best_metric = summary.get("best_metric")

    checks = {
        "entrypoint_status_passed": result.get("status") == "passed",
        "experiment_id_correct": (
            result.get("experiment_id") == specification.experiment_id
        ),
        "model_name_correct": (
            result.get("model_name") == specification.model_name
        ),
        "smoke_flag_true": result.get("smoke_test") is True,
        "cuda_device_used": str(result.get("device", "")).startswith("cuda"),
        "amp_enabled": result.get("amp_enabled") is True,
        "run_manifest_completed": manifest.get("status") == "completed",
        "failure_report_absent": not failure_path.exists(),
        "completed_epochs_correct": (
            int(summary.get("completed_epochs", -1)) == expected_epochs
        ),
        "final_epoch_correct": int(summary.get("final_epoch", -1)) == 0,
        "global_step_correct": (
            int(summary.get("final_global_step", -1)) == expected_steps
        ),
        "best_epoch_correct": int(summary.get("best_epoch", -1)) == 0,
        "best_metric_finite": (
            isinstance(best_metric, (int, float))
            and math.isfinite(float(best_metric))
        ),
        "latest_checkpoint_checksum_valid": (
            latest_verification.get("checksum_matches") is True
        ),
        "best_checkpoint_checksum_valid": (
            best_verification.get("checksum_matches") is True
        ),
        "elapsed_time_positive": (
            math.isfinite(elapsed_seconds) and elapsed_seconds > 0.0
        ),
        "peak_allocated_memory_positive": (
            math.isfinite(peak_allocated_mebibytes)
            and peak_allocated_mebibytes > 0.0
        ),
        "peak_reserved_memory_positive": (
            math.isfinite(peak_reserved_mebibytes)
            and peak_reserved_mebibytes > 0.0
        ),
    }

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise BaselineGPUSmokeError(
            f"{specification.experiment_id} smoke validation failed: "
            + ", ".join(failed)
        )

    return {
        "experiment_id": specification.experiment_id,
        "model_name": specification.model_name,
        "status": "passed",
        "device": str(device),
        "amp_enabled": bool(result["amp_enabled"]),
        "configuration_canonical_sha256": validated.canonical_sha256,
        "configuration_source_sha256": validated.source_file_sha256,
        "smoke_protocol": {
            "train_samples": int(smoke["train_samples"]),
            "validation_samples": int(smoke["validation_samples"]),
            "batch_size": int(smoke["batch_size"]),
            "epochs": expected_epochs,
            "expected_training_steps": expected_steps,
        },
        "completed_epochs": int(summary["completed_epochs"]),
        "final_epoch": int(summary["final_epoch"]),
        "final_global_step": int(summary["final_global_step"]),
        "best_metric": float(best_metric),
        "best_epoch": int(summary["best_epoch"]),
        "elapsed_seconds": float(elapsed_seconds),
        "peak_allocated_mebibytes": float(peak_allocated_mebibytes),
        "peak_reserved_mebibytes": float(peak_reserved_mebibytes),
        "output_root": str(output_root),
        "run_manifest": str(manifest_path),
        "training_summary": str(summary_path),
        "latest_checkpoint": latest_verification,
        "best_checkpoint": best_verification,
        "checks": checks,
        "not_for_scientific_comparison": True,
    }


def _release_cuda_memory(device: torch.device) -> None:
    """Release unreachable objects and clear the CUDA caching allocator."""

    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()


def run_step08a_gpu_smoke_tests(
    *,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    num_workers: int = 2,
    requested_experiments: Sequence[str] | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Run and validate the selected approved baseline GPU smoke tests."""

    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    resolved_device = _resolve_cuda_device(device)
    resolved_config_directory = Path(config_directory).expanduser().resolve()
    resolved_source_root = Path(source_root).expanduser().resolve()
    resolved_output_root = Path(output_root).expanduser().resolve()
    selected = _select_specifications(requested_experiments)

    if not resolved_config_directory.is_dir():
        raise FileNotFoundError(
            f"Configuration directory not found: {resolved_config_directory}"
        )
    if not resolved_source_root.is_dir():
        raise FileNotFoundError(
            f"ISIC 2018 source root not found: {resolved_source_root}"
        )

    if resolved_output_root.exists():
        if overwrite_existing:
            shutil.rmtree(resolved_output_root)
        elif any(resolved_output_root.iterdir()):
            raise FileExistsError(
                f"Output root is not empty: {resolved_output_root}. "
                "Pass --overwrite-existing to recreate it."
            )

    resolved_output_root.mkdir(parents=True, exist_ok=True)

    from src.train.train_baseline import run_baseline_training

    validated_configs: dict[str, "ValidatedBaselineExperimentConfig"] = {}
    for specification in selected:
        config_path = resolved_config_directory / specification.config_filename
        validated_configs[specification.experiment_id] = (
            _validate_configuration_assignment(specification, config_path)
        )

    report: dict[str, Any] = {
        "status": "running",
        "all_checks_passed": False,
        "protocol_version": STEP08A_PROTOCOL_VERSION,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "purpose": (
            "sequential_real_data_gpu_smoke_validation_of_all_approved_"
            "baseline_architectures_before_full_training"
        ),
        "scientific_use": (
            "Integration validation only; smoke metrics are prohibited from "
            "the final model-comparison table."
        ),
        "device": _device_summary(resolved_device),
        "config_directory": str(resolved_config_directory),
        "source_root": str(resolved_source_root),
        "output_root": str(resolved_output_root),
        "num_workers_override": num_workers,
        "selected_experiments": [spec.experiment_id for spec in selected],
        "experiment_model_assignments": {
            spec.experiment_id: spec.model_name for spec in selected
        },
        "experiments": {},
    }

    aggregate_path = resolved_output_root / AGGREGATE_REPORT_FILENAME
    _atomic_write_json(aggregate_path, report)

    summary_rows: list[dict[str, Any]] = []

    try:
        for specification in selected:
            validated = validated_configs[specification.experiment_id]
            config_path = (
                resolved_config_directory / specification.config_filename
            )
            experiment_output = (
                resolved_output_root / specification.output_directory_name
            )

            print(
                f"\n[{specification.experiment_id}] Starting "
                f"{specification.model_name} GPU smoke test",
                flush=True,
            )

            _release_cuda_memory(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
            torch.cuda.synchronize(resolved_device)

            started = time.perf_counter()

            result = run_baseline_training(
                config_path=config_path,
                source_roots=[resolved_source_root],
                smoke_test=True,
                device=resolved_device,
                num_workers_override=num_workers,
                output_root_override=experiment_output,
                resume_checkpoint=None,
                overwrite_existing=False,
                allow_cpu_full_training=False,
            )

            torch.cuda.synchronize(resolved_device)
            elapsed = time.perf_counter() - started
            peak_allocated = (
                torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(resolved_device) / (1024**2)
            )

            experiment_report = _validate_completed_smoke_run(
                specification=specification,
                validated=validated,
                output_root=experiment_output,
                result=result,
                device=resolved_device,
                elapsed_seconds=elapsed,
                peak_allocated_mebibytes=peak_allocated,
                peak_reserved_mebibytes=peak_reserved,
            )

            report["experiments"][specification.experiment_id] = (
                experiment_report
            )
            summary_rows.append(experiment_report)
            _atomic_write_json(aggregate_path, report)
            _write_summary_csv(
                resolved_output_root / SUMMARY_CSV_FILENAME,
                summary_rows,
            )

            print(
                json.dumps(
                    {
                        "status": "passed",
                        "experiment_id": specification.experiment_id,
                        "model_name": specification.model_name,
                        "completed_epochs": experiment_report[
                            "completed_epochs"
                        ],
                        "final_global_step": experiment_report[
                            "final_global_step"
                        ],
                        "best_metric": experiment_report["best_metric"],
                        "elapsed_seconds": experiment_report[
                            "elapsed_seconds"
                        ],
                        "peak_allocated_mebibytes": experiment_report[
                            "peak_allocated_mebibytes"
                        ],
                    },
                    indent=2,
                ),
                flush=True,
            )

            del result
            _release_cuda_memory(resolved_device)

    except BaseException as error:
        failure = {
            "status": "failed",
            "protocol_version": STEP08A_PROTOCOL_VERSION,
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "completed_experiments": [row["experiment_id"] for row in summary_rows],
        }
        _atomic_write_json(
            resolved_output_root / FAILURE_REPORT_FILENAME,
            failure,
        )
        report["status"] = "failed"
        report["completed_at_utc"] = _utc_now()
        report["failure"] = failure
        _atomic_write_json(aggregate_path, report)
        raise

    report["status"] = "passed"
    report["all_checks_passed"] = True
    report["completed_at_utc"] = _utc_now()
    report["validated_experiment_count"] = len(summary_rows)
    report["summary_csv"] = str(
        resolved_output_root / SUMMARY_CSV_FILENAME
    )
    report["aggregate_report"] = str(aggregate_path)
    _atomic_write_json(aggregate_path, report)
    _write_summary_csv(
        resolved_output_root / SUMMARY_CSV_FILENAME,
        summary_rows,
    )

    printable = {
        "status": report["status"],
        "all_checks_passed": report["all_checks_passed"],
        "validated_experiment_count": report["validated_experiment_count"],
        "selected_experiments": report["selected_experiments"],
        "device": report["device"]["device_name"],
        "aggregate_report": report["aggregate_report"],
        "summary_csv": report["summary_csv"],
    }
    print("\n" + json.dumps(printable, indent=2), flush=True)
    return report


def run_step08a_self_test() -> dict[str, Any]:
    """Run artifact-independent orchestration checks without requiring CUDA."""

    approved_ids = [spec.experiment_id for spec in APPROVED_BASELINES]
    approved_models = [spec.model_name for spec in APPROVED_BASELINES]
    selected = _select_specifications(["e05", "E01", "E03"])

    checks = {
        "five_approved_baselines": len(APPROVED_BASELINES) == 5,
        "experiment_order_correct": approved_ids
        == ["E01", "E02", "E03", "E04", "E05"],
        "model_order_correct": approved_models
        == ["unet", "unetpp", "deeplabv3plus", "transunet", "swin_unet"],
        "subset_preserves_canonical_order": [
            spec.experiment_id for spec in selected
        ]
        == ["E01", "E03", "E05"],
        "output_names_unique": len(
            {spec.output_directory_name for spec in APPROVED_BASELINES}
        )
        == 5,
        "all_config_names_yaml": all(
            spec.config_filename.endswith(".yaml")
            for spec in APPROVED_BASELINES
        ),
        "protocol_version_present": bool(STEP08A_PROTOCOL_VERSION),
    }

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol_version": STEP08A_PROTOCOL_VERSION,
        "checks": checks,
        "approved_baselines": [asdict(spec) for spec in APPROVED_BASELINES],
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the Step08A command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run sequential one-epoch GPU smoke tests for approved E01-E05 "
            "baseline experiments."
        )
    )
    parser.add_argument(
        "--config-directory",
        default=str(DEFAULT_CONFIG_DIRECTORY),
        help="Directory containing E01-E05 experiment YAML files.",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Mounted ISIC 2018 source root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Writable Step08A aggregate output root.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for every smoke run. Default: cuda:0.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker override for smoke tests. Default: 2.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional subset such as --experiments E01 E03. "
            "Default: all E01-E05 in canonical order."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Delete and recreate a non-empty Step08A output root.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run artifact-independent orchestration checks without CUDA.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.self_test:
        result = run_step08a_self_test()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] != "passed":
            return 1
        return 0

    run_step08a_gpu_smoke_tests(
        config_directory=arguments.config_directory,
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        device=arguments.device,
        num_workers=arguments.num_workers,
        requested_experiments=arguments.experiments,
        overwrite_existing=bool(arguments.overwrite_existing),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
