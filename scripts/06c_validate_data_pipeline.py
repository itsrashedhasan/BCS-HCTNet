"""Step 06C: Validate the real BCS-HCTNet data pipeline.

Run from the repository root:

    python3 scripts/06c_validate_data_pipeline.py

This CPU-only stage validates the complete data foundation using the
persistent Kaggle artifacts:

- E00 experiment configuration;
- Step 04 train and validation manifests;
- Step 05A mask, contour, boundary-band, and SDM targets;
- original ISIC 2018 source images;
- synchronized transforms;
- deterministic PyTorch DataLoaders.

The configured smoke-test subset contains:

- 8 training samples;
- 4 validation samples;
- batch size 2.

No model training is executed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPOSITORY_ROOT),
)


from src.data.dataloader import (
    DataLoaderBundle,
    build_loader_bundle,
)
from src.train.experiment_config import (
    validate_experiment_configuration,
)


VALIDATION_PROTOCOL_VERSION = (
    "BCS-HCTNet-data-pipeline-validation-v1"
)

DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "e00_bcs_hctnet_foundation.yaml"
)

DEFAULT_ISIC2018_ROOT = Path(
    "/kaggle/input/datasets/"
    "asrafulislam7/isic-2018/ISIC_2018"
)

DEFAULT_OUTPUT_PATH = Path(
    "/kaggle/working/outputs/"
    "experiments/E00/data_foundation/"
    "DATA_PIPELINE_VALIDATION.json"
)

REQUIRED_BATCH_KEYS = {
    "image",
    "mask",
    "contour",
    "boundary_band",
    "sdm",
    "image_id",
    "index",
    "split",
    "source_image_path",
    "target_mask_path",
    "target_contour_path",
    "target_boundary_band_path",
    "target_sdm_path",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the real BCS-HCTNet "
            "data pipeline."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the E00 experiment "
            "configuration."
        ),
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root containing original ISIC 2018 "
            "images. May be supplied more than once."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for the JSON validation report."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker count for this "
            "validation. Default: 0."
        ),
    )

    parser.add_argument(
        "--minimum-sign-agreement",
        type=float,
        default=0.95,
        help=(
            "Minimum required fraction of mask "
            "and SDM sign agreement."
        ),
    )

    return parser.parse_args()


def require_finite_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Require a finite PyTorch tensor."""

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor, "
            f"received {type(tensor).__name__}."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise RuntimeError(
            f"{name} contains non-finite values."
        )


def require_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    """Require an exact tensor shape."""

    observed_shape = tuple(
        int(value)
        for value in tensor.shape
    )

    if observed_shape != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected "
            f"{expected_shape}, found "
            f"{observed_shape}."
        )


def require_binary_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Require exact binary values 0 and 1."""

    values = {
        float(value)
        for value in torch.unique(
            tensor.detach().cpu()
        )
    }

    if not values.issubset(
        {
            0.0,
            1.0,
        }
    ):
        raise RuntimeError(
            f"{name} is not binary. Values: "
            f"{sorted(values)[:20]}."
        )


def safe_fraction(
    numerator: int,
    denominator: int,
) -> float | None:
    """Calculate a fraction or return None."""

    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def finite_or_none(
    value: torch.Tensor,
) -> float | None:
    """Convert a scalar tensor to finite float."""

    number = float(
        value.detach().cpu()
    )

    if not math.isfinite(
        number
    ):
        return None

    return number


def validate_sample_geometry(
    *,
    mask: torch.Tensor,
    contour: torch.Tensor,
    boundary_band: torch.Tensor,
    sdm: torch.Tensor,
    image_id: str,
) -> dict[str, Any]:
    """Validate target geometry for one sample."""

    mask_foreground = (
        mask >= 0.5
    )

    mask_background = (
        ~mask_foreground
    )

    contour_foreground = (
        contour >= 0.5
    )

    boundary_foreground = (
        boundary_band >= 0.5
    )

    contour_outside_band = (
        contour_foreground
        & (
            ~boundary_foreground
        )
    )

    inside_pixels = int(
        mask_foreground.sum().item()
    )

    outside_pixels = int(
        mask_background.sum().item()
    )

    inside_nonnegative = int(
        (
            sdm[
                mask_foreground
            ]
            >= -1e-6
        ).sum().item()
    )

    outside_nonpositive = int(
        (
            sdm[
                mask_background
            ]
            <= 1e-6
        ).sum().item()
    )

    inside_sign_agreement = (
        safe_fraction(
            inside_nonnegative,
            inside_pixels,
        )
    )

    outside_sign_agreement = (
        safe_fraction(
            outside_nonpositive,
            outside_pixels,
        )
    )

    inside_mean = (
        finite_or_none(
            sdm[
                mask_foreground
            ].mean()
        )
        if inside_pixels > 0
        else None
    )

    outside_mean = (
        finite_or_none(
            sdm[
                mask_background
            ].mean()
        )
        if outside_pixels > 0
        else None
    )

    return {
        "image_id": image_id,
        "mask_foreground_pixels": (
            inside_pixels
        ),
        "mask_background_pixels": (
            outside_pixels
        ),
        "mask_foreground_ratio": float(
            mask_foreground.float()
            .mean()
            .item()
        ),
        "contour_pixels": int(
            contour_foreground.sum().item()
        ),
        "boundary_band_pixels": int(
            boundary_foreground.sum().item()
        ),
        "contour_outside_boundary_band_pixels": (
            int(
                contour_outside_band
                .sum()
                .item()
            )
        ),
        "sdm_min": float(
            sdm.min().item()
        ),
        "sdm_max": float(
            sdm.max().item()
        ),
        "sdm_mean": float(
            sdm.mean().item()
        ),
        "sdm_inside_mean": (
            inside_mean
        ),
        "sdm_outside_mean": (
            outside_mean
        ),
        "inside_sign_agreement": (
            inside_sign_agreement
        ),
        "outside_sign_agreement": (
            outside_sign_agreement
        ),
    }


def validate_batch(
    *,
    batch: Mapping[str, Any],
    phase: str,
    batch_number: int,
    expected_height: int,
    expected_width: int,
    minimum_sign_agreement: float,
) -> dict[str, Any]:
    """Validate one real DataLoader batch."""

    missing_keys = sorted(
        REQUIRED_BATCH_KEYS
        - set(batch)
    )

    if missing_keys:
        raise KeyError(
            f"{phase} batch {batch_number} is "
            f"missing keys: {missing_keys}."
        )

    image = batch["image"]
    mask = batch["mask"]
    contour = batch["contour"]
    boundary_band = batch[
        "boundary_band"
    ]
    sdm = batch["sdm"]

    for name, tensor in {
        "image": image,
        "mask": mask,
        "contour": contour,
        "boundary_band": boundary_band,
        "sdm": sdm,
    }.items():
        require_finite_tensor(
            tensor,
            (
                f"{phase} batch "
                f"{batch_number} {name}"
            ),
        )

    batch_size = int(
        image.shape[0]
    )

    require_shape(
        image,
        (
            batch_size,
            3,
            expected_height,
            expected_width,
        ),
        (
            f"{phase} batch "
            f"{batch_number} image"
        ),
    )

    for name, tensor in {
        "mask": mask,
        "contour": contour,
        "boundary_band": boundary_band,
        "sdm": sdm,
    }.items():
        require_shape(
            tensor,
            (
                batch_size,
                1,
                expected_height,
                expected_width,
            ),
            (
                f"{phase} batch "
                f"{batch_number} {name}"
            ),
        )

    require_binary_tensor(
        mask,
        (
            f"{phase} batch "
            f"{batch_number} mask"
        ),
    )

    require_binary_tensor(
        contour,
        (
            f"{phase} batch "
            f"{batch_number} contour"
        ),
    )

    require_binary_tensor(
        boundary_band,
        (
            f"{phase} batch "
            f"{batch_number} boundary band"
        ),
    )

    sdm_min = float(
        sdm.min().item()
    )

    sdm_max = float(
        sdm.max().item()
    )

    if (
        sdm_min < -1.000001
        or sdm_max > 1.000001
    ):
        raise RuntimeError(
            f"{phase} batch {batch_number} "
            "contains SDM values outside "
            f"[-1, 1]: min={sdm_min}, "
            f"max={sdm_max}."
        )

    image_min = float(
        image.min().item()
    )

    image_max = float(
        image.max().item()
    )

    if (
        image_min < -5.0
        or image_max > 5.0
    ):
        raise RuntimeError(
            f"{phase} batch {batch_number} "
            "contains implausible normalized "
            f"image values: min={image_min}, "
            f"max={image_max}."
        )

    image_ids = [
        str(value)
        for value in batch[
            "image_id"
        ]
    ]

    if len(
        image_ids
    ) != batch_size:
        raise RuntimeError(
            f"{phase} batch {batch_number} "
            "metadata count does not match "
            "tensor batch size."
        )

    sample_reports: list[
        dict[str, Any]
    ] = []

    for sample_index, image_id in enumerate(
        image_ids
    ):
        sample_report = (
            validate_sample_geometry(
                mask=mask[
                    sample_index
                ],
                contour=contour[
                    sample_index
                ],
                boundary_band=(
                    boundary_band[
                        sample_index
                    ]
                ),
                sdm=sdm[
                    sample_index
                ],
                image_id=image_id,
            )
        )

        if (
            sample_report[
                "contour_outside_boundary_band_pixels"
            ]
            != 0
        ):
            raise RuntimeError(
                f"{phase} sample {image_id} "
                "contains contour pixels outside "
                "the boundary band."
            )

        inside_agreement = (
            sample_report[
                "inside_sign_agreement"
            ]
        )

        outside_agreement = (
            sample_report[
                "outside_sign_agreement"
            ]
        )

        if (
            inside_agreement is not None
            and inside_agreement
            < minimum_sign_agreement
        ):
            raise RuntimeError(
                f"{phase} sample {image_id} "
                "has insufficient inside-mask "
                "SDM sign agreement: "
                f"{inside_agreement:.6f}."
            )

        if (
            outside_agreement is not None
            and outside_agreement
            < minimum_sign_agreement
        ):
            raise RuntimeError(
                f"{phase} sample {image_id} "
                "has insufficient outside-mask "
                "SDM sign agreement: "
                f"{outside_agreement:.6f}."
            )

        sample_reports.append(
            sample_report
        )

    return {
        "phase": phase,
        "batch_number": (
            batch_number
        ),
        "batch_size": (
            batch_size
        ),
        "image_ids": image_ids,
        "image_min": image_min,
        "image_max": image_max,
        "sdm_min": sdm_min,
        "sdm_max": sdm_max,
        "samples": (
            sample_reports
        ),
    }


def validate_loader(
    *,
    loader,
    phase: str,
    expected_rows: int,
    expected_height: int,
    expected_width: int,
    minimum_sign_agreement: float,
) -> dict[str, Any]:
    """Validate every batch from one loader."""

    batch_reports: list[
        dict[str, Any]
    ] = []

    observed_ids: list[str] = []

    observed_rows = 0

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):
        batch_report = validate_batch(
            batch=batch,
            phase=phase,
            batch_number=batch_number,
            expected_height=(
                expected_height
            ),
            expected_width=(
                expected_width
            ),
            minimum_sign_agreement=(
                minimum_sign_agreement
            ),
        )

        batch_reports.append(
            batch_report
        )

        observed_ids.extend(
            batch_report[
                "image_ids"
            ]
        )

        observed_rows += int(
            batch_report[
                "batch_size"
            ]
        )

    if observed_rows != expected_rows:
        raise RuntimeError(
            f"{phase} loader expected "
            f"{expected_rows} rows, found "
            f"{observed_rows}."
        )

    duplicate_ids = sorted(
        {
            image_id
            for image_id in observed_ids
            if observed_ids.count(
                image_id
            ) > 1
        }
    )

    if duplicate_ids:
        raise RuntimeError(
            f"{phase} loader produced duplicate "
            f"image IDs: {duplicate_ids}."
        )

    all_sample_reports = [
        sample
        for batch_report in batch_reports
        for sample in batch_report[
            "samples"
        ]
    ]

    inside_agreements = [
        float(
            sample[
                "inside_sign_agreement"
            ]
        )
        for sample in all_sample_reports
        if sample[
            "inside_sign_agreement"
        ] is not None
    ]

    outside_agreements = [
        float(
            sample[
                "outside_sign_agreement"
            ]
        )
        for sample in all_sample_reports
        if sample[
            "outside_sign_agreement"
        ] is not None
    ]

    return {
        "phase": phase,
        "expected_rows": (
            expected_rows
        ),
        "observed_rows": (
            observed_rows
        ),
        "batch_count": len(
            batch_reports
        ),
        "image_ids": (
            observed_ids
        ),
        "minimum_inside_sign_agreement": (
            min(
                inside_agreements
            )
            if inside_agreements
            else None
        ),
        "minimum_outside_sign_agreement": (
            min(
                outside_agreements
            )
            if outside_agreements
            else None
        ),
        "batches": (
            batch_reports
        ),
    }


def verify_validation_determinism(
    bundle: DataLoaderBundle,
) -> dict[str, Any]:
    """Load one validation sample twice and compare outputs."""

    if len(
        bundle.active_validation_dataset
    ) <= 0:
        raise RuntimeError(
            "Active validation dataset is empty."
        )

    first = bundle.active_validation_dataset[
        0
    ]

    second = bundle.active_validation_dataset[
        0
    ]

    tensor_keys = [
        "image",
        "mask",
        "contour",
        "boundary_band",
        "sdm",
    ]

    checks = {
        key: torch.equal(
            first[key],
            second[key],
        )
        for key in tensor_keys
    }

    checks[
        "image_id"
    ] = (
        first["image_id"]
        == second["image_id"]
    )

    return {
        "image_id": (
            first["image_id"]
        ),
        "checks": checks,
        "all_checks_passed": all(
            checks.values()
        ),
    }


def verify_resolved_files(
    bundle: DataLoaderBundle,
) -> dict[str, Any]:
    """Verify all active sample files exist."""

    datasets_and_indices = [
        (
            "train",
            bundle.train_dataset,
            bundle.train_indices,
        ),
        (
            "validation",
            bundle.validation_dataset,
            bundle.validation_indices,
        ),
    ]

    missing_files: list[
        dict[str, str]
    ] = []

    checked_files = 0

    for (
        phase,
        dataset,
        indices,
    ) in datasets_and_indices:
        for index in indices:
            record = dataset.record(
                index
            )

            for label, path in {
                "image": (
                    record.image_path
                ),
                "mask": (
                    record.mask_path
                ),
                "contour": (
                    record.contour_path
                ),
                "boundary_band": (
                    record.boundary_band_path
                ),
                "sdm": (
                    record.sdm_path
                ),
            }.items():
                checked_files += 1

                if not path.is_file():
                    missing_files.append(
                        {
                            "phase": phase,
                            "image_id": (
                                record.image_id
                            ),
                            "file_type": label,
                            "path": str(path),
                        }
                    )

    return {
        "checked_files": (
            checked_files
        ),
        "missing_files": (
            missing_files
        ),
        "all_files_exist": (
            len(
                missing_files
            )
            == 0
        ),
    }


def main() -> int:
    """Run Step 06C."""

    arguments = parse_arguments()

    if arguments.num_workers < 0:
        raise ValueError(
            "--num-workers must be "
            "non-negative."
        )

    if not (
        0.0
        <= arguments.minimum_sign_agreement
        <= 1.0
    ):
        raise ValueError(
            "--minimum-sign-agreement must "
            "be in [0, 1]."
        )

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    output_path = (
        arguments.output
        .expanduser()
        .resolve()
    )

    source_roots = (
        [
            path.expanduser().resolve()
            for path in arguments.source_root
        ]
        if arguments.source_root
        else [
            DEFAULT_ISIC2018_ROOT.resolve()
        ]
    )

    print(
        "=== Step 06C: Validate Real "
        "Data Pipeline ==="
    )

    print(
        "Repository root :",
        REPOSITORY_ROOT,
    )

    print(
        "Configuration   :",
        config_path,
    )

    print(
        "Source roots    :"
    )

    for source_root in source_roots:
        print(
            " -",
            source_root,
        )

    print(
        "Output report   :",
        output_path,
    )

    print(
        "Data workers    :",
        arguments.num_workers,
    )

    print(
        "Execution mode  : CPU-only"
    )

    for source_root in source_roots:
        if not source_root.is_dir():
            raise FileNotFoundError(
                "Source image root not found: "
                f"{source_root}"
            )

    validated = (
        validate_experiment_configuration(
            config_path
        )
    )

    bundle = build_loader_bundle(
        validated=validated,
        source_roots=source_roots,
        smoke_test=True,
        num_workers_override=(
            arguments.num_workers
        ),
    )

    bundle_summary = bundle.summary()

    print(
        "\n=== Loader Summary ==="
    )

    print(
        "Full training rows     :",
        bundle_summary[
            "full_train_rows"
        ],
    )

    print(
        "Full validation rows   :",
        bundle_summary[
            "full_validation_rows"
        ],
    )

    print(
        "Active training rows   :",
        bundle_summary[
            "active_train_rows"
        ],
    )

    print(
        "Active validation rows :",
        bundle_summary[
            "active_validation_rows"
        ],
    )

    print(
        "Training batches       :",
        bundle_summary[
            "train_batches"
        ],
    )

    print(
        "Validation batches     :",
        bundle_summary[
            "validation_batches"
        ],
    )

    image_height = int(
        validated.payload[
            "data"
        ][
            "image_size"
        ][
            "height"
        ]
    )

    image_width = int(
        validated.payload[
            "data"
        ][
            "image_size"
        ][
            "width"
        ]
    )

    file_validation = (
        verify_resolved_files(
            bundle
        )
    )

    if not file_validation[
        "all_files_exist"
    ]:
        raise RuntimeError(
            "Some active data files could not "
            "be resolved."
        )

    print(
        "\nResolved active files :",
        file_validation[
            "checked_files"
        ],
    )

    train_validation = (
        validate_loader(
            loader=bundle.train_loader,
            phase="train",
            expected_rows=len(
                bundle.active_train_dataset
            ),
            expected_height=(
                image_height
            ),
            expected_width=(
                image_width
            ),
            minimum_sign_agreement=(
                arguments
                .minimum_sign_agreement
            ),
        )
    )

    validation_validation = (
        validate_loader(
            loader=(
                bundle.validation_loader
            ),
            phase="validation",
            expected_rows=len(
                bundle
                .active_validation_dataset
            ),
            expected_height=(
                image_height
            ),
            expected_width=(
                image_width
            ),
            minimum_sign_agreement=(
                arguments
                .minimum_sign_agreement
            ),
        )
    )

    deterministic_validation = (
        verify_validation_determinism(
            bundle
        )
    )

    train_ids = set(
        train_validation[
            "image_ids"
        ]
    )

    validation_ids = set(
        validation_validation[
            "image_ids"
        ]
    )

    collisions = sorted(
        train_ids
        & validation_ids
    )

    checks = {
        "training_readiness": (
            validated.training_readiness[
                "training_allowed"
            ]
            is True
        ),
        "full_train_count": (
            len(
                bundle.train_dataset
            )
            == 2594
        ),
        "full_validation_count": (
            len(
                bundle.validation_dataset
            )
            == 100
        ),
        "smoke_train_count": (
            len(
                bundle.active_train_dataset
            )
            == 8
        ),
        "smoke_validation_count": (
            len(
                bundle
                .active_validation_dataset
            )
            == 4
        ),
        "smoke_train_batches": (
            len(
                bundle.train_loader
            )
            == 4
        ),
        "smoke_validation_batches": (
            len(
                bundle.validation_loader
            )
            == 2
        ),
        "all_active_files_exist": (
            file_validation[
                "all_files_exist"
            ]
        ),
        "train_rows_loaded": (
            train_validation[
                "observed_rows"
            ]
            == 8
        ),
        "validation_rows_loaded": (
            validation_validation[
                "observed_rows"
            ]
            == 4
        ),
        "no_train_validation_collisions": (
            len(
                collisions
            )
            == 0
        ),
        "validation_transform_deterministic": (
            deterministic_validation[
                "all_checks_passed"
            ]
        ),
    }

    report = {
        "protocol_version": (
            VALIDATION_PROTOCOL_VERSION
        ),
        "configuration_path": str(
            validated.source_path
        ),
        "configuration_sha256": (
            validated.canonical_sha256
        ),
        "source_roots": [
            str(path)
            for path in source_roots
        ],
        "execution": {
            "device": "cpu",
            "num_workers": (
                arguments.num_workers
            ),
            "smoke_test": True,
            "training_executed": False,
        },
        "bundle_summary": (
            bundle_summary
        ),
        "resolved_file_validation": (
            file_validation
        ),
        "train_validation": (
            train_validation
        ),
        "validation_validation": (
            validation_validation
        ),
        "validation_determinism": (
            deterministic_validation
        ),
        "train_validation_collisions": (
            collisions
        ),
        "minimum_required_sign_agreement": (
            arguments
            .minimum_sign_agreement
        ),
        "checks": checks,
        "all_checks_passed": all(
            checks.values()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Step 06C Results ==="
    )

    print(
        "Training sample IDs:"
    )

    for image_id in train_validation[
        "image_ids"
    ]:
        print(
            " -",
            image_id,
        )

    print(
        "\nValidation sample IDs:"
    )

    for image_id in validation_validation[
        "image_ids"
    ]:
        print(
            " -",
            image_id,
        )

    print(
        "\nMinimum train inside-sign "
        "agreement:",
        train_validation[
            "minimum_inside_sign_agreement"
        ],
    )

    print(
        "Minimum train outside-sign "
        "agreement:",
        train_validation[
            "minimum_outside_sign_agreement"
        ],
    )

    print(
        "Minimum validation inside-sign "
        "agreement:",
        validation_validation[
            "minimum_inside_sign_agreement"
        ],
    )

    print(
        "Minimum validation outside-sign "
        "agreement:",
        validation_validation[
            "minimum_outside_sign_agreement"
        ],
    )

    print(
        "\nValidation deterministic:",
        deterministic_validation[
            "all_checks_passed"
        ],
    )

    print(
        "Train/validation collisions:",
        len(
            collisions
        ),
    )

    print(
        "\n=== Checks ==="
    )

    for name, passed in checks.items():
        print(
            f"{name}: {passed}"
        )

    print(
        "\nSaved report:",
        output_path,
    )

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "Step 06C data-pipeline "
            "validation failed."
        )

    print(
        "\nStep 06C real data-pipeline "
        "validation: PASSED"
    )

    print(
        "GPU required: False"
    )

    print(
        "Training executed: False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )