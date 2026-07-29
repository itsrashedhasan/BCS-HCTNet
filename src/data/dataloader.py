"""Deterministic dataset and DataLoader construction for BCS-HCTNet.

This module connects:

- validated E00 configuration;
- approved Step 04 split manifests;
- persistent Step 05A targets;
- synchronized multi-target transforms;
- deterministic PyTorch DataLoaders.

It supports a bounded smoke-test mode without modifying the underlying
approved manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import BCSHCTNetDataset
from src.data.transforms import build_transforms
from src.train.experiment_config import (
    ValidatedExperimentConfig,
)
from src.utils.seed import (
    create_torch_generator,
    seed_everything,
    seed_worker,
)


DATALOADER_PROTOCOL_VERSION = (
    "BCS-HCTNet-dataloader-v1"
)


@dataclass(frozen=True)
class DataLoaderSettings:
    """Validated PyTorch DataLoader settings."""

    train_batch_size: int
    evaluation_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    drop_last_train: bool
    drop_last_evaluation: bool
    shuffle_train: bool
    shuffle_evaluation: bool
    seed: int

    def validate(self) -> None:
        """Validate DataLoader settings."""

        for name, value in {
            "train_batch_size": (
                self.train_batch_size
            ),
            "evaluation_batch_size": (
                self.evaluation_batch_size
            ),
            "prefetch_factor": (
                self.prefetch_factor
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive "
                    f"integer, received {value!r}."
                )

        if (
            isinstance(
                self.num_workers,
                bool,
            )
            or not isinstance(
                self.num_workers,
                int,
            )
            or self.num_workers < 0
        ):
            raise ValueError(
                "num_workers must be a "
                "non-negative integer."
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative "
                "integer."
            )

        for name, value in {
            "pin_memory": self.pin_memory,
            "persistent_workers": (
                self.persistent_workers
            ),
            "drop_last_train": (
                self.drop_last_train
            ),
            "drop_last_evaluation": (
                self.drop_last_evaluation
            ),
            "shuffle_train": (
                self.shuffle_train
            ),
            "shuffle_evaluation": (
                self.shuffle_evaluation
            ),
        }.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be Boolean."
                )

        if (
            self.num_workers == 0
            and self.persistent_workers
        ):
            raise ValueError(
                "persistent_workers cannot be "
                "enabled when num_workers is zero."
            )

        if not self.shuffle_train:
            raise ValueError(
                "Training loader must shuffle."
            )

        if self.shuffle_evaluation:
            raise ValueError(
                "Evaluation loader must not "
                "shuffle."
            )

        if self.drop_last_evaluation:
            raise ValueError(
                "Evaluation loader must preserve "
                "all samples."
            )

    def with_overrides(
        self,
        *,
        train_batch_size: int | None = None,
        evaluation_batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> "DataLoaderSettings":
        """Return settings with safe runtime overrides."""

        updated_workers = (
            self.num_workers
            if num_workers is None
            else int(num_workers)
        )

        updated = replace(
            self,
            train_batch_size=(
                self.train_batch_size
                if train_batch_size is None
                else int(train_batch_size)
            ),
            evaluation_batch_size=(
                self.evaluation_batch_size
                if evaluation_batch_size is None
                else int(
                    evaluation_batch_size
                )
            ),
            num_workers=updated_workers,
            persistent_workers=(
                self.persistent_workers
                if updated_workers > 0
                else False
            ),
        )

        updated.validate()

        return updated

    def to_dict(self) -> dict[str, Any]:
        """Return serializable settings."""

        self.validate()

        return {
            "train_batch_size": (
                self.train_batch_size
            ),
            "evaluation_batch_size": (
                self.evaluation_batch_size
            ),
            "num_workers": (
                self.num_workers
            ),
            "pin_memory": (
                self.pin_memory
            ),
            "persistent_workers": (
                self.persistent_workers
            ),
            "prefetch_factor": (
                self.prefetch_factor
            ),
            "drop_last_train": (
                self.drop_last_train
            ),
            "drop_last_evaluation": (
                self.drop_last_evaluation
            ),
            "shuffle_train": (
                self.shuffle_train
            ),
            "shuffle_evaluation": (
                self.shuffle_evaluation
            ),
            "seed": self.seed,
        }


@dataclass
class DataLoaderBundle:
    """Datasets and loaders for one experiment run."""

    train_dataset: BCSHCTNetDataset
    validation_dataset: BCSHCTNetDataset

    active_train_dataset: Dataset
    active_validation_dataset: Dataset

    train_loader: DataLoader
    validation_loader: DataLoader

    settings: DataLoaderSettings
    smoke_test: bool
    train_indices: list[int]
    validation_indices: list[int]

    def summary(self) -> dict[str, Any]:
        """Return loader and dataset provenance."""

        return {
            "protocol_version": (
                DATALOADER_PROTOCOL_VERSION
            ),
            "smoke_test": (
                self.smoke_test
            ),
            "settings": (
                self.settings.to_dict()
            ),
            "full_train_rows": len(
                self.train_dataset
            ),
            "full_validation_rows": len(
                self.validation_dataset
            ),
            "active_train_rows": len(
                self.active_train_dataset
            ),
            "active_validation_rows": len(
                self.active_validation_dataset
            ),
            "train_batches": len(
                self.train_loader
            ),
            "validation_batches": len(
                self.validation_loader
            ),
            "train_indices": (
                self.train_indices
            ),
            "validation_indices": (
                self.validation_indices
            ),
            "train_dataset": (
                self.train_dataset.summary()
            ),
            "validation_dataset": (
                self.validation_dataset.summary()
            ),
        }


def settings_from_payload(
    payload: Mapping[str, Any],
) -> DataLoaderSettings:
    """Build DataLoader settings from experiment configuration."""

    loader = payload[
        "data"
    ][
        "loader"
    ]

    reproducibility = payload[
        "reproducibility"
    ]

    settings = DataLoaderSettings(
        train_batch_size=int(
            loader[
                "train_batch_size"
            ]
        ),
        evaluation_batch_size=int(
            loader[
                "evaluation_batch_size"
            ]
        ),
        num_workers=int(
            loader[
                "num_workers"
            ]
        ),
        pin_memory=bool(
            loader[
                "pin_memory"
            ]
        ),
        persistent_workers=bool(
            loader[
                "persistent_workers"
            ]
        ),
        prefetch_factor=int(
            loader[
                "prefetch_factor"
            ]
        ),
        drop_last_train=bool(
            loader[
                "drop_last_train"
            ]
        ),
        drop_last_evaluation=bool(
            loader[
                "drop_last_evaluation"
            ]
        ),
        shuffle_train=bool(
            loader[
                "shuffle_train"
            ]
        ),
        shuffle_evaluation=bool(
            loader[
                "shuffle_evaluation"
            ]
        ),
        seed=int(
            reproducibility[
                "seed_torch"
            ]
        ),
    )

    settings.validate()

    return settings


def evenly_spaced_indices(
    dataset_length: int,
    sample_count: int,
) -> list[int]:
    """Choose deterministic indices spanning a dataset."""

    if (
        isinstance(dataset_length, bool)
        or not isinstance(
            dataset_length,
            int,
        )
        or dataset_length <= 0
    ):
        raise ValueError(
            "dataset_length must be a "
            "positive integer."
        )

    if (
        isinstance(sample_count, bool)
        or not isinstance(
            sample_count,
            int,
        )
        or sample_count <= 0
    ):
        raise ValueError(
            "sample_count must be a "
            "positive integer."
        )

    if sample_count > dataset_length:
        raise ValueError(
            f"Cannot select {sample_count} "
            f"samples from a dataset containing "
            f"{dataset_length} rows."
        )

    if sample_count == 1:
        return [0]

    maximum_index = (
        dataset_length - 1
    )

    indices = [
        round(
            position
            * maximum_index
            / (
                sample_count - 1
            )
        )
        for position in range(
            sample_count
        )
    ]

    if len(
        set(indices)
    ) != sample_count:
        raise RuntimeError(
            "Deterministic sample selection "
            "produced duplicate indices."
        )

    return [
        int(index)
        for index in indices
    ]


def create_data_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    settings: DataLoaderSettings,
    generator_seed: int,
) -> DataLoader:
    """Construct one deterministic PyTorch DataLoader."""

    if len(dataset) <= 0:
        raise ValueError(
            "Cannot create a DataLoader for "
            "an empty dataset."
        )

    generator = create_torch_generator(
        seed=generator_seed,
        device="cpu",
    )

    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            batch_size
        ),
        "shuffle": bool(
            shuffle
        ),
        "num_workers": (
            settings.num_workers
        ),
        "pin_memory": (
            settings.pin_memory
        ),
        "drop_last": bool(
            drop_last
        ),
        "worker_init_fn": (
            seed_worker
        ),
        "generator": generator,
    }

    if settings.num_workers > 0:
        arguments[
            "persistent_workers"
        ] = settings.persistent_workers

        arguments[
            "prefetch_factor"
        ] = settings.prefetch_factor

    return DataLoader(
        **arguments
    )


def build_loader_bundle(
    *,
    validated: ValidatedExperimentConfig,
    source_roots: Sequence[
        str | Path
    ],
    smoke_test: bool,
    num_workers_override: int | None = None,
) -> DataLoaderBundle:
    """Build train and validation datasets and DataLoaders."""

    if not isinstance(
        smoke_test,
        bool,
    ):
        raise TypeError(
            "smoke_test must be Boolean."
        )

    payload = validated.payload

    reproducibility = payload[
        "reproducibility"
    ]

    seed_everything(
        seed=int(
            reproducibility[
                "seed_python"
            ]
        ),
        deterministic_algorithms=bool(
            reproducibility[
                "deterministic_algorithms"
            ]
        ),
        cudnn_deterministic=bool(
            reproducibility[
                "cudnn_deterministic"
            ]
        ),
        cudnn_benchmark=bool(
            reproducibility[
                "cudnn_benchmark"
            ]
        ),
    )

    settings = settings_from_payload(
        payload
    )

    if num_workers_override is not None:
        settings = settings.with_overrides(
            num_workers=(
                num_workers_override
            )
        )

    transforms = build_transforms(
        payload
    )

    target_height = int(
        payload[
            "data"
        ][
            "image_size"
        ][
            "height"
        ]
    )

    target_width = int(
        payload[
            "data"
        ][
            "image_size"
        ][
            "width"
        ]
    )

    sdm_clip_distance = float(
        payload[
            "data"
        ][
            "targets"
        ][
            "signed_distance_map"
        ][
            "clip_distance_pixels"
        ]
    )

    target_manifest = (
        validated.manifests[
            "derived_targets"
        ]
    )

    train_manifest = (
        validated.manifests[
            "train"
        ]
    )

    validation_manifest = (
        validated.manifests[
            "validation"
        ]
    )

    train_dataset = BCSHCTNetDataset(
        split_manifest_path=(
            train_manifest.absolute_path
        ),
        target_manifest_path=(
            target_manifest.absolute_path
        ),
        target_artifact_root=(
            validated.artifact_roots[
                "step05a"
            ]
        ),
        source_roots=source_roots,
        transform=transforms[
            "train"
        ],
        expected_target_height=(
            target_height
        ),
        expected_target_width=(
            target_width
        ),
        sdm_clip_distance_pixels=(
            sdm_clip_distance
        ),
        expected_rows=(
            train_manifest.expected_rows
        ),
    )

    validation_dataset = (
        BCSHCTNetDataset(
            split_manifest_path=(
                validation_manifest
                .absolute_path
            ),
            target_manifest_path=(
                target_manifest
                .absolute_path
            ),
            target_artifact_root=(
                validated.artifact_roots[
                    "step05a"
                ]
            ),
            source_roots=source_roots,
            transform=transforms[
                "validation"
            ],
            expected_target_height=(
                target_height
            ),
            expected_target_width=(
                target_width
            ),
            sdm_clip_distance_pixels=(
                sdm_clip_distance
            ),
            expected_rows=(
                validation_manifest
                .expected_rows
            ),
        )
    )

    train_ids = {
        record.image_id
        for record in (
            train_dataset.records
        )
    }

    validation_ids = {
        record.image_id
        for record in (
            validation_dataset.records
        )
    }

    collisions = sorted(
        train_ids
        & validation_ids
    )

    if collisions:
        raise RuntimeError(
            "Train-validation identifier "
            f"collisions detected: "
            f"{collisions[:20]}."
        )

    if smoke_test:
        smoke = payload[
            "smoke_test"
        ]

        train_indices = (
            evenly_spaced_indices(
                len(
                    train_dataset
                ),
                int(
                    smoke[
                        "train_samples"
                    ]
                ),
            )
        )

        validation_indices = (
            evenly_spaced_indices(
                len(
                    validation_dataset
                ),
                int(
                    smoke[
                        "validation_samples"
                    ]
                ),
            )
        )

        active_train_dataset: Dataset = (
            Subset(
                train_dataset,
                train_indices,
            )
        )

        active_validation_dataset: Dataset = (
            Subset(
                validation_dataset,
                validation_indices,
            )
        )

        settings = settings.with_overrides(
            train_batch_size=int(
                smoke[
                    "batch_size"
                ]
            ),
            evaluation_batch_size=int(
                smoke[
                    "batch_size"
                ]
            ),
        )

    else:
        train_indices = list(
            range(
                len(
                    train_dataset
                )
            )
        )

        validation_indices = list(
            range(
                len(
                    validation_dataset
                )
            )
        )

        active_train_dataset = (
            train_dataset
        )

        active_validation_dataset = (
            validation_dataset
        )

    train_loader = create_data_loader(
        dataset=active_train_dataset,
        batch_size=(
            settings.train_batch_size
        ),
        shuffle=(
            settings.shuffle_train
        ),
        drop_last=(
            settings.drop_last_train
        ),
        settings=settings,
        generator_seed=(
            settings.seed
        ),
    )

    validation_loader = (
        create_data_loader(
            dataset=(
                active_validation_dataset
            ),
            batch_size=(
                settings
                .evaluation_batch_size
            ),
            shuffle=(
                settings
                .shuffle_evaluation
            ),
            drop_last=(
                settings
                .drop_last_evaluation
            ),
            settings=settings,
            generator_seed=(
                settings.seed + 1
            ),
        )
    )

    return DataLoaderBundle(
        train_dataset=train_dataset,
        validation_dataset=(
            validation_dataset
        ),
        active_train_dataset=(
            active_train_dataset
        ),
        active_validation_dataset=(
            active_validation_dataset
        ),
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        settings=settings,
        smoke_test=smoke_test,
        train_indices=train_indices,
        validation_indices=(
            validation_indices
        ),
    )


class _SyntheticDataset(Dataset):
    """Small deterministic dataset for loader testing."""

    def __init__(
        self,
        length: int,
    ) -> None:
        self.length = int(
            length
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        value = float(
            index
        )

        return {
            "image": torch.full(
                (
                    3,
                    16,
                    16,
                ),
                value,
                dtype=torch.float32,
            ),
            "mask": torch.full(
                (
                    1,
                    16,
                    16,
                ),
                float(
                    index % 2
                ),
                dtype=torch.float32,
            ),
            "contour": torch.zeros(
                (
                    1,
                    16,
                    16,
                ),
                dtype=torch.float32,
            ),
            "boundary_band": (
                torch.zeros(
                    (
                        1,
                        16,
                        16,
                    ),
                    dtype=torch.float32,
                )
            ),
            "sdm": torch.full(
                (
                    1,
                    16,
                    16,
                ),
                -1.0
                + (
                    2.0
                    * index
                    / max(
                        1,
                        self.length - 1,
                    )
                ),
                dtype=torch.float32,
            ),
            "image_id": (
                f"SELF_TEST_{index:03d}"
            ),
            "index": index,
        }


def _collect_image_ids(
    loader: DataLoader,
) -> list[str]:
    """Collect image IDs from a DataLoader."""

    identifiers: list[str] = []

    for batch in loader:
        batch_ids = batch[
            "image_id"
        ]

        identifiers.extend(
            str(value)
            for value in batch_ids
        )

    return identifiers


def run_dataloader_self_test() -> dict[str, Any]:
    """Test deterministic loader behavior on CPU."""

    settings = DataLoaderSettings(
        train_batch_size=3,
        evaluation_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last_train=False,
        drop_last_evaluation=False,
        shuffle_train=True,
        shuffle_evaluation=False,
        seed=42,
    )

    settings.validate()

    dataset = _SyntheticDataset(
        length=12
    )

    first_train_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            drop_last=False,
            settings=settings,
            generator_seed=42,
        )
    )

    second_train_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            drop_last=False,
            settings=settings,
            generator_seed=42,
        )
    )

    evaluation_loader = (
        create_data_loader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            settings=settings,
            generator_seed=43,
        )
    )

    first_order = _collect_image_ids(
        first_train_loader
    )

    second_order = _collect_image_ids(
        second_train_loader
    )

    evaluation_order = (
        _collect_image_ids(
            evaluation_loader
        )
    )

    shape_loader = create_data_loader(
        dataset=dataset,
        batch_size=3,
        shuffle=False,
        drop_last=False,
        settings=settings,
        generator_seed=44,
    )

    first_batch = next(
        iter(
            shape_loader
        )
    )

    selected_indices = (
        evenly_spaced_indices(
            dataset_length=12,
            sample_count=4,
        )
    )

    checks = {
        "training_order_reproducible": (
            first_order
            == second_order
        ),
        "training_order_shuffled": (
            first_order
            != [
                f"SELF_TEST_{index:03d}"
                for index in range(12)
            ]
        ),
        "all_training_samples_preserved": (
            sorted(
                first_order
            )
            == [
                f"SELF_TEST_{index:03d}"
                for index in range(12)
            ]
        ),
        "evaluation_order_fixed": (
            evaluation_order
            == [
                f"SELF_TEST_{index:03d}"
                for index in range(12)
            ]
        ),
        "train_batch_count": (
            len(
                first_train_loader
            )
            == 4
        ),
        "evaluation_batch_count": (
            len(
                evaluation_loader
            )
            == 3
        ),
        "image_batch_shape": (
            tuple(
                first_batch[
                    "image"
                ].shape
            )
            == (
                3,
                3,
                16,
                16,
            )
        ),
        "mask_batch_shape": (
            tuple(
                first_batch[
                    "mask"
                ].shape
            )
            == (
                3,
                1,
                16,
                16,
            )
        ),
        "spanning_indices_correct": (
            selected_indices
            == [
                0,
                4,
                7,
                11,
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
            DATALOADER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "settings": (
            settings.to_dict()
        ),
        "first_training_order": (
            first_order
        ),
        "evaluation_order": (
            evaluation_order
        ),
        "selected_indices": (
            selected_indices
        ),
    }


if __name__ == "__main__":
    result = run_dataloader_self_test()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if result[
        "status"
    ] != "passed":
        raise SystemExit(1)