"""Persistent CSV and JSON experiment logging.

This module provides a lightweight Kaggle-friendly experiment logger without
requiring TensorBoard or an external tracking service.

For every completed epoch it maintains:

- an append-safe epoch-history CSV;
- a JSON copy of the latest epoch record;
- experiment metadata and provenance;
- a logger-state JSON file;
- deterministic schema validation during resume.

The complete CSV is rewritten atomically after every update. This is slightly
more expensive than a direct append but protects the history from partially
written rows when a Kaggle session is interrupted.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


EXPERIMENT_LOGGER_PROTOCOL_VERSION = (
    "BCS-HCTNet-experiment-logger-v1"
)

LOGGER_SCHEMA_VERSION = 1

RESERVED_RECORD_FIELDS = (
    "epoch",
    "global_step",
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write a text file atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
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
            newline="",
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


def _normalize_scalar(
    value: object,
    context: str,
) -> str | int | float | bool | None:
    """Convert an approved scalar into a serializable value."""

    if value is None:
        return None

    if isinstance(
        value,
        Tensor,
    ):
        if value.numel() != 1:
            raise ValueError(
                f"{context} tensor must contain "
                "exactly one value."
            )

        if not torch.isfinite(
            value
        ).all():
            raise ValueError(
                f"{context} tensor is non-finite."
            )

        value = value.detach().cpu().item()

    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"{context} must be finite."
            )

        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        f"{context} must be a scalar, string, "
        f"Path, or None; received "
        f"{type(value).__name__}."
    )


def normalize_record(
    record: Mapping[str, object],
) -> dict[
    str,
    str | int | float | bool | None,
]:
    """Normalize and validate one logger record."""

    if not isinstance(
        record,
        Mapping,
    ):
        raise TypeError(
            "record must be a mapping."
        )

    if not record:
        raise ValueError(
            "record cannot be empty."
        )

    normalized: dict[
        str,
        str | int | float | bool | None,
    ] = {}

    for key, value in record.items():
        normalized_key = str(
            key
        ).strip()

        if not normalized_key:
            raise ValueError(
                "Record field names cannot be empty."
            )

        if normalized_key in normalized:
            raise ValueError(
                "Record field names are not unique "
                "after normalization."
            )

        normalized[
            normalized_key
        ] = _normalize_scalar(
            value,
            f"record[{normalized_key!r}]",
        )

    for required_field in (
        RESERVED_RECORD_FIELDS
    ):
        if required_field not in normalized:
            raise KeyError(
                f"Record is missing required field "
                f"{required_field!r}."
            )

    epoch = normalized[
        "epoch"
    ]

    global_step = normalized[
        "global_step"
    ]

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(
            "record['epoch'] must be a "
            "non-negative integer."
        )

    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            "record['global_step'] must be a "
            "non-negative integer."
        )

    return normalized


def _validate_fieldnames(
    fieldnames: Sequence[str],
) -> tuple[str, ...]:
    """Validate an ordered CSV field schema."""

    if isinstance(
        fieldnames,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "fieldnames must be a sequence "
            "of field names."
        )

    normalized = tuple(
        str(field).strip()
        for field in fieldnames
    )

    if not normalized:
        raise ValueError(
            "fieldnames cannot be empty."
        )

    if any(
        not field
        for field in normalized
    ):
        raise ValueError(
            "fieldnames cannot contain "
            "empty values."
        )

    if len(
        set(normalized)
    ) != len(
        normalized
    ):
        raise ValueError(
            "fieldnames must be unique."
        )

    for required_field in (
        RESERVED_RECORD_FIELDS
    ):
        if required_field not in normalized:
            raise ValueError(
                f"fieldnames must include "
                f"{required_field!r}."
            )

    return normalized


def _parse_csv_scalar(
    value: str,
) -> str | int | float | bool | None:
    """Parse one CSV value for resumed in-memory records."""

    if value == "":
        return None

    if value == "True":
        return True

    if value == "False":
        return False

    try:
        return int(
            value
        )

    except ValueError:
        pass

    try:
        number = float(
            value
        )

    except ValueError:
        return value

    if not math.isfinite(
        number
    ):
        raise ValueError(
            "Existing CSV contains a "
            "non-finite numeric value."
        )

    return number


class ExperimentLogger:
    """Persistent epoch-level experiment logger."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        experiment_id: str,
        fieldnames: Sequence[str] | None = None,
        resume: bool = False,
    ) -> None:
        """Initialize a new or resumed logger."""

        normalized_experiment_id = str(
            experiment_id
        ).strip()

        if not normalized_experiment_id:
            raise ValueError(
                "experiment_id cannot be empty."
            )

        if not isinstance(
            resume,
            bool,
        ):
            raise TypeError(
                "resume must be Boolean."
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

        self.experiment_id = (
            normalized_experiment_id
        )

        self.history_path = (
            self.output_directory
            / "epoch_history.csv"
        )

        self.latest_record_path = (
            self.output_directory
            / "latest_epoch.json"
        )

        self.metadata_path = (
            self.output_directory
            / "experiment_metadata.json"
        )

        self.state_path = (
            self.output_directory
            / "logger_state.json"
        )

        self._fieldnames: (
            tuple[str, ...] | None
        ) = (
            _validate_fieldnames(
                fieldnames
            )
            if fieldnames is not None
            else None
        )

        self._rows: list[
            dict[
                str,
                str | int | float | bool | None,
            ]
        ] = []

        if resume:
            self._resume_existing_log()

        else:
            existing_paths = [
                path
                for path in (
                    self.history_path,
                    self.latest_record_path,
                    self.state_path,
                )
                if path.exists()
            ]

            if existing_paths:
                raise FileExistsError(
                    "Existing logger artifacts found. "
                    "Use resume=True or choose a new "
                    f"output directory: {existing_paths}"
                )

            self._write_state()

    def __len__(
        self,
    ) -> int:
        """Return the number of completed epoch records."""

        return len(
            self._rows
        )

    @property
    def fieldnames(
        self,
    ) -> tuple[str, ...] | None:
        """Return the fixed logging schema."""

        return self._fieldnames

    @property
    def last_epoch(
        self,
    ) -> int | None:
        """Return the latest logged epoch."""

        if not self._rows:
            return None

        return int(
            self._rows[-1][
                "epoch"
            ]
        )

    @property
    def last_global_step(
        self,
    ) -> int | None:
        """Return the latest logged global step."""

        if not self._rows:
            return None

        return int(
            self._rows[-1][
                "global_step"
            ]
        )

    def rows(
        self,
    ) -> list[dict[str, Any]]:
        """Return defensive copies of all records."""

        return [
            dict(row)
            for row in self._rows
        ]

    def _resume_existing_log(
        self,
    ) -> None:
        """Load and validate an existing logger history."""

        if not self.state_path.is_file():
            raise FileNotFoundError(
                "Cannot resume because logger state "
                f"is missing: {self.state_path}"
            )

        state = json.loads(
            self.state_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            state,
            Mapping,
        ):
            raise TypeError(
                "Logger state must contain "
                "a JSON object."
            )

        if (
            state.get(
                "protocol_version"
            )
            != EXPERIMENT_LOGGER_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Logger protocol version mismatch."
            )

        if (
            state.get(
                "schema_version"
            )
            != LOGGER_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "Logger schema version mismatch."
            )

        if (
            state.get(
                "experiment_id"
            )
            != self.experiment_id
        ):
            raise RuntimeError(
                "Logger experiment ID mismatch."
            )

        saved_fieldnames = state.get(
            "fieldnames"
        )

        if saved_fieldnames is not None:
            saved_schema = (
                _validate_fieldnames(
                    saved_fieldnames
                )
            )

            if (
                self._fieldnames is not None
                and self._fieldnames
                != saved_schema
            ):
                raise RuntimeError(
                    "Requested field schema differs "
                    "from the saved logger schema."
                )

            self._fieldnames = saved_schema

        if self.history_path.is_file():
            with self.history_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as input_file:
                reader = csv.DictReader(
                    input_file
                )

                if reader.fieldnames is None:
                    raise RuntimeError(
                        "Existing epoch-history CSV "
                        "has no header."
                    )

                csv_schema = (
                    _validate_fieldnames(
                        reader.fieldnames
                    )
                )

                if (
                    self._fieldnames is not None
                    and csv_schema
                    != self._fieldnames
                ):
                    raise RuntimeError(
                        "CSV schema differs from "
                        "logger-state schema."
                    )

                self._fieldnames = csv_schema

                for raw_row in reader:
                    parsed_row = {
                        key: _parse_csv_scalar(
                            value
                        )
                        for key, value
                        in raw_row.items()
                    }

                    normalized_row = (
                        normalize_record(
                            parsed_row
                        )
                    )

                    self._rows.append(
                        normalized_row
                    )

        expected_record_count = int(
            state.get(
                "record_count",
                0,
            )
        )

        if expected_record_count != len(
            self._rows
        ):
            raise RuntimeError(
                "Logger-state record count differs "
                "from epoch-history CSV."
            )

        self._validate_record_order()

    def _validate_record_order(
        self,
    ) -> None:
        """Validate monotonically increasing epochs and steps."""

        previous_epoch: int | None = None
        previous_global_step: int | None = None

        for row in self._rows:
            epoch = int(
                row["epoch"]
            )

            global_step = int(
                row["global_step"]
            )

            if (
                previous_epoch is not None
                and epoch <= previous_epoch
            ):
                raise RuntimeError(
                    "Epoch history must be strictly "
                    "increasing."
                )

            if (
                previous_global_step is not None
                and global_step
                <= previous_global_step
            ):
                raise RuntimeError(
                    "Global-step history must be "
                    "strictly increasing."
                )

            previous_epoch = epoch
            previous_global_step = (
                global_step
            )

    def set_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write experiment metadata and provenance."""

        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping."
            )

        if not isinstance(
            overwrite,
            bool,
        ):
            raise TypeError(
                "overwrite must be Boolean."
            )

        if (
            self.metadata_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                "Experiment metadata already exists. "
                "Use overwrite=True to replace it."
            )

        normalized_metadata = {
            str(key): _normalize_scalar(
                value,
                f"metadata[{key!r}]",
            )
            for key, value in metadata.items()
        }

        report = {
            "protocol_version": (
                EXPERIMENT_LOGGER_PROTOCOL_VERSION
            ),
            "schema_version": (
                LOGGER_SCHEMA_VERSION
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "created_at_utc": (
                utc_timestamp()
            ),
            "metadata": (
                normalized_metadata
            ),
        }

        _atomic_write_text(
            self.metadata_path,
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

        return self.metadata_path

    def log_epoch(
        self,
        record: Mapping[str, object],
    ) -> dict[str, Any]:
        """Validate and persist one completed epoch."""

        normalized_record = (
            normalize_record(
                record
            )
        )

        record_schema = tuple(
            normalized_record
        )

        if self._fieldnames is None:
            self._fieldnames = (
                _validate_fieldnames(
                    record_schema
                )
            )

        elif record_schema != self._fieldnames:
            missing = sorted(
                set(
                    self._fieldnames
                )
                - set(
                    record_schema
                )
            )

            unexpected = sorted(
                set(
                    record_schema
                )
                - set(
                    self._fieldnames
                )
            )

            raise RuntimeError(
                "Epoch record schema mismatch. "
                f"Missing fields: {missing}; "
                f"unexpected fields: {unexpected}; "
                "field order must also remain fixed."
            )

        epoch = int(
            normalized_record[
                "epoch"
            ]
        )

        global_step = int(
            normalized_record[
                "global_step"
            ]
        )

        if (
            self.last_epoch is not None
            and epoch <= self.last_epoch
        ):
            raise RuntimeError(
                "Epoch must increase strictly. "
                f"Latest logged epoch is "
                f"{self.last_epoch}; received {epoch}."
            )

        if (
            self.last_global_step is not None
            and global_step
            <= self.last_global_step
        ):
            raise RuntimeError(
                "global_step must increase strictly. "
                f"Latest logged global step is "
                f"{self.last_global_step}; received "
                f"{global_step}."
            )

        self._rows.append(
            normalized_record
        )

        try:
            self._write_history()
            self._write_latest_record()
            self._write_state()

        except BaseException:
            self._rows.pop()
            raise

        return dict(
            normalized_record
        )

    def _write_history(
        self,
    ) -> None:
        """Write the complete epoch history atomically."""

        if self._fieldnames is None:
            return

        with tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            newline="",
        ) as temporary_file:
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=list(
                    self._fieldnames
                ),
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(
                self._rows
            )

            temporary_file.seek(
                0
            )

            content = (
                temporary_file.read()
            )

        _atomic_write_text(
            self.history_path,
            content,
        )

    def _write_latest_record(
        self,
    ) -> None:
        """Write the latest completed epoch record."""

        if not self._rows:
            return

        report = {
            "protocol_version": (
                EXPERIMENT_LOGGER_PROTOCOL_VERSION
            ),
            "schema_version": (
                LOGGER_SCHEMA_VERSION
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "updated_at_utc": (
                utc_timestamp()
            ),
            "record": dict(
                self._rows[-1]
            ),
        }

        _atomic_write_text(
            self.latest_record_path,
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    def _write_state(
        self,
    ) -> None:
        """Write logger resume state."""

        state = {
            "protocol_version": (
                EXPERIMENT_LOGGER_PROTOCOL_VERSION
            ),
            "schema_version": (
                LOGGER_SCHEMA_VERSION
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "updated_at_utc": (
                utc_timestamp()
            ),
            "fieldnames": (
                list(
                    self._fieldnames
                )
                if self._fieldnames
                is not None
                else None
            ),
            "record_count": len(
                self._rows
            ),
            "last_epoch": (
                self.last_epoch
            ),
            "last_global_step": (
                self.last_global_step
            ),
            "files": {
                "history_csv": (
                    self.history_path.name
                ),
                "latest_epoch_json": (
                    self.latest_record_path.name
                ),
                "metadata_json": (
                    self.metadata_path.name
                ),
            },
        }

        _atomic_write_text(
            self.state_path,
            json.dumps(
                state,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        """Return logger status and artifact paths."""

        return {
            "protocol_version": (
                EXPERIMENT_LOGGER_PROTOCOL_VERSION
            ),
            "schema_version": (
                LOGGER_SCHEMA_VERSION
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "output_directory": str(
                self.output_directory
            ),
            "record_count": len(
                self._rows
            ),
            "fieldnames": (
                list(
                    self._fieldnames
                )
                if self._fieldnames
                is not None
                else None
            ),
            "first_epoch": (
                int(
                    self._rows[0][
                        "epoch"
                    ]
                )
                if self._rows
                else None
            ),
            "last_epoch": (
                self.last_epoch
            ),
            "last_global_step": (
                self.last_global_step
            ),
            "artifacts": {
                "history_csv": str(
                    self.history_path
                ),
                "latest_epoch_json": str(
                    self.latest_record_path
                ),
                "metadata_json": str(
                    self.metadata_path
                ),
                "logger_state_json": str(
                    self.state_path
                ),
            },
        }


CSVExperimentLogger = ExperimentLogger


def run_experiment_logger_self_test() -> dict[str, Any]:
    """Run a CPU-only logger creation and resume test."""

    import tempfile

    fieldnames = (
        "epoch",
        "global_step",
        "learning_rate",
        "train_loss",
        "validation_loss",
        "validation_dice",
        "is_best",
    )

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        output_directory = (
            Path(
                temporary_directory
            )
            / "logs"
        )

        logger = ExperimentLogger(
            output_directory,
            experiment_id="LOGGER_SELF_TEST",
            fieldnames=fieldnames,
            resume=False,
        )

        metadata_path = (
            logger.set_metadata(
                {
                    "model_name": (
                        "synthetic_model"
                    ),
                    "seed": 42,
                    "device": "cpu",
                }
            )
        )

        logger.log_epoch(
            {
                "epoch": 0,
                "global_step": 10,
                "learning_rate": 1e-3,
                "train_loss": 0.8,
                "validation_loss": 0.7,
                "validation_dice": (
                    torch.tensor(
                        0.60
                    )
                ),
                "is_best": True,
            }
        )

        logger.log_epoch(
            {
                "epoch": 1,
                "global_step": 20,
                "learning_rate": 8e-4,
                "train_loss": 0.6,
                "validation_loss": 0.5,
                "validation_dice": (
                    np.float64(
                        0.72
                    )
                ),
                "is_best": True,
            }
        )

        initial_summary = (
            logger.summary()
        )

        resumed_logger = (
            ExperimentLogger(
                output_directory,
                experiment_id=(
                    "LOGGER_SELF_TEST"
                ),
                fieldnames=fieldnames,
                resume=True,
            )
        )

        resumed_logger.log_epoch(
            {
                "epoch": 2,
                "global_step": 30,
                "learning_rate": 6e-4,
                "train_loss": 0.5,
                "validation_loss": 0.45,
                "validation_dice": 0.75,
                "is_best": True,
            }
        )

        duplicate_epoch_rejected = False

        try:
            resumed_logger.log_epoch(
                {
                    "epoch": 2,
                    "global_step": 40,
                    "learning_rate": 5e-4,
                    "train_loss": 0.4,
                    "validation_loss": 0.4,
                    "validation_dice": 0.76,
                    "is_best": True,
                }
            )

        except RuntimeError:
            duplicate_epoch_rejected = (
                True
            )

        schema_mismatch_rejected = False

        try:
            resumed_logger.log_epoch(
                {
                    "epoch": 3,
                    "global_step": 40,
                    "learning_rate": 5e-4,
                    "train_loss": 0.4,
                    "validation_loss": 0.4,
                    "validation_dice": 0.76,
                    "is_best": True,
                    "unexpected_field": 123,
                }
            )

        except RuntimeError:
            schema_mismatch_rejected = (
                True
            )

        with resumed_logger.history_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            csv_rows = list(
                csv.DictReader(
                    input_file
                )
            )

        latest_report = json.loads(
            resumed_logger
            .latest_record_path
            .read_text(
                encoding="utf-8"
            )
        )

        state_report = json.loads(
            resumed_logger
            .state_path
            .read_text(
                encoding="utf-8"
            )
        )

        metadata_report = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        final_summary = (
            resumed_logger.summary()
        )

        artifact_exists = {
            "history": (
                resumed_logger
                .history_path
                .is_file()
            ),
            "latest": (
                resumed_logger
                .latest_record_path
                .is_file()
            ),
            "metadata": (
                resumed_logger
                .metadata_path
                .is_file()
            ),
            "state": (
                resumed_logger
                .state_path
                .is_file()
            ),
        }

    checks = {
        "initial_record_count": (
            initial_summary[
                "record_count"
            ]
            == 2
        ),
        "resume_record_count": (
            final_summary[
                "record_count"
            ]
            == 3
        ),
        "resume_last_epoch": (
            final_summary[
                "last_epoch"
            ]
            == 2
        ),
        "resume_last_global_step": (
            final_summary[
                "last_global_step"
            ]
            == 30
        ),
        "field_schema_preserved": (
            tuple(
                final_summary[
                    "fieldnames"
                ]
            )
            == fieldnames
        ),
        "csv_row_count": (
            len(
                csv_rows
            )
            == 3
        ),
        "csv_epoch_order": (
            [
                int(
                    row[
                        "epoch"
                    ]
                )
                for row in csv_rows
            ]
            == [
                0,
                1,
                2,
            ]
        ),
        "latest_epoch_correct": (
            latest_report[
                "record"
            ][
                "epoch"
            ]
            == 2
        ),
        "latest_dice_correct": (
            latest_report[
                "record"
            ][
                "validation_dice"
            ]
            == 0.75
        ),
        "state_record_count": (
            state_report[
                "record_count"
            ]
            == 3
        ),
        "metadata_preserved": (
            metadata_report[
                "metadata"
            ][
                "model_name"
            ]
            == "synthetic_model"
            and metadata_report[
                "metadata"
            ][
                "seed"
            ]
            == 42
        ),
        "duplicate_epoch_rejected": (
            duplicate_epoch_rejected
        ),
        "schema_mismatch_rejected": (
            schema_mismatch_rejected
        ),
        "all_artifacts_exist": all(
            artifact_exists.values()
        ),
        "protocol_version_correct": (
            final_summary[
                "protocol_version"
            ]
            == (
                EXPERIMENT_LOGGER_PROTOCOL_VERSION
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
            EXPERIMENT_LOGGER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "fieldnames": list(
            fieldnames
        ),
        "final_record_count": (
            final_summary[
                "record_count"
            ]
        ),
        "final_epoch": (
            final_summary[
                "last_epoch"
            ]
        ),
        "final_global_step": (
            final_summary[
                "last_global_step"
            ]
        ),
    }