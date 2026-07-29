"""Learning-rate scheduler construction for BCS-HCTNet experiments.

Supported scheduler policies
----------------------------
- ``cosine``:
    Linear warm-up followed by cosine annealing.
- ``step``:
    Reduce the learning rate by a fixed factor every configured number
    of epochs.
- ``plateau``:
    Reduce the learning rate when a monitored validation metric stops
    improving.
- ``none``:
    Keep the optimizer learning rate unchanged.

The approved E00 protocol uses cosine annealing. This module nevertheless
supports the other policies so baselines, ablations, and future experiments
can use one common scheduler interface.

Scheduler stepping
------------------
``epoch`` schedulers:
    Step once after every completed training epoch.

``metric`` schedulers:
    Step after validation using the monitored validation metric.

The custom warm-up cosine scheduler stores all required state for exact
checkpoint restoration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.optim import Optimizer


SCHEDULER_PROTOCOL_VERSION = (
    "BCS-HCTNet-learning-rate-scheduler-v1"
)

SchedulerStepMode = Literal[
    "epoch",
    "metric",
    "none",
]


def _require_optimizer(
    optimizer: object,
) -> Optimizer:
    """Require a PyTorch optimizer."""

    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise TypeError(
            "optimizer must be a "
            "torch.optim.Optimizer."
        )

    if not optimizer.param_groups:
        raise ValueError(
            "optimizer must contain at least "
            "one parameter group."
        )

    return optimizer


def _require_positive_integer(
    value: object,
    context: str,
) -> int:
    """Require a positive integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{context} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{context} must be positive."
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
    ):
        raise TypeError(
            f"{context} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{context} must be non-negative."
        )

    return value


def _require_positive_number(
    value: object,
    context: str,
) -> float:
    """Require a finite positive number."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric."
        ) from error

    if (
        not math.isfinite(number)
        or number <= 0.0
    ):
        raise ValueError(
            f"{context} must be positive "
            "and finite."
        )

    return number


def _require_nonnegative_number(
    value: object,
    context: str,
) -> float:
    """Require a finite non-negative number."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{context} must be numeric."
        )

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{context} must be numeric."
        ) from error

    if (
        not math.isfinite(number)
        or number < 0.0
    ):
        raise ValueError(
            f"{context} must be non-negative "
            "and finite."
        )

    return number


def _require_probability_open_upper(
    value: object,
    context: str,
) -> float:
    """Require a finite number in the interval (0, 1)."""

    number = _require_positive_number(
        value,
        context,
    )

    if number >= 1.0:
        raise ValueError(
            f"{context} must be less than 1."
        )

    return number


def _require_factor(
    value: object,
    context: str,
) -> float:
    """Require a finite factor in the interval (0, 1]."""

    number = _require_positive_number(
        value,
        context,
    )

    if number > 1.0:
        raise ValueError(
            f"{context} must not exceed 1."
        )

    return number


def _normalize_scheduler_name(
    name: object,
) -> str:
    """Normalize supported scheduler names and aliases."""

    normalized = str(
        name
    ).strip().lower().replace(
        "-",
        "_",
    )

    aliases = {
        "cosine": "cosine",
        "cosine_annealing": "cosine",
        "warmup_cosine": "cosine",
        "linear_warmup_cosine": "cosine",
        "step": "step",
        "step_lr": "step",
        "plateau": "plateau",
        "reduce_on_plateau": "plateau",
        "reduce_lr_on_plateau": "plateau",
        "none": "none",
        "constant": "none",
        "disabled": "none",
    }

    if normalized not in aliases:
        raise ValueError(
            "Unsupported scheduler name "
            f"{name!r}. Supported values are "
            "'cosine', 'step', 'plateau', "
            "and 'none'."
        )

    return aliases[
        normalized
    ]


class WarmupCosineScheduler:
    """Linear warm-up followed by cosine learning-rate annealing.

    Epoch numbering starts at zero. The optimizer is immediately assigned
    the learning rate for epoch zero during initialization.

    Calling ``step()`` advances to the next epoch. An explicit epoch may
    also be supplied for deterministic restoration or testing.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int = 0,
        minimum_learning_rate: float = 0.0,
        warmup_start_factor: float = 0.1,
    ) -> None:
        """Initialize the scheduler."""

        self.optimizer = _require_optimizer(
            optimizer
        )

        self.total_epochs = (
            _require_positive_integer(
                total_epochs,
                "total_epochs",
            )
        )

        self.warmup_epochs = (
            _require_nonnegative_integer(
                warmup_epochs,
                "warmup_epochs",
            )
        )

        if (
            self.warmup_epochs
            >= self.total_epochs
        ):
            raise ValueError(
                "warmup_epochs must be smaller "
                "than total_epochs."
            )

        self.minimum_learning_rate = (
            _require_nonnegative_number(
                minimum_learning_rate,
                "minimum_learning_rate",
            )
        )

        self.warmup_start_factor = (
            _require_factor(
                warmup_start_factor,
                "warmup_start_factor",
            )
        )

        self.base_learning_rates = [
            _require_positive_number(
                parameter_group[
                    "lr"
                ],
                (
                    "optimizer parameter-group "
                    "learning rate"
                ),
            )
            for parameter_group
            in self.optimizer.param_groups
        ]

        for base_learning_rate in (
            self.base_learning_rates
        ):
            if (
                self.minimum_learning_rate
                > base_learning_rate
            ):
                raise ValueError(
                    "minimum_learning_rate cannot "
                    "exceed any optimizer base "
                    "learning rate."
                )

        self.current_epoch = 0

        self._last_learning_rates = (
            self._learning_rates_for_epoch(
                self.current_epoch
            )
        )

        self._apply_learning_rates(
            self._last_learning_rates
        )

    def _factor_for_epoch(
        self,
        epoch: int,
    ) -> float:
        """Calculate the warm-up or cosine factor."""

        if (
            self.warmup_epochs > 0
            and epoch < self.warmup_epochs
        ):
            if self.warmup_epochs == 1:
                return 1.0

            warmup_progress = (
                epoch
                / float(
                    self.warmup_epochs - 1
                )
            )

            return (
                self.warmup_start_factor
                + (
                    1.0
                    - self.warmup_start_factor
                )
                * warmup_progress
            )

        cosine_epochs = (
            self.total_epochs
            - self.warmup_epochs
        )

        if cosine_epochs <= 1:
            return 0.0

        cosine_epoch = (
            epoch
            - self.warmup_epochs
        )

        cosine_progress = (
            cosine_epoch
            / float(
                cosine_epochs - 1
            )
        )

        cosine_progress = min(
            max(
                cosine_progress,
                0.0,
            ),
            1.0,
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * cosine_progress
            )
        )

    def _learning_rates_for_epoch(
        self,
        epoch: int,
    ) -> list[float]:
        """Calculate every parameter-group learning rate."""

        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError(
                "epoch must be a non-negative "
                "integer."
            )

        bounded_epoch = min(
            epoch,
            self.total_epochs - 1,
        )

        if (
            self.warmup_epochs > 0
            and bounded_epoch
            < self.warmup_epochs
        ):
            factor = self._factor_for_epoch(
                bounded_epoch
            )

            return [
                (
                    base_learning_rate
                    * factor
                )
                for base_learning_rate
                in self.base_learning_rates
            ]

        cosine_factor = (
            self._factor_for_epoch(
                bounded_epoch
            )
        )

        return [
            (
                self.minimum_learning_rate
                + (
                    base_learning_rate
                    - self.minimum_learning_rate
                )
                * cosine_factor
            )
            for base_learning_rate
            in self.base_learning_rates
        ]

    def _apply_learning_rates(
        self,
        learning_rates: list[float],
    ) -> None:
        """Assign learning rates to optimizer parameter groups."""

        if len(
            learning_rates
        ) != len(
            self.optimizer.param_groups
        ):
            raise RuntimeError(
                "Learning-rate count does not "
                "match optimizer parameter groups."
            )

        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            parameter_group[
                "lr"
            ] = float(
                learning_rate
            )

    def step(
        self,
        epoch: int | None = None,
    ) -> list[float]:
        """Advance or explicitly set the scheduled epoch."""

        if epoch is None:
            next_epoch = (
                self.current_epoch + 1
            )

        else:
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 0
            ):
                raise ValueError(
                    "epoch must be None or a "
                    "non-negative integer."
                )

            next_epoch = epoch

        self.current_epoch = min(
            next_epoch,
            self.total_epochs - 1,
        )

        self._last_learning_rates = (
            self._learning_rates_for_epoch(
                self.current_epoch
            )
        )

        self._apply_learning_rates(
            self._last_learning_rates
        )

        return self.get_last_lr()

    def get_last_lr(
        self,
    ) -> list[float]:
        """Return the current optimizer learning rates."""

        return [
            float(
                parameter_group[
                    "lr"
                ]
            )
            for parameter_group
            in self.optimizer.param_groups
        ]

    @property
    def last_epoch(
        self,
    ) -> int:
        """Provide PyTorch-scheduler-compatible epoch metadata."""

        return self.current_epoch

    def state_dict(
        self,
    ) -> dict[str, Any]:
        """Return serializable scheduler state."""

        return {
            "protocol_version": (
                SCHEDULER_PROTOCOL_VERSION
            ),
            "scheduler_type": (
                "warmup_cosine"
            ),
            "total_epochs": (
                self.total_epochs
            ),
            "warmup_epochs": (
                self.warmup_epochs
            ),
            "minimum_learning_rate": (
                self.minimum_learning_rate
            ),
            "warmup_start_factor": (
                self.warmup_start_factor
            ),
            "base_learning_rates": list(
                self.base_learning_rates
            ),
            "current_epoch": (
                self.current_epoch
            ),
            "last_learning_rates": list(
                self._last_learning_rates
            ),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> None:
        """Restore scheduler state and optimizer learning rates."""

        if not isinstance(
            state_dict,
            Mapping,
        ):
            raise TypeError(
                "Scheduler state must be "
                "a mapping."
            )

        required_keys = {
            "protocol_version",
            "scheduler_type",
            "total_epochs",
            "warmup_epochs",
            "minimum_learning_rate",
            "warmup_start_factor",
            "base_learning_rates",
            "current_epoch",
            "last_learning_rates",
        }

        missing = sorted(
            required_keys
            - set(
                state_dict
            )
        )

        if missing:
            raise KeyError(
                "Scheduler state is missing "
                f"keys: {missing}."
            )

        if (
            state_dict[
                "protocol_version"
            ]
            != SCHEDULER_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Scheduler protocol version "
                "does not match."
            )

        if (
            state_dict[
                "scheduler_type"
            ]
            != "warmup_cosine"
        ):
            raise RuntimeError(
                "Scheduler state type does "
                "not match warmup cosine."
            )

        expected_values = {
            "total_epochs": (
                self.total_epochs
            ),
            "warmup_epochs": (
                self.warmup_epochs
            ),
            "minimum_learning_rate": (
                self.minimum_learning_rate
            ),
            "warmup_start_factor": (
                self.warmup_start_factor
            ),
        }

        for name, expected_value in (
            expected_values.items()
        ):
            observed_value = state_dict[
                name
            ]

            if observed_value != expected_value:
                raise RuntimeError(
                    f"Scheduler state {name} "
                    "does not match the current "
                    "scheduler configuration. "
                    f"Expected {expected_value!r}, "
                    f"found {observed_value!r}."
                )

        base_learning_rates = [
            float(value)
            for value in state_dict[
                "base_learning_rates"
            ]
        ]

        if (
            base_learning_rates
            != self.base_learning_rates
        ):
            raise RuntimeError(
                "Scheduler base learning rates "
                "do not match."
            )

        current_epoch = (
            _require_nonnegative_integer(
                state_dict[
                    "current_epoch"
                ],
                "current_epoch",
            )
        )

        last_learning_rates = [
            float(value)
            for value in state_dict[
                "last_learning_rates"
            ]
        ]

        expected_learning_rates = (
            self._learning_rates_for_epoch(
                current_epoch
            )
        )

        if len(
            last_learning_rates
        ) != len(
            expected_learning_rates
        ):
            raise RuntimeError(
                "Saved scheduler learning-rate "
                "count is invalid."
            )

        for observed, expected in zip(
            last_learning_rates,
            expected_learning_rates,
            strict=True,
        ):
            if not math.isclose(
                observed,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    "Saved scheduler learning "
                    "rates are inconsistent with "
                    "the saved epoch."
                )

        self.current_epoch = min(
            current_epoch,
            self.total_epochs - 1,
        )

        self._last_learning_rates = (
            expected_learning_rates
        )

        self._apply_learning_rates(
            self._last_learning_rates
        )


@dataclass(frozen=True)
class SchedulerBundle:
    """Scheduler plus its stepping protocol."""

    scheduler: object | None
    name: str
    step_mode: SchedulerStepMode
    monitor: str | None
    configuration: dict[str, Any]

    def state_dict(
        self,
    ) -> dict[str, Any] | None:
        """Return the underlying scheduler state."""

        if self.scheduler is None:
            return None

        state_function = getattr(
            self.scheduler,
            "state_dict",
            None,
        )

        if not callable(
            state_function
        ):
            raise TypeError(
                "Scheduler does not provide "
                "state_dict()."
            )

        state = state_function()

        if not isinstance(
            state,
            Mapping,
        ):
            raise TypeError(
                "Scheduler state_dict() must "
                "return a mapping."
            )

        return dict(
            state
        )

    def architecture_summary(
        self,
    ) -> dict[str, Any]:
        """Return scheduler metadata."""

        return {
            "protocol_version": (
                SCHEDULER_PROTOCOL_VERSION
            ),
            "name": self.name,
            "step_mode": (
                self.step_mode
            ),
            "monitor": self.monitor,
            "enabled": (
                self.scheduler is not None
            ),
            "configuration": dict(
                self.configuration
            ),
        }


def build_scheduler(
    optimizer: Optimizer,
    configuration: Mapping[str, Any] | None,
    *,
    total_epochs: int,
) -> SchedulerBundle:
    """Build a scheduler from an experiment configuration."""

    resolved_optimizer = (
        _require_optimizer(
            optimizer
        )
    )

    resolved_total_epochs = (
        _require_positive_integer(
            total_epochs,
            "total_epochs",
        )
    )

    scheduler_configuration = (
        dict(configuration)
        if configuration is not None
        else {
            "name": "none",
        }
    )

    scheduler_name = (
        _normalize_scheduler_name(
            scheduler_configuration.get(
                "name",
                "none",
            )
        )
    )

    if scheduler_name == "none":
        return SchedulerBundle(
            scheduler=None,
            name="none",
            step_mode="none",
            monitor=None,
            configuration={
                "name": "none",
            },
        )

    if scheduler_name == "cosine":
        warmup_epochs = (
            _require_nonnegative_integer(
                scheduler_configuration.get(
                    "warmup_epochs",
                    0,
                ),
                "scheduler.warmup_epochs",
            )
        )

        minimum_learning_rate = (
            _require_nonnegative_number(
                scheduler_configuration.get(
                    "minimum_learning_rate",
                    scheduler_configuration.get(
                        "min_lr",
                        0.0,
                    ),
                ),
                (
                    "scheduler."
                    "minimum_learning_rate"
                ),
            )
        )

        warmup_start_factor = (
            _require_factor(
                scheduler_configuration.get(
                    "warmup_start_factor",
                    0.1,
                ),
                (
                    "scheduler."
                    "warmup_start_factor"
                ),
            )
        )

        scheduler = WarmupCosineScheduler(
            resolved_optimizer,
            total_epochs=(
                resolved_total_epochs
            ),
            warmup_epochs=(
                warmup_epochs
            ),
            minimum_learning_rate=(
                minimum_learning_rate
            ),
            warmup_start_factor=(
                warmup_start_factor
            ),
        )

        return SchedulerBundle(
            scheduler=scheduler,
            name="cosine",
            step_mode="epoch",
            monitor=None,
            configuration={
                "name": "cosine",
                "total_epochs": (
                    resolved_total_epochs
                ),
                "warmup_epochs": (
                    warmup_epochs
                ),
                "minimum_learning_rate": (
                    minimum_learning_rate
                ),
                "warmup_start_factor": (
                    warmup_start_factor
                ),
            },
        )

    if scheduler_name == "step":
        step_size = (
            _require_positive_integer(
                scheduler_configuration.get(
                    "step_size",
                    30,
                ),
                "scheduler.step_size",
            )
        )

        gamma = _require_probability_open_upper(
            scheduler_configuration.get(
                "gamma",
                0.1,
            ),
            "scheduler.gamma",
        )

        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                resolved_optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        )

        return SchedulerBundle(
            scheduler=scheduler,
            name="step",
            step_mode="epoch",
            monitor=None,
            configuration={
                "name": "step",
                "step_size": (
                    step_size
                ),
                "gamma": gamma,
            },
        )

    if scheduler_name == "plateau":
        mode = str(
            scheduler_configuration.get(
                "mode",
                "max",
            )
        ).strip().lower()

        if mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "scheduler.mode must be "
                "'min' or 'max'."
            )

        factor = (
            _require_probability_open_upper(
                scheduler_configuration.get(
                    "factor",
                    0.5,
                ),
                "scheduler.factor",
            )
        )

        patience = (
            _require_nonnegative_integer(
                scheduler_configuration.get(
                    "patience",
                    10,
                ),
                "scheduler.patience",
            )
        )

        threshold = (
            _require_nonnegative_number(
                scheduler_configuration.get(
                    "threshold",
                    1e-4,
                ),
                "scheduler.threshold",
            )
        )

        minimum_learning_rate = (
            _require_nonnegative_number(
                scheduler_configuration.get(
                    "minimum_learning_rate",
                    scheduler_configuration.get(
                        "min_lr",
                        0.0,
                    ),
                ),
                (
                    "scheduler."
                    "minimum_learning_rate"
                ),
            )
        )

        monitor = str(
            scheduler_configuration.get(
                "monitor",
                (
                    "validation_loss"
                    if mode == "min"
                    else "validation_dice"
                ),
            )
        ).strip()

        if not monitor:
            raise ValueError(
                "scheduler.monitor cannot "
                "be empty."
            )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                resolved_optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=(
                    minimum_learning_rate
                ),
            )
        )

        return SchedulerBundle(
            scheduler=scheduler,
            name="plateau",
            step_mode="metric",
            monitor=monitor,
            configuration={
                "name": "plateau",
                "mode": mode,
                "factor": factor,
                "patience": patience,
                "threshold": threshold,
                "minimum_learning_rate": (
                    minimum_learning_rate
                ),
                "monitor": monitor,
            },
        )

    raise AssertionError(
        "Unreachable scheduler branch."
    )


def step_scheduler(
    scheduler_bundle: SchedulerBundle,
    *,
    metric: float | None = None,
) -> list[float] | None:
    """Step a scheduler according to its approved protocol."""

    if not isinstance(
        scheduler_bundle,
        SchedulerBundle,
    ):
        raise TypeError(
            "scheduler_bundle must be a "
            "SchedulerBundle."
        )

    scheduler = (
        scheduler_bundle.scheduler
    )

    if scheduler is None:
        return None

    step_function = getattr(
        scheduler,
        "step",
        None,
    )

    if not callable(
        step_function
    ):
        raise TypeError(
            "Scheduler does not provide step()."
        )

    if (
        scheduler_bundle.step_mode
        == "metric"
    ):
        if metric is None:
            raise ValueError(
                "A validation metric is required "
                "for the plateau scheduler."
            )

        metric_value = float(
            metric
        )

        if not math.isfinite(
            metric_value
        ):
            raise ValueError(
                "Scheduler metric must be finite."
            )

        step_function(
            metric_value
        )

    elif (
        scheduler_bundle.step_mode
        == "epoch"
    ):
        if metric is not None:
            metric_value = float(
                metric
            )

            if not math.isfinite(
                metric_value
            ):
                raise ValueError(
                    "Ignored scheduler metric "
                    "must still be finite."
                )

        step_function()

    else:
        raise RuntimeError(
            "Invalid scheduler step mode: "
            f"{scheduler_bundle.step_mode!r}."
        )

    optimizer = getattr(
        scheduler,
        "optimizer",
        None,
    )

    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise RuntimeError(
            "Scheduler does not expose a valid "
            "optimizer."
        )

    learning_rates = [
        float(
            parameter_group[
                "lr"
            ]
        )
        for parameter_group
        in optimizer.param_groups
    ]

    if not all(
        math.isfinite(
            learning_rate
        )
        and learning_rate >= 0.0
        for learning_rate
        in learning_rates
    ):
        raise RuntimeError(
            "Scheduler produced invalid "
            "learning rates."
        )

    return learning_rates


def current_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    """Return current optimizer learning rates."""

    resolved_optimizer = (
        _require_optimizer(
            optimizer
        )
    )

    learning_rates = [
        float(
            parameter_group[
                "lr"
            ]
        )
        for parameter_group
        in resolved_optimizer.param_groups
    ]

    if not all(
        math.isfinite(
            learning_rate
        )
        and learning_rate >= 0.0
        for learning_rate
        in learning_rates
    ):
        raise RuntimeError(
            "Optimizer contains invalid "
            "learning rates."
        )

    return learning_rates


def run_scheduler_self_test() -> dict[str, Any]:
    """Run deterministic CPU scheduler and restoration tests."""

    torch.manual_seed(
        42
    )

    model = torch.nn.Linear(
        3,
        1,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=5,
        warmup_epochs=2,
        minimum_learning_rate=0.01,
        warmup_start_factor=0.5,
    )

    observed_learning_rates = [
        current_learning_rates(
            optimizer
        )[0]
    ]

    for _ in range(4):
        scheduler.step()

        observed_learning_rates.append(
            current_learning_rates(
                optimizer
            )[0]
        )

    expected_learning_rates = [
        0.05,
        0.1,
        0.1,
        0.055,
        0.01,
    ]

    schedule_correct = all(
        math.isclose(
            observed,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for observed, expected in zip(
            observed_learning_rates,
            expected_learning_rates,
            strict=True,
        )
    )

    restore_model_a = torch.nn.Linear(
        3,
        1,
    )

    restore_optimizer_a = (
        torch.optim.AdamW(
            restore_model_a.parameters(),
            lr=0.001,
        )
    )

    restore_scheduler_a = (
        WarmupCosineScheduler(
            restore_optimizer_a,
            total_epochs=8,
            warmup_epochs=2,
            minimum_learning_rate=1e-5,
            warmup_start_factor=0.2,
        )
    )

    restore_scheduler_a.step()
    restore_scheduler_a.step()
    restore_scheduler_a.step()

    saved_state = (
        restore_scheduler_a.state_dict()
    )

    restore_model_b = torch.nn.Linear(
        3,
        1,
    )

    restore_optimizer_b = (
        torch.optim.AdamW(
            restore_model_b.parameters(),
            lr=0.001,
        )
    )

    restore_scheduler_b = (
        WarmupCosineScheduler(
            restore_optimizer_b,
            total_epochs=8,
            warmup_epochs=2,
            minimum_learning_rate=1e-5,
            warmup_start_factor=0.2,
        )
    )

    restore_scheduler_b.load_state_dict(
        saved_state
    )

    restored_epoch_matches = (
        restore_scheduler_b.current_epoch
        == restore_scheduler_a.current_epoch
    )

    restored_lr_matches = all(
        math.isclose(
            first,
            second,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for first, second in zip(
            restore_scheduler_a.get_last_lr(),
            restore_scheduler_b.get_last_lr(),
            strict=True,
        )
    )

    restore_scheduler_a.step()
    restore_scheduler_b.step()

    continued_lr_matches = all(
        math.isclose(
            first,
            second,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for first, second in zip(
            restore_scheduler_a.get_last_lr(),
            restore_scheduler_b.get_last_lr(),
            strict=True,
        )
    )

    cosine_model = torch.nn.Linear(
        2,
        1,
    )

    cosine_optimizer = (
        torch.optim.SGD(
            cosine_model.parameters(),
            lr=0.01,
        )
    )

    cosine_bundle = build_scheduler(
        cosine_optimizer,
        {
            "name": "cosine",
            "warmup_epochs": 1,
            "minimum_learning_rate": (
                1e-5
            ),
            "warmup_start_factor": 0.1,
        },
        total_epochs=10,
    )

    step_model = torch.nn.Linear(
        2,
        1,
    )

    step_optimizer = torch.optim.SGD(
        step_model.parameters(),
        lr=0.01,
    )

    step_bundle = build_scheduler(
        step_optimizer,
        {
            "name": "step",
            "step_size": 3,
            "gamma": 0.5,
        },
        total_epochs=10,
    )

    plateau_model = torch.nn.Linear(
        2,
        1,
    )

    plateau_optimizer = (
        torch.optim.SGD(
            plateau_model.parameters(),
            lr=0.01,
        )
    )

    plateau_bundle = build_scheduler(
        plateau_optimizer,
        {
            "name": "plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
            "monitor": "validation_dice",
        },
        total_epochs=10,
    )

    none_model = torch.nn.Linear(
        2,
        1,
    )

    none_optimizer = torch.optim.SGD(
        none_model.parameters(),
        lr=0.01,
    )

    none_bundle = build_scheduler(
        none_optimizer,
        {
            "name": "none",
        },
        total_epochs=10,
    )

    cosine_step_result = step_scheduler(
        cosine_bundle
    )

    step_step_result = step_scheduler(
        step_bundle
    )

    plateau_step_result = step_scheduler(
        plateau_bundle,
        metric=0.75,
    )

    none_step_result = step_scheduler(
        none_bundle
    )

    checks = {
        "warmup_cosine_schedule": (
            schedule_correct
        ),
        "initial_warmup_learning_rate": (
            math.isclose(
                observed_learning_rates[0],
                0.05,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "warmup_reaches_base_lr": (
            math.isclose(
                observed_learning_rates[1],
                0.1,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "cosine_reaches_minimum_lr": (
            math.isclose(
                observed_learning_rates[-1],
                0.01,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "state_epoch_restored": (
            restored_epoch_matches
        ),
        "state_lr_restored": (
            restored_lr_matches
        ),
        "restored_schedule_continues": (
            continued_lr_matches
        ),
        "cosine_bundle_valid": (
            cosine_bundle.name
            == "cosine"
            and cosine_bundle.step_mode
            == "epoch"
            and cosine_bundle.scheduler
            is not None
        ),
        "step_bundle_valid": (
            step_bundle.name
            == "step"
            and step_bundle.step_mode
            == "epoch"
            and step_bundle.scheduler
            is not None
        ),
        "plateau_bundle_valid": (
            plateau_bundle.name
            == "plateau"
            and plateau_bundle.step_mode
            == "metric"
            and plateau_bundle.monitor
            == "validation_dice"
        ),
        "none_bundle_valid": (
            none_bundle.name
            == "none"
            and none_bundle.step_mode
            == "none"
            and none_bundle.scheduler
            is None
        ),
        "cosine_step_returns_lr": (
            isinstance(
                cosine_step_result,
                list,
            )
            and len(
                cosine_step_result
            )
            == 1
        ),
        "step_scheduler_returns_lr": (
            isinstance(
                step_step_result,
                list,
            )
            and len(
                step_step_result
            )
            == 1
        ),
        "plateau_step_returns_lr": (
            isinstance(
                plateau_step_result,
                list,
            )
            and len(
                plateau_step_result
            )
            == 1
        ),
        "none_step_returns_none": (
            none_step_result is None
        ),
        "protocol_version_present": (
            cosine_bundle
            .architecture_summary()[
                "protocol_version"
            ]
            == SCHEDULER_PROTOCOL_VERSION
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
            SCHEDULER_PROTOCOL_VERSION
        ),
        "checks": checks,
        "expected_learning_rates": (
            expected_learning_rates
        ),
        "observed_learning_rates": (
            observed_learning_rates
        ),
        "restored_epoch": (
            restore_scheduler_b.current_epoch
        ),
        "restored_learning_rates": (
            restore_scheduler_b.get_last_lr()
        ),
        "scheduler_bundles": {
            "cosine": (
                cosine_bundle
                .architecture_summary()
            ),
            "step": (
                step_bundle
                .architecture_summary()
            ),
            "plateau": (
                plateau_bundle
                .architecture_summary()
            ),
            "none": (
                none_bundle
                .architecture_summary()
            ),
        },
    }