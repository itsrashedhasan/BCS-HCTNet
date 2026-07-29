"""Regression tests for baseline-stage segmentation losses.

This suite validates:

- differentiable per-image soft Dice score and loss;
- BCEWithLogits calculated independently per image;
- the shared BCE-Dice baseline objective;
- supported input shapes;
- reduction behavior;
- gradient propagation;
- component reporting;
- configuration metadata;
- invalid-input rejection;
- built-in deterministic self-tests.

These tests cover only the mask-only losses required by the five baseline
architectures. Boundary, contour, SDM, consistency, and composite losses are
tested later with the proposed BCS-HCTNet architecture.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from src.losses.bce_dice_loss import (
    BCE_DICE_LOSS_PROTOCOL_VERSION,
    BCEAndDiceLoss,
    BCEDiceLoss,
    BaselineMaskLoss,
    bce_dice_loss,
    binary_cross_entropy_per_image,
    compute_bce_dice_loss,
    prepare_binary_logits,
    run_bce_dice_loss_self_test,
)
from src.losses.dice_loss import (
    DICE_LOSS_PROTOCOL_VERSION,
    DiceLoss,
    SoftDiceLoss,
    dice_loss,
    prepare_dice_prediction,
    prepare_dice_target,
    run_dice_loss_self_test,
    soft_dice_score,
)


def test_protocol_versions_and_aliases() -> None:
    """Loss protocol identifiers and class aliases must remain stable."""

    assert (
        DICE_LOSS_PROTOCOL_VERSION
        == "BCS-HCTNet-soft-dice-loss-v1"
    )

    assert (
        BCE_DICE_LOSS_PROTOCOL_VERSION
        == "BCS-HCTNet-bce-dice-loss-v1"
    )

    assert DiceLoss is SoftDiceLoss
    assert BCEAndDiceLoss is BCEDiceLoss
    assert BaselineMaskLoss is BCEDiceLoss


@pytest.mark.parametrize(
    (
        "input_shape",
        "expected_shape",
    ),
    (
        (
            (
                5,
                7,
            ),
            (
                1,
                1,
                5,
                7,
            ),
        ),
        (
            (
                2,
                5,
                7,
            ),
            (
                2,
                1,
                5,
                7,
            ),
        ),
        (
            (
                2,
                1,
                5,
                7,
            ),
            (
                2,
                1,
                5,
                7,
            ),
        ),
    ),
)
def test_prepare_dice_target_normalizes_supported_shapes(
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Supported target shapes must normalize to BCHW format."""

    target = torch.zeros(
        input_shape,
        dtype=torch.uint8,
    )

    prepared = prepare_dice_target(
        target
    )

    assert tuple(
        prepared.shape
    ) == expected_shape

    assert prepared.dtype == torch.float32

    assert torch.equal(
        prepared,
        torch.zeros_like(
            prepared
        ),
    )


def test_prepare_dice_target_rejects_invalid_range() -> None:
    """Targets outside the unit interval must fail immediately."""

    invalid_target = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [2.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        prepare_dice_target(
            invalid_target
        )


def test_prepare_dice_prediction_logits_match_sigmoid() -> None:
    """Logit preparation must be equivalent to an explicit sigmoid."""

    logits = torch.tensor(
        [
            [
                [
                    [-2.0, 0.0],
                    [1.0, 3.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    prepared = prepare_dice_prediction(
        logits,
        from_logits=True,
    )

    expected = torch.sigmoid(
        logits
    )

    assert torch.equal(
        prepared,
        expected,
    )


def test_prepare_dice_prediction_rejects_invalid_probability() -> None:
    """Probability mode must reject values outside the unit interval."""

    probability = torch.tensor(
        [
            [
                [
                    [0.0, 0.5],
                    [1.1, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        prepare_dice_prediction(
            probability,
            from_logits=False,
        )


def test_soft_dice_perfect_and_empty_empty() -> None:
    """Perfect and empty-empty predictions must receive Dice one."""

    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    target = prediction.clone()

    scores = soft_dice_score(
        prediction,
        target,
        from_logits=False,
        smooth=1.0,
        reduction="none",
    )

    losses = dice_loss(
        prediction,
        target,
        from_logits=False,
        smooth=1.0,
        reduction="none",
    )

    assert torch.equal(
        scores,
        torch.ones_like(
            scores
        ),
    )

    assert torch.equal(
        losses,
        torch.zeros_like(
            losses
        ),
    )


def test_soft_dice_disjoint_without_smoothing() -> None:
    """Disjoint non-empty masks must receive Dice zero without smoothing."""

    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    score = soft_dice_score(
        prediction,
        target,
        from_logits=False,
        smooth=0.0,
        epsilon=1e-7,
        reduction="mean",
    )

    assert score.item() == pytest.approx(
        0.0,
        abs=0.0,
    )


def test_soft_dice_reductions() -> None:
    """None, mean, and sum reductions must agree."""

    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    scores = soft_dice_score(
        prediction,
        target,
        from_logits=False,
        reduction="none",
    )

    mean_score = soft_dice_score(
        prediction,
        target,
        from_logits=False,
        reduction="mean",
    )

    sum_score = soft_dice_score(
        prediction,
        target,
        from_logits=False,
        reduction="sum",
    )

    assert tuple(
        scores.shape
    ) == (
        2,
    )

    assert torch.allclose(
        mean_score,
        scores.mean(),
        rtol=0.0,
        atol=1e-7,
    )

    assert torch.allclose(
        sum_score,
        scores.sum(),
        rtol=0.0,
        atol=1e-7,
    )


def test_dice_loss_is_one_minus_score() -> None:
    """Dice loss must equal one minus the corresponding Dice score."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        3,
        1,
        6,
        7,
    )

    target = (
        torch.rand(
            3,
            1,
            6,
            7,
        )
        > 0.5
    ).float()

    scores = soft_dice_score(
        logits,
        target,
        from_logits=True,
        reduction="none",
    )

    losses = dice_loss(
        logits,
        target,
        from_logits=True,
        reduction="none",
    )

    assert torch.allclose(
        losses,
        1.0 - scores,
        rtol=0.0,
        atol=1e-7,
    )


def test_soft_dice_module_configuration() -> None:
    """The module wrapper must expose reproducible configuration metadata."""

    criterion = SoftDiceLoss(
        from_logits=True,
        smooth=1.0,
        epsilon=1e-7,
        reduction="mean",
    )

    configuration = (
        criterion.configuration()
    )

    assert configuration[
        "protocol_version"
    ] == DICE_LOSS_PROTOCOL_VERSION

    assert configuration[
        "name"
    ] == "soft_dice_loss"

    assert configuration[
        "from_logits"
    ] is True

    assert configuration[
        "per_image"
    ] is True

    assert configuration[
        "hard_thresholding"
    ] is False


def test_soft_dice_backpropagates() -> None:
    """Soft Dice loss must propagate finite non-zero gradients."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )

    target = (
        torch.rand(
            2,
            1,
            8,
            8,
        )
        > 0.5
    ).float()

    criterion = SoftDiceLoss()

    loss = criterion(
        logits,
        target,
    )

    assert isinstance(
        loss,
        Tensor,
    )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )

    assert loss.requires_grad

    loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(
        logits.grad
    ).all()

    assert float(
        logits.grad.abs().sum().item()
    ) > 0.0


def test_dice_shape_mismatch_rejected() -> None:
    """Prediction and target dimensions must match."""

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        dice_loss(
            torch.zeros(
                1,
                1,
                4,
                4,
            ),
            torch.zeros(
                1,
                1,
                3,
                4,
            ),
        )


@pytest.mark.parametrize(
    (
        "input_shape",
        "expected_shape",
    ),
    (
        (
            (
                5,
                7,
            ),
            (
                1,
                1,
                5,
                7,
            ),
        ),
        (
            (
                2,
                5,
                7,
            ),
            (
                2,
                1,
                5,
                7,
            ),
        ),
        (
            (
                2,
                1,
                5,
                7,
            ),
            (
                2,
                1,
                5,
                7,
            ),
        ),
    ),
)
def test_prepare_binary_logits_supported_shapes(
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Binary logits must normalize to one-channel BCHW tensors."""

    logits = torch.zeros(
        input_shape,
        dtype=torch.int64,
    )

    prepared = prepare_binary_logits(
        logits
    )

    assert tuple(
        prepared.shape
    ) == expected_shape

    assert prepared.dtype == torch.float32

    assert torch.isfinite(
        prepared
    ).all()


def test_prepare_binary_logits_rejects_multichannel() -> None:
    """The binary segmentation criterion must reject multiple channels."""

    with pytest.raises(
        ValueError,
        match="exactly one channel",
    ):
        prepare_binary_logits(
            torch.zeros(
                2,
                2,
                8,
                8,
            )
        )


def test_binary_cross_entropy_per_image_matches_pytorch() -> None:
    """Per-image BCE must match direct PyTorch pixel-loss averaging."""

    logits = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [-1.0, 2.0],
                ]
            ],
            [
                [
                    [2.0, -2.0],
                    [0.5, -0.5],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    observed = (
        binary_cross_entropy_per_image(
            logits,
            target,
        )
    )

    expected = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        .flatten(
            start_dim=1
        )
        .mean(
            dim=1
        )
    )

    assert tuple(
        observed.shape
    ) == (
        2,
    )

    assert torch.allclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-7,
    )


def test_positive_class_weight_increases_positive_loss() -> None:
    """A larger positive-class weight must increase all-positive BCE."""

    logits = torch.zeros(
        1,
        1,
        4,
        4,
    )

    target = torch.ones_like(
        logits
    )

    unweighted = (
        binary_cross_entropy_per_image(
            logits,
            target,
            pos_weight=None,
        )
    )

    weighted = (
        binary_cross_entropy_per_image(
            logits,
            target,
            pos_weight=2.0,
        )
    )

    assert weighted.item() > unweighted.item()

    assert weighted.item() == pytest.approx(
        2.0 * unweighted.item(),
        rel=1e-6,
        abs=1e-7,
    )


def test_compute_bce_dice_components_and_weighted_total() -> None:
    """The reported total must equal the configured weighted components."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        3,
        1,
        8,
        8,
    )

    target = (
        torch.rand(
            3,
            1,
            8,
            8,
        )
        > 0.5
    ).float()

    bce_weight = 0.7
    dice_weight = 1.3

    components = compute_bce_dice_loss(
        logits,
        target,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        reduction="mean",
    )

    assert tuple(
        components
    ) == (
        "total_loss",
        "bce_loss",
        "dice_loss",
    )

    assert all(
        value.ndim == 0
        for value in components.values()
    )

    assert all(
        torch.isfinite(
            value
        )
        for value in components.values()
    )

    expected_total = (
        bce_weight
        * components[
            "bce_loss"
        ]
        + dice_weight
        * components[
            "dice_loss"
        ]
    )

    tolerance = (
        10.0
        * torch.finfo(
            components[
                "total_loss"
            ].dtype
        ).eps
    )

    assert torch.allclose(
        components[
            "total_loss"
        ],
        expected_total,
        rtol=0.0,
        atol=tolerance,
    )


def test_compute_bce_dice_reductions() -> None:
    """Per-image, mean, and sum combined losses must agree."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        4,
        1,
        5,
        7,
    )

    target = (
        torch.rand(
            4,
            1,
            5,
            7,
        )
        > 0.5
    ).float()

    per_image = compute_bce_dice_loss(
        logits,
        target,
        reduction="none",
    )

    mean_components = (
        compute_bce_dice_loss(
            logits,
            target,
            reduction="mean",
        )
    )

    sum_components = (
        compute_bce_dice_loss(
            logits,
            target,
            reduction="sum",
        )
    )

    for component_name in (
        "total_loss",
        "bce_loss",
        "dice_loss",
    ):
        assert tuple(
            per_image[
                component_name
            ].shape
        ) == (
            4,
        )

        assert torch.allclose(
            mean_components[
                component_name
            ],
            per_image[
                component_name
            ].mean(),
            rtol=0.0,
            atol=1e-7,
        )

        assert torch.allclose(
            sum_components[
                component_name
            ],
            per_image[
                component_name
            ].sum(),
            rtol=0.0,
            atol=1e-6,
        )


def test_bce_dice_function_matches_total_component() -> None:
    """The functional convenience API must return the reported total."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        2,
        1,
        6,
        6,
    )

    target = (
        torch.rand(
            2,
            1,
            6,
            6,
        )
        > 0.5
    ).float()

    functional_loss = bce_dice_loss(
        logits,
        target,
        bce_weight=0.5,
        dice_weight=1.5,
        reduction="mean",
    )

    component_loss = (
        compute_bce_dice_loss(
            logits,
            target,
            bce_weight=0.5,
            dice_weight=1.5,
            reduction="mean",
        )[
            "total_loss"
        ]
    )

    assert torch.equal(
        functional_loss,
        component_loss,
    )


def test_bce_dice_module_component_and_scalar_modes() -> None:
    """The module must support trainer components and direct scalar output."""

    logits = torch.zeros(
        2,
        1,
        4,
        4,
    )

    target = torch.ones_like(
        logits
    )

    component_criterion = BCEDiceLoss(
        return_components=True,
    )

    component_output = (
        component_criterion(
            logits,
            target,
        )
    )

    assert isinstance(
        component_output,
        dict,
    )

    assert tuple(
        component_output
    ) == (
        "total_loss",
        "bce_loss",
        "dice_loss",
    )

    scalar_criterion = BCEDiceLoss(
        return_components=False,
    )

    scalar_output = scalar_criterion(
        logits,
        target,
    )

    assert isinstance(
        scalar_output,
        Tensor,
    )

    assert scalar_output.ndim == 0

    assert torch.equal(
        scalar_output,
        component_output[
            "total_loss"
        ],
    )

    configuration = (
        component_criterion.configuration()
    )

    assert configuration[
        "protocol_version"
    ] == BCE_DICE_LOSS_PROTOCOL_VERSION

    assert configuration[
        "learning_type"
    ] == "fully_supervised"

    assert configuration[
        "per_image"
    ] is True

    assert configuration[
        "from_logits"
    ] is True

    assert configuration[
        "hard_thresholding"
    ] is False

    assert configuration[
        "uses_auxiliary_targets"
    ] is False


def test_bce_dice_backpropagates() -> None:
    """The combined objective must produce finite non-zero gradients."""

    torch.manual_seed(
        42
    )

    logits = torch.randn(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )

    target = (
        torch.rand(
            2,
            1,
            8,
            8,
        )
        > 0.5
    ).float()

    criterion = BCEDiceLoss(
        return_components=True,
    )

    components = criterion(
        logits,
        target,
    )

    assert isinstance(
        components,
        dict,
    )

    total_loss = components[
        "total_loss"
    ]

    assert torch.isfinite(
        total_loss
    )

    assert total_loss.requires_grad

    total_loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(
        logits.grad
    ).all()

    assert float(
        logits.grad.abs().sum().item()
    ) > 0.0


@pytest.mark.parametrize(
    "constructor_arguments",
    (
        {
            "bce_weight": 0.0,
            "dice_weight": 0.0,
        },
        {
            "pos_weight": 0.0,
        },
        {
            "reduction": "none",
            "return_components": True,
        },
    ),
)
def test_invalid_bce_dice_configurations_rejected(
    constructor_arguments: dict[
        str,
        object,
    ],
) -> None:
    """Unsafe or trainer-incompatible configurations must fail."""

    with pytest.raises(
        ValueError
    ):
        BCEDiceLoss(
            **constructor_arguments
        )


def test_bce_dice_shape_mismatch_rejected() -> None:
    """Combined loss must reject mismatched prediction and target sizes."""

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        bce_dice_loss(
            torch.zeros(
                1,
                1,
                8,
                8,
            ),
            torch.zeros(
                1,
                1,
                7,
                8,
            ),
        )


def test_loss_self_tests_pass() -> None:
    """Both module-level deterministic self-tests must remain passing."""

    dice_result = (
        run_dice_loss_self_test()
    )

    bce_dice_result = (
        run_bce_dice_loss_self_test()
    )

    assert dice_result[
        "status"
    ] == "passed"

    assert all(
        dice_result[
            "checks"
        ].values()
    )

    assert bce_dice_result[
        "status"
    ] == "passed"

    assert all(
        bce_dice_result[
            "checks"
        ].values()
    )