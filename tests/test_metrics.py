"""Regression tests for segmentation evaluation metrics.

These tests verify the common evaluation protocol used by every baseline
model and the proposed BCS-HCTNet model.

Covered behavior
----------------
- overlap metrics with known binary examples;
- logits and probability equivalence;
- empty-mask conventions;
- ISIC thresholded Jaccard at the exact 0.65 cutoff;
- boundary F1 tolerance behavior;
- HD95 and ASSD conventions;
- one-empty-mask finite distance penalty;
- per-image metric tracker accumulation;
- duplicate sample rejection;
- CSV and JSON export.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from src.metrics.boundary import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    assd_score,
    boundary_f1_score,
    compute_boundary_metrics,
    hd95_score,
    image_diagonal_penalty,
)
from src.metrics.metric_tracker import (
    BOUNDED_METRIC_NAMES,
    DISTANCE_METRIC_NAMES,
    METRIC_NAMES,
    SegmentationMetricTracker,
)
from src.metrics.overlap import (
    binary_confusion_counts,
    compute_overlap_metrics,
    dice_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    prepare_binary_prediction,
    prepare_binary_target,
    recall_score,
    specificity_score,
)
from src.metrics.thresholded_jaccard import (
    DEFAULT_JACCARD_QUALITY_THRESHOLD,
    apply_jaccard_quality_threshold,
    compute_thresholded_jaccard_details,
    thresholded_jaccard_score,
)


def build_overlap_examples() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Create three small examples with analytically known metrics."""

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
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    probability = torch.tensor(
        [
            [
                [
                    [0.9, 0.8],
                    [0.1, 0.2],
                ]
            ],
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ],
            [
                [
                    [0.9, 0.8],
                    [0.1, 0.2],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    return probability, target


def build_boundary_examples() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Create exact, shifted, empty, and one-empty examples."""

    target = torch.zeros(
        4,
        1,
        8,
        8,
        dtype=torch.float32,
    )

    probability = torch.full(
        (
            4,
            1,
            8,
            8,
        ),
        fill_value=0.1,
        dtype=torch.float32,
    )

    # Exact 4 x 4 square.
    target[
        0,
        0,
        2:6,
        2:6,
    ] = 1.0

    probability[
        0,
        0,
        2:6,
        2:6,
    ] = 0.9

    # Same square shifted one pixel right.
    target[
        1,
        0,
        2:6,
        2:6,
    ] = 1.0

    probability[
        1,
        0,
        2:6,
        3:7,
    ] = 0.9

    # Sample 2 remains empty-empty.

    # Sample 3 has a target but an empty prediction.
    target[
        3,
        0,
        2:6,
        2:6,
    ] = 1.0

    return probability, target


def test_overlap_metrics_known_values() -> None:
    """Overlap metrics must match analytically known values."""

    probability, target = (
        build_overlap_examples()
    )

    metrics = compute_overlap_metrics(
        probability,
        target,
        from_logits=False,
        reduction="none",
    )

    expected = {
        "dice": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "iou": torch.tensor(
            [
                1.0,
                1.0,
                1.0 / 3.0,
            ],
            dtype=torch.float64,
        ),
        "precision": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "recall": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "specificity": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
        "accuracy": torch.tensor(
            [
                1.0,
                1.0,
                0.5,
            ],
            dtype=torch.float64,
        ),
    }

    assert tuple(
        metrics
    ) == tuple(
        expected
    )

    for name in expected:
        torch.testing.assert_close(
            metrics[name],
            expected[name],
            atol=1e-12,
            rtol=0.0,
        )


def test_overlap_metric_convenience_functions() -> None:
    """Individual metric functions must match the combined function."""

    probability, target = (
        build_overlap_examples()
    )

    combined = compute_overlap_metrics(
        probability,
        target,
        from_logits=False,
        reduction="none",
    )

    individual = {
        "dice": dice_score(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
        "iou": iou_score(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
        "precision": precision_score(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
        "recall": recall_score(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
        "specificity": specificity_score(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
        "accuracy": pixel_accuracy(
            probability,
            target,
            from_logits=False,
            reduction="none",
        ),
    }

    for name in combined:
        torch.testing.assert_close(
            individual[name],
            combined[name],
            atol=0.0,
            rtol=0.0,
        )


def test_logits_and_probabilities_are_equivalent() -> None:
    """Equivalent logits and probabilities must produce identical metrics."""

    probability, target = (
        build_overlap_examples()
    )

    logits = torch.logit(
        probability.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    )

    probability_metrics = (
        compute_overlap_metrics(
            probability,
            target,
            from_logits=False,
            reduction="none",
        )
    )

    logit_metrics = (
        compute_overlap_metrics(
            logits,
            target,
            from_logits=True,
            reduction="none",
        )
    )

    for name in probability_metrics:
        assert torch.equal(
            probability_metrics[name],
            logit_metrics[name],
        )


def test_confusion_counts_known_values() -> None:
    """Binary confusion counts must be correct per image."""

    probability, target = (
        build_overlap_examples()
    )

    counts = binary_confusion_counts(
        probability,
        target,
        from_logits=False,
    )

    expected = {
        "true_positive": [
            2.0,
            0.0,
            1.0,
        ],
        "false_positive": [
            0.0,
            0.0,
            1.0,
        ],
        "false_negative": [
            0.0,
            0.0,
            1.0,
        ],
        "true_negative": [
            2.0,
            4.0,
            1.0,
        ],
    }

    for name, values in expected.items():
        assert torch.equal(
            counts[name],
            torch.tensor(
                values,
                dtype=torch.float64,
            ),
        )


def test_empty_prediction_and_target_are_perfect() -> None:
    """The approved empty-empty convention must remain stable."""

    probability = torch.zeros(
        2,
        1,
        8,
        8,
        dtype=torch.float32,
    )

    target = torch.zeros_like(
        probability
    )

    metrics = compute_overlap_metrics(
        probability,
        target,
        from_logits=False,
        reduction="none",
    )

    for values in metrics.values():
        assert torch.equal(
            values,
            torch.ones(
                2,
                dtype=torch.float64,
            ),
        )


def test_invalid_probability_range_is_rejected() -> None:
    """Probability inputs outside [0, 1] must not be accepted."""

    prediction = torch.tensor(
        [
            [
                [
                    [1.2],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    target = torch.zeros_like(
        prediction
    )

    with pytest.raises(
        ValueError,
        match="Probability predictions",
    ):
        compute_overlap_metrics(
            prediction,
            target,
            from_logits=False,
        )


def test_invalid_target_range_is_rejected() -> None:
    """Ground-truth values outside [0, 1] must not be accepted."""

    prediction = torch.zeros(
        1,
        1,
        2,
        2,
        dtype=torch.float32,
    )

    target = torch.full_like(
        prediction,
        fill_value=255.0,
    )

    with pytest.raises(
        ValueError,
        match="Target mask values",
    ):
        compute_overlap_metrics(
            prediction,
            target,
            from_logits=False,
        )


def test_binary_input_shape_normalization() -> None:
    """Two-dimensional masks must normalize to [1, 1, H, W]."""

    mask = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    prediction = prepare_binary_prediction(
        mask,
        from_logits=False,
    )

    target = prepare_binary_target(
        mask
    )

    assert tuple(
        prediction.shape
    ) == (
        1,
        1,
        2,
        2,
    )

    assert tuple(
        target.shape
    ) == (
        1,
        1,
        2,
        2,
    )

    assert torch.equal(
        prediction,
        target,
    )


def test_thresholded_jaccard_exact_cutoff() -> None:
    """IoU equal to 0.65 must be retained, not zeroed."""

    target = torch.ones(
        3,
        1,
        4,
        5,
        dtype=torch.float32,
    )

    probability = torch.full(
        (
            3,
            1,
            4,
            5,
        ),
        fill_value=0.1,
        dtype=torch.float32,
    )

    probability[
        0
    ].reshape(
        -1
    )[:13] = 0.9

    probability[
        1
    ].reshape(
        -1
    )[:12] = 0.9

    probability[
        2
    ].reshape(
        -1
    )[:20] = 0.9

    result = thresholded_jaccard_score(
        probability,
        target,
        from_logits=False,
        reduction="none",
    )

    expected = torch.tensor(
        [
            0.65,
            0.0,
            1.0,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        result,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_thresholded_jaccard_details() -> None:
    """Detailed thresholded-Jaccard output must report pass statistics."""

    ordinary = torch.tensor(
        [
            0.2,
            0.65,
            0.8,
        ],
        dtype=torch.float64,
    )

    thresholded = (
        apply_jaccard_quality_threshold(
            ordinary,
            quality_threshold=0.65,
        )
    )

    torch.testing.assert_close(
        thresholded,
        torch.tensor(
            [
                0.0,
                0.65,
                0.8,
            ],
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=0.0,
    )

    probability = torch.tensor(
        [
            [
                [
                    [0.9, 0.1],
                    [0.1, 0.1],
                ]
            ],
            [
                [
                    [0.1, 0.1],
                    [0.1, 0.1],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
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

    details = (
        compute_thresholded_jaccard_details(
            probability,
            target,
            from_logits=False,
        )
    )

    assert details[
        "number_of_images"
    ] == 2

    assert details[
        "number_passing_threshold"
    ] == 2

    assert details[
        "pass_rate"
    ] == 1.0


def test_official_threshold_constant() -> None:
    """The approved ISIC quality threshold must remain 0.65."""

    assert (
        DEFAULT_JACCARD_QUALITY_THRESHOLD
        == 0.65
    )


def test_identical_boundaries_have_perfect_scores() -> None:
    """Identical non-empty masks must have perfect boundary agreement."""

    probability, target = (
        build_boundary_examples()
    )

    metrics = compute_boundary_metrics(
        probability[:1],
        target[:1],
        from_logits=False,
        boundary_tolerance=0.0,
        reduction="none",
    )

    assert metrics[
        "boundary_f1"
    ].item() == 1.0

    assert metrics[
        "hd95"
    ].item() == 0.0

    assert metrics[
        "assd"
    ].item() == 0.0


def test_boundary_tolerance_changes_shifted_result() -> None:
    """A one-pixel tolerance must accept a one-pixel boundary shift."""

    probability, target = (
        build_boundary_examples()
    )

    exact_f1 = boundary_f1_score(
        probability[1:2],
        target[1:2],
        from_logits=False,
        tolerance=0.0,
        reduction="none",
    )

    tolerant_f1 = boundary_f1_score(
        probability[1:2],
        target[1:2],
        from_logits=False,
        tolerance=1.0,
        reduction="none",
    )

    assert exact_f1.item() < 1.0
    assert tolerant_f1.item() == 1.0


def test_shifted_boundaries_have_positive_distance() -> None:
    """A shifted segmentation must have positive HD95 and ASSD."""

    probability, target = (
        build_boundary_examples()
    )

    hd95 = hd95_score(
        probability[1:2],
        target[1:2],
        from_logits=False,
        reduction="none",
    )

    assd = assd_score(
        probability[1:2],
        target[1:2],
        from_logits=False,
        reduction="none",
    )

    assert hd95.item() > 0.0
    assert assd.item() > 0.0


def test_boundary_empty_empty_convention() -> None:
    """Empty prediction and target must have perfect finite boundary scores."""

    probability, target = (
        build_boundary_examples()
    )

    metrics = compute_boundary_metrics(
        probability[2:3],
        target[2:3],
        from_logits=False,
        reduction="none",
    )

    assert metrics[
        "boundary_f1"
    ].item() == 1.0

    assert metrics[
        "hd95"
    ].item() == 0.0

    assert metrics[
        "assd"
    ].item() == 0.0


def test_boundary_one_empty_penalty() -> None:
    """Exactly one empty mask must receive the finite diagonal penalty."""

    probability, target = (
        build_boundary_examples()
    )

    metrics = compute_boundary_metrics(
        probability[3:4],
        target[3:4],
        from_logits=False,
        reduction="none",
    )

    expected_penalty = (
        image_diagonal_penalty(
            (
                8,
                8,
            )
        )
    )

    assert metrics[
        "boundary_f1"
    ].item() == 0.0

    assert metrics[
        "hd95"
    ].item() == pytest.approx(
        expected_penalty,
        abs=1e-12,
    )

    assert metrics[
        "assd"
    ].item() == pytest.approx(
        expected_penalty,
        abs=1e-12,
    )


def test_boundary_metric_defaults() -> None:
    """The approved boundary tolerance must remain two pixels."""

    assert (
        DEFAULT_BOUNDARY_TOLERANCE_PIXELS
        == 2.0
    )


def test_metric_tracker_accumulates_per_image_rows() -> None:
    """Tracker updates must preserve sample order and metric completeness."""

    probability, target = (
        build_boundary_examples()
    )

    tracker = SegmentationMetricTracker(
        from_logits=False,
        split_name="validation",
        dataset_name="synthetic",
        boundary_tolerance_pixels=1.0,
    )

    tracker.update(
        prediction=probability[:2],
        target=target[:2],
        sample_ids=[
            "sample_001",
            "sample_002",
        ],
    )

    tracker.update(
        prediction=probability[2:],
        target=target[2:],
        sample_ids=[
            "sample_003",
            "sample_004",
        ],
    )

    rows = tracker.rows()

    assert len(
        tracker
    ) == 4

    assert tracker.sample_ids == (
        "sample_001",
        "sample_002",
        "sample_003",
        "sample_004",
    )

    assert len(
        rows
    ) == 4

    for row in rows:
        for metric_name in METRIC_NAMES:
            assert metric_name in row
            assert math.isfinite(
                float(
                    row[
                        metric_name
                    ]
                )
            )

        for metric_name in (
            BOUNDED_METRIC_NAMES
        ):
            assert (
                0.0
                <= float(
                    row[
                        metric_name
                    ]
                )
                <= 1.0
            )

        for metric_name in (
            DISTANCE_METRIC_NAMES
        ):
            assert (
                float(
                    row[
                        metric_name
                    ]
                )
                >= 0.0
            )


def test_metric_tracker_rejects_duplicate_samples() -> None:
    """The same sample must not be counted twice."""

    probability, target = (
        build_overlap_examples()
    )

    tracker = SegmentationMetricTracker(
        from_logits=False,
    )

    tracker.update(
        prediction=probability[:1],
        target=target[:1],
        sample_ids=[
            "duplicate_sample",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="already evaluated",
    ):
        tracker.update(
            prediction=probability[:1],
            target=target[:1],
            sample_ids=[
                "duplicate_sample",
            ],
        )


def test_metric_tracker_summary() -> None:
    """Dataset summary must aggregate per-image rows."""

    probability, target = (
        build_overlap_examples()
    )

    tracker = SegmentationMetricTracker(
        from_logits=False,
        split_name="validation",
        dataset_name="synthetic",
    )

    tracker.update(
        prediction=probability,
        target=target,
        sample_ids=[
            "sample_a",
            "sample_b",
            "sample_c",
        ],
    )

    summary = tracker.summary()

    assert summary[
        "number_of_images"
    ] == 3

    assert summary[
        "dataset"
    ] == "synthetic"

    assert summary[
        "split"
    ] == "validation"

    assert tuple(
        summary[
            "metrics"
        ]
    ) == METRIC_NAMES

    for metric_name in METRIC_NAMES:
        metric_summary = summary[
            "metrics"
        ][
            metric_name
        ]

        assert metric_summary[
            "count"
        ] == 3

        for statistic_name in (
            "mean",
            "standard_deviation",
            "median",
            "minimum",
            "maximum",
        ):
            assert math.isfinite(
                float(
                    metric_summary[
                        statistic_name
                    ]
                )
            )


def test_metric_tracker_exports(
    tmp_path: Path,
) -> None:
    """Tracker must export reproducible CSV and JSON artifacts."""

    probability, target = (
        build_overlap_examples()
    )

    tracker = SegmentationMetricTracker(
        from_logits=False,
        split_name="validation",
        dataset_name="synthetic",
    )

    tracker.update(
        prediction=probability,
        target=target,
        sample_ids=[
            "sample_a",
            "sample_b",
            "sample_c",
        ],
    )

    paths = tracker.export(
        tmp_path,
        file_prefix="pytest",
        additional_metadata={
            "model_name": "synthetic_model",
            "seed": 42,
        },
    )

    csv_path = paths[
        "per_image_csv"
    ]

    summary_path = paths[
        "summary_json"
    ]

    assert csv_path.is_file()
    assert summary_path.is_file()

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        rows = list(
            csv.DictReader(
                input_file
            )
        )

    report = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(
        rows
    ) == 3

    assert [
        row[
            "sample_id"
        ]
        for row in rows
    ] == [
        "sample_a",
        "sample_b",
        "sample_c",
    ]

    assert report[
        "summary"
    ][
        "number_of_images"
    ] == 3

    assert report[
        "additional_metadata"
    ][
        "model_name"
    ] == "synthetic_model"

    assert report[
        "additional_metadata"
    ][
        "seed"
    ] == 42


def test_empty_tracker_cannot_be_summarized() -> None:
    """An empty tracker must fail rather than reporting misleading values."""

    tracker = SegmentationMetricTracker()

    with pytest.raises(
        RuntimeError,
        match="empty",
    ):
        tracker.summary()