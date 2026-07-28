"""Step 03D: Build a ranked perceptual-overlap review cohort.

This CPU-only stage reads:

1. Persistent Step 02 manifests.
2. Persistent Step 03C perceptual-neighbor results.
3. Original ISIC 2018 and IMA++ images.

It then:

- derives screening limits from known same-ID control pairs;
- selects suspicious candidate-reference pairs;
- calculates confirmatory similarity metrics;
- assigns review-priority tiers;
- creates ranked CSV files;
- creates side-by-side visual contact sheets.

Important:
    This stage does not automatically exclude any image.
    Final exclusions require review of the ranked evidence.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
)
from skimage.metrics import structural_similarity

from src.data.analyze_exact_pixels import (
    resolve_image_path,
    select_unique_image_rows,
)
from src.data.analyze_overlap import (
    find_step02_manifest_directory,
    normalize_image_id,
    read_csv_rows,
)
from src.data.analyze_perceptual_hashes import (
    ahash64,
    dhash64,
    hamming_distance,
    phash64,
)
from src.utils import config


EXPECTED_CALIBRATION_PAIRS = 3568
EXPECTED_CANDIDATES = 11399
MAX_CONTACT_SHEET_PAIRS = 200
PAIRS_PER_CONTACT_SHEET = 2
ANALYSIS_SIZE = 256


def find_step03_intermediate_directory() -> Path:
    """Find the persistent Step 03 intermediate artifact directory."""

    input_root = Path("/kaggle/input")

    required_files = {
        "step03c_known_overlap_hash_calibration.csv",
        "step03c_candidate_nearest_hash_neighbors.csv",
    }

    candidates: list[Path] = []

    for neighbor_path in input_root.rglob(
        "step03c_candidate_nearest_hash_neighbors.csv"
    ):
        manifest_directory = neighbor_path.parent

        if all(
            (
                manifest_directory
                / required_file
            ).exists()
            for required_file in required_files
        ):
            candidates.append(
                manifest_directory.resolve()
            )

    candidates = sorted(set(candidates))

    preferred = [
        candidate
        for candidate in candidates
        if "bcs-hctnet-step03-intermediate"
        in str(candidate).lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not find the persistent Step 03 "
            "intermediate manifest directory."
        )

    raise RuntimeError(
        "Multiple Step 03 intermediate directories found: "
        f"{[str(path) for path in candidates]}"
    )


def integer_quantile(
    values: Sequence[int],
    quantile: float,
) -> int:
    """Return a conservative integer quantile."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "Cannot calculate a quantile from an empty list."
        )

    try:
        result = np.quantile(
            array,
            quantile,
            method="higher",
        )

    except TypeError:
        result = np.quantile(
            array,
            quantile,
            interpolation="higher",
        )

    return int(result)


def resolve_saved_path(value: str) -> Path:
    """Resolve a path stored in a persistent CSV."""

    value = value.strip()

    if not value:
        raise FileNotFoundError(
            "A required image path is empty."
        )

    if "::" in value:
        raise RuntimeError(
            "ZIP-member paths are not supported in Step 03D."
        )

    direct_path = Path(value)

    if direct_path.is_file():
        return direct_path

    kaggle_marker = "/kaggle/input/"

    if kaggle_marker in value:
        relative_value = value.split(
            kaggle_marker,
            1,
        )[1]

        rebased_path = (
            Path("/kaggle/input")
            / relative_value
        )

        if rebased_path.is_file():
            return rebased_path

    raise FileNotFoundError(
        f"Could not resolve saved image path: {value}"
    )


def load_rgb_array(
    image_path: Path,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an EXIF-corrected RGB image."""

    with Image.open(image_path) as opened_image:
        oriented = ImageOps.exif_transpose(
            opened_image
        )

        rgb_image = oriented.convert("RGB")
        rgb_image.load()

        original_size = rgb_image.size

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.uint8,
        )

    return rgb_array, original_size


def resize_for_analysis(
    rgb_array: np.ndarray,
) -> np.ndarray:
    """Resize an RGB image to the fixed analysis resolution."""

    return cv2.resize(
        rgb_array,
        (ANALYSIS_SIZE, ANALYSIS_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def normalized_histogram_correlation(
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
) -> float:
    """Calculate mean RGB histogram correlation."""

    correlations: list[float] = []

    for channel_index in range(3):
        first_histogram = cv2.calcHist(
            [first_rgb],
            [channel_index],
            None,
            [64],
            [0, 256],
        )

        second_histogram = cv2.calcHist(
            [second_rgb],
            [channel_index],
            None,
            [64],
            [0, 256],
        )

        cv2.normalize(
            first_histogram,
            first_histogram,
        )

        cv2.normalize(
            second_histogram,
            second_histogram,
        )

        correlation = cv2.compareHist(
            first_histogram,
            second_histogram,
            cv2.HISTCMP_CORREL,
        )

        correlations.append(float(correlation))

    mean_correlation = float(
        np.mean(correlations)
    )

    return max(
        0.0,
        min(
            1.0,
            (mean_correlation + 1.0) / 2.0,
        ),
    )


def orb_match_ratio(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
) -> float:
    """Calculate a conservative ORB feature-match ratio."""

    orb = cv2.ORB_create(
        nfeatures=750,
        scaleFactor=1.2,
        nlevels=8,
    )

    first_keypoints, first_descriptors = (
        orb.detectAndCompute(
            first_gray,
            None,
        )
    )

    second_keypoints, second_descriptors = (
        orb.detectAndCompute(
            second_gray,
            None,
        )
    )

    if (
        first_descriptors is None
        or second_descriptors is None
        or not first_keypoints
        or not second_keypoints
    ):
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True,
    )

    matches = matcher.match(
        first_descriptors,
        second_descriptors,
    )

    if not matches:
        return 0.0

    good_matches = [
        match
        for match in matches
        if match.distance <= 32
    ]

    denominator = max(
        1,
        min(
            len(first_keypoints),
            len(second_keypoints),
        ),
    )

    return min(
        1.0,
        len(good_matches) / denominator,
    )


def compare_images(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Calculate confirmatory image-similarity evidence."""

    candidate_rgb, candidate_size = (
        load_rgb_array(candidate_path)
    )

    reference_rgb, reference_size = (
        load_rgb_array(reference_path)
    )

    candidate_resized = resize_for_analysis(
        candidate_rgb
    )

    reference_resized = resize_for_analysis(
        reference_rgb
    )

    candidate_gray = cv2.cvtColor(
        candidate_resized,
        cv2.COLOR_RGB2GRAY,
    )

    reference_gray = cv2.cvtColor(
        reference_resized,
        cv2.COLOR_RGB2GRAY,
    )

    pixel_ssim = float(
        structural_similarity(
            candidate_gray,
            reference_gray,
            data_range=255,
        )
    )

    candidate_edges = cv2.Canny(
        candidate_gray,
        60,
        160,
    )

    reference_edges = cv2.Canny(
        reference_gray,
        60,
        160,
    )

    edge_ssim = float(
        structural_similarity(
            candidate_edges,
            reference_edges,
            data_range=255,
        )
    )

    histogram_similarity = (
        normalized_histogram_correlation(
            candidate_resized,
            reference_resized,
        )
    )

    feature_match_ratio = orb_match_ratio(
        candidate_gray,
        reference_gray,
    )

    normalized_mae = float(
        np.mean(
            np.abs(
                candidate_resized.astype(
                    np.float32
                )
                - reference_resized.astype(
                    np.float32
                )
            )
        )
        / 255.0
    )

    candidate_phash = phash64(
        candidate_gray
    )

    reference_phash = phash64(
        reference_gray
    )

    candidate_dhash = dhash64(
        candidate_gray
    )

    reference_dhash = dhash64(
        reference_gray
    )

    candidate_ahash = ahash64(
        candidate_gray
    )

    reference_ahash = ahash64(
        reference_gray
    )

    phash_distance = hamming_distance(
        candidate_phash,
        reference_phash,
    )

    dhash_distance = hamming_distance(
        candidate_dhash,
        reference_dhash,
    )

    ahash_distance = hamming_distance(
        candidate_ahash,
        reference_ahash,
    )

    combined_hash_score = (
        2 * phash_distance
        + dhash_distance
        + ahash_distance
    )

    verification_score = (
        0.45 * max(0.0, pixel_ssim)
        + 0.20 * max(0.0, edge_ssim)
        + 0.20 * histogram_similarity
        + 0.15 * feature_match_ratio
    )

    return {
        "candidate_width": candidate_size[0],
        "candidate_height": candidate_size[1],
        "reference_width": reference_size[0],
        "reference_height": reference_size[1],
        "phash_distance": phash_distance,
        "dhash_distance": dhash_distance,
        "ahash_distance": ahash_distance,
        "combined_hash_score": (
            combined_hash_score
        ),
        "pixel_ssim": round(
            pixel_ssim,
            8,
        ),
        "edge_ssim": round(
            edge_ssim,
            8,
        ),
        "histogram_similarity": round(
            histogram_similarity,
            8,
        ),
        "orb_match_ratio": round(
            feature_match_ratio,
            8,
        ),
        "normalized_mae": round(
            normalized_mae,
            8,
        ),
        "verification_score": round(
            verification_score,
            8,
        ),
    }


def write_csv(
    output_path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Write UTF-8 CSV rows."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font when available."""

    candidates = [
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/"
            "liberation2/LiberationSans-Regular.ttf"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(
                str(candidate),
                size=size,
            )

    return ImageFont.load_default()


def fit_image(
    image_path: Path,
    target_size: tuple[int, int],
) -> Image.Image:
    """Fit an image within a fixed display box."""

    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(
            opened_image
        ).convert("RGB")

        contained = ImageOps.contain(
            image,
            target_size,
            method=Image.Resampling.LANCZOS,
        )

    canvas = Image.new(
        "RGB",
        target_size,
        "black",
    )

    left = (
        target_size[0]
        - contained.width
    ) // 2

    top = (
        target_size[1]
        - contained.height
    ) // 2

    canvas.paste(
        contained,
        (left, top),
    )

    return canvas


def create_pair_panel(
    row: dict[str, object],
    rank: int,
) -> Image.Image:
    """Create one side-by-side review panel."""

    panel_width = 1100
    panel_height = 390

    panel = Image.new(
        "RGB",
        (panel_width, panel_height),
        "white",
    )

    draw = ImageDraw.Draw(panel)

    title_font = load_font(20)
    body_font = load_font(16)

    candidate_image = fit_image(
        Path(str(row["candidate_image_path"])),
        (500, 270),
    )

    reference_image = fit_image(
        Path(str(row["reference_image_path"])),
        (500, 270),
    )

    panel.paste(
        candidate_image,
        (25, 80),
    )

    panel.paste(
        reference_image,
        (575, 80),
    )

    title = (
        f"Rank {rank} | Tier {row['review_tier']} | "
        f"Candidate {row['imaplusplus_image_id']} | "
        f"ISIC 2018 {row['isic2018_image_id']}"
    )

    draw.text(
        (25, 15),
        title,
        fill="black",
        font=title_font,
    )

    metrics = (
        f"Reason: {row['screening_reasons']} | "
        f"pHash={row['phash_distance']} "
        f"dHash={row['dhash_distance']} "
        f"aHash={row['ahash_distance']} "
        f"combined={row['combined_hash_score']} | "
        f"SSIM={float(row['pixel_ssim']):.4f} "
        f"edge={float(row['edge_ssim']):.4f} "
        f"hist={float(row['histogram_similarity']):.4f} "
        f"ORB={float(row['orb_match_ratio']):.4f} "
        f"score={float(row['verification_score']):.4f}"
    )

    draw.text(
        (25, 47),
        metrics,
        fill="black",
        font=body_font,
    )

    draw.text(
        (25, 355),
        "IMA++ candidate",
        fill="black",
        font=body_font,
    )

    draw.text(
        (575, 355),
        "ISIC 2018 reference",
        fill="black",
        font=body_font,
    )

    return panel


def create_contact_sheets(
    ranked_rows: Sequence[dict[str, object]],
    output_directory: Path,
) -> list[dict[str, object]]:
    """Create visual review sheets for the top-ranked pairs."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_rows = list(
        ranked_rows[:MAX_CONTACT_SHEET_PAIRS]
    )

    index_rows: list[dict[str, object]] = []

    total_sheets = math.ceil(
        len(selected_rows)
        / PAIRS_PER_CONTACT_SHEET
    )

    for sheet_index in range(total_sheets):
        start = (
            sheet_index
            * PAIRS_PER_CONTACT_SHEET
        )

        sheet_rows = selected_rows[
            start : start + PAIRS_PER_CONTACT_SHEET
        ]

        sheet = Image.new(
            "RGB",
            (
                1100,
                390 * PAIRS_PER_CONTACT_SHEET,
            ),
            "white",
        )

        for local_index, row in enumerate(
            sheet_rows
        ):
            global_rank = (
                start + local_index + 1
            )

            panel = create_pair_panel(
                row,
                global_rank,
            )

            sheet.paste(
                panel,
                (
                    0,
                    local_index * 390,
                ),
            )

            index_rows.append(
                {
                    "rank": global_rank,
                    "contact_sheet": (
                        f"review_sheet_"
                        f"{sheet_index + 1:03d}.jpg"
                    ),
                    "panel_position": (
                        local_index + 1
                    ),
                    "imaplusplus_image_id": (
                        row[
                            "imaplusplus_image_id"
                        ]
                    ),
                    "isic2018_image_id": (
                        row[
                            "isic2018_image_id"
                        ]
                    ),
                    "review_tier": (
                        row["review_tier"]
                    ),
                }
            )

        output_path = (
            output_directory
            / (
                f"review_sheet_"
                f"{sheet_index + 1:03d}.jpg"
            )
        )

        sheet.save(
            output_path,
            quality=92,
            optimize=True,
        )

    return index_rows


def main() -> int:
    """Build the ranked Step 03D review evidence."""

    config.ensure_all_dirs()

    step02_manifest_directory = (
        find_step02_manifest_directory()
    )

    step03_manifest_directory = (
        find_step03_intermediate_directory()
    )

    calibration_path = (
        step03_manifest_directory
        / "step03c_known_overlap_hash_calibration.csv"
    )

    neighbors_path = (
        step03_manifest_directory
        / "step03c_candidate_nearest_hash_neighbors.csv"
    )

    isic_manifest_path = (
        step02_manifest_directory
        / "isic2018_all.csv"
    )

    print(
        "=== Step 03D: Ranked Perceptual Review ==="
    )

    print(
        "Step 02 manifests: "
        f"{step02_manifest_directory}"
    )

    print(
        "Step 03 intermediate: "
        f"{step03_manifest_directory}"
    )

    calibration_rows = read_csv_rows(
        calibration_path
    )

    neighbor_rows = read_csv_rows(
        neighbors_path
    )

    isic_manifest_rows = read_csv_rows(
        isic_manifest_path
    )

    if len(calibration_rows) != (
        EXPECTED_CALIBRATION_PAIRS
    ):
        raise RuntimeError(
            "Unexpected calibration-pair count: "
            f"{len(calibration_rows)}"
        )

    if len(neighbor_rows) != (
        EXPECTED_CANDIDATES
    ):
        raise RuntimeError(
            "Unexpected candidate-neighbor count: "
            f"{len(neighbor_rows)}"
        )

    calibration_phash = [
        int(row["phash_distance"])
        for row in calibration_rows
    ]

    calibration_dhash = [
        int(row["dhash_distance"])
        for row in calibration_rows
    ]

    calibration_ahash = [
        int(row["ahash_distance"])
        for row in calibration_rows
    ]

    calibration_combined = [
        (
            2 * int(row["phash_distance"])
            + int(row["dhash_distance"])
            + int(row["ahash_distance"])
        )
        for row in calibration_rows
    ]

    thresholds = {
        "phash_p99": integer_quantile(
            calibration_phash,
            0.99,
        ),
        "phash_max": max(
            calibration_phash
        ),
        "dhash_p99": integer_quantile(
            calibration_dhash,
            0.99,
        ),
        "dhash_max": max(
            calibration_dhash
        ),
        "ahash_p99": integer_quantile(
            calibration_ahash,
            0.99,
        ),
        "ahash_max": max(
            calibration_ahash
        ),
        "combined_p99": integer_quantile(
            calibration_combined,
            0.99,
        ),
        "combined_max": max(
            calibration_combined
        ),
    }

    isic_unique_rows = (
        select_unique_image_rows(
            isic_manifest_rows
        )
    )

    isic_paths: dict[str, Path] = {}

    for image_id, row in (
        isic_unique_rows.items()
    ):
        isic_paths[
            normalize_image_id(image_id)
        ] = resolve_image_path(row)

    selected_pairs: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}

    for row in neighbor_rows:
        candidate_id = normalize_image_id(
            row["imaplusplus_image_id"]
        )

        candidate_path = resolve_saved_path(
            row["imaplusplus_image_path"]
        )

        possible_references: list[
            tuple[str, str]
        ] = []

        if int(
            row["nearest_phash_distance"]
        ) <= thresholds["phash_max"]:
            possible_references.append(
                (
                    normalize_image_id(
                        row[
                            "nearest_phash_isic2018_id"
                        ]
                    ),
                    "nearest_phash_within_control_max",
                )
            )

        if int(
            row["nearest_dhash_distance"]
        ) <= thresholds["dhash_max"]:
            possible_references.append(
                (
                    normalize_image_id(
                        row[
                            "nearest_dhash_isic2018_id"
                        ]
                    ),
                    "nearest_dhash_within_control_max",
                )
            )

        if int(
            row["combined_score"]
        ) <= thresholds["combined_max"]:
            possible_references.append(
                (
                    normalize_image_id(
                        row[
                            "nearest_combined_isic2018_id"
                        ]
                    ),
                    "combined_score_within_control_max",
                )
            )

        for reference_id, reason in (
            possible_references
        ):
            if reference_id not in isic_paths:
                raise RuntimeError(
                    "ISIC 2018 reference ID is missing "
                    f"from the manifest: {reference_id}"
                )

            key = (
                candidate_id,
                reference_id,
            )

            if key not in selected_pairs:
                selected_pairs[key] = {
                    "imaplusplus_image_id": (
                        candidate_id
                    ),
                    "isic2018_image_id": (
                        reference_id
                    ),
                    "candidate_image_path": str(
                        candidate_path
                    ),
                    "reference_image_path": str(
                        isic_paths[reference_id]
                    ),
                    "screening_reasons": set(),
                }

            selected_pairs[key][
                "screening_reasons"
            ].add(reason)

    print(
        "Suspicious candidate-reference pairs: "
        f"{len(selected_pairs)}"
    )

    reviewed_rows: list[
        dict[str, object]
    ] = []

    total_pairs = len(selected_pairs)

    for index, pair in enumerate(
        selected_pairs.values(),
        start=1,
    ):
        metrics = compare_images(
            Path(
                str(
                    pair[
                        "candidate_image_path"
                    ]
                )
            ),
            Path(
                str(
                    pair[
                        "reference_image_path"
                    ]
                )
            ),
        )

        phash_distance = int(
            metrics["phash_distance"]
        )

        dhash_distance = int(
            metrics["dhash_distance"]
        )

        combined_score = int(
            metrics[
                "combined_hash_score"
            ]
        )

        if (
            phash_distance
            <= thresholds["phash_p99"]
            and dhash_distance
            <= thresholds["dhash_p99"]
            and combined_score
            <= thresholds["combined_p99"]
        ):
            review_tier = 1

        elif (
            phash_distance
            <= thresholds["phash_max"]
            and dhash_distance
            <= thresholds["dhash_max"]
            and combined_score
            <= thresholds["combined_max"]
        ):
            review_tier = 2

        else:
            review_tier = 3

        output_row = {
            "imaplusplus_image_id": (
                pair["imaplusplus_image_id"]
            ),
            "isic2018_image_id": (
                pair["isic2018_image_id"]
            ),
            "screening_reasons": ";".join(
                sorted(
                    pair[
                        "screening_reasons"
                    ]
                )
            ),
            "review_tier": review_tier,
            **metrics,
            "candidate_image_path": (
                pair["candidate_image_path"]
            ),
            "reference_image_path": (
                pair["reference_image_path"]
            ),
            "review_decision": "",
            "review_notes": "",
        }

        reviewed_rows.append(output_row)

        if (
            index % 100 == 0
            or index == total_pairs
        ):
            print(
                "  confirmatory comparison: "
                f"{index}/{total_pairs}"
            )

    reviewed_rows.sort(
        key=lambda row: (
            int(row["review_tier"]),
            -float(
                row["verification_score"]
            ),
            int(
                row["combined_hash_score"]
            ),
            str(
                row["imaplusplus_image_id"]
            ),
            str(
                row["isic2018_image_id"]
            ),
        )
    )

    for rank, row in enumerate(
        reviewed_rows,
        start=1,
    ):
        row["rank"] = rank

    manifest_output_directory = Path(
        config.MANIFEST_DIR
    )

    report_output_directory = Path(
        config.REPORTS_DIR
    )

    figure_output_directory = (
        report_output_directory.parent
        / "figures"
        / "step03d_perceptual_review"
    )

    ranked_csv_path = (
        manifest_output_directory
        / "step03d_ranked_perceptual_review.csv"
    )

    decision_template_path = (
        manifest_output_directory
        / "step03d_manual_review_decisions.csv"
    )

    fieldnames = [
        "rank",
        "imaplusplus_image_id",
        "isic2018_image_id",
        "screening_reasons",
        "review_tier",
        "phash_distance",
        "dhash_distance",
        "ahash_distance",
        "combined_hash_score",
        "pixel_ssim",
        "edge_ssim",
        "histogram_similarity",
        "orb_match_ratio",
        "normalized_mae",
        "verification_score",
        "candidate_width",
        "candidate_height",
        "reference_width",
        "reference_height",
        "candidate_image_path",
        "reference_image_path",
        "review_decision",
        "review_notes",
    ]

    write_csv(
        ranked_csv_path,
        reviewed_rows,
        fieldnames,
    )

    decision_rows = [
        {
            "rank": row["rank"],
            "imaplusplus_image_id": (
                row["imaplusplus_image_id"]
            ),
            "isic2018_image_id": (
                row["isic2018_image_id"]
            ),
            "review_tier": (
                row["review_tier"]
            ),
            "verification_score": (
                row["verification_score"]
            ),
            "review_decision": "",
            "review_notes": "",
        }
        for row in reviewed_rows
    ]

    write_csv(
        decision_template_path,
        decision_rows,
        [
            "rank",
            "imaplusplus_image_id",
            "isic2018_image_id",
            "review_tier",
            "verification_score",
            "review_decision",
            "review_notes",
        ],
    )

    contact_sheet_index = (
        create_contact_sheets(
            reviewed_rows,
            figure_output_directory,
        )
    )

    contact_sheet_index_path = (
        manifest_output_directory
        / "step03d_contact_sheet_index.csv"
    )

    write_csv(
        contact_sheet_index_path,
        contact_sheet_index,
        [
            "rank",
            "contact_sheet",
            "panel_position",
            "imaplusplus_image_id",
            "isic2018_image_id",
            "review_tier",
        ],
    )

    tier_counts: dict[str, int] = (
        defaultdict(int)
    )

    for row in reviewed_rows:
        tier_counts[
            str(row["review_tier"])
        ] += 1

    checks = {
        "calibration_count_correct": (
            len(calibration_rows)
            == EXPECTED_CALIBRATION_PAIRS
        ),
        "candidate_count_correct": (
            len(neighbor_rows)
            == EXPECTED_CANDIDATES
        ),
        "at_least_one_pair_screened": (
            bool(reviewed_rows)
        ),
        "all_selected_pairs_compared": (
            len(reviewed_rows)
            == len(selected_pairs)
        ),
        "no_automatic_exclusion": (
            all(
                not str(
                    row["review_decision"]
                ).strip()
                for row in reviewed_rows
            )
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "03D_ranked_perceptual_"
            "review_cohort"
        ),
        "threshold_source": (
            "Observed same-ID positive-control "
            "distance distributions"
        ),
        "thresholds": thresholds,
        "counts": {
            "calibration_pairs": len(
                calibration_rows
            ),
            "candidates_screened": len(
                neighbor_rows
            ),
            "candidate_reference_pairs_selected": (
                len(selected_pairs)
            ),
            "review_tier_counts": dict(
                sorted(
                    tier_counts.items()
                )
            ),
            "contact_sheet_pairs": len(
                contact_sheet_index
            ),
            "contact_sheet_files": len(
                {
                    row["contact_sheet"]
                    for row
                    in contact_sheet_index
                }
            ),
        },
        "confirmatory_metrics": {
            "pixel_ssim": True,
            "edge_ssim": True,
            "rgb_histogram_correlation": True,
            "orb_feature_match_ratio": True,
            "normalized_mae": True,
            "perceptual_hash_distances": True,
        },
        "automatic_exclusion_performed": False,
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            "ranked_review_csv": str(
                ranked_csv_path
            ),
            "manual_decision_template": str(
                decision_template_path
            ),
            "contact_sheet_index": str(
                contact_sheet_index_path
            ),
            "contact_sheet_directory": str(
                figure_output_directory
            ),
        },
        "final_clean_manifest_created": False,
        "training_allowed": False,
        "training_block_reason": (
            "Ranked suspicious pairs still require "
            "review and a final exclusion ledger."
        ),
    }

    report_path = (
        report_output_directory
        / "step03d_perceptual_review_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Step 03D Results ===")
    print(
        "Selected candidate-reference pairs : "
        f"{len(reviewed_rows)}"
    )
    print(
        "Tier 1 pairs                       : "
        f"{tier_counts.get('1', 0)}"
    )
    print(
        "Tier 2 pairs                       : "
        f"{tier_counts.get('2', 0)}"
    )
    print(
        "Tier 3 pairs                       : "
        f"{tier_counts.get('3', 0)}"
    )
    print(
        "Contact-sheet pairs generated      : "
        f"{len(contact_sheet_index)}"
    )
    print(
        "All validation checks passed       : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {ranked_csv_path}")
    print(f" - {decision_template_path}")
    print(f" - {contact_sheet_index_path}")
    print(f" - {figure_output_directory}")
    print(f" - {report_path}")

    print(
        "\nNo candidate was excluded automatically."
    )

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())