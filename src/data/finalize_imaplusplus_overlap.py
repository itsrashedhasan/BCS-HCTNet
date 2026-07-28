"""Step 03E: Finalize clean IMA++ overlap-screened manifests.

Primary policy
--------------
Exclude:
1. Exact ISIC 2018 identifier overlaps.
2. Exact decoded-pixel overlaps.
3. Perceptual-review Tier 1 candidates.
4. Perceptual-review Tier 2 candidates.

Sensitivity policy
------------------
Exclude:
1. Exact identifier overlaps.
2. Exact decoded-pixel overlaps.
3. Perceptual-review Tier 1 candidates.

Tier 2 candidates remain in the sensitivity cohort. Tier 3 candidates remain
in both cohorts.

The perceptual exclusions are documented as conservative high-risk overlap
exclusions. They are not mislabeled as mathematically proven duplicates.

This stage reads persistent Step 02, Step 03 intermediate, and Step 03D review
artifacts from /kaggle/input. It writes non-destructive final Step 03 outputs
to /kaggle/working.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.data.analyze_exact_pixels import (
    select_unique_image_rows,
)
from src.data.analyze_overlap import (
    EXPECTED_IMAPLUSPLUS_ROWS,
    EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES,
    EXPECTED_ISIC2018_UNIQUE_IMAGES,
    find_step02_manifest_directory,
    normalize_image_id,
    read_csv_rows,
    sha256_file,
)
from src.data.build_perceptual_review import (
    find_step03_intermediate_directory,
)
from src.utils import config


EXPECTED_EXACT_ID_OVERLAP = 3568
EXPECTED_RANKED_PAIRS = 769
EXPECTED_TIER1_PAIRS = 81
EXPECTED_TIER2_PAIRS = 25
EXPECTED_TIER3_PAIRS = 663

POLICY_VERSION = "BCS-HCTNet-IMA-overlap-policy-v1"


def find_step03d_review_directory() -> Path:
    """Locate the persistent Step 03D review manifest directory."""

    input_root = Path("/kaggle/input")

    candidates: list[Path] = []

    for ranked_path in input_root.rglob(
        "step03d_ranked_perceptual_review.csv"
    ):
        directory = ranked_path.parent

        required_files = [
            directory
            / "step03d_contact_sheet_index.csv",
            directory
            / "step03d_manual_review_decisions.csv",
        ]

        if all(path.is_file() for path in required_files):
            candidates.append(directory.resolve())

    candidates = sorted(set(candidates))

    preferred = [
        candidate
        for candidate in candidates
        if "bcs-hctnet-step03d-review-artifacts"
        in str(candidate).lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not find the persistent Step 03D "
            "review manifest directory."
        )

    raise RuntimeError(
        "Multiple Step 03D review directories were found: "
        f"{[str(path) for path in candidates]}"
    )


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


def safe_float(
    value: object,
    default: float = 0.0,
) -> float:
    """Convert a CSV value to float safely."""

    text = str(value).strip()

    if not text:
        return default

    return float(text)


def aggregate_perceptual_pairs(
    ranked_rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, object]]:
    """Aggregate multiple reference pairs to one decision per IMA++ image."""

    aggregated: dict[str, dict[str, object]] = {}

    for row in ranked_rows:
        candidate_id = normalize_image_id(
            row["imaplusplus_image_id"]
        )

        reference_id = normalize_image_id(
            row["isic2018_image_id"]
        )

        tier = int(row["review_tier"])

        verification_score = safe_float(
            row["verification_score"]
        )

        combined_hash_score = int(
            float(row["combined_hash_score"])
        )

        if candidate_id not in aggregated:
            aggregated[candidate_id] = {
                "imaplusplus_image_id": candidate_id,
                "minimum_review_tier": tier,
                "pair_count": 0,
                "matching_isic2018_ids": set(),
                "screening_reasons": set(),
                "best_verification_score": (
                    verification_score
                ),
                "best_pixel_ssim": safe_float(
                    row["pixel_ssim"]
                ),
                "best_edge_ssim": safe_float(
                    row["edge_ssim"]
                ),
                "best_histogram_similarity": (
                    safe_float(
                        row[
                            "histogram_similarity"
                        ]
                    )
                ),
                "best_orb_match_ratio": (
                    safe_float(
                        row["orb_match_ratio"]
                    )
                ),
                "minimum_combined_hash_score": (
                    combined_hash_score
                ),
                "best_pair_reference_id": (
                    reference_id
                ),
            }

        aggregate = aggregated[candidate_id]

        aggregate["pair_count"] = (
            int(aggregate["pair_count"]) + 1
        )

        aggregate[
            "matching_isic2018_ids"
        ].add(reference_id)

        reasons = str(
            row.get(
                "screening_reasons",
                "",
            )
        ).split(";")

        for reason in reasons:
            reason = reason.strip()

            if reason:
                aggregate[
                    "screening_reasons"
                ].add(reason)

        aggregate["minimum_review_tier"] = min(
            int(
                aggregate[
                    "minimum_review_tier"
                ]
            ),
            tier,
        )

        aggregate[
            "minimum_combined_hash_score"
        ] = min(
            int(
                aggregate[
                    "minimum_combined_hash_score"
                ]
            ),
            combined_hash_score,
        )

        if verification_score > float(
            aggregate[
                "best_verification_score"
            ]
        ):
            aggregate[
                "best_verification_score"
            ] = verification_score

            aggregate[
                "best_pixel_ssim"
            ] = safe_float(
                row["pixel_ssim"]
            )

            aggregate[
                "best_edge_ssim"
            ] = safe_float(
                row["edge_ssim"]
            )

            aggregate[
                "best_histogram_similarity"
            ] = safe_float(
                row[
                    "histogram_similarity"
                ]
            )

            aggregate[
                "best_orb_match_ratio"
            ] = safe_float(
                row["orb_match_ratio"]
            )

            aggregate[
                "best_pair_reference_id"
            ] = reference_id

    finalized: dict[str, dict[str, object]] = {}

    for candidate_id, aggregate in aggregated.items():
        tier = int(
            aggregate["minimum_review_tier"]
        )

        if tier == 1:
            primary_action = "exclude"
            sensitivity_action = "exclude"
            decision_class = (
                "probable_near_duplicate"
            )

        elif tier == 2:
            primary_action = "exclude"
            sensitivity_action = "retain"
            decision_class = (
                "possible_high_risk_near_duplicate"
            )

        else:
            primary_action = "retain"
            sensitivity_action = "retain"
            decision_class = (
                "screened_not_excluded"
            )

        finalized[candidate_id] = {
            "imaplusplus_image_id": (
                candidate_id
            ),
            "minimum_review_tier": tier,
            "pair_count": int(
                aggregate["pair_count"]
            ),
            "matching_isic2018_ids": (
                ";".join(
                    sorted(
                        aggregate[
                            "matching_isic2018_ids"
                        ]
                    )
                )
            ),
            "screening_reasons": (
                ";".join(
                    sorted(
                        aggregate[
                            "screening_reasons"
                        ]
                    )
                )
            ),
            "best_pair_reference_id": (
                aggregate[
                    "best_pair_reference_id"
                ]
            ),
            "best_verification_score": round(
                float(
                    aggregate[
                        "best_verification_score"
                    ]
                ),
                8,
            ),
            "best_pixel_ssim": round(
                float(
                    aggregate[
                        "best_pixel_ssim"
                    ]
                ),
                8,
            ),
            "best_edge_ssim": round(
                float(
                    aggregate[
                        "best_edge_ssim"
                    ]
                ),
                8,
            ),
            "best_histogram_similarity": round(
                float(
                    aggregate[
                        "best_histogram_similarity"
                    ]
                ),
                8,
            ),
            "best_orb_match_ratio": round(
                float(
                    aggregate[
                        "best_orb_match_ratio"
                    ]
                ),
                8,
            ),
            "minimum_combined_hash_score": int(
                aggregate[
                    "minimum_combined_hash_score"
                ]
            ),
            "decision_class": decision_class,
            "primary_policy_action": (
                primary_action
            ),
            "tier1_only_sensitivity_action": (
                sensitivity_action
            ),
            "policy_version": POLICY_VERSION,
        }

    return finalized


def main() -> int:
    """Finalize overlap-screened IMA++ manifests."""

    config.ensure_all_dirs()

    step02_directory = (
        find_step02_manifest_directory()
    )

    step03_directory = (
        find_step03_intermediate_directory()
    )

    step03d_directory = (
        find_step03d_review_directory()
    )

    isic_manifest_path = (
        step02_directory
        / "isic2018_all.csv"
    )

    ima_manifest_path = (
        step02_directory
        / "imaplusplus_all.csv"
    )

    exact_pixel_ids_path = (
        step03_directory
        / "imaplusplus_exact_pixel_overlap_ids.csv"
    )

    ranked_review_path = (
        step03d_directory
        / "step03d_ranked_perceptual_review.csv"
    )

    print(
        "=== Step 03E: Finalize IMA++ "
        "Overlap-Screened Cohorts ==="
    )

    print(
        "Step 02 manifests : "
        f"{step02_directory}"
    )

    print(
        "Step 03 artifacts : "
        f"{step03_directory}"
    )

    print(
        "Step 03D artifacts: "
        f"{step03d_directory}"
    )

    isic_rows = read_csv_rows(
        isic_manifest_path
    )

    ima_rows = read_csv_rows(
        ima_manifest_path
    )

    exact_pixel_rows = read_csv_rows(
        exact_pixel_ids_path
    )

    ranked_rows = read_csv_rows(
        ranked_review_path
    )

    if len(ima_rows) != EXPECTED_IMAPLUSPLUS_ROWS:
        raise RuntimeError(
            "Unexpected IMA++ annotation-row count: "
            f"expected {EXPECTED_IMAPLUSPLUS_ROWS}, "
            f"found {len(ima_rows)}."
        )

    if len(ranked_rows) != EXPECTED_RANKED_PAIRS:
        raise RuntimeError(
            "Unexpected ranked-pair count: "
            f"expected {EXPECTED_RANKED_PAIRS}, "
            f"found {len(ranked_rows)}."
        )

    pair_tier_counts: dict[int, int] = (
        defaultdict(int)
    )

    for row in ranked_rows:
        pair_tier_counts[
            int(row["review_tier"])
        ] += 1

    expected_pair_counts = {
        1: EXPECTED_TIER1_PAIRS,
        2: EXPECTED_TIER2_PAIRS,
        3: EXPECTED_TIER3_PAIRS,
    }

    if dict(pair_tier_counts) != (
        expected_pair_counts
    ):
        raise RuntimeError(
            "Perceptual tier pair counts changed: "
            f"expected {expected_pair_counts}, "
            f"found {dict(pair_tier_counts)}."
        )

    isic_unique_rows = (
        select_unique_image_rows(
            isic_rows
        )
    )

    ima_unique_rows = (
        select_unique_image_rows(
            ima_rows
        )
    )

    if len(isic_unique_rows) != (
        EXPECTED_ISIC2018_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "Unexpected ISIC 2018 unique-image count."
        )

    if len(ima_unique_rows) != (
        EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
    ):
        raise RuntimeError(
            "Unexpected IMA++ unique-image count."
        )

    isic_ids = set(isic_unique_rows)
    ima_ids = set(ima_unique_rows)

    exact_id_overlap_ids = (
        isic_ids & ima_ids
    )

    if len(exact_id_overlap_ids) != (
        EXPECTED_EXACT_ID_OVERLAP
    ):
        raise RuntimeError(
            "Exact-ID overlap count changed: "
            f"expected {EXPECTED_EXACT_ID_OVERLAP}, "
            f"found {len(exact_id_overlap_ids)}."
        )

    exact_pixel_overlap_ids = {
        normalize_image_id(
            row["imaplusplus_image_id"]
        )
        for row in exact_pixel_rows
        if row.get(
            "imaplusplus_image_id",
            "",
        ).strip()
    }

    perceptual_decisions = (
        aggregate_perceptual_pairs(
            ranked_rows
        )
    )

    tier1_ids = {
        image_id
        for image_id, decision
        in perceptual_decisions.items()
        if int(
            decision[
                "minimum_review_tier"
            ]
        ) == 1
    }

    tier2_ids = {
        image_id
        for image_id, decision
        in perceptual_decisions.items()
        if int(
            decision[
                "minimum_review_tier"
            ]
        ) == 2
    }

    tier3_ids = {
        image_id
        for image_id, decision
        in perceptual_decisions.items()
        if int(
            decision[
                "minimum_review_tier"
            ]
        ) == 3
    }

    if exact_id_overlap_ids & (
        tier1_ids | tier2_ids | tier3_ids
    ):
        raise RuntimeError(
            "Perceptual review contains an image "
            "already excluded by exact identifier."
        )

    if exact_id_overlap_ids & (
        exact_pixel_overlap_ids
    ):
        raise RuntimeError(
            "Exact-pixel set unexpectedly overlaps "
            "the exact-ID exclusion set."
        )

    primary_excluded_ids = (
        exact_id_overlap_ids
        | exact_pixel_overlap_ids
        | tier1_ids
        | tier2_ids
    )

    sensitivity_excluded_ids = (
        exact_id_overlap_ids
        | exact_pixel_overlap_ids
        | tier1_ids
    )

    primary_retained_ids = (
        ima_ids - primary_excluded_ids
    )

    sensitivity_retained_ids = (
        ima_ids - sensitivity_excluded_ids
    )

    annotation_counts: dict[str, int] = (
        defaultdict(int)
    )

    for row in ima_rows:
        image_id = normalize_image_id(
            row["image_id"]
        )

        annotation_counts[image_id] += 1

    original_fieldnames = list(
        ima_rows[0].keys()
    )

    additional_fields = [
        "overlap_screening_status",
        "overlap_policy_version",
        "perceptual_review_tier",
    ]

    output_fieldnames = (
        original_fieldnames
        + [
            field
            for field in additional_fields
            if field not in original_fieldnames
        ]
    )

    primary_rows: list[
        dict[str, object]
    ] = []

    sensitivity_rows: list[
        dict[str, object]
    ] = []

    for original_row in ima_rows:
        image_id = normalize_image_id(
            original_row["image_id"]
        )

        decision = perceptual_decisions.get(
            image_id
        )

        tier_value = (
            str(
                decision[
                    "minimum_review_tier"
                ]
            )
            if decision is not None
            else ""
        )

        if image_id in primary_retained_ids:
            row = dict(original_row)
            row["image_id"] = image_id
            row[
                "overlap_screening_status"
            ] = (
                "retained_primary_clean_cohort"
            )
            row[
                "overlap_policy_version"
            ] = POLICY_VERSION
            row[
                "perceptual_review_tier"
            ] = tier_value

            primary_rows.append(row)

        if image_id in sensitivity_retained_ids:
            row = dict(original_row)
            row["image_id"] = image_id

            if image_id in tier2_ids:
                status = (
                    "retained_tier1_only_"
                    "sensitivity_cohort_tier2"
                )
            else:
                status = (
                    "retained_tier1_only_"
                    "sensitivity_cohort"
                )

            row[
                "overlap_screening_status"
            ] = status
            row[
                "overlap_policy_version"
            ] = POLICY_VERSION
            row[
                "perceptual_review_tier"
            ] = tier_value

            sensitivity_rows.append(row)

    exclusion_ledger: list[
        dict[str, object]
    ] = []

    for image_id in sorted(
        primary_excluded_ids
    ):
        if image_id in exact_id_overlap_ids:
            exclusion_stage = "03A"
            exclusion_class = (
                "exact_identifier_overlap"
            )
            decision_basis = (
                "IMA++ image identifier is also "
                "present in ISIC 2018."
            )
            review_tier = ""
            matching_ids = image_id
            pair_count = ""
            verification_score = ""
            pixel_ssim = ""
            edge_ssim = ""
            histogram_similarity = ""
            orb_match_ratio = ""
            combined_score = ""
            sensitivity_action = "exclude"

        elif image_id in exact_pixel_overlap_ids:
            exclusion_stage = "03B"
            exclusion_class = (
                "exact_decoded_pixel_overlap"
            )
            decision_basis = (
                "Decoded RGB dimensions and pixels "
                "match an ISIC 2018 image."
            )
            review_tier = ""
            matching_ids = ""
            pair_count = ""
            verification_score = ""
            pixel_ssim = ""
            edge_ssim = ""
            histogram_similarity = ""
            orb_match_ratio = ""
            combined_score = ""
            sensitivity_action = "exclude"

        else:
            decision = perceptual_decisions[
                image_id
            ]

            review_tier = int(
                decision[
                    "minimum_review_tier"
                ]
            )

            exclusion_stage = "03D/03E"

            if review_tier == 1:
                exclusion_class = (
                    "probable_perceptual_"
                    "near_duplicate"
                )
            else:
                exclusion_class = (
                    "possible_high_risk_"
                    "perceptual_near_duplicate"
                )

            decision_basis = (
                "Conservative exclusion based on "
                "positive-control-calibrated "
                f"perceptual review Tier {review_tier}."
            )

            matching_ids = decision[
                "matching_isic2018_ids"
            ]

            pair_count = decision[
                "pair_count"
            ]

            verification_score = decision[
                "best_verification_score"
            ]

            pixel_ssim = decision[
                "best_pixel_ssim"
            ]

            edge_ssim = decision[
                "best_edge_ssim"
            ]

            histogram_similarity = decision[
                "best_histogram_similarity"
            ]

            orb_match_ratio = decision[
                "best_orb_match_ratio"
            ]

            combined_score = decision[
                "minimum_combined_hash_score"
            ]

            sensitivity_action = decision[
                "tier1_only_sensitivity_action"
            ]

        exclusion_ledger.append(
            {
                "imaplusplus_image_id": image_id,
                "exclusion_stage": exclusion_stage,
                "exclusion_class": exclusion_class,
                "decision_basis": decision_basis,
                "review_tier": review_tier,
                "matching_isic2018_ids": (
                    matching_ids
                ),
                "candidate_reference_pair_count": (
                    pair_count
                ),
                "best_verification_score": (
                    verification_score
                ),
                "best_pixel_ssim": pixel_ssim,
                "best_edge_ssim": edge_ssim,
                "best_histogram_similarity": (
                    histogram_similarity
                ),
                "best_orb_match_ratio": (
                    orb_match_ratio
                ),
                "minimum_combined_hash_score": (
                    combined_score
                ),
                "annotation_rows_removed_primary": (
                    annotation_counts[image_id]
                ),
                "primary_policy_action": "exclude",
                "tier1_only_sensitivity_action": (
                    sensitivity_action
                ),
                "policy_version": POLICY_VERSION,
            }
        )

    perceptual_decision_rows = [
        perceptual_decisions[image_id]
        for image_id in sorted(
            perceptual_decisions
        )
    ]

    manifest_output_directory = Path(
        config.MANIFEST_DIR
    )

    report_output_directory = Path(
        config.REPORTS_DIR
    )

    manifest_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_manifest_path = (
        manifest_output_directory
        / "imaplusplus_final_clean_primary.csv"
    )

    sensitivity_manifest_path = (
        manifest_output_directory
        / (
            "imaplusplus_clean_"
            "sensitivity_tier1_only.csv"
        )
    )

    exclusion_ledger_path = (
        manifest_output_directory
        / "imaplusplus_overlap_exclusion_ledger.csv"
    )

    perceptual_decisions_path = (
        manifest_output_directory
        / (
            "imaplusplus_perceptual_"
            "candidate_decisions.csv"
        )
    )

    write_csv(
        primary_manifest_path,
        primary_rows,
        output_fieldnames,
    )

    write_csv(
        sensitivity_manifest_path,
        sensitivity_rows,
        output_fieldnames,
    )

    write_csv(
        exclusion_ledger_path,
        exclusion_ledger,
        [
            "imaplusplus_image_id",
            "exclusion_stage",
            "exclusion_class",
            "decision_basis",
            "review_tier",
            "matching_isic2018_ids",
            "candidate_reference_pair_count",
            "best_verification_score",
            "best_pixel_ssim",
            "best_edge_ssim",
            "best_histogram_similarity",
            "best_orb_match_ratio",
            "minimum_combined_hash_score",
            "annotation_rows_removed_primary",
            "primary_policy_action",
            "tier1_only_sensitivity_action",
            "policy_version",
        ],
    )

    write_csv(
        perceptual_decisions_path,
        perceptual_decision_rows,
        [
            "imaplusplus_image_id",
            "minimum_review_tier",
            "pair_count",
            "matching_isic2018_ids",
            "screening_reasons",
            "best_pair_reference_id",
            "best_verification_score",
            "best_pixel_ssim",
            "best_edge_ssim",
            "best_histogram_similarity",
            "best_orb_match_ratio",
            "minimum_combined_hash_score",
            "decision_class",
            "primary_policy_action",
            "tier1_only_sensitivity_action",
            "policy_version",
        ],
    )

    primary_annotation_rows_removed = (
        len(ima_rows) - len(primary_rows)
    )

    sensitivity_annotation_rows_removed = (
        len(ima_rows) - len(sensitivity_rows)
    )

    checks = {
        "original_unique_count_correct": (
            len(ima_ids)
            == EXPECTED_IMAPLUSPLUS_UNIQUE_IMAGES
        ),
        "exact_id_count_correct": (
            len(exact_id_overlap_ids)
            == EXPECTED_EXACT_ID_OVERLAP
        ),
        "ranked_pair_count_correct": (
            len(ranked_rows)
            == EXPECTED_RANKED_PAIRS
        ),
        "primary_unique_partition_complete": (
            len(primary_retained_ids)
            + len(primary_excluded_ids)
            == len(ima_ids)
        ),
        "sensitivity_unique_partition_complete": (
            len(sensitivity_retained_ids)
            + len(sensitivity_excluded_ids)
            == len(ima_ids)
        ),
        "primary_row_partition_complete": (
            len(primary_rows)
            + primary_annotation_rows_removed
            == len(ima_rows)
        ),
        "sensitivity_row_partition_complete": (
            len(sensitivity_rows)
            + sensitivity_annotation_rows_removed
            == len(ima_rows)
        ),
        "primary_contains_no_exact_id_overlap": (
            not (
                primary_retained_ids
                & exact_id_overlap_ids
            )
        ),
        "primary_contains_no_exact_pixel_overlap": (
            not (
                primary_retained_ids
                & exact_pixel_overlap_ids
            )
        ),
        "primary_contains_no_tier1_candidate": (
            not (
                primary_retained_ids
                & tier1_ids
            )
        ),
        "primary_contains_no_tier2_candidate": (
            not (
                primary_retained_ids
                & tier2_ids
            )
        ),
        "tier3_candidates_retained_primary": (
            tier3_ids
            <= primary_retained_ids
        ),
        "sensitivity_retains_all_tier2_candidates": (
            tier2_ids
            <= sensitivity_retained_ids
        ),
        "primary_is_subset_of_sensitivity": (
            primary_retained_ids
            <= sensitivity_retained_ids
        ),
        "ledger_has_one_row_per_primary_exclusion": (
            len(exclusion_ledger)
            == len(primary_excluded_ids)
        ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "stage": (
            "03E_finalize_imaplusplus_"
            "overlap_screening"
        ),
        "policy_version": POLICY_VERSION,
        "policy": {
            "primary_cohort_excludes": [
                "exact identifier overlaps",
                "exact decoded-pixel overlaps",
                "perceptual Tier 1 candidates",
                "perceptual Tier 2 candidates",
            ],
            "tier1_only_sensitivity_excludes": [
                "exact identifier overlaps",
                "exact decoded-pixel overlaps",
                "perceptual Tier 1 candidates",
            ],
            "tier3_action": "retain",
            "interpretation": (
                "Tier 1 and Tier 2 removals are "
                "conservative high-risk overlap "
                "exclusions, not claims that every "
                "pair is a proven exact duplicate."
            ),
        },
        "inputs": {
            "isic2018_manifest": {
                "path": str(
                    isic_manifest_path
                ),
                "sha256": sha256_file(
                    isic_manifest_path
                ),
            },
            "imaplusplus_manifest": {
                "path": str(
                    ima_manifest_path
                ),
                "sha256": sha256_file(
                    ima_manifest_path
                ),
            },
            "exact_pixel_ids": {
                "path": str(
                    exact_pixel_ids_path
                ),
                "sha256": sha256_file(
                    exact_pixel_ids_path
                ),
            },
            "ranked_perceptual_review": {
                "path": str(
                    ranked_review_path
                ),
                "sha256": sha256_file(
                    ranked_review_path
                ),
            },
        },
        "original_imaplusplus": {
            "unique_images": len(ima_ids),
            "annotation_rows": len(ima_rows),
        },
        "exclusions": {
            "exact_id_unique_images": len(
                exact_id_overlap_ids
            ),
            "exact_pixel_unique_images": len(
                exact_pixel_overlap_ids
            ),
            "tier1_unique_images": len(
                tier1_ids
            ),
            "tier2_unique_images": len(
                tier2_ids
            ),
            "tier3_unique_images_retained": len(
                tier3_ids
            ),
            "primary_total_unique_images": len(
                primary_excluded_ids
            ),
            "primary_total_annotation_rows": (
                primary_annotation_rows_removed
            ),
            "sensitivity_total_unique_images": len(
                sensitivity_excluded_ids
            ),
            "sensitivity_total_annotation_rows": (
                sensitivity_annotation_rows_removed
            ),
        },
        "primary_clean_cohort": {
            "unique_images": len(
                primary_retained_ids
            ),
            "annotation_rows": len(
                primary_rows
            ),
        },
        "tier1_only_sensitivity_cohort": {
            "unique_images": len(
                sensitivity_retained_ids
            ),
            "annotation_rows": len(
                sensitivity_rows
            ),
        },
        "perceptual_review": {
            "ranked_pairs": len(
                ranked_rows
            ),
            "unique_screened_candidates": len(
                perceptual_decisions
            ),
            "tier1_pairs": pair_tier_counts[1],
            "tier2_pairs": pair_tier_counts[2],
            "tier3_pairs": pair_tier_counts[3],
        },
        "checks": checks,
        "all_checks_passed": (
            all_checks_passed
        ),
        "outputs": {
            "primary_clean_manifest": str(
                primary_manifest_path
            ),
            "tier1_only_sensitivity_manifest": str(
                sensitivity_manifest_path
            ),
            "exclusion_ledger": str(
                exclusion_ledger_path
            ),
            "perceptual_candidate_decisions": str(
                perceptual_decisions_path
            ),
        },
        "training_allowed": False,
        "training_block_reason": (
            "Fixed development splits, derived "
            "target generation, and target QC "
            "remain incomplete."
        ),
    }

    report_path = (
        report_output_directory
        / (
            "imaplusplus_final_overlap_"
            "screening_report.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Step 03E Results ===")

    print(
        "Exact-ID exclusions                 : "
        f"{len(exact_id_overlap_ids)}"
    )

    print(
        "Exact-pixel exclusions              : "
        f"{len(exact_pixel_overlap_ids)}"
    )

    print(
        "Tier 1 unique candidate exclusions  : "
        f"{len(tier1_ids)}"
    )

    print(
        "Tier 2 unique candidate exclusions  : "
        f"{len(tier2_ids)}"
    )

    print(
        "Tier 3 unique candidates retained   : "
        f"{len(tier3_ids)}"
    )

    print(
        "Primary clean unique images         : "
        f"{len(primary_retained_ids)}"
    )

    print(
        "Primary clean annotation rows       : "
        f"{len(primary_rows)}"
    )

    print(
        "Tier-1-only sensitivity images      : "
        f"{len(sensitivity_retained_ids)}"
    )

    print(
        "Tier-1-only sensitivity rows        : "
        f"{len(sensitivity_rows)}"
    )

    print(
        "All validation checks passed        : "
        f"{all_checks_passed}"
    )

    print("\nOutputs:")
    print(f" - {primary_manifest_path}")
    print(f" - {sensitivity_manifest_path}")
    print(f" - {exclusion_ledger_path}")
    print(f" - {perceptual_decisions_path}")
    print(f" - {report_path}")

    print(
        "\nNo persistent input artifact was modified."
    )

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())