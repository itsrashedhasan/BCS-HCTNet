"""
Step 01 -- Audit all datasets for corrupt files, missing masks, mismatched
image/mask counts. Uses the exact folder structures confirmed from the
user's actual downloaded data (July 2026 tree scan).

Each audit_* function returns a dict report. Nothing here modifies data --
this only reads and reports.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.io import list_valid_files, is_image_openable
from src.utils import config


def audit_isic2018(raw_dir: str) -> dict:
    """
    Expects the official folder names:
      ISIC2018_Task1-2_Training_Input / ISIC2018_Task1_Training_GroundTruth
      ISIC2018_Task1-2_Validation_Input / ISIC2018_Task1_Validation_GroundTruth
      ISIC2018_Task1-2_Test_Input / ISIC2018_Task1_Test_GroundTruth (test has no GT publicly)
    Mask naming: ISIC_0000000.jpg -> ISIC_0000000_segmentation.png
    """
    report = {"dataset": "isic2018", "splits": {}}

    split_folders = {
        "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
        "val": ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
        "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
    }

    for split, (img_folder, mask_folder) in split_folders.items():
        img_dir = os.path.join(raw_dir, img_folder)
        mask_dir = os.path.join(raw_dir, mask_folder)

        split_report = {"n_images": 0, "n_masks": 0, "missing_masks": [],
                         "corrupt_images": [], "note": None}

        if not os.path.isdir(img_dir):
            split_report["note"] = f"folder not found: {img_folder}"
            report["splits"][split] = split_report
            continue

        images = list_valid_files(img_dir, extensions=(".jpg", ".jpeg"))
        split_report["n_images"] = len(images)

        if os.path.isdir(mask_dir):
            masks = set(list_valid_files(mask_dir, extensions=(".png",)))
            split_report["n_masks"] = len(masks)
            for img_name in images:
                image_id = os.path.splitext(img_name)[0]
                expected_mask = f"{image_id}_segmentation.png"
                if expected_mask not in masks:
                    split_report["missing_masks"].append(img_name)
        else:
            split_report["note"] = "no ground truth folder (expected for official test split)"

        # spot-check first 50 images for corruption (full scan is slow; increase if needed)
        for img_name in images[:50]:
            if not is_image_openable(os.path.join(img_dir, img_name)):
                split_report["corrupt_images"].append(img_name)

        report["splits"][split] = split_report

    return report


def audit_ph2(raw_dir: str) -> dict:
    """
    Expects trainx/ (images, .bmp) and trainy/ (masks, "{ID}_lesion.bmp").
    User has a duplicate copy at PH2/ph2_dataset/trainx+trainy -- we only
    read from PH2/trainx and PH2/trainy directly.
    """
    report = {"dataset": "ph2", "n_images": 0, "n_masks": 0,
              "missing_masks": [], "corrupt_images": []}

    img_dir = os.path.join(raw_dir, "trainx")
    mask_dir = os.path.join(raw_dir, "trainy")

    if not os.path.isdir(img_dir):
        report["note"] = f"folder not found: {img_dir}"
        return report

    images = list_valid_files(img_dir, extensions=(".bmp",))
    report["n_images"] = len(images)

    if os.path.isdir(mask_dir):
        masks = set(list_valid_files(mask_dir, extensions=(".bmp",)))
        report["n_masks"] = len(masks)
        for img_name in images:
            image_id = os.path.splitext(img_name)[0]  # e.g. "IMD002"
            expected_mask = f"{image_id}_lesion.bmp"
            if expected_mask not in masks:
                report["missing_masks"].append(img_name)

    for img_name in images[:50]:
        if not is_image_openable(os.path.join(img_dir, img_name)):
            report["corrupt_images"].append(img_name)

    return report


def audit_isic2017(raw_dir: str) -> dict:
    """
    ISIC 2017 zips extract into a DOUBLE-nested folder on Windows
    (e.g. ISIC-2017_Training_Data/ISIC-2017_Training_Data/*.jpg) because
    each zip's root already contains a folder of that name.
    Data folders also mix in "_superpixels.png" files alongside the real
    .jpg images -- these are filtered out automatically since we only
    look for the .jpg extension.
    """
    report = {"dataset": "isic2017", "splits": {}}

    split_folders = {
        "train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
        "val": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
        "test": ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth"),
    }

    for split, (img_folder, mask_folder) in split_folders.items():
        img_dir = os.path.join(raw_dir, img_folder, img_folder)
        mask_dir = os.path.join(raw_dir, mask_folder, mask_folder)

        split_report = {"n_images": 0, "n_masks": 0, "missing_masks": [],
                         "corrupt_images": [], "note": None}

        if not os.path.isdir(img_dir):
            split_report["note"] = f"folder not found: {img_dir}"
            report["splits"][split] = split_report
            continue

        images = list_valid_files(img_dir, extensions=(".jpg", ".jpeg"))
        split_report["n_images"] = len(images)

        if os.path.isdir(mask_dir):
            masks = set(list_valid_files(mask_dir, extensions=(".png",)))
            split_report["n_masks"] = len(masks)
            for img_name in images:
                image_id = os.path.splitext(img_name)[0]
                expected_mask = f"{image_id}_segmentation.png"
                if expected_mask not in masks:
                    split_report["missing_masks"].append(img_name)
        else:
            split_report["note"] = f"ground truth folder not found: {mask_dir}"

        for img_name in images[:50]:
            if not is_image_openable(os.path.join(img_dir, img_name)):
                split_report["corrupt_images"].append(img_name)

        report["splits"][split] = split_report

    return report


def audit_imaplusplus(raw_dir: str) -> dict:
    """
    Expects ISIC-images/ (raw images, .jpg) and 14201693/segs/ (masks, .png,
    named "{ISIC_ID}_{annotator}_{...}.png" -- multiple masks per image).
    Also reads the provided train/val/test CSVs and metadata if present.
    """
    report = {"dataset": "imaplusplus", "n_images": 0, "n_unique_ids_in_segs": 0,
              "images_without_any_mask": [], "corrupt_images": [],
              "metadata_files_found": []}

    img_dir = os.path.join(raw_dir, "ISIC-images")
    segs_dir = os.path.join(raw_dir, "14201693", "segs")
    meta_dir = os.path.join(raw_dir, "14201693")

    if not os.path.isdir(img_dir):
        report["note"] = f"folder not found: {img_dir}"
        return report

    images = list_valid_files(img_dir, extensions=(".jpg", ".jpeg"))
    report["n_images"] = len(images)

    if os.path.isdir(segs_dir):
        seg_files = list_valid_files(segs_dir, extensions=(".png",))
        # extract the ISIC id prefix from filenames like
        # "ISIC_0000000_A04_T3_S1_....png" -> "ISIC_0000000"
        seg_ids = set()
        for f in seg_files:
            parts = f.split("_")
            if len(parts) >= 2:
                seg_ids.add(f"{parts[0]}_{parts[1]}")
        report["n_unique_ids_in_segs"] = len(seg_ids)

        for img_name in images:
            image_id = os.path.splitext(img_name)[0]
            if image_id not in seg_ids:
                report["images_without_any_mask"].append(img_name)
    else:
        report["note"] = f"segs folder not found: {segs_dir}"

    if os.path.isdir(meta_dir):
        for expected in ["img_metadata.csv", "seg_metadata.csv", "train.csv", "val.csv", "test.csv"]:
            if os.path.exists(os.path.join(meta_dir, expected)):
                report["metadata_files_found"].append(expected)

    # spot-check first 50 images for corruption
    for img_name in images[:50]:
        if not is_image_openable(os.path.join(img_dir, img_name)):
            report["corrupt_images"].append(img_name)

    return report


def run_all_audits():
    """Runs every audit that's ready, prints a summary, and saves a JSON report."""
    import json

    config.ensure_all_dirs()
    all_reports = {}

    for name, audit_fn in [
        ("isic2018", audit_isic2018),
        ("isic2017", audit_isic2017),
        ("ph2", audit_ph2),
        ("imaplusplus", audit_imaplusplus),
    ]:
        raw_dir = config.get_raw_dir(name)
        print(f"\n--- Auditing {name} ({raw_dir}) ---")
        try:
            report = audit_fn(raw_dir)
            all_reports[name] = report
            print(json.dumps(report, indent=2)[:2000])  # print a preview
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            all_reports[name] = {"error": str(e)}

    out_path = os.path.join(config.REPORTS_DIR, "audit_report.json")
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nFull report saved to: {out_path}")

    return all_reports


if __name__ == "__main__":
    run_all_audits()
