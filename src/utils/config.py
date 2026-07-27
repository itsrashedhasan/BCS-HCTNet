"""
Central path configuration for BCS-HCTNet.

This is the ONE file that knows whether we're running:
  - on Kaggle (paths under /kaggle/input/... and /kaggle/working/...)
  - locally on your PC (paths under the project's own data/ and outputs/ folders)

Every other script/module should import paths from here instead of
hardcoding "data/raw/..." directly. That way the exact same code runs
in both places with zero changes.

HOW TO USE ON KAGGLE:
  1. Add each dataset as a Kaggle "Dataset" input to your notebook.
  2. Update the KAGGLE_DATASET_DIRS dict below to match the exact
     folder names Kaggle gives your inputs (visible in the notebook's
     right-hand "Data" panel, under /kaggle/input/<name>/...).
  3. Everything else (manifests, splits, derived targets, checkpoints,
     outputs) is written to /kaggle/working/ automatically, which is
     what you download / commit as a new dataset version at the end
     of a session.
"""

import os

# ---------------------------------------------------------------------------
# 1. Detect environment
# ---------------------------------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle/input")

# ---------------------------------------------------------------------------
# 2. Project root
# ---------------------------------------------------------------------------
if IS_KAGGLE:
    PROJECT_ROOT = "/kaggle/working"
else:
    # this file lives at <PROJECT_ROOT>/src/utils/config.py
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

# ---------------------------------------------------------------------------
# 3. Raw dataset locations
# ---------------------------------------------------------------------------
if IS_KAGGLE:
    # EDIT THESE to match your actual Kaggle input dataset folder names.
    # Check the right-hand "Data" panel in your Kaggle notebook for the
    # exact names once you've added each dataset as an input.
    KAGGLE_DATASET_DIRS = {
        "isic2018": "/kaggle/input/isic2018-task1",
        "ph2": "/kaggle/input/ph2-dataset",
        "imaplusplus": "/kaggle/input/imaplusplus",
        "isic2017": "/kaggle/input/isic2017-task1",
    }
    RAW_DIR = None  # not used directly on Kaggle; see get_raw_dir() below
else:
    RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    KAGGLE_DATASET_DIRS = {}


def get_raw_dir(dataset_name: str) -> str:
    """
    Returns the raw data folder for a given dataset name:
    one of "isic2018", "ph2", "imaplusplus", "isic2017".
    """
    if IS_KAGGLE:
        if dataset_name not in KAGGLE_DATASET_DIRS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Update KAGGLE_DATASET_DIRS in src/utils/config.py."
            )
        return KAGGLE_DATASET_DIRS[dataset_name]
    else:
        return os.path.join(RAW_DIR, dataset_name)


# ---------------------------------------------------------------------------
# 4. Everything we generate (manifests, splits, derived targets, outputs)
#    always lives under PROJECT_ROOT, whether that's /kaggle/working or local.
# ---------------------------------------------------------------------------
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
DERIVED_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
XAI_DIR = os.path.join(OUTPUT_DIR, "xai")

ALL_GENERATED_DIRS = [
    MANIFEST_DIR, SPLIT_DIR, DERIVED_DIR, CACHE_DIR,
    OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PREDICTIONS_DIR,
    TABLES_DIR, FIGURES_DIR, REPORTS_DIR, XAI_DIR,
]


def ensure_all_dirs():
    """Create every generated directory if it doesn't already exist.
    Safe to call at the start of any script."""
    for d in ALL_GENERATED_DIRS:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


if __name__ == "__main__":
    # Quick sanity check you can run to confirm paths look right:
    #   python src/utils/config.py
    print(f"IS_KAGGLE      = {IS_KAGGLE}")
    print(f"PROJECT_ROOT   = {PROJECT_ROOT}")
    print(f"MANIFEST_DIR   = {MANIFEST_DIR}")
    print(f"CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    if not IS_KAGGLE:
        print(f"RAW_DIR        = {RAW_DIR}")
        for name in ["isic2018", "ph2", "imaplusplus", "isic2017"]:
            print(f"  {name}: {get_raw_dir(name)}")
