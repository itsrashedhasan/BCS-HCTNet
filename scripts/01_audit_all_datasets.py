"""
Step 01 -- Audit all 4 raw datasets for corrupt files, missing masks, empty masks.

Run with:  python scripts/01_audit_all_datasets.py
(On Kaggle: run this exact code in a notebook cell instead.)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.audit_dataset import run_all_audits

if __name__ == "__main__":
    run_all_audits()
