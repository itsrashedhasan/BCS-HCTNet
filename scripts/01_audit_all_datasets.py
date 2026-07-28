"""Step 01: Audit all four datasets without modifying them.

Run from the repository root:

    python scripts/01_audit_all_datasets.py

The audit automatically discovers arbitrary Kaggle nesting and reads ZIP
members in place. No manual dataset path editing is required for this step.
"""

from __future__ import annotations

import os
import sys


# Add the repository root to Python's import path.
REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.audit_dataset import run_all_audits


if __name__ == "__main__":
    run_all_audits()