"""Step 03D: Build the ranked perceptual-overlap review cohort.

Run from the repository root:

    python3 scripts/03d_build_perceptual_review.py

This CPU-only stage reads persistent Step 02 and Step 03 intermediate
artifacts from Kaggle input. It writes ranked review CSVs and contact-sheet
figures to /kaggle/working.

No image is excluded automatically.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.build_perceptual_review import main


if __name__ == "__main__":
    raise SystemExit(main())