"""Step 05A: Generate fixed-resolution ISIC 2018 derived targets.

Run from the repository root:

    python3 scripts/05a_generate_isic_targets.py

This CPU-only stage generates resized masks, contour targets, boundary bands,
and signed-distance maps for all locked ISIC 2018 train, validation, and
internal-test samples.

Persistent Step 04 inputs are read-only. Generated outputs are written under
/kaggle/working and must be persisted as a Kaggle Dataset after validation.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(
    0,
    REPOSITORY_ROOT,
)


from src.targets.generate_targets import main


if __name__ == "__main__":
    raise SystemExit(main())