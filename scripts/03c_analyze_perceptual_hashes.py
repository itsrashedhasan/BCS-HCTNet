"""Step 03C: Calibrate and screen perceptual image hashes.

Run from the repository root:

    python3 scripts/03c_analyze_perceptual_hashes.py

This is a CPU-only, non-destructive analysis. It reads persistent Step 02
manifests from Kaggle input and writes temporary results to Kaggle working.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.analyze_perceptual_hashes import main


if __name__ == "__main__":
    raise SystemExit(main())