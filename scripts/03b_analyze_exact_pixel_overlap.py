"""Step 03B: Analyze exact decoded-pixel overlap.

Run from the repository root:

    python3 scripts/03b_analyze_exact_pixel_overlap.py

This launcher reads the persistent Step 02 manifests from Kaggle input and
writes non-destructive Step 03B outputs to /kaggle/working.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.analyze_exact_pixels import main


if __name__ == "__main__":
    raise SystemExit(main())