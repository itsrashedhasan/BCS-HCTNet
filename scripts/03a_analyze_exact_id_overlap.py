"""Step 03A: Analyze exact ISIC-ID overlap between IMA++ and ISIC 2018.

Run from the repository root:

    python3 scripts/03a_analyze_exact_id_overlap.py

This stage reads the persistent Step 02 manifests from Kaggle input and writes
non-destructive overlap-analysis outputs to /kaggle/working.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.analyze_overlap import main


if __name__ == "__main__":
    raise SystemExit(main())