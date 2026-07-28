"""Step 05B: Run numerical and visual QC for derived targets.

Run from the repository root:

    python3 scripts/05b_run_target_quality_control.py

This CPU-only stage reads the persistent Step 04 split artifacts and Step 05A
target artifacts. It reproduces every target numerically and creates
stratified visual contact sheets for final review.

No persistent input artifact is modified.
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


from src.targets.quality_control import main


if __name__ == "__main__":
    raise SystemExit(main())