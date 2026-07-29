"""Step 05C: Finalize target QC and unlock model development.

Run from the repository root:

    python3 scripts/05c_finalize_target_quality_control.py

This CPU-only stage records the completed visual review, preserves the official
ISIC 2018 internal-test cohort, creates two prespecified sensitivity cohorts,
and writes the final training-readiness decision.

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


from src.targets.finalize_quality_control import main


if __name__ == "__main__":
    raise SystemExit(main())