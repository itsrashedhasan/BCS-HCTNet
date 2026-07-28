"""Step 04: Lock official ISIC 2018 and external evaluation splits.

Run from the repository root:

    python3 scripts/04_create_fixed_splits.py

This CPU-only stage preserves the official ISIC 2018 train, validation, and
test releases. It also locks PH2, cleaned IMA++, and ISIC 2017 as
evaluation-only cohorts.

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


from src.data.split_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())