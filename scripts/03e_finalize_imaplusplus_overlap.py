"""Step 03E: Finalize overlap-screened IMA++ cohorts.

Run from the repository root:

    python3 scripts/03e_finalize_imaplusplus_overlap.py

This CPU-only stage reads persistent Step 02, Step 03 intermediate, and
Step 03D review artifacts from Kaggle input.

It creates:

- the primary conservative clean IMA++ cohort;
- a Tier-1-only sensitivity cohort;
- the complete overlap-exclusion ledger;
- aggregated perceptual-review decisions.

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


from src.data.finalize_imaplusplus_overlap import main


if __name__ == "__main__":
    raise SystemExit(main())