"""Step 02: Build manifests for all four datasets.

Run from the repository root:

    python3 scripts/02_build_all_manifests.py

This launcher calls the validated manifest builder in
src/data/build_manifest.py.
"""

from __future__ import annotations

import os
import sys


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPOSITORY_ROOT)


from src.data.build_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())