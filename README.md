# BCS-HCTNet

Boundary-Reliability-Conditioned Hybrid CNN–Transformer with Contour–Distance
Shape Consistency for Trustworthy Skin Lesion Segmentation.

Trained on ISIC 2018 only. Evaluated (frozen, no retraining) on PH2,
IMA++ non-overlap subset, and ISIC 2017.

## Status

This is a **skeleton**. Folder structure and file names are final; most file
contents are placeholders with a docstring describing what will go there.
Files get filled in one at a time, in the order listed in `scripts/`.

## Where things run

| Folder | Purpose | Runs on |
|---|---|---|
| `data/raw/` | Your 4 raw datasets (never committed to git) | Kaggle input / local disk |
| `src/` | All reusable code (models, losses, data loading, eval) | Anywhere |
| `scripts/00-06` | Data audit, manifests, splits, target generation | CPU is enough |
| `scripts/07-11` | Training and ablation | **GPU (Kaggle)** |
| `scripts/12-14` | Subgroup analysis, stats, final tables | CPU is enough |
| `outputs/` | Checkpoints, logs, tables, figures | Generated, not committed |

`src/utils/config.py` automatically detects whether it's running on Kaggle
(`/kaggle/input` exists) or locally, and points all paths to the right place.
Nothing else needs to change between environments.

## Getting this onto GitHub (no command line needed)

1. Download **GitHub Desktop** (github.com/apps/desktop) and sign in with
   your GitHub account.
2. On github.com, create a new empty repository (no README, no .gitignore —
   this project already has them).
3. In GitHub Desktop: `File > Clone Repository`, pick the repo you just made.
4. Copy every file/folder from this project into the cloned folder on your
   PC (File Explorer, drag and drop).
5. In GitHub Desktop you'll see all the new files listed. Type a commit
   message like "initial skeleton", click **Commit to main**, then
   **Push origin**.

That's it — no `git` commands typed anywhere.

## Using this on Kaggle

1. Create a new Kaggle Notebook.
2. Settings (right panel) → Accelerator → GPU (T4 x2 or P100).
3. First code cell:
   ```
   !git clone https://github.com/<your-username>/<your-repo>.git
   %cd <your-repo>
   !pip install -r requirements.txt
   ```
4. Add each dataset as a Notebook input (Add Data → search or upload your
   own private dataset). Note the exact folder name Kaggle gives it under
   `/kaggle/input/...`.
5. Open `src/utils/config.py` and update `KAGGLE_DATASET_DIRS` to match
   those exact folder names.
6. From here on, each `scripts/NN_*.py` step gets run as a notebook cell,
   e.g.:
   ```
   !python scripts/01_audit_all_datasets.py
   ```

## Session limits on Kaggle

Kaggle GPU sessions are capped (~9-12h per session, ~30 GPU-hours/week).
Training scripts checkpoint every epoch to `outputs/checkpoints/`. To
resume across sessions:
1. At the end of a session, "Save Version" so `outputs/` becomes part of
   the notebook's output.
2. Next session, add your own previous notebook output as an input dataset.
3. Point the training script at the last checkpoint to resume instead of
   starting over.

## Full step order

See `scripts/00_check_environment.py` through `scripts/14_export_final_tables.py` —
run in numeric order. Each one's docstring explains what it does.
