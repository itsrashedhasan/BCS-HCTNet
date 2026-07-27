"""
Image/mask read-write helpers, including filtering out the macOS "AppleDouble"
junk files (._filename, .DS_Store) that show up when a zip extracted on a Mac
gets copied to Windows. These are metadata sidecar files, never real data --
we just need to skip them everywhere we list a folder.
"""

import os
from PIL import Image


def is_junk_file(filename: str) -> bool:
    """True for macOS AppleDouble/metadata files that should be ignored."""
    return filename.startswith("._") or filename == ".DS_Store"


def list_valid_files(folder: str, extensions=None) -> list:
    """
    Returns a sorted list of real filenames in `folder`, skipping junk files.
    `extensions`: optional tuple like (".jpg", ".png") to filter by type.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = []
    for name in os.listdir(folder):
        if is_junk_file(name):
            continue
        if extensions and not name.lower().endswith(extensions):
            continue
        if os.path.isfile(os.path.join(folder, name)):
            files.append(name)
    return sorted(files)


def is_image_openable(path: str) -> bool:
    """Returns True if PIL can open and verify the image without error."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
