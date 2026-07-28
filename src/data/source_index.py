"""Discover normal files and ZIP members under Kaggle input directories.

Dataset code must not rely on fixed Kaggle dataset names or nesting depths.
This module creates a read-only index of all available files.

It never extracts, modifies, renames, or deletes dataset files.
"""

from __future__ import annotations

import os
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Sequence

from PIL import Image


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_junk_file(name: str) -> bool:
    """Return True for operating-system metadata and hidden junk files."""

    base_name = os.path.basename(name.rstrip("/"))

    return (
        base_name.startswith("._")
        or base_name
        in {
            ".DS_Store",
            "Thumbs.db",
            "__MACOSX",
        }
    )


@dataclass(frozen=True)
class SourceFile:
    """Represent a normal file or a file stored inside a ZIP archive."""

    kind: str
    container: str
    member: str = ""

    @property
    def virtual_path(self) -> str:
        """Return a readable path for reports."""

        if self.kind == "zip":
            return f"{self.container}::{self.member}"

        return self.container

    @property
    def name(self) -> str:
        """Return only the filename."""

        source_path = (
            self.member
            if self.kind == "zip"
            else self.container
        )

        return os.path.basename(source_path)

    @property
    def suffix(self) -> str:
        """Return the lowercase file extension."""

        return Path(self.name).suffix.lower()

    def to_dict(self) -> dict:
        """Convert the reference into JSON-compatible information."""

        result = asdict(self)
        result["virtual_path"] = self.virtual_path

        return result


def default_scan_roots() -> list[str]:
    """Return broad dataset locations for Kaggle or local development."""

    if os.path.isdir("/kaggle/input"):
        return ["/kaggle/input"]

    repository_root = Path(__file__).resolve().parents[2]
    local_raw_directory = repository_root / "data" / "raw"

    if local_raw_directory.exists():
        return [str(local_raw_directory)]

    return []


def remove_overlapping_roots(
    roots: Sequence[str],
) -> list[str]:
    """Prevent scanning the same directory multiple times."""

    existing_roots = sorted(
        {
            os.path.realpath(root)
            for root in roots
            if os.path.isdir(root)
        },
        key=len,
    )

    selected_roots: list[str] = []

    for root in existing_roots:
        is_already_covered = any(
            root == parent
            or root.startswith(parent + os.sep)
            for parent in selected_roots
        )

        if not is_already_covered:
            selected_roots.append(root)

    return selected_roots


def build_source_index(
    roots: Sequence[str] | None = None,
) -> list[SourceFile]:
    """Index files recursively, including files inside ZIP archives."""

    scan_roots = remove_overlapping_roots(
        roots or default_scan_roots()
    )

    entries: list[SourceFile] = []

    for root in scan_roots:
        for current_directory, directories, files in os.walk(root):
            directories[:] = [
                directory
                for directory in directories
                if not is_junk_file(directory)
            ]

            for filename in files:
                if is_junk_file(filename):
                    continue

                file_path = os.path.join(
                    current_directory,
                    filename,
                )

                if filename.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(file_path) as archive:
                            for archive_member in archive.infolist():
                                if archive_member.is_dir():
                                    continue

                                if is_junk_file(
                                    archive_member.filename
                                ):
                                    continue

                                entries.append(
                                    SourceFile(
                                        kind="zip",
                                        container=file_path,
                                        member=archive_member.filename,
                                    )
                                )

                    except (
                        OSError,
                        zipfile.BadZipFile,
                    ):
                        # Broken archives are reported by Step 00.
                        continue

                else:
                    entries.append(
                        SourceFile(
                            kind="file",
                            container=file_path,
                        )
                    )

    return entries


@contextmanager
def open_source(
    source_file: SourceFile,
) -> Iterator[BinaryIO]:
    """Open a normal file or ZIP member as a binary stream."""

    if source_file.kind == "file":
        with open(source_file.container, "rb") as stream:
            yield stream

        return

    archive = zipfile.ZipFile(source_file.container)
    stream = archive.open(source_file.member, "r")

    try:
        yield stream

    finally:
        stream.close()
        archive.close()


def is_image_openable(
    source_file: SourceFile,
) -> bool:
    """Check whether an image can be read without retaining pixel data."""

    if source_file.suffix not in IMAGE_SUFFIXES:
        return False

    try:
        with open_source(source_file) as stream:
            with Image.open(stream) as image:
                image.verify()

        return True

    except Exception:
        return False


def image_entries(
    entries: Iterable[SourceFile],
) -> list[SourceFile]:
    """Return only image files from an existing source index."""

    return [
        entry
        for entry in entries
        if entry.suffix in IMAGE_SUFFIXES
    ]