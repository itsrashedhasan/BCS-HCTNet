"""Step 00: Check the Python, Kaggle, GPU, and dataset environment.

Run from the repository root:

    python scripts/00_check_environment.py

This script only reads system information. It does not modify, extract,
rename, or delete datasets.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# Allow imports from the repository root.
PROJECT_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_REPOSITORY_ROOT))

from src.utils import config


def human_readable_bytes(value: int) -> str:
    """Convert a byte count into a readable value."""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{value} B"


def build_directory_tree(
    root: str,
    max_depth: int = 4,
    limit: int = 250,
) -> list[str]:
    """Return a limited directory listing for debugging dataset paths."""

    lines: list[str] = []
    root_path = Path(root)

    if not root_path.exists():
        return [f"MISSING: {root}"]

    for current_directory, directories, files in os.walk(root):
        current_path = Path(current_directory)
        relative_path = current_path.relative_to(root_path)
        depth = len(relative_path.parts)

        if depth >= max_depth:
            directories[:] = []

        directories[:] = sorted(
            directory
            for directory in directories
            if not directory.startswith(".") and directory != "__MACOSX"
        )

        for directory in directories:
            lines.append(str(current_path / directory))

            if len(lines) >= limit:
                lines.append(f"... output capped at {limit} entries")
                return lines

        for filename in sorted(files):
            if filename.startswith("._"):
                continue

            if filename in {".DS_Store", "Thumbs.db"}:
                continue

            file_path = current_path / filename

            try:
                file_size = human_readable_bytes(file_path.stat().st_size)
            except OSError:
                file_size = "unknown size"

            lines.append(f"{file_path}  [{file_size}]")

            if len(lines) >= limit:
                lines.append(f"... output capped at {limit} entries")
                return lines

    return lines


def check_import(module_name: str) -> dict:
    """Check whether a Python package can be imported."""

    try:
        module = importlib.import_module(module_name)

        return {
            "ok": True,
            "version": getattr(module, "__version__", "installed"),
        }

    except Exception as error:
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }


def inspect_zip_archives(root: str) -> list[dict]:
    """Find ZIP archives and inspect their contents without extracting them."""

    reports: list[dict] = []

    if not os.path.isdir(root):
        return reports

    for current_directory, _, files in os.walk(root):
        for filename in files:
            if not filename.lower().endswith(".zip"):
                continue

            archive_path = os.path.join(current_directory, filename)

            archive_report: dict = {
                "path": archive_path,
            }

            try:
                with zipfile.ZipFile(archive_path) as archive:
                    members = [
                        member
                        for member in archive.namelist()
                        if not member.endswith("/")
                    ]

                    bad_member = archive.testzip()

                    archive_report.update(
                        {
                            "ok": bad_member is None,
                            "n_files": len(members),
                            "bad_member": bad_member,
                            "sample_members": members[:12],
                        }
                    )

            except Exception as error:
                archive_report.update(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

            reports.append(archive_report)

    return reports


def get_torch_information() -> dict:
    """Return PyTorch and CUDA information."""

    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(device_index)
                for device_index in range(torch.cuda.device_count())
            ],
        }

    except Exception as error:
        return {
            "error": f"{type(error).__name__}: {error}",
        }


def get_disk_information(root: str) -> dict:
    """Return disk usage information."""

    try:
        total, used, free = shutil.disk_usage(root)

        return {
            "root": root,
            "total": human_readable_bytes(total),
            "used": human_readable_bytes(used),
            "free": human_readable_bytes(free),
        }

    except OSError as error:
        return {
            "root": root,
            "error": str(error),
        }


def main() -> int:
    """Run the complete environment check."""

    config.ensure_all_dirs()

    if os.path.isdir("/kaggle/input"):
        input_root = "/kaggle/input"
    else:
        input_root = str(PROJECT_REPOSITORY_ROOT / "data" / "raw")

    dependency_modules = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "cv2",
        "skimage",
        "scipy",
        "PIL",
        "sklearn",
        "statsmodels",
        "yaml",
        "tqdm",
    ]

    dependency_results = {
        module_name: check_import(module_name)
        for module_name in dependency_modules
    }

    torch_information = get_torch_information()

    if os.path.isdir("/kaggle/working"):
        disk_root = "/kaggle/working"
    else:
        disk_root = config.PROJECT_ROOT

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_kaggle": config.IS_KAGGLE,
        "python": sys.version,
        "platform": platform.platform(),
        "current_working_directory": os.getcwd(),
        "repository_root": str(PROJECT_REPOSITORY_ROOT),
        "configured_project_root": config.PROJECT_ROOT,
        "input_root": input_root,
        "input_root_exists": os.path.isdir(input_root),
        "dependencies": dependency_results,
        "torch": torch_information,
        "disk": get_disk_information(disk_root),
        "archives": inspect_zip_archives(input_root),
        "input_tree": build_directory_tree(input_root),
    }

    output_path = os.path.join(
        config.REPORTS_DIR,
        "environment_report.json",
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print("=== BCS-HCTNet Environment Check ===")
    print(f"Kaggle detected : {report['is_kaggle']}")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Working folder  : {os.getcwd()}")
    print(f"Repository root : {PROJECT_REPOSITORY_ROOT}")
    print(f"Input root      : {input_root}")
    print(f"Input exists    : {report['input_root_exists']}")
    print(
        "PyTorch         : "
        f"{torch_information.get('version', torch_information.get('error'))}"
    )
    print(
        "CUDA available  : "
        f"{torch_information.get('cuda_available', False)}"
    )
    print(
        "CUDA version    : "
        f"{torch_information.get('cuda_version')}"
    )
    print(
        "GPU devices     : "
        f"{torch_information.get('devices', [])}"
    )
    print(
        "Free disk       : "
        f"{report.get('disk', {}).get('free', 'unknown')}"
    )

    missing_dependencies = [
        module_name
        for module_name, result in dependency_results.items()
        if not result["ok"]
    ]

    print(
        "Missing imports : "
        f"{missing_dependencies if missing_dependencies else 'none'}"
    )

    print("\n=== Actual input directory tree ===")

    for line in report["input_tree"]:
        print(line)

    print("\n=== ZIP archives found ===")

    if not report["archives"]:
        print("None")

    for archive_report in report["archives"]:
        status = "OK" if archive_report.get("ok") else "BROKEN"

        print(
            f"[{status}] {archive_report['path']} "
            f"| files={archive_report.get('n_files', '?')}"
        )

        if archive_report.get("bad_member"):
            print(
                "    First corrupted member: "
                f"{archive_report['bad_member']}"
            )

        if archive_report.get("error"):
            print(f"    Error: {archive_report['error']}")

        for member in archive_report.get("sample_members", []):
            print(f"    {member}")

    print(f"\nFull report saved to: {output_path}")

    required_check_failed = (
        not report["input_root_exists"]
        or not dependency_results["PIL"]["ok"]
    )

    return 1 if required_check_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())