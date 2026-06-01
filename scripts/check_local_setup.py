"""Check the local ShapeBench-CUDA development environment.

This script is intentionally CPU-compatible. Missing CUDA is reported, not
treated as a failure, because local development may happen without a GPU.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


PROJECT_FOLDERS = [
    "tasks",
    "prompts",
    "generated",
    "generated/baseline",
    "generated/shape_aware",
    "harness",
    "scripts",
    "results",
    "results/raw",
    "results/tables",
    "results/figures",
    "report",
    "docs",
    "tests",
]

CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pytest", "pytest"),
    ("yaml", "pyyaml"),
    ("jinja2", "jinja2"),
    ("rich", "rich"),
    ("tabulate", "tabulate"),
    ("jsonlines", "jsonlines"),
    ("orjson", "orjson"),
    ("dotenv", "python-dotenv"),
]


def _version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "installed"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _project_root()
    failures: list[str] = []

    print("ShapeBench-CUDA local setup check")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Project root: {root}")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', '<not active>')}")
    print()

    print("Core packages:")
    for module_name, package_name in CORE_PACKAGES:
        try:
            print(f"  {package_name}: {_version(module_name)}")
        except ImportError:
            print(f"  {package_name}: MISSING")
            failures.append(f"missing package: {package_name}")
    print()

    print("PyTorch/CUDA:")
    try:
        import torch

        print(f"  torch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  torch CUDA version: {torch.version.cuda}")
        else:
            print("  CUDA unavailable locally: expected on non-GPU development machines")
    except ImportError:
        print("  torch: not installed")
        print("  CUDA check skipped because torch is not installed")
    print()

    print("Project folders:")
    for folder in PROJECT_FOLDERS:
        path = root / folder
        exists = path.is_dir()
        print(f"  {folder}: {'ok' if exists else 'MISSING'}")
        if not exists:
            failures.append(f"missing folder: {folder}")
    print()

    if failures:
        print("Local setup check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Local setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

