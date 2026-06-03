#!/usr/bin/env python
"""Prepare a generated attempt for the solution.py evaluation contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.attempt_contract import prepare_attempt_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", help="Attempt directory containing extracted/manifest.json")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing fallback solution.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = prepare_attempt_contract(args.attempt_dir, overwrite=args.overwrite)
    print(f"Prepared evaluation contract: {contract.extracted_dir / contract.entrypoint_file}")
    print(f"Entrypoint: {contract.entrypoint_file}:{contract.entrypoint_function}")
    print(f"CUDA source: {contract.cuda_source}")
    print(f"Extension function: {contract.extension_function}")
    print(f"Created fallback solution: {contract.created_fallback_solution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
