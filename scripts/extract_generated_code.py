#!/usr/bin/env python
"""Extract generated code blocks from an LLM attempt directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.code_extractor import extract_attempt_code  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", help="Attempt directory containing response.md")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extracted = extract_attempt_code(args.attempt_dir, overwrite=args.overwrite)
    print(f"Extracted {len(extracted)} files:")
    for item in extracted:
        print(f"  {item.path} ({item.confidence}: {item.reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
