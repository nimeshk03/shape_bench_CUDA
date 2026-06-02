#!/usr/bin/env python
"""Render a concrete LLM prompt for a ShapeBench-CUDA task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.prompt_renderer import PROMPT_FILES, write_rendered_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, help="Task directory, for example tasks/task_001")
    parser.add_argument("--mode", required=True, choices=sorted(PROMPT_FILES), help="Prompt mode")
    parser.add_argument("--output", help="Optional output markdown file")
    parser.add_argument("--output-dir", default="generated/prompts", help="Default output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = write_rendered_prompt(
        args.task_dir,
        args.mode,
        args.output,
        output_dir=args.output_dir,
    )
    print(f"Rendered prompt written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
