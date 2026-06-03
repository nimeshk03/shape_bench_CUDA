#!/usr/bin/env python
"""Evaluate one prepared generated attempt across task shape variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.evaluator import default_output_path, evaluate_attempt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", help="Attempt directory containing extracted/eval_contract.json")
    parser.add_argument("--output", help="JSONL output path. Defaults to results/raw/<task>_<mode>_<attempt>.jsonl")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for task inputs. auto chooses CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to task create_inputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path(args.attempt_dir)
    run = evaluate_attempt(
        args.attempt_dir,
        output_path=output_path,
        device=args.device,
        seed=args.seed,
    )
    print(f"Wrote results: {run.output_path}")
    print(f"Task: {run.summary.task_id}")
    print(f"Prompt mode: {run.summary.prompt_mode}")
    print(f"Passed shapes: {run.summary.passed_shapes}/{run.summary.total_shapes}")
    print(f"Original passed: {run.summary.original_passed}")
    print(f"Failure reasons: {run.summary.failure_reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
