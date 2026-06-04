#!/usr/bin/env python
"""Run the first ShapeBench-CUDA correctness batch on a CUDA GPU machine."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.evaluator import DEFAULT_BENCHMARK_ITERS, DEFAULT_BENCHMARK_WARMUP  # noqa: E402
from harness.gpu_eval_batch import DEFAULT_ATTEMPTS, run_gpu_eval_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt",
        action="append",
        dest="attempts",
        help="Attempt directory to evaluate. Can be passed multiple times. Defaults to the first clean task_001 pair.",
    )
    parser.add_argument(
        "--summary-output",
        help="Summary JSON path. Defaults to results/tables/gpu_eval_batch_summary.json.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device passed to the evaluator. On a GPU machine, auto should resolve to cuda.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to task create_inputs.")
    parser.add_argument("--no-benchmark", action="store_true", help="Run correctness only and leave timing fields empty.")
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=DEFAULT_BENCHMARK_WARMUP,
        help="Warmup calls per callable before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=DEFAULT_BENCHMARK_ITERS,
        help="Timed calls per callable.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the batch to run when CUDA is unavailable. Intended only for local smoke tests.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip nvidia-smi, nvcc, and torch CUDA preflight checks.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging and print only the final summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempts = args.attempts or list(DEFAULT_ATTEMPTS)
    progress_log = None if args.quiet else _progress_log
    try:
        batch = run_gpu_eval_batch(
            project_root=ROOT,
            attempt_dirs=attempts,
            summary_output=args.summary_output,
            device=args.device,
            seed=args.seed,
            require_cuda=not args.allow_cpu,
            run_preflight=not args.skip_preflight,
            benchmark=not args.no_benchmark,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_iters=args.benchmark_iters,
            log=progress_log,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote summary: {batch.summary_output}")
    print(f"Device: {batch.device}")
    print(f"CUDA required: {batch.require_cuda}")
    for attempt in batch.attempts:
        print(
            f"{attempt.prompt_mode} {attempt.attempt_dir}: "
            f"{attempt.passed_shapes}/{attempt.total_shapes} shapes passed; "
            f"original_passed={attempt.original_passed}; "
            f"failures={attempt.failure_reasons}"
        )
    return 0


def _progress_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[gpu-eval {timestamp}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
