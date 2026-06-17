#!/usr/bin/env python
"""Run a ShapeBench-CUDA GPU experiment inside a Google Colab runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.evaluator import (  # noqa: E402
    DEFAULT_BENCHMARK_ITERS,
    DEFAULT_BENCHMARK_WARMUP,
    default_output_path,
    evaluate_attempt,
)
from harness.experiment_artifact import export_experiment_artifact  # noqa: E402
from harness.experiment_config import load_experiment_config  # noqa: E402
from harness.gpu_eval_batch import BatchAttemptSummary, GpuEvalBatchRun, collect_preflight  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment config JSON, for example configs/phase1_task013_016.json.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--run-root",
        default=ROOT / "results" / "colab_runs",
        type=Path,
        help="Directory for Colab run directories.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=DEFAULT_BENCHMARK_WARMUP)
    parser.add_argument("--benchmark-iters", type=int, default=DEFAULT_BENCHMARK_ITERS)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow execution without CUDA. Intended only for debugging notebook setup.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--export-artifact",
        action="store_true",
        help="Export compact artifacts to results/experiments/<run_id> after the run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root / run_id
    raw_dir = run_dir / "results" / "raw"
    tables_dir = run_dir / "results" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(args.experiment, project_root=ROOT)
    source_commit = _git_output(["git", "rev-parse", "HEAD"]) or "unknown"
    metadata = {
        "backend": "colab",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": str(Path(args.experiment)),
        "experiment_name": experiment.name,
        "repo_ref": "HEAD",
        "source_commit": source_commit,
    }
    _write_metadata(run_dir, metadata)

    log(f"run id: {run_id}")
    log(f"run dir: {run_dir}")
    log(f"experiment: {experiment.name} ({len(experiment.attempts)} attempts)")
    log("preflight start" if not args.skip_preflight else "preflight skipped")
    try:
        preflight = (
            collect_preflight(require_cuda=not args.allow_cpu)
            if not args.skip_preflight
            else {"skipped": True}
        )
    except RuntimeError as exc:
        metadata = {**metadata, "remote_exit_code": 1, "error": str(exc)}
        _write_metadata(run_dir, metadata)
        print(str(exc), file=sys.stderr)
        return 1
    log("preflight complete")

    attempt_summaries: list[BatchAttemptSummary] = []
    remote_exit_code = 0
    for index, attempt in enumerate(experiment.attempts, start=1):
        attempt_dir = ROOT / attempt
        output_path = raw_dir / default_output_path(attempt_dir).name
        log(f"attempt {index}/{len(experiment.attempts)} start: {attempt}")
        try:
            run = evaluate_attempt(
                attempt_dir,
                output_path=output_path,
                device=args.device,
                seed=args.seed,
                benchmark=not args.no_benchmark,
                benchmark_warmup=args.benchmark_warmup,
                benchmark_iters=args.benchmark_iters,
                log=log,
            )
        except Exception as exc:
            metadata = {**metadata, "remote_exit_code": 1, "error": str(exc)}
            _write_metadata(run_dir, metadata)
            print(f"Colab evaluation failed: {exc}", file=sys.stderr)
            return 1

        log(
            f"attempt {index}/{len(experiment.attempts)} done: "
            f"{run.summary.passed_shapes}/{run.summary.total_shapes} shapes passed; "
            f"failures={run.summary.failure_reasons}"
        )
        attempt_summaries.append(
            BatchAttemptSummary(
                attempt_dir=Path(attempt),
                output_path=output_path.relative_to(ROOT),
                task_id=run.summary.task_id,
                prompt_mode=run.summary.prompt_mode,
                total_shapes=run.summary.total_shapes,
                passed_shapes=run.summary.passed_shapes,
                original_passed=run.summary.original_passed,
                failure_reasons=run.summary.failure_reasons,
            )
        )

    summary_output = tables_dir / "gpu_eval_batch_summary.json"
    batch = GpuEvalBatchRun(
        created_at=datetime.now(timezone.utc).isoformat(),
        device=args.device,
        seed=args.seed,
        require_cuda=not args.allow_cpu,
        benchmark={
            "enabled": not args.no_benchmark,
            "warmup": args.benchmark_warmup,
            "iterations": args.benchmark_iters,
        },
        preflight=preflight,
        attempts=attempt_summaries,
        summary_output=summary_output.relative_to(ROOT),
    )
    summary_output.write_text(json.dumps(batch.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    metadata = {**metadata, "remote_exit_code": remote_exit_code, "summary_output": str(summary_output.relative_to(ROOT))}
    _write_metadata(run_dir, metadata)
    log(f"wrote summary: {summary_output}")

    if args.export_artifact:
        output_path = export_experiment_artifact(run_dir)
        log(f"exported artifact: {output_path}")

    print(f"Colab run dir: {run_dir}")
    return remote_exit_code


def _write_metadata(run_dir: Path, metadata: dict[str, object]) -> None:
    payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    (run_dir / "run_metadata.json").write_text(payload, encoding="utf-8")
    # Compatibility with the existing compact artifact exporter.
    (run_dir / "vast_run_metadata.json").write_text(payload, encoding="utf-8")


def _git_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[colab-eval {timestamp}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
