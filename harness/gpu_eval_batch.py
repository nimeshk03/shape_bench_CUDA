"""GPU evaluation batch helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.evaluator import default_output_path, evaluate_attempt


DEFAULT_ATTEMPTS = (
    "generated/baseline/task_001/attempt_003",
    "generated/shape_aware/task_001/attempt_002",
)


@dataclass(frozen=True)
class BatchAttemptSummary:
    attempt_dir: Path
    output_path: Path
    task_id: str
    prompt_mode: str
    total_shapes: int
    passed_shapes: int
    original_passed: bool
    failure_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_dir": str(self.attempt_dir),
            "output_path": str(self.output_path),
            "task_id": self.task_id,
            "prompt_mode": self.prompt_mode,
            "total_shapes": self.total_shapes,
            "passed_shapes": self.passed_shapes,
            "original_passed": self.original_passed,
            "failure_reasons": self.failure_reasons,
        }


@dataclass(frozen=True)
class GpuEvalBatchRun:
    created_at: str
    device: str
    seed: int
    require_cuda: bool
    preflight: dict[str, Any]
    attempts: list[BatchAttemptSummary]
    summary_output: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "device": self.device,
            "seed": self.seed,
            "require_cuda": self.require_cuda,
            "preflight": self.preflight,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "summary_output": str(self.summary_output),
        }


def run_gpu_eval_batch(
    *,
    project_root: str | Path,
    attempt_dirs: list[str | Path] | None = None,
    summary_output: str | Path | None = None,
    device: str = "auto",
    seed: int = 0,
    require_cuda: bool = True,
    run_preflight: bool = True,
) -> GpuEvalBatchRun:
    """Run a small GPU correctness batch and write a summary JSON file."""
    root = Path(project_root)
    attempts = [_resolve_attempt(root, attempt) for attempt in (attempt_dirs or list(DEFAULT_ATTEMPTS))]
    preflight = collect_preflight(require_cuda=require_cuda) if run_preflight else {"skipped": True}
    summaries: list[BatchAttemptSummary] = []

    for attempt_dir in attempts:
        output_path = default_output_path(attempt_dir)
        run = evaluate_attempt(attempt_dir, output_path=output_path, device=device, seed=seed)
        summaries.append(
            BatchAttemptSummary(
                attempt_dir=_display_path(attempt_dir, root),
                output_path=_display_path(output_path, root),
                task_id=run.summary.task_id,
                prompt_mode=run.summary.prompt_mode,
                total_shapes=run.summary.total_shapes,
                passed_shapes=run.summary.passed_shapes,
                original_passed=run.summary.original_passed,
                failure_reasons=run.summary.failure_reasons,
            )
        )

    destination = Path(summary_output) if summary_output else root / "results" / "tables" / "gpu_eval_batch_summary.json"
    batch = GpuEvalBatchRun(
        created_at=datetime.now(timezone.utc).isoformat(),
        device=device,
        seed=seed,
        require_cuda=require_cuda,
        preflight=preflight,
        attempts=summaries,
        summary_output=_display_path(destination, root),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(batch.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return batch


def collect_preflight(*, require_cuda: bool = True) -> dict[str, Any]:
    """Collect GPU setup checks and optionally require CUDA readiness."""
    checks = {
        "python": sys.version.split()[0],
        "nvidia_smi": _run_command(["nvidia-smi"]),
        "nvcc": _run_command(["nvcc", "--version"]),
        "torch": _torch_cuda_check(),
    }
    failures: list[str] = []
    if require_cuda:
        if not checks["nvidia_smi"]["ok"]:
            failures.append("nvidia-smi is not available")
        if not checks["nvcc"]["ok"]:
            failures.append("nvcc is not available")
        if not checks["torch"]["cuda_available"]:
            failures.append("torch.cuda.is_available() is false")
    checks["failures"] = failures
    if failures:
        raise RuntimeError("GPU preflight failed: " + "; ".join(failures))
    return checks


def _torch_cuda_check() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {exc.timeout}s",
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _resolve_attempt(root: Path, attempt: str | Path) -> Path:
    path = Path(attempt)
    if not path.is_absolute():
        path = root / path
    return path


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
