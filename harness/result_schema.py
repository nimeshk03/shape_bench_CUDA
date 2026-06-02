"""Structured result records for ShapeBench-CUDA runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FAILURE_REASONS = (
    "compilation_failure",
    "original_shape_correctness_failure",
    "shape_variant_correctness_failure",
    "runtime_error",
    "out_of_bounds_memory_access",
    "incorrect_indexing",
    "assumes_fixed_dimension",
    "assumes_power_of_two_size",
    "boundary_condition_bug",
    "performance_regression",
    "timeout",
)


@dataclass(frozen=True)
class ShapeBenchResult:
    """One task/prompt/shape evaluation result."""

    task_id: str
    prompt_mode: str
    shape_category: str
    shape: tuple[int, ...]
    correct: bool
    failure_reason: str | None = None
    pytorch_eager_ms: float | None = None
    torch_compile_ms: float | None = None
    generated_ms: float | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_compile: float | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    gpu_name: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure_reason is not None and self.failure_reason not in FAILURE_REASONS:
            raise ValueError(f"unknown failure_reason: {self.failure_reason}")
        if not self.shape:
            raise ValueError("shape must contain at least one dimension")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")
        if self.correct and self.failure_reason is not None:
            raise ValueError("correct results must not include a failure_reason")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "task_id": self.task_id,
            "prompt_mode": self.prompt_mode,
            "shape_category": self.shape_category,
            "shape": list(self.shape),
            "correct": self.correct,
            "failure_reason": self.failure_reason,
            "pytorch_eager_ms": self.pytorch_eager_ms,
            "torch_compile_ms": self.torch_compile_ms,
            "generated_ms": self.generated_ms,
            "speedup_vs_eager": self.speedup_vs_eager,
            "speedup_vs_compile": self.speedup_vs_compile,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "gpu_name": self.gpu_name,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "extra": self.extra,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShapeBenchResult":
        copied = dict(data)
        copied["shape"] = tuple(copied["shape"])
        copied.setdefault("extra", {})
        return cls(**copied)


@dataclass(frozen=True)
class TaskSummary:
    """Aggregate results for one task and one prompt mode."""

    task_id: str
    prompt_mode: str
    total_shapes: int
    passed_shapes: int
    original_passed: bool
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def robustness_score(self) -> float:
        if self.total_shapes == 0:
            return 0.0
        return self.passed_shapes / self.total_shapes

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt_mode": self.prompt_mode,
            "total_shapes": self.total_shapes,
            "passed_shapes": self.passed_shapes,
            "original_passed": self.original_passed,
            "robustness_score": self.robustness_score,
            "failure_reasons": self.failure_reasons,
        }

    @classmethod
    def from_results(cls, results: list[ShapeBenchResult]) -> "TaskSummary":
        if not results:
            raise ValueError("cannot summarize an empty result list")
        task_ids = {result.task_id for result in results}
        prompt_modes = {result.prompt_mode for result in results}
        if len(task_ids) != 1 or len(prompt_modes) != 1:
            raise ValueError("TaskSummary requires one task_id and one prompt_mode")

        failure_reasons: dict[str, int] = {}
        original_results = [result for result in results if result.shape_category == "original"]
        if len(original_results) != 1:
            raise ValueError(
                "TaskSummary requires exactly one original shape result, "
                f"got {len(original_results)}"
            )

        for result in results:
            if result.failure_reason is not None:
                failure_reasons[result.failure_reason] = failure_reasons.get(result.failure_reason, 0) + 1

        return cls(
            task_id=results[0].task_id,
            prompt_mode=results[0].prompt_mode,
            total_shapes=len(results),
            passed_shapes=sum(result.correct for result in results),
            original_passed=original_results[0].correct,
            failure_reasons=failure_reasons,
        )


@dataclass(frozen=True)
class ExperimentSummary:
    """Aggregate summaries for a full experiment."""

    name: str
    task_summaries: list[TaskSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_summaries": [summary.to_dict() for summary in self.task_summaries],
        }
