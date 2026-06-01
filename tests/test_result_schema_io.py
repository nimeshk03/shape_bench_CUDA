from __future__ import annotations

import pytest

from harness.result_io import read_jsonl, write_jsonl
from harness.result_schema import ShapeBenchResult, TaskSummary


def test_shape_bench_result_round_trip() -> None:
    result = ShapeBenchResult(
        task_id="task_001",
        prompt_mode="baseline",
        shape_category="original",
        shape=(1024, 1024),
        correct=True,
        pytorch_eager_ms=1.5,
        generated_ms=0.75,
        speedup_vs_eager=2.0,
    )

    assert ShapeBenchResult.from_dict(result.to_dict()) == result


def test_shape_bench_result_rejects_failure_reason_on_correct_result() -> None:
    with pytest.raises(ValueError, match="correct results"):
        ShapeBenchResult(
            task_id="task_001",
            prompt_mode="baseline",
            shape_category="original",
            shape=(1024,),
            correct=True,
            failure_reason="runtime_error",
        )


def test_task_summary_from_results() -> None:
    results = [
        ShapeBenchResult(
            task_id="task_001",
            prompt_mode="shape_aware",
            shape_category="original",
            shape=(8, 8),
            correct=True,
        ),
        ShapeBenchResult(
            task_id="task_001",
            prompt_mode="shape_aware",
            shape_category="odd",
            shape=(7, 9),
            correct=False,
            failure_reason="boundary_condition_bug",
        ),
    ]

    summary = TaskSummary.from_results(results)

    assert summary.original_passed is True
    assert summary.passed_shapes == 1
    assert summary.total_shapes == 2
    assert summary.robustness_score == 0.5
    assert summary.failure_reasons == {"boundary_condition_bug": 1}


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    result = ShapeBenchResult(
        task_id="task_001",
        prompt_mode="baseline",
        shape_category="original",
        shape=(16,),
        correct=True,
    )

    write_jsonl(path, [result])
    loaded = read_jsonl(path, ShapeBenchResult)

    assert loaded == [result]

