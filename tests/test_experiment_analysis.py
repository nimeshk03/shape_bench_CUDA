from __future__ import annotations

import json

from harness.experiment_analysis import (
    analyze_experiments,
    load_experiment_artifacts,
    render_initial_findings,
)


def _write_artifact(root, run_id: str, records: list[dict]) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "source_commit": "abcdef123456",
                "gpu_eval_batch_summary": {
                    "preflight": {
                        "torch": {
                            "gpu_name": "Test GPU",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "raw_results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(
    task_id: str,
    prompt_mode: str,
    attempt: int,
    shape_category: str,
    correct: bool,
    *,
    speedup: float | None = 2.0,
    failure_reason: str | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "prompt_mode": prompt_mode,
        "shape_category": shape_category,
        "shape": [4, 4],
        "correct": correct,
        "failure_reason": failure_reason,
        "generated_ms": 1.0 if speedup is not None and correct else None,
        "pytorch_eager_ms": speedup if speedup is not None and correct else None,
        "speedup_vs_eager": speedup if correct else None,
        "max_abs_error": 0.0 if correct else 1.25,
        "extra": {
            "attempt": attempt,
        },
    }


def test_analyze_experiments_computes_prompt_rates_and_failures(tmp_path) -> None:
    records = [
        _record("task_001", "baseline", 1, "original", True, speedup=2.0),
        _record("task_001", "baseline", 1, "odd", True, speedup=4.0),
        _record("task_001", "shape_aware", 1, "original", True, speedup=1.0),
        _record(
            "task_001",
            "shape_aware",
            1,
            "odd",
            False,
            failure_reason="shape_variant_correctness_failure",
        ),
        _record(
            "task_001",
            "shape_aware",
            2,
            "original",
            False,
            failure_reason="original_shape_correctness_failure",
        ),
        _record(
            "task_001",
            "shape_aware",
            2,
            "odd",
            False,
            failure_reason="shape_variant_correctness_failure",
        ),
    ]
    _write_artifact(tmp_path, "run_001", records)

    artifacts = load_experiment_artifacts(tmp_path)
    summary = analyze_experiments(artifacts, task_names={"task_001": "elementwise_add_relu"})

    assert summary.total_attempts == 3
    assert summary.total_shape_checks == 6
    assert summary.passed_shape_checks == 3

    prompt_stats = {row.prompt_mode: row for row in summary.prompt_stats}
    assert prompt_stats["baseline"].original_pass_rate == 1.0
    assert prompt_stats["baseline"].multi_shape_pass_rate == 1.0
    assert prompt_stats["baseline"].robustness_score == 1.0
    assert prompt_stats["baseline"].median_speedup == 3.0

    assert prompt_stats["shape_aware"].original_passed == 1
    assert prompt_stats["shape_aware"].original_total == 2
    assert prompt_stats["shape_aware"].multi_shape_passed == 0
    assert prompt_stats["shape_aware"].shape_variant_only_failures == 1
    assert len(summary.failure_cases) == 2
    assert summary.failure_cases[0].original_passed is True
    assert summary.failure_cases[0].failure_class == "shape_variant_only_failure"
    assert summary.failure_cases[1].original_passed is False
    assert summary.failure_cases[1].failure_class == "original_and_variant_correctness_failure"

    taxonomy = {row.failure_class: row for row in summary.failure_taxonomy}
    assert taxonomy["shape_variant_only_failure"].attempts == 1
    assert taxonomy["shape_variant_only_failure"].failed_shape_checks == 1
    assert taxonomy["shape_variant_only_failure"].original_failures == 0
    assert taxonomy["shape_variant_only_failure"].variant_failures == 1
    assert taxonomy["original_and_variant_correctness_failure"].attempts == 1
    assert taxonomy["original_and_variant_correctness_failure"].failed_shape_checks == 2
    assert taxonomy["original_and_variant_correctness_failure"].original_failures == 1
    assert taxonomy["original_and_variant_correctness_failure"].variant_failures == 1


def test_analyze_experiments_classifies_compilation_failures_first(tmp_path) -> None:
    records = [
        _record(
            "task_002",
            "baseline",
            1,
            "original",
            False,
            failure_reason="compilation_failure",
        ),
        _record(
            "task_002",
            "baseline",
            1,
            "odd",
            False,
            failure_reason="shape_variant_correctness_failure",
        ),
    ]
    _write_artifact(tmp_path, "run_001", records)

    summary = analyze_experiments(load_experiment_artifacts(tmp_path))

    assert len(summary.failure_cases) == 1
    assert summary.failure_cases[0].failure_class == "compilation_failure"
    assert summary.failure_taxonomy[0].failure_class == "compilation_failure"
    assert summary.failure_taxonomy[0].attempts == 1
    assert summary.failure_taxonomy[0].failed_shape_checks == 2
    assert summary.failure_taxonomy[0].failure_reasons == {
        "compilation_failure": 1,
        "shape_variant_correctness_failure": 1,
    }


def test_render_initial_findings_contains_expected_sections(tmp_path) -> None:
    records = [
        _record("task_001", "baseline", 1, "original", True),
        _record("task_001", "baseline", 1, "odd", True),
    ]
    _write_artifact(tmp_path, "run_001", records)
    summary = analyze_experiments(load_experiment_artifacts(tmp_path), task_names={"task_001": "starter"})

    report = render_initial_findings(summary)

    assert "# ShapeBench-CUDA Initial Findings" in report
    assert "## Experiment Setup" in report
    assert "## Task List" in report
    assert "## Prompt-Mode Results" in report
    assert "## Failure Taxonomy" in report
    assert "## Failure Taxonomy By Task And Prompt" in report
    assert "## Failure Cases" in report
    assert "## Lessons Learned" in report
    assert "`task_001`" in report
    assert "100.0%" in report
