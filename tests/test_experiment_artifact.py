from __future__ import annotations

import hashlib
import json

import pytest

from harness.experiment_artifact import export_experiment_artifact


def _write_minimal_run(run_dir) -> None:
    raw_dir = run_dir / "results" / "raw"
    tables_dir = run_dir / "results" / "tables"
    raw_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (run_dir / "vast_run_metadata.json").write_text(
        json.dumps({"repo_ref": "HEAD", "remote_exit_code": 0}),
        encoding="utf-8",
    )
    (tables_dir / "gpu_eval_batch_summary.json").write_text(
        json.dumps({"attempts": [], "benchmark": {"enabled": True}}),
        encoding="utf-8",
    )


def test_export_experiment_artifact_writes_versionable_summary(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()
    _write_minimal_run(run_dir)

    records = [
        {
            "task_id": "task_003",
            "prompt_mode": "baseline",
            "shape_category": "original",
            "shape": [1024, 1024],
            "correct": True,
            "generated_ms": 2.0,
            "pytorch_eager_ms": 4.0,
            "speedup_vs_eager": 2.0,
            "max_abs_error": 0.0,
        },
        {
            "task_id": "task_003",
            "prompt_mode": "baseline",
            "shape_category": "odd",
            "shape": [1007, 1013],
            "correct": False,
            "generated_ms": None,
            "pytorch_eager_ms": None,
            "speedup_vs_eager": None,
            "max_abs_error": 1.0,
        },
    ]
    (raw_dir / "task_003_baseline_attempt_001_correctness.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    output_path = export_experiment_artifact(run_dir, source_commit="abc123")

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    raw_path = output_path.parent / "raw_results.jsonl"
    raw_payload = raw_path.read_text(encoding="utf-8")

    assert output_path == tmp_path / "results" / "experiments" / "run_001" / "summary.json"
    assert artifact["schema_version"] == 2
    assert artifact["source_commit"] == "abc123"
    assert artifact["vast_run_metadata"]["source_commit"] == "abc123"
    assert "exported_at" not in artifact
    assert artifact["raw_results"] == {
        "path": "raw_results.jsonl",
        "record_count": 2,
        "sha256": hashlib.sha256(raw_payload.encode("utf-8")).hexdigest(),
    }
    assert [json.loads(line) for line in raw_payload.splitlines()] == records
    assert artifact["aggregates"] == [
        {
            "task_id": "task_003",
            "prompt_mode": "baseline",
            "checks": 2,
            "passed": 1,
            "mean_generated_ms": 2.0,
            "mean_pytorch_eager_ms": 4.0,
            "mean_speedup_vs_eager": 2.0,
            "min_speedup_vs_eager": 2.0,
            "max_speedup_vs_eager": 2.0,
            "max_abs_error": 1.0,
        }
    ]


def test_export_experiment_artifact_requires_source_commit_for_legacy_metadata(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    tables_dir = run_dir / "results" / "tables"
    raw_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()
    (run_dir / "vast_run_metadata.json").write_text(json.dumps({"repo_ref": "HEAD"}), encoding="utf-8")
    (tables_dir / "gpu_eval_batch_summary.json").write_text(json.dumps({}), encoding="utf-8")
    (raw_dir / "x_correctness.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "prompt_mode": "baseline",
                "shape_category": "original",
                "shape": [1024, 1024],
                "correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        export_experiment_artifact(run_dir)
    except ValueError as exc:
        assert "source_commit is required" in str(exc)
    else:
        raise AssertionError("expected legacy metadata without source commit to be rejected")


def test_export_experiment_artifact_rejects_missing_required_raw_field(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()
    _write_minimal_run(run_dir)
    (raw_dir / "x_correctness.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "shape_category": "original",
                "shape": [1024, 1024],
                "correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"x_correctness\.jsonl:1: missing field 'prompt_mode'"):
        export_experiment_artifact(run_dir, source_commit="abc123")


def test_export_experiment_artifact_rejects_invalid_raw_shape(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()
    _write_minimal_run(run_dir)
    (raw_dir / "x_correctness.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "prompt_mode": "baseline",
                "shape_category": "original",
                "shape": [1024, 0],
                "correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"field 'shape' must be a non-empty list of positive integers"):
        export_experiment_artifact(run_dir, source_commit="abc123")


def test_export_experiment_artifact_rejects_invalid_raw_timing_type(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()
    _write_minimal_run(run_dir)
    (raw_dir / "x_correctness.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "prompt_mode": "baseline",
                "shape_category": "original",
                "shape": [1024, 1024],
                "correct": True,
                "generated_ms": "fast",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"field 'generated_ms' must be a number or null"):
        export_experiment_artifact(run_dir, source_commit="abc123")
