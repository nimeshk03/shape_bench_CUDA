from __future__ import annotations

import json

from harness.experiment_artifact import export_experiment_artifact


def test_export_experiment_artifact_writes_versionable_summary(tmp_path) -> None:
    run_dir = tmp_path / "results" / "vast_runs" / "run_001"
    raw_dir = run_dir / "results" / "raw"
    tables_dir = run_dir / "results" / "tables"
    raw_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (tmp_path / "tasks").mkdir()
    (tmp_path / "generated").mkdir()

    (run_dir / "vast_run_metadata.json").write_text(
        json.dumps({"repo_ref": "HEAD", "remote_exit_code": 0}),
        encoding="utf-8",
    )
    (tables_dir / "gpu_eval_batch_summary.json").write_text(
        json.dumps({"attempts": [], "benchmark": {"enabled": True}}),
        encoding="utf-8",
    )
    records = [
        {
            "task_id": "task_003",
            "prompt_mode": "baseline",
            "correct": True,
            "generated_ms": 2.0,
            "pytorch_eager_ms": 4.0,
            "speedup_vs_eager": 2.0,
            "max_abs_error": 0.0,
        },
        {
            "task_id": "task_003",
            "prompt_mode": "baseline",
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
    assert output_path == tmp_path / "results" / "experiments" / "run_001.json"
    assert artifact["schema_version"] == 1
    assert artifact["source_commit"] == "abc123"
    assert len(artifact["raw_results"]) == 2
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
