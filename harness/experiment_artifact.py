"""Export compact, versionable experiment artifacts from ignored run dirs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def export_experiment_artifact(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    source_commit: str | None = None,
) -> Path:
    """Write a compact JSON artifact for a completed Vast/GPU run."""
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "vast_run_metadata.json")
    summary = _read_json(run_path / "results" / "tables" / "gpu_eval_batch_summary.json")
    raw_results = _read_raw_results(run_path / "results" / "raw")

    resolved_source_commit = source_commit or metadata.get("source_commit")
    artifact = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_path),
        "source_commit": resolved_source_commit,
        "vast_run_metadata": metadata,
        "gpu_eval_batch_summary": summary,
        "aggregates": _aggregate(raw_results),
        "raw_results": raw_results,
    }

    destination = Path(output_path) if output_path is not None else _default_output_path(run_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _default_output_path(run_path: Path) -> Path:
    project_root = _project_root_from_run(run_path)
    return project_root / "results" / "experiments" / f"{run_path.name}.json"


def _project_root_from_run(run_path: Path) -> Path:
    current = run_path.resolve()
    for parent in (current, *current.parents):
        if (parent / "tasks").is_dir() and (parent / "generated").is_dir():
            return parent
    raise ValueError(f"could not locate project root from run directory: {run_path}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON artifact input: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_raw_results(raw_dir: Path) -> list[dict[str, Any]]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw results directory: {raw_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*_correctness.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object line in {path}")
                records.append(record)
    if not records:
        raise ValueError(f"no correctness JSONL records found in {raw_dir}")
    return records


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["task_id"]), str(record["prompt_mode"]))].append(record)

    aggregate_rows = []
    for (task_id, prompt_mode), group in sorted(groups.items()):
        generated_ms = _numeric_values(group, "generated_ms")
        eager_ms = _numeric_values(group, "pytorch_eager_ms")
        speedups = _numeric_values(group, "speedup_vs_eager")
        max_errors = _numeric_values(group, "max_abs_error")
        aggregate_rows.append(
            {
                "task_id": task_id,
                "prompt_mode": prompt_mode,
                "checks": len(group),
                "passed": sum(1 for record in group if record.get("correct") is True),
                "mean_generated_ms": _mean(generated_ms),
                "mean_pytorch_eager_ms": _mean(eager_ms),
                "mean_speedup_vs_eager": _mean(speedups),
                "min_speedup_vs_eager": min(speedups) if speedups else None,
                "max_speedup_vs_eager": max(speedups) if speedups else None,
                "max_abs_error": max(max_errors) if max_errors else None,
            }
        )
    return aggregate_rows


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
