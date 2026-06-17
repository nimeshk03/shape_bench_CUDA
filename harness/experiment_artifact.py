"""Export compact, versionable experiment artifacts from ignored run dirs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
SUMMARY_FILENAME = "summary.json"
RAW_RESULTS_FILENAME = "raw_results.jsonl"
REQUIRED_RAW_STRING_FIELDS = ("task_id", "prompt_mode", "shape_category")
OPTIONAL_RAW_NUMERIC_FIELDS = (
    "generated_ms",
    "pytorch_eager_ms",
    "speedup_vs_eager",
    "max_abs_error",
)


def export_experiment_artifact(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    source_commit: str | None = None,
    exported_at: str | None = None,
) -> Path:
    """Write a compact artifact directory for a completed Vast/GPU run."""
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "vast_run_metadata.json")
    summary = _read_json(run_path / "results" / "tables" / "gpu_eval_batch_summary.json")
    raw_results = _read_raw_results(run_path / "results" / "raw")

    resolved_source_commit = source_commit or metadata.get("source_commit")
    if not isinstance(resolved_source_commit, str) or not resolved_source_commit.strip():
        raise ValueError("source_commit is required; pass --source-commit for legacy runs")
    metadata = {**metadata, "source_commit": resolved_source_commit}

    destination = Path(output_path) if output_path is not None else _default_output_path(run_path)
    if destination.suffix:
        raise ValueError("output_path must be a directory for schema_version 2 artifacts")
    destination.mkdir(parents=True, exist_ok=True)

    raw_payload = _jsonl_payload(raw_results)
    raw_path = destination / RAW_RESULTS_FILENAME
    raw_path.write_text(raw_payload, encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    artifact_summary = {
        "schema_version": SCHEMA_VERSION,
        "source_run_dir": str(run_path),
        "source_commit": resolved_source_commit,
        "run_metadata": metadata,
        "vast_run_metadata": metadata,
        "gpu_eval_batch_summary": summary,
        "aggregates": _aggregate(raw_results),
        "raw_results": {
            "path": RAW_RESULTS_FILENAME,
            "record_count": len(raw_results),
            "sha256": raw_sha256,
        },
    }
    if exported_at is not None:
        artifact_summary["exported_at"] = exported_at

    (destination / SUMMARY_FILENAME).write_text(
        json.dumps(artifact_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination / SUMMARY_FILENAME


def _default_output_path(run_path: Path) -> Path:
    project_root = _project_root_from_run(run_path)
    return project_root / "results" / "experiments" / run_path.name


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
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"invalid raw result in {path}:{line_number}: expected JSON object")
                _validate_raw_record(record, path, line_number)
                records.append(record)
    if not records:
        raise ValueError(f"no correctness JSONL records found in {raw_dir}")
    return records


def _validate_raw_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    for field in REQUIRED_RAW_STRING_FIELDS:
        if field not in record:
            _raise_invalid_raw_result(path, line_number, f"missing field '{field}'")
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            _raise_invalid_raw_result(path, line_number, f"field '{field}' must be a non-empty string")

    if "shape" not in record:
        _raise_invalid_raw_result(path, line_number, "missing field 'shape'")
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
    ):
        _raise_invalid_raw_result(path, line_number, "field 'shape' must be a non-empty list of positive integers")

    if "correct" not in record:
        _raise_invalid_raw_result(path, line_number, "missing field 'correct'")
    if not isinstance(record["correct"], bool):
        _raise_invalid_raw_result(path, line_number, "field 'correct' must be a boolean")

    for field in OPTIONAL_RAW_NUMERIC_FIELDS:
        value = record.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            _raise_invalid_raw_result(path, line_number, f"field '{field}' must be a number or null")


def _raise_invalid_raw_result(path: Path, line_number: int, message: str) -> None:
    raise ValueError(f"invalid raw result in {path}:{line_number}: {message}")


def _jsonl_payload(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


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
