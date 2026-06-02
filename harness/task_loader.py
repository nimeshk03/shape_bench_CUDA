"""Task metadata loading for ShapeBench-CUDA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.shape_registry import Shape, ShapeRegistry, validate_shape, validate_shape_registry


REQUIRED_METADATA_FIELDS = (
    "task_id",
    "name",
    "description",
    "category",
    "input_kind",
    "input_names",
    "original_shape",
    "dtype",
    "atol",
    "rtol",
    "expected_output",
)

SUPPORTED_DTYPES = {"float32"}


@dataclass(frozen=True)
class TaskDefinition:
    task_dir: Path
    metadata: dict[str, Any]
    shapes: ShapeRegistry
    model_path: Path


def load_metadata(task_dir: str | Path) -> dict[str, Any]:
    """Load a task metadata.json file."""
    path = Path(task_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing task metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"task metadata must be a JSON object: {path}")
    return validate_metadata(metadata, source=path)


def validate_metadata(metadata: dict[str, Any], *, source: str | Path = "metadata") -> dict[str, Any]:
    """Validate and normalize task metadata."""
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"task metadata missing required fields in {source}: {', '.join(missing)}")

    normalized = dict(metadata)
    for field in ("task_id", "name", "description", "category", "input_kind", "dtype", "expected_output"):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise ValueError(f"task metadata field {field!r} must be a non-empty string in {source}")

    input_names = normalized["input_names"]
    if (
        not isinstance(input_names, list)
        or not input_names
        or any(not isinstance(name, str) or not name.strip() for name in input_names)
    ):
        raise ValueError(f"task metadata field 'input_names' must be a non-empty string list in {source}")

    if normalized["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(
            f"task metadata field 'dtype' must be one of {sorted(SUPPORTED_DTYPES)}, "
            f"got {normalized['dtype']!r} in {source}"
        )

    normalized["original_shape"] = list(validate_shape(normalized["original_shape"]))
    normalized["atol"] = _validate_positive_number(normalized["atol"], "atol", source)
    normalized["rtol"] = _validate_positive_number(normalized["rtol"], "rtol", source)
    return normalized


def load_shapes(task_dir: str | Path) -> ShapeRegistry:
    """Load and validate a task shapes.json file."""
    path = Path(task_dir) / "shapes.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing task shapes file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        shapes = json.load(handle)
    if not isinstance(shapes, dict):
        raise ValueError(f"task shapes must be a JSON object: {path}")
    return validate_shape_registry(shapes)


def load_task(task_dir: str | Path) -> TaskDefinition:
    """Load task metadata and shape variants."""
    directory = Path(task_dir)
    model_path = directory / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing task model file: {model_path}")
    metadata = load_metadata(directory)
    shapes = load_shapes(directory)
    _validate_original_shape_matches(metadata["original_shape"], shapes["original"], directory)
    return TaskDefinition(
        task_dir=directory,
        metadata=metadata,
        shapes=shapes,
        model_path=model_path,
    )


def _validate_positive_number(value: Any, field: str, source: str | Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"task metadata field {field!r} must be a non-negative number in {source}")
    if value < 0:
        raise ValueError(f"task metadata field {field!r} must be non-negative in {source}")
    return float(value)


def _validate_original_shape_matches(metadata_shape: list[int], registry_shape: Shape, task_dir: Path) -> None:
    if tuple(metadata_shape) != registry_shape:
        raise ValueError(
            "metadata original_shape must match shapes original for "
            f"{task_dir}: {metadata_shape} != {list(registry_shape)}"
        )
