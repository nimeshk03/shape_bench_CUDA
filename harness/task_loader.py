"""Task metadata loading for ShapeBench-CUDA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.shape_registry import ShapeRegistry, validate_shape_registry


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
    if not metadata.get("task_id"):
        raise ValueError(f"task metadata must include task_id: {path}")
    return metadata


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
    return TaskDefinition(
        task_dir=directory,
        metadata=load_metadata(directory),
        shapes=load_shapes(directory),
        model_path=model_path,
    )
