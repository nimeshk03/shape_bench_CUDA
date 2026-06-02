from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

from harness.task_loader import load_task


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "task_001"


def test_task_001_metadata_and_shapes_load() -> None:
    task = load_task(TASK_DIR)

    assert task.metadata["task_id"] == "task_001"
    assert task.metadata["name"] == "elementwise_add_relu"
    assert task.shapes["original"] == (1024, 1024)
    assert task.shapes["odd"] == (1007, 1013)


def test_task_001_model_matches_reference_for_all_shapes() -> None:
    task = load_task(TASK_DIR)
    module = _load_task_module(task.model_path)
    model = module.Model()

    for shape in task.shapes.values():
        x, y = module.create_inputs(shape, seed=123)
        output = model(x, y)
        expected = module.reference(x, y)

        assert output.shape == torch.Size(shape)
        assert torch.allclose(output, expected)
        assert torch.all(output >= 0)


def test_task_001_inputs_are_deterministic() -> None:
    module = _load_task_module(TASK_DIR / "model.py")

    first = module.create_inputs((8, 8), seed=42)
    second = module.create_inputs((8, 8), seed=42)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def _load_task_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("task_001_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
