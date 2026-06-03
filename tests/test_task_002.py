from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

from harness.task_loader import load_task


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "task_002"


def test_task_002_metadata_and_shapes_load() -> None:
    task = load_task(TASK_DIR)

    assert task.metadata["task_id"] == "task_002"
    assert task.metadata["name"] == "rowwise_sum"
    assert task.metadata["category"] == "reduction"
    assert task.metadata["input_names"] == ["x"]
    assert task.shapes["original"] == (1024, 1024)
    assert task.shapes["odd"] == (1007, 1013)


def test_task_002_model_matches_reference_for_all_shapes() -> None:
    task = load_task(TASK_DIR)
    module = _load_task_module(task.model_path)
    model = module.Model()

    for shape in task.shapes.values():
        (x,) = module.create_inputs(shape, seed=123)
        output = model(x)
        expected = module.reference(x)

        assert output.shape == torch.Size([shape[0]])
        assert torch.allclose(output, expected)


def test_task_002_inputs_are_deterministic() -> None:
    module = _load_task_module(TASK_DIR / "model.py")

    first = module.create_inputs((8, 7), seed=42)
    second = module.create_inputs((8, 7), seed=42)

    assert len(first) == 1
    assert torch.equal(first[0], second[0])


def test_task_002_rejects_non_2d_shapes() -> None:
    module = _load_task_module(TASK_DIR / "model.py")

    try:
        module.create_inputs((8,), seed=42)
    except ValueError as exc:
        assert "expects a 2D shape" in str(exc)
    else:
        raise AssertionError("expected task_002 to reject non-2D shapes")


def _load_task_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("task_002_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
