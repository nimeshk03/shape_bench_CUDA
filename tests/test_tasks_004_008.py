from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from harness.task_loader import load_task


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("task_id", "name", "category", "input_names", "original_shape"),
    [
        ("task_004", "matrix_multiply", "matmul", ["a", "b"], (256, 256, 256)),
        ("task_005", "rowwise_softmax", "reduction", ["x"], (1024, 1024)),
        ("task_006", "rowwise_layer_norm", "normalization", ["x"], (1024, 1024)),
        ("task_007", "broadcast_affine_clamp", "broadcast", ["x", "scale", "bias"], (1024, 1024)),
        ("task_008", "batched_transpose", "layout", ["x"], (16, 256, 512)),
    ],
)
def test_task_metadata_and_shapes_load(task_id, name, category, input_names, original_shape) -> None:
    task = load_task(ROOT / "tasks" / task_id)

    assert task.metadata["task_id"] == task_id
    assert task.metadata["name"] == name
    assert task.metadata["category"] == category
    assert task.metadata["input_names"] == input_names
    assert task.shapes["original"] == original_shape
    assert set(task.shapes) == {
        "original",
        "smaller",
        "larger",
        "odd",
        "batch_variant",
        "non_power_of_two",
    }


@pytest.mark.parametrize("task_id", ["task_004", "task_005", "task_006", "task_007", "task_008"])
def test_task_models_match_reference_for_all_shapes(task_id) -> None:
    task = load_task(ROOT / "tasks" / task_id)
    module = _load_task_module(task.model_path, task_id)
    model = module.Model()

    for shape in task.shapes.values():
        inputs = module.create_inputs(shape, seed=123)
        output = model(*inputs)
        expected = module.reference(*inputs)

        assert output.shape == _expected_output_shape(task_id, shape)
        assert torch.allclose(output, expected, atol=task.metadata["atol"], rtol=task.metadata["rtol"])


@pytest.mark.parametrize("task_id", ["task_004", "task_005", "task_006", "task_007", "task_008"])
def test_task_inputs_are_deterministic(task_id) -> None:
    task = load_task(ROOT / "tasks" / task_id)
    module = _load_task_module(task.model_path, task_id)

    first = module.create_inputs(task.shapes["odd"], seed=42)
    second = module.create_inputs(task.shapes["odd"], seed=42)

    assert len(first) == len(task.metadata["input_names"])
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second, strict=True))


@pytest.mark.parametrize(
    ("task_id", "bad_shape", "message"),
    [
        ("task_004", (8, 8), "expects a 3D shape descriptor"),
        ("task_005", (8,), "expects a 2D shape"),
        ("task_006", (8,), "expects a 2D shape"),
        ("task_007", (8,), "expects a 2D shape"),
        ("task_008", (8, 8), "expects a 3D shape"),
    ],
)
def test_tasks_reject_wrong_rank_shapes(task_id, bad_shape, message) -> None:
    module = _load_task_module(ROOT / "tasks" / task_id / "model.py", task_id)

    with pytest.raises(ValueError, match=message):
        module.create_inputs(bad_shape, seed=42)


def test_task_005_softmax_rows_sum_to_one() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_005" / "model.py", "task_005")
    (x,) = module.create_inputs((8, 7), seed=42)

    output = module.reference(x)

    assert torch.allclose(output.sum(dim=1), torch.ones(8), atol=1e-6)


def test_task_006_layer_norm_rows_have_zero_mean_and_unit_variance() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_006" / "model.py", "task_006")
    (x,) = module.create_inputs((8, 16), seed=42)

    output = module.reference(x)

    assert torch.allclose(output.mean(dim=1), torch.zeros(8), atol=1e-6)
    assert torch.allclose(output.var(dim=1, unbiased=False), torch.ones(8), atol=1e-4)


def _expected_output_shape(task_id: str, shape: tuple[int, ...]) -> torch.Size:
    if task_id == "task_004":
        return torch.Size([shape[0], shape[2]])
    if task_id in {"task_005", "task_006", "task_007"}:
        return torch.Size(shape)
    if task_id == "task_008":
        return torch.Size([shape[0], shape[2], shape[1]])
    raise AssertionError(f"unknown task id: {task_id}")


def _load_task_module(path: Path, task_id: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{task_id}_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
