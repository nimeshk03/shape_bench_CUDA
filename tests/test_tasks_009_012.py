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
        ("task_009", "noncontiguous_affine_relu", "non_contiguous", ["x", "scale", "bias"], (1024, 1024)),
        ("task_010", "dynamic_lastdim_sum_squares", "reduction", ["x"], (16, 256, 512)),
        ("task_011", "batched_matrix_multiply", "batched_matmul", ["a", "b"], (8, 128, 128, 128)),
        ("task_012", "strided_batched_transpose", "stride_layout", ["x"], (16, 256, 512)),
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


@pytest.mark.parametrize("task_id", ["task_009", "task_010", "task_011", "task_012"])
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


@pytest.mark.parametrize("task_id", ["task_009", "task_010", "task_011", "task_012"])
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
        ("task_009", (8,), "expects a 2D shape"),
        ("task_010", (8, 8), "expects a 3D shape"),
        ("task_011", (8, 8, 8), "expects a 4D shape descriptor"),
        ("task_012", (8, 8), "expects a 3D shape"),
    ],
)
def test_tasks_reject_wrong_rank_shapes(task_id, bad_shape, message) -> None:
    module = _load_task_module(ROOT / "tasks" / task_id / "model.py", task_id)

    with pytest.raises(ValueError, match=message):
        module.create_inputs(bad_shape, seed=42)


def test_task_009_input_is_non_contiguous() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_009" / "model.py", "task_009")
    x, scale, bias = module.create_inputs((8, 7), seed=42)

    assert not x.is_contiguous()
    assert x.stride()[1] == 2
    assert scale.shape == torch.Size([7])
    assert bias.shape == torch.Size([7])


def test_task_010_reduces_last_dimension() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_010" / "model.py", "task_010")
    (x,) = module.create_inputs((3, 5, 7), seed=42)

    output = module.reference(x)

    assert output.shape == torch.Size([3, 5])
    assert torch.all(output >= 0)


def test_task_011_uses_independent_batched_matmul() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_011" / "model.py", "task_011")
    a, b = module.create_inputs((2, 3, 4, 5), seed=42)

    output = module.reference(a, b)

    assert output.shape == torch.Size([2, 3, 5])
    assert torch.allclose(output[0], a[0] @ b[0])
    assert torch.allclose(output[1], a[1] @ b[1])


def test_task_012_input_is_non_contiguous_and_output_is_transposed_contiguous() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_012" / "model.py", "task_012")
    (x,) = module.create_inputs((3, 5, 7), seed=42)

    output = module.reference(x)

    assert not x.is_contiguous()
    assert x.stride()[1] == 28
    assert x.stride()[2] == 2
    assert output.shape == torch.Size([3, 7, 5])
    assert output.is_contiguous()


def _expected_output_shape(task_id: str, shape: tuple[int, ...]) -> torch.Size:
    if task_id == "task_009":
        return torch.Size(shape)
    if task_id == "task_010":
        return torch.Size([shape[0], shape[1]])
    if task_id == "task_011":
        return torch.Size([shape[0], shape[1], shape[3]])
    if task_id == "task_012":
        return torch.Size([shape[0], shape[2], shape[1]])
    raise AssertionError(f"unknown task id: {task_id}")


def _load_task_module(path: Path, task_id: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{task_id}_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
