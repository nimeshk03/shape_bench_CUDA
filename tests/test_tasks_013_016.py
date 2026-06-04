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
        ("task_013", "diagnostic_batched_transpose", "diagnostic_layout", ["x"], (4, 64, 96)),
        ("task_014", "tile_aligned_to_irregular_transpose", "shape_variant_trap", ["x"], (8, 128, 128)),
        ("task_015", "offset_strided_affine_relu", "strong_non_contiguous", ["x", "scale", "bias"], (8, 128, 192)),
        ("task_016", "irregular_lastdim_layer_norm", "irregular_reduction", ["x", "weight", "bias"], (8, 128, 512)),
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


@pytest.mark.parametrize("task_id", ["task_013", "task_014", "task_015", "task_016"])
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


@pytest.mark.parametrize("task_id", ["task_013", "task_014", "task_015", "task_016"])
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
        ("task_013", (8, 8), "expects a 3D shape"),
        ("task_014", (8, 8), "expects a 3D shape"),
        ("task_015", (8, 8), "expects a 3D shape"),
        ("task_016", (8, 8), "expects a 3D shape"),
    ],
)
def test_tasks_reject_wrong_rank_shapes(task_id, bad_shape, message) -> None:
    module = _load_task_module(ROOT / "tasks" / task_id / "model.py", task_id)

    with pytest.raises(ValueError, match=message):
        module.create_inputs(bad_shape, seed=42)


def test_task_013_values_encode_source_indices() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_013" / "model.py", "task_013")
    (x,) = module.create_inputs((2, 3, 4), seed=42)

    assert x[1, 2, 3].item() == 1_002_003
    output = module.reference(x)
    assert output[1, 3, 2].item() == 1_002_003


def test_task_014_uses_square_original_and_irregular_variants() -> None:
    task = load_task(ROOT / "tasks" / "task_014")

    assert task.shapes["original"][1] == task.shapes["original"][2]
    assert task.shapes["odd"][1] != task.shapes["odd"][2]
    assert task.shapes["non_power_of_two"][2] == 257


def test_task_015_input_has_offset_and_irregular_strides() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_015" / "model.py", "task_015")
    x, scale, bias = module.create_inputs((3, 5, 7), seed=42)

    assert not x.is_contiguous()
    assert x.storage_offset() > 0
    assert x.stride()[1] == 52
    assert x.stride()[2] == 3
    assert scale.shape == torch.Size([7])
    assert bias.shape == torch.Size([5])


def test_task_016_layer_norm_uses_dynamic_last_dimension() -> None:
    module = _load_task_module(ROOT / "tasks" / "task_016" / "model.py", "task_016")
    x, weight, bias = module.create_inputs((2, 3, 7), seed=42)

    output = module.reference(x, weight, bias)

    assert output.shape == torch.Size([2, 3, 7])
    assert output.is_contiguous()
    assert weight.shape == torch.Size([7])
    assert bias.shape == torch.Size([7])


def _expected_output_shape(task_id: str, shape: tuple[int, ...]) -> torch.Size:
    if task_id in {"task_013", "task_014"}:
        return torch.Size([shape[0], shape[2], shape[1]])
    if task_id in {"task_015", "task_016"}:
        return torch.Size(shape)
    raise AssertionError(f"unknown task id: {task_id}")


def _load_task_module(path: Path, task_id: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{task_id}_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
