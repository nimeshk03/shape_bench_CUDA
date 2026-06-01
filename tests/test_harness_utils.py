from __future__ import annotations

import json

from harness.compare_outputs import compare_tensors
from harness.cuda_checks import get_gpu_info, is_cuda_available
from harness.run_benchmark import benchmark_callable
from harness.task_loader import load_metadata, load_shapes, load_task


def test_compare_tensors_passes_close_values() -> None:
    comparison = compare_tensors([1.0, 2.0], [1.0, 2.001], atol=1e-2, rtol=1e-2)

    assert comparison.passed is True
    assert comparison.max_abs_error is not None


def test_compare_tensors_reports_shape_mismatch() -> None:
    comparison = compare_tensors([1.0, 2.0], [1.0])

    assert comparison.passed is False
    assert "shape mismatch" in comparison.message


def test_cuda_checks_are_cpu_safe() -> None:
    info = get_gpu_info()

    assert isinstance(is_cuda_available(), bool)
    assert "cuda_available" in info
    assert "gpu_name" in info


def test_benchmark_callable_runs_on_cpu() -> None:
    result = benchmark_callable(lambda value: value + 1, (1,), warmup=1, iters=3)

    assert result.iterations == 3
    assert result.average_ms >= 0


def test_task_loader_reads_task_definition(tmp_path) -> None:
    task_dir = tmp_path / "task_001"
    task_dir.mkdir()
    (task_dir / "model.py").write_text("def model(x):\n    return x\n", encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps({"task_id": "task_001", "name": "demo"}),
        encoding="utf-8",
    )
    (task_dir / "shapes.json").write_text(
        json.dumps(
            {
                "original": [8, 8],
                "smaller": [4, 8],
                "larger": [16, 8],
                "odd": [7, 9],
                "batch_variant": [2, 8],
                "non_power_of_two": [7, 9],
            }
        ),
        encoding="utf-8",
    )

    assert load_metadata(task_dir)["task_id"] == "task_001"
    assert load_shapes(task_dir)["odd"] == (7, 9)
    task = load_task(task_dir)
    assert task.model_path.name == "model.py"
