from __future__ import annotations

import json

from harness.evaluator import default_output_path, evaluate_attempt
from harness.result_io import read_jsonl
from harness.result_schema import ShapeBenchResult


def test_evaluate_attempt_passes_cpu_solution_and_writes_jsonl(tmp_path) -> None:
    attempt_dir = _make_project(tmp_path, solution_source=_solution_source("torch.relu(x + y)"))
    output_path = tmp_path / "results" / "attempt.jsonl"

    run = evaluate_attempt(
        attempt_dir,
        output_path=output_path,
        device="cpu",
        seed=123,
        benchmark_warmup=1,
        benchmark_iters=2,
    )

    assert run.summary.passed_shapes == 6
    assert run.summary.total_shapes == 6
    assert run.summary.original_passed is True
    assert run.summary.failure_reasons == {}
    loaded = read_jsonl(output_path, ShapeBenchResult)
    assert len(loaded) == 6
    assert all(result.correct for result in loaded)
    assert loaded[0].extra["attempt"] == 1
    assert loaded[0].extra["device"] == "cpu"
    assert loaded[0].pytorch_eager_ms is not None
    assert loaded[0].generated_ms is not None
    assert loaded[0].speedup_vs_eager is not None
    assert loaded[0].extra["benchmark"]["enabled"] is True
    assert loaded[0].extra["benchmark"]["iterations"] == 2
    assert loaded[0].extra["input_layouts"] == [
        {
            "device": "cpu",
            "dtype": "torch.float32",
            "index": 0,
            "is_contiguous": True,
            "name": "x",
            "shape": [4, 4],
            "storage_offset": 0,
            "stride": [4, 1],
            "type": "Tensor",
        },
        {
            "device": "cpu",
            "dtype": "torch.float32",
            "index": 1,
            "is_contiguous": True,
            "name": "y",
            "shape": [4, 4],
            "storage_offset": 0,
            "stride": [4, 1],
            "type": "Tensor",
        },
    ]


def test_evaluate_attempt_can_disable_benchmarking(tmp_path) -> None:
    attempt_dir = _make_project(tmp_path, solution_source=_solution_source("torch.relu(x + y)"))

    run = evaluate_attempt(attempt_dir, device="cpu", benchmark=False)

    assert run.summary.passed_shapes == 6
    assert all(result.generated_ms is None for result in run.results)
    assert all(result.pytorch_eager_ms is None for result in run.results)
    assert all(result.speedup_vs_eager is None for result in run.results)
    assert {result.extra["benchmark"]["enabled"] for result in run.results} == {False}


def test_evaluate_attempt_records_correctness_failures(tmp_path) -> None:
    attempt_dir = _make_project(tmp_path, solution_source=_solution_source("x + y"))

    run = evaluate_attempt(attempt_dir, device="cpu", benchmark_warmup=1, benchmark_iters=2)

    assert run.summary.passed_shapes == 0
    assert run.summary.failure_reasons == {
        "original_shape_correctness_failure": 1,
        "shape_variant_correctness_failure": 5,
    }
    original = next(result for result in run.results if result.shape_category == "original")
    assert original.failure_reason == "original_shape_correctness_failure"
    assert original.max_abs_error is not None
    assert original.generated_ms is None


def test_evaluate_attempt_records_import_compile_failure_for_all_shapes(tmp_path) -> None:
    attempt_dir = _make_project(
        tmp_path,
        solution_source='raise RuntimeError("CUDA_HOME environment variable is not set")\n',
    )

    run = evaluate_attempt(attempt_dir, device="cpu")

    assert run.summary.passed_shapes == 0
    assert run.summary.failure_reasons == {"compilation_failure": 6}
    assert {result.extra["phase"] for result in run.results} == {"import"}
    assert all("CUDA_HOME" in result.extra["error"] for result in run.results)


def test_evaluate_attempt_allows_solution_local_helper_imports(tmp_path) -> None:
    attempt_dir = _make_project(
        tmp_path,
        solution_source=(
            "from helper import relu_add\n\n"
            "def forward(x, y):\n"
            "    return relu_add(x, y)\n"
        ),
    )
    (attempt_dir / "extracted" / "helper.py").write_text(
        "import torch\n\n"
        "def relu_add(x, y):\n"
        "    return torch.relu(x + y)\n",
        encoding="utf-8",
    )

    run = evaluate_attempt(attempt_dir, device="cpu", benchmark_warmup=1, benchmark_iters=2)

    assert run.summary.passed_shapes == 6


def test_default_output_path_is_results_raw_jsonl(tmp_path) -> None:
    attempt_dir = _make_project(tmp_path, solution_source=_solution_source("torch.relu(x + y)"))

    assert default_output_path(attempt_dir) == (
        tmp_path / "results" / "raw" / "task_001_baseline_attempt_001_correctness.jsonl"
    )


def _make_project(tmp_path, *, solution_source: str):
    task_dir = tmp_path / "tasks" / "task_001"
    task_dir.mkdir(parents=True)
    (tmp_path / "generated").mkdir()
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "name": "demo",
                "description": "demo task",
                "category": "elementwise",
                "input_kind": "matrix",
                "input_names": ["x", "y"],
                "original_shape": [4, 4],
                "dtype": "float32",
                "atol": 1e-4,
                "rtol": 1e-4,
                "expected_output": "same shape tensor",
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "shapes.json").write_text(
        json.dumps(
            {
                "original": [4, 4],
                "smaller": [2, 4],
                "larger": [8, 4],
                "odd": [5, 5],
                "batch_variant": [1, 4],
                "non_power_of_two": [3, 5],
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "model.py").write_text(
        """
from __future__ import annotations

import torch


def create_inputs(shape, *, device="cpu", seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    y = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return x.to(device), y.to(device)


def reference(x, y):
    return torch.relu(x + y)
""",
        encoding="utf-8",
    )

    attempt_dir = tmp_path / "generated" / "baseline" / "task_001" / "attempt_001"
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(solution_source, encoding="utf-8")
    (extracted_dir / "eval_contract.json").write_text(
        json.dumps(
            {
                "attempt": 1,
                "attempt_dir": str(attempt_dir),
                "created_fallback_solution": False,
                "cuda_source": None,
                "entrypoint_file": "solution.py",
                "entrypoint_function": "forward",
                "extension_function": None,
                "extension_name": None,
                "extracted_dir": "extracted",
                "input_names": ["x", "y"],
                "prompt_mode": "baseline",
                "task_id": "task_001",
            }
        ),
        encoding="utf-8",
    )
    return attempt_dir


def _solution_source(expression: str) -> str:
    return f"""
from __future__ import annotations

import torch


def forward(x, y):
    return {expression}
"""
