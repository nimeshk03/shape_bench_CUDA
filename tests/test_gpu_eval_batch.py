from __future__ import annotations

import json

from harness.gpu_eval_batch import run_gpu_eval_batch


def test_run_gpu_eval_batch_writes_summary_for_attempts(tmp_path) -> None:
    first_attempt = _make_project_attempt(tmp_path, prompt_mode="baseline", attempt=1)
    second_attempt = _make_project_attempt(tmp_path, prompt_mode="shape_aware", attempt=2)
    summary_output = tmp_path / "results" / "tables" / "batch.json"

    batch = run_gpu_eval_batch(
        project_root=tmp_path,
        attempt_dirs=[first_attempt, second_attempt],
        summary_output=summary_output,
        device="cpu",
        require_cuda=False,
        run_preflight=False,
        benchmark_warmup=1,
        benchmark_iters=2,
    )

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert batch.summary_output == summary_output.relative_to(tmp_path)
    assert summary["require_cuda"] is False
    assert summary["benchmark"] == {"enabled": True, "iterations": 2, "warmup": 1}
    assert summary["preflight"] == {"skipped": True}
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][0]["prompt_mode"] == "baseline"
    assert summary["attempts"][0]["passed_shapes"] == 6
    assert summary["attempts"][1]["prompt_mode"] == "shape_aware"
    assert summary["attempts"][1]["passed_shapes"] == 6


def test_run_gpu_eval_batch_logs_attempt_and_shape_progress(tmp_path) -> None:
    attempt = _make_project_attempt(tmp_path, prompt_mode="baseline", attempt=1)
    messages: list[str] = []

    run_gpu_eval_batch(
        project_root=tmp_path,
        attempt_dirs=[attempt],
        device="cpu",
        require_cuda=False,
        run_preflight=False,
        benchmark_warmup=1,
        benchmark_iters=1,
        log=messages.append,
    )

    assert any("batch start: attempts=1" in message for message in messages)
    assert any("attempt 1/1 start" in message for message in messages)
    assert any("task_001 baseline attempt_001: original shape=[4, 4]: start" in message for message in messages)
    assert any("benchmark generated start" in message for message in messages)
    assert messages[-1] == "batch complete"


def _make_project_attempt(tmp_path, *, prompt_mode: str, attempt: int):
    task_dir = tmp_path / "tasks" / "task_001"
    task_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated").mkdir(exist_ok=True)
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

    attempt_dir = tmp_path / "generated" / prompt_mode / "task_001" / f"attempt_{attempt:03d}"
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(
        "import torch\n\n"
        "def forward(x, y):\n"
        "    return torch.relu(x + y)\n",
        encoding="utf-8",
    )
    (extracted_dir / "eval_contract.json").write_text(
        json.dumps(
            {
                "attempt": attempt,
                "attempt_dir": str(attempt_dir),
                "created_fallback_solution": False,
                "cuda_source": None,
                "entrypoint_file": "solution.py",
                "entrypoint_function": "forward",
                "extension_function": None,
                "extension_name": None,
                "extracted_dir": "extracted",
                "input_names": ["x", "y"],
                "prompt_mode": prompt_mode,
                "task_id": "task_001",
            }
        ),
        encoding="utf-8",
    )
    return attempt_dir
