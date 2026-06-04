"""Correctness evaluator for prepared generated attempts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from harness.compare_outputs import compare_tensors
from harness.cuda_checks import get_gpu_info
from harness.result_io import write_jsonl
from harness.run_benchmark import benchmark_callable
from harness.result_schema import ShapeBenchResult, TaskSummary
from harness.task_loader import TaskDefinition, load_task


DEFAULT_CONTRACT_FILE = "eval_contract.json"
DEFAULT_BENCHMARK_WARMUP = 10
DEFAULT_BENCHMARK_ITERS = 50
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class EvaluationRun:
    results: list[ShapeBenchResult]
    summary: TaskSummary
    output_path: Path | None


def evaluate_attempt(
    attempt_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "auto",
    seed: int = 0,
    benchmark: bool = True,
    benchmark_warmup: int = DEFAULT_BENCHMARK_WARMUP,
    benchmark_iters: int = DEFAULT_BENCHMARK_ITERS,
    log: LogFn | None = None,
) -> EvaluationRun:
    """Evaluate one prepared attempt across all task shape variants."""
    if benchmark_warmup < 0:
        raise ValueError("benchmark_warmup must be non-negative")
    if benchmark_iters <= 0:
        raise ValueError("benchmark_iters must be positive")
    attempt_path = Path(attempt_dir)
    _log(log, f"loading contract: {attempt_path}")
    contract = _load_contract(attempt_path)
    project_root = _project_root_from_attempt(attempt_path)
    task = load_task(project_root / "tasks" / contract["task_id"])
    resolved_device = _resolve_device(device)
    _ensure_torch_extensions_dir()
    environment = _environment_extra(resolved_device)
    attempt_label = _attempt_label(contract)
    _log(
        log,
        (
            f"{attempt_label}: start task={contract['task_id']} "
            f"device={resolved_device} benchmark={benchmark}"
        ),
    )

    solution_path = attempt_path / contract["extracted_dir"] / contract["entrypoint_file"]
    try:
        _log(log, f"{attempt_label}: importing solution {solution_path}")
        solution_module = _load_module(solution_path, prefix="shapebench_solution")
        forward = getattr(solution_module, contract["entrypoint_function"])
        _log(log, f"{attempt_label}: solution import ready")
    except Exception as exc:
        _log(log, f"{attempt_label}: solution import failed: {_short_error(exc)}")
        results = _failure_results_for_all_shapes(
            task,
            contract,
            failure_reason=_classify_exception(exc),
            message=str(exc),
            device=resolved_device,
            environment=environment,
            phase="import",
        )
        return _finish_run(results, output_path)

    try:
        _log(log, f"{attempt_label}: importing task model {task.model_path}")
        task_module = _load_module(task.model_path, prefix="shapebench_task")
        _log(log, f"{attempt_label}: task model import ready")
    except Exception as exc:
        _log(log, f"{attempt_label}: task model import failed: {_short_error(exc)}")
        results = _failure_results_for_all_shapes(
            task,
            contract,
            failure_reason="runtime_error",
            message=f"failed to load task model: {exc}",
            device=resolved_device,
            environment=environment,
            phase="task_import",
        )
        return _finish_run(results, output_path)

    results = [
        _evaluate_shape(
            task=task,
            task_module=task_module,
            forward=forward,
            contract=contract,
            shape_category=shape_category,
            shape=shape,
            device=resolved_device,
            seed=seed,
            environment=environment,
            benchmark=benchmark,
            benchmark_warmup=benchmark_warmup,
            benchmark_iters=benchmark_iters,
            log=log,
        )
        for shape_category, shape in task.shapes.items()
    ]
    _log(log, f"{attempt_label}: complete")
    return _finish_run(results, output_path)


def default_output_path(attempt_dir: str | Path) -> Path:
    """Return the default JSONL output path for one attempt."""
    attempt_path = Path(attempt_dir)
    contract = _load_contract(attempt_path)
    filename = (
        f"{contract['task_id']}_{contract['prompt_mode']}_"
        f"attempt_{int(contract['attempt']):03d}_correctness.jsonl"
    )
    return _project_root_from_attempt(attempt_path) / "results" / "raw" / filename


def _evaluate_shape(
    *,
    task: TaskDefinition,
    task_module: ModuleType,
    forward: Callable[..., Any],
    contract: dict[str, Any],
    shape_category: str,
    shape: tuple[int, ...],
    device: str,
    seed: int,
    environment: dict[str, Any],
    benchmark: bool,
    benchmark_warmup: int,
    benchmark_iters: int,
    log: LogFn | None,
) -> ShapeBenchResult:
    base_extra = _base_extra(contract, device=device, environment=environment)
    base_extra["seed"] = seed
    attempt_label = _attempt_label(contract)
    shape_label = f"{shape_category} shape={list(shape)}"
    _log(log, f"{attempt_label}: {shape_label}: start")
    try:
        _log(log, f"{attempt_label}: {shape_label}: creating inputs/reference")
        inputs = _create_inputs(task_module, shape, device=device, seed=seed)
        expected = _reference_output(task_module, inputs)
        _log(log, f"{attempt_label}: {shape_label}: running generated forward")
        actual = forward(*inputs)
        _log(log, f"{attempt_label}: {shape_label}: comparing outputs")
        comparison = compare_tensors(
            expected,
            actual,
            atol=float(task.metadata["atol"]),
            rtol=float(task.metadata["rtol"]),
        )
    except Exception as exc:
        _log(log, f"{attempt_label}: {shape_label}: failed during correctness: {_short_error(exc)}")
        return _result(
            task=task,
            contract=contract,
            shape_category=shape_category,
            shape=shape,
            correct=False,
            failure_reason=_classify_exception(exc),
            max_abs_error=None,
            mean_abs_error=None,
            extra={
                **base_extra,
                "phase": "shape_evaluation",
                "error": str(exc),
            },
        )

    failure_reason = None
    if not comparison.passed:
        failure_reason = (
            "original_shape_correctness_failure"
            if shape_category == "original"
            else "shape_variant_correctness_failure"
        )
        _log(
            log,
            (
                f"{attempt_label}: {shape_label}: correctness failed "
                f"max_abs_error={comparison.max_abs_error}"
            ),
        )
    else:
        _log(
            log,
            (
                f"{attempt_label}: {shape_label}: correctness passed "
                f"max_abs_error={comparison.max_abs_error}"
            ),
        )
    timing = _benchmark_shape(
        task_module=task_module,
        forward=forward,
        inputs=inputs,
        enabled=benchmark and comparison.passed,
        warmup=benchmark_warmup,
        iters=benchmark_iters,
        log=log,
        label=f"{attempt_label}: {shape_label}",
    )
    _log(log, f"{attempt_label}: {shape_label}: complete")
    return _result(
        task=task,
        contract=contract,
        shape_category=shape_category,
        shape=shape,
        correct=comparison.passed,
        failure_reason=failure_reason,
        max_abs_error=comparison.max_abs_error,
        mean_abs_error=comparison.mean_abs_error,
        pytorch_eager_ms=timing["pytorch_eager_ms"],
        generated_ms=timing["generated_ms"],
        speedup_vs_eager=timing["speedup_vs_eager"],
        extra={
            **base_extra,
            "phase": "shape_evaluation",
            "comparison": comparison.message,
            "benchmark": timing["extra"],
        },
    )


def _failure_results_for_all_shapes(
    task: TaskDefinition,
    contract: dict[str, Any],
    *,
    failure_reason: str,
    message: str,
    device: str,
    environment: dict[str, Any],
    phase: str,
) -> list[ShapeBenchResult]:
    return [
        _result(
            task=task,
            contract=contract,
            shape_category=shape_category,
            shape=shape,
            correct=False,
            failure_reason=failure_reason,
            max_abs_error=None,
            mean_abs_error=None,
            pytorch_eager_ms=None,
            generated_ms=None,
            speedup_vs_eager=None,
            extra={
                **_base_extra(contract, device=device, environment=environment),
                "phase": phase,
                "error": message,
            },
        )
        for shape_category, shape in task.shapes.items()
    ]


def _finish_run(results: list[ShapeBenchResult], output_path: str | Path | None) -> EvaluationRun:
    summary = TaskSummary.from_results(results)
    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        write_jsonl(destination, results)
    return EvaluationRun(results=results, summary=summary, output_path=destination)


def _result(
    *,
    task: TaskDefinition,
    contract: dict[str, Any],
    shape_category: str,
    shape: tuple[int, ...],
    correct: bool,
    failure_reason: str | None,
    max_abs_error: float | None,
    mean_abs_error: float | None,
    extra: dict[str, Any],
    pytorch_eager_ms: float | None = None,
    generated_ms: float | None = None,
    speedup_vs_eager: float | None = None,
) -> ShapeBenchResult:
    environment = extra.get("environment", {})
    return ShapeBenchResult(
        task_id=task.metadata["task_id"],
        prompt_mode=contract["prompt_mode"],
        shape_category=shape_category,
        shape=shape,
        correct=correct,
        failure_reason=failure_reason,
        pytorch_eager_ms=pytorch_eager_ms,
        generated_ms=generated_ms,
        speedup_vs_eager=speedup_vs_eager,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        gpu_name=environment.get("gpu_name"),
        cuda_version=environment.get("cuda_version"),
        torch_version=environment.get("torch_version"),
        extra=extra,
    )


def _load_contract(attempt_dir: Path) -> dict[str, Any]:
    contract_path = attempt_dir / "extracted" / DEFAULT_CONTRACT_FILE
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing evaluation contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError(f"evaluation contract must be a JSON object: {contract_path}")
    for field in (
        "entrypoint_file",
        "entrypoint_function",
        "extracted_dir",
        "task_id",
        "prompt_mode",
        "attempt",
        "input_names",
    ):
        if field not in contract:
            raise ValueError(f"evaluation contract missing {field!r}: {contract_path}")
    return contract


def _load_module(path: Path, *, prefix: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"missing Python module: {path}")
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module_dir = str(path.parent)
    inserted_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if inserted_path:
            sys.path.remove(module_dir)
    return module


def _create_inputs(task_module: ModuleType, shape: tuple[int, ...], *, device: str, seed: int) -> tuple[Any, ...]:
    if not hasattr(task_module, "create_inputs"):
        raise AttributeError("task model.py must define create_inputs")
    inputs = task_module.create_inputs(shape, device=device, seed=seed)
    if not isinstance(inputs, tuple):
        raise TypeError("create_inputs must return a tuple")
    return inputs


def _reference_output(task_module: ModuleType, inputs: tuple[Any, ...]) -> Any:
    if hasattr(task_module, "reference"):
        return task_module.reference(*inputs)
    if hasattr(task_module, "Model"):
        return task_module.Model()(*inputs)
    raise AttributeError("task model.py must define reference or Model")


def _benchmark_shape(
    *,
    task_module: ModuleType,
    forward: Callable[..., Any],
    inputs: tuple[Any, ...],
    enabled: bool,
    warmup: int,
    iters: int,
    log: LogFn | None,
    label: str,
) -> dict[str, Any]:
    if not enabled:
        _log(log, f"{label}: benchmark skipped")
        return {
            "pytorch_eager_ms": None,
            "generated_ms": None,
            "speedup_vs_eager": None,
            "extra": {
                "enabled": False,
                "warmup": warmup,
                "iterations": iters,
            },
        }

    extra: dict[str, Any] = {
        "enabled": True,
        "warmup": warmup,
        "iterations": iters,
    }
    try:
        _log(log, f"{label}: benchmark PyTorch eager start warmup={warmup} iters={iters}")
        eager = benchmark_callable(
            lambda *args: _reference_output(task_module, args),
            inputs,
            warmup=warmup,
            iters=iters,
        )
        _log(log, f"{label}: benchmark generated start warmup={warmup} iters={iters}")
        generated = benchmark_callable(
            forward,
            inputs,
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:
        _log(log, f"{label}: benchmark failed: {_short_error(exc)}")
        extra["error"] = str(exc)
        return {
            "pytorch_eager_ms": None,
            "generated_ms": None,
            "speedup_vs_eager": None,
            "extra": extra,
        }

    speedup = None
    if generated.average_ms > 0:
        speedup = eager.average_ms / generated.average_ms
    _log(
        log,
        (
            f"{label}: benchmark done "
            f"pytorch_eager_ms={eager.average_ms:.6f} "
            f"generated_ms={generated.average_ms:.6f} "
            f"speedup={speedup:.3f}" if speedup is not None else
            f"{label}: benchmark done "
            f"pytorch_eager_ms={eager.average_ms:.6f} "
            f"generated_ms={generated.average_ms:.6f} speedup=None"
        ),
    )
    extra["pytorch_eager_total_ms"] = eager.total_ms
    extra["generated_total_ms"] = generated.total_ms
    return {
        "pytorch_eager_ms": eager.average_ms,
        "generated_ms": generated.average_ms,
        "speedup_vs_eager": speedup,
        "extra": extra,
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if get_gpu_info()["cuda_available"] else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return device


def _environment_extra(device: str) -> dict[str, Any]:
    info = get_gpu_info()
    return {
        "requested_device": device,
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "torch_version": info["torch_version"],
        "cuda_available": info["cuda_available"],
        "cuda_version": info["cuda_version"],
        "gpu_name": info["gpu_name"],
        "device_count": info["device_count"],
    }


def _base_extra(contract: dict[str, Any], *, device: str, environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt": int(contract["attempt"]),
        "entrypoint": f"{contract['entrypoint_file']}:{contract['entrypoint_function']}",
        "created_fallback_solution": contract.get("created_fallback_solution"),
        "cuda_source": contract.get("cuda_source"),
        "extension_function": contract.get("extension_function"),
        "extension_name": contract.get("extension_name"),
        "input_names": list(contract["input_names"]),
        "device": device,
        "environment": environment,
    }


def _classify_exception(exc: Exception) -> str:
    message = str(exc).lower()
    compile_markers = (
        "cuda_home",
        "nvcc",
        "ninja",
        "compile",
        "compilation",
        "cpp_extension",
        "no cuda",
        "not compiled with cuda",
        "torch_extensions",
    )
    if any(marker in message for marker in compile_markers):
        return "compilation_failure"
    return "runtime_error"


def _attempt_label(contract: dict[str, Any]) -> str:
    return (
        f"{contract['task_id']} {contract['prompt_mode']} "
        f"attempt_{int(contract['attempt']):03d}"
    )


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def _short_error(exc: Exception, *, limit: int = 240) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _ensure_torch_extensions_dir() -> None:
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(Path(tempfile.gettempdir()) / "shape_bench_torch_extensions"),
    )


def _project_root_from_attempt(attempt_dir: Path) -> Path:
    current = attempt_dir.resolve()
    for parent in (current, *current.parents):
        if (parent / "tasks").is_dir() and (parent / "generated").is_dir():
            return parent
    raise ValueError(f"could not locate project root from attempt directory: {attempt_dir}")
