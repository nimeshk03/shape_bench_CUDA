"""Simple benchmark helpers for CPU or CUDA callables."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkResult:
    average_ms: float
    total_ms: float
    warmup: int
    iterations: int


def benchmark_callable(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...] | list[Any] | Any = (),
    *,
    warmup: int = 10,
    iters: int = 100,
    synchronize_cuda: bool = True,
) -> BenchmarkResult:
    """Benchmark a callable with optional CUDA synchronization."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")

    args = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    for _ in range(warmup):
        fn(*args)

    _sync_cuda_if_available(synchronize_cuda)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    _sync_cuda_if_available(synchronize_cuda)

    total_ms = (time.perf_counter() - start) * 1000.0
    return BenchmarkResult(
        average_ms=total_ms / iters,
        total_ms=total_ms,
        warmup=warmup,
        iterations=iters,
    )


def _sync_cuda_if_available(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()

