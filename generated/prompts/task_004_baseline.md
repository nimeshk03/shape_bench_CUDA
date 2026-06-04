# ShapeBench-CUDA Rendered Prompt

Prompt mode: `baseline`
Task id: `task_004`

## Generation Instructions

# Baseline CUDA Generation Prompt

You are given a PyTorch model and input specification. Write a high-performance CUDA/C++ extension implementation that matches the PyTorch model's output.

## Goal

Optimize the given model for CUDA execution while preserving numerical correctness.

## Requirements

- Implement the operation using CUDA/C++ extension code.
- Match the PyTorch reference output within the provided tolerance.
- Use the provided input and output specifications.
- Return output tensors with the same shape and dtype expected by the PyTorch reference.
- Avoid using PyTorch tensor operations inside the generated CUDA kernel.
- Include necessary boundary checks for valid memory access.
- Prefer clear, maintainable CUDA code before complex optimization.
- Optimize for runtime performance against PyTorch eager execution.
- If reasonable, also optimize against `torch.compile`.

## Output

Provide only the generated implementation code and any minimal build glue required by the harness.

Do not include unrelated explanation, training code, reinforcement learning code, or large benchmark scripts.

## Task Metadata

```json
{
  "atol": 0.01,
  "category": "matmul",
  "description": "Compute matrix multiplication C = A @ B for A shaped [M, K] and B shaped [K, N].",
  "dtype": "float32",
  "expected_output": "2D tensor with shape [M, N]",
  "input_kind": "matrix_pair",
  "input_names": [
    "a",
    "b"
  ],
  "name": "matrix_multiply",
  "notes": "Tiling-oriented task intended to expose hardcoded dimensions and block-boundary assumptions.",
  "original_shape": [
    256,
    256,
    256
  ],
  "rtol": 0.01,
  "task_id": "task_004"
}
```

## Original Shape

Baseline generation should optimize for this original shape.

```json
{
  "original": [
    256,
    256,
    256
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 004: matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for C = A @ B."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic matrix-multiply inputs from [M, K, N]."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_004 expects a 3D shape descriptor [M, K, N], got {normalized_shape}")
    m, k, n = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((m, k), generator=generator, dtype=dtype)
    b = torch.randn((k, n), generator=generator, dtype=dtype)
    return a.to(device), b.to(device)


def reference(a: Tensor, b: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.matmul(a, b)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
