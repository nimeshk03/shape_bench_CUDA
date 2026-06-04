# ShapeBench-CUDA Rendered Prompt

Prompt mode: `baseline`
Task id: `task_006`

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
  "atol": 0.001,
  "category": "normalization",
  "description": "Normalize each row of a 2D float32 tensor using row mean and variance.",
  "dtype": "float32",
  "expected_output": "2D tensor with same shape as input",
  "input_kind": "matrix",
  "input_names": [
    "x"
  ],
  "name": "rowwise_layer_norm",
  "notes": "Two-pass row reduction task intended to expose assumptions about row length and reduction synchronization.",
  "original_shape": [
    1024,
    1024
  ],
  "rtol": 0.001,
  "task_id": "task_006"
}
```

## Original Shape

Baseline generation should optimize for this original shape.

```json
{
  "original": [
    1024,
    1024
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 006: row-wise layer normalization without affine parameters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


EPSILON = 1e-5


class Model(nn.Module):
    """Reference PyTorch module for row-wise layer normalization."""

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + EPSILON)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 2D input for row-wise normalization."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_006 expects a 2D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    mean = torch.mean(x, dim=1, keepdim=True)
    variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
    return (x - mean) * torch.rsqrt(variance + EPSILON)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
