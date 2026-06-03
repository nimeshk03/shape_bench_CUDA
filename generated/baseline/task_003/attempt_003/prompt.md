# ShapeBench-CUDA Rendered Prompt

Prompt mode: `baseline`
Task id: `task_003`

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
  "atol": 0.0,
  "category": "layout",
  "description": "Transpose a 2D float32 tensor from shape [rows, cols] to [cols, rows].",
  "dtype": "float32",
  "expected_output": "2D tensor with rows and columns swapped",
  "input_kind": "matrix",
  "input_names": [
    "x"
  ],
  "name": "matrix_transpose",
  "notes": "Layout task intended to expose row/column indexing mistakes, square-shape assumptions, odd dimensions, and non-power-of-two boundary errors.",
  "original_shape": [
    1024,
    2048
  ],
  "rtol": 0.0,
  "task_id": "task_003"
}
```

## Original Shape

Baseline generation should optimize for this original shape.

```json
{
  "original": [
    1024,
    2048
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 003: 2D matrix transpose."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for transposing a 2D tensor."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(x, 0, 1).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 2D input for matrix transpose."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_003 expects a 2D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.transpose(x, 0, 1).contiguous()
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
