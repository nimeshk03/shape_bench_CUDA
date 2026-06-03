# ShapeBench-CUDA Rendered Prompt

Prompt mode: `baseline`
Task id: `task_001`

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
  "atol": 0.0001,
  "category": "elementwise",
  "description": "Compute relu(x + y) for two tensors with the same shape.",
  "dtype": "float32",
  "expected_output": "Tensor with same shape as inputs",
  "input_kind": "matrix",
  "input_names": [
    "x",
    "y"
  ],
  "name": "elementwise_add_relu",
  "notes": "Starter task for validating shape-generalization harness behavior before CUDA experiments.",
  "original_shape": [
    1024,
    1024
  ],
  "rtol": 0.0001,
  "task_id": "task_001"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    256,
    1024
  ],
  "larger": [
    2048,
    1024
  ],
  "non_power_of_two": [
    1000,
    1007
  ],
  "odd": [
    1007,
    1013
  ],
  "original": [
    1024,
    1024
  ],
  "smaller": [
    512,
    1024
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 001: elementwise add followed by ReLU."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for relu(x + y)."""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.relu(x + y)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic inputs for local correctness checks."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    normalized_shape = tuple(int(dim) for dim in shape)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    y = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return x.to(device), y.to(device)


def reference(x: Tensor, y: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.relu(x + y)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
