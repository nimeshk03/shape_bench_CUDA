# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_011`

## Generation Instructions

# Shape-Aware CUDA Generation Prompt

You are given a PyTorch model, input specification, original shape, and shape variants. Write a high-performance CUDA/C++ extension implementation that matches the PyTorch model's output across all listed shapes.

## Goal

Optimize the given model for CUDA execution while preserving numerical correctness across shape changes.

## Requirements

- Implement the operation using CUDA/C++ extension code.
- Match the PyTorch reference output within the provided tolerance.
- Use runtime tensor shape information instead of hardcoded dimensions.
- Handle the original shape and all provided shape variants.
- Handle smaller and larger shapes.
- Handle odd dimensions.
- Handle non-power-of-two dimensions.
- Handle batch-size changes when a batch dimension is present.
- Include boundary checks so tail elements and partial blocks are correct.
- Avoid assuming dimensions are block-aligned, warp-aligned, or powers of two.
- Avoid using PyTorch tensor operations inside the generated CUDA kernel.
- Return output tensors with the same shape and dtype expected by the PyTorch reference.
- Preserve performance where possible, but do not trade away correctness on shape variants.

## Output

Provide only the generated implementation code and any minimal build glue required by the harness.

Do not include unrelated explanation, training code, reinforcement learning code, or large benchmark scripts.

## Task Metadata

```json
{
  "atol": 0.01,
  "category": "batched_matmul",
  "description": "Compute C = A @ B for batched matrices A shaped [batch, M, K] and B shaped [batch, K, N].",
  "dtype": "float32",
  "expected_output": "3D tensor with shape [batch, M, N]",
  "input_kind": "batched_matrix_pair",
  "input_names": [
    "a",
    "b"
  ],
  "name": "batched_matrix_multiply",
  "notes": "Batched matmul task intended to expose hardcoded batch, tile, and dimension assumptions.",
  "original_shape": [
    8,
    128,
    128,
    128
  ],
  "rtol": 0.01,
  "task_id": "task_011"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    3,
    128,
    64,
    256
  ],
  "larger": [
    16,
    128,
    256,
    128
  ],
  "non_power_of_two": [
    7,
    125,
    127,
    129
  ],
  "odd": [
    5,
    63,
    65,
    67
  ],
  "original": [
    8,
    128,
    128,
    128
  ],
  "smaller": [
    4,
    64,
    128,
    64
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 011: batched matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for batched matrix multiplication."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.bmm(a, b)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic batched matrix multiplication inputs."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 4:
        raise ValueError(f"task_011 expects a 4D shape descriptor, got {normalized_shape}")
    batch, rows, shared, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((batch, rows, shared), generator=generator, dtype=dtype)
    b = torch.randn((batch, shared, cols), generator=generator, dtype=dtype)
    return a.to(device), b.to(device)


def reference(a: Tensor, b: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.bmm(a, b)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
