# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_013`

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
  "atol": 0.0,
  "category": "diagnostic_layout",
  "description": "Transpose the last two dimensions of a 3D float32 tensor whose values encode their original batch, row, and column indices.",
  "dtype": "float32",
  "expected_output": "3D contiguous tensor with shape [batch, cols, rows]",
  "input_kind": "encoded_batched_matrix",
  "input_names": [
    "x"
  ],
  "name": "diagnostic_batched_transpose",
  "notes": "Diagnostic layout task intended to make wrong transpose/indexing bugs easier to identify than with random inputs.",
  "original_shape": [
    4,
    64,
    96
  ],
  "rtol": 0.0,
  "task_id": "task_013"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    1,
    64,
    128
  ],
  "larger": [
    8,
    128,
    192
  ],
  "non_power_of_two": [
    5,
    65,
    97
  ],
  "odd": [
    3,
    63,
    95
  ],
  "original": [
    4,
    64,
    96
  ],
  "smaller": [
    2,
    32,
    96
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 013: diagnostic batched transpose with index-encoded values."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for transposing encoded batched matrices."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic index-encoded 3D input for batched transpose."""
    del seed
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_013 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    b = torch.arange(batch, dtype=dtype).view(batch, 1, 1)
    r = torch.arange(rows, dtype=dtype).view(1, rows, 1)
    c = torch.arange(cols, dtype=dtype).view(1, 1, cols)
    x = b * 1_000_000 + r * 1_000 + c
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.transpose(x, 1, 2).contiguous()
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
