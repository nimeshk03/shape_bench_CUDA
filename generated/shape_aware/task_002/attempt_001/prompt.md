# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_002`

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
  "atol": 0.001,
  "category": "reduction",
  "description": "Compute the sum of each row in a 2D float32 tensor.",
  "dtype": "float32",
  "expected_output": "1D tensor with one sum per input row",
  "input_kind": "matrix",
  "input_names": [
    "x"
  ],
  "name": "rowwise_sum",
  "notes": "Reduction task intended to expose shape assumptions around row length, odd widths, and non-power-of-two dimensions.",
  "original_shape": [
    1024,
    1024
  ],
  "rtol": 0.001,
  "task_id": "task_002"
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
"""Task 002: row-wise sum reduction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for summing each row of a 2D tensor."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=1)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 2D input for row-wise reductions."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_002 expects a 2D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.sum(x, dim=1)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
