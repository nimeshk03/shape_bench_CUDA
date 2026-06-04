# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_012`

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
  "category": "stride_layout",
  "description": "Transpose the last two dimensions of a 3D float32 tensor that is created as a non-contiguous strided view.",
  "dtype": "float32",
  "expected_output": "3D contiguous tensor with shape [batch, cols, rows]",
  "input_kind": "strided_batched_matrix",
  "input_names": [
    "x"
  ],
  "name": "strided_batched_transpose",
  "notes": "Stride-sensitive layout task intended to expose kernels that ignore input strides while transposing.",
  "original_shape": [
    16,
    256,
    512
  ],
  "rtol": 0.0,
  "task_id": "task_012"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    4,
    256,
    1024
  ],
  "larger": [
    32,
    512,
    512
  ],
  "non_power_of_two": [
    10,
    250,
    257
  ],
  "odd": [
    7,
    127,
    131
  ],
  "original": [
    16,
    256,
    512
  ],
  "smaller": [
    8,
    128,
    512
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 012: batched transpose of a non-contiguous strided view."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for strided-view batched transpose."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 3D non-contiguous input for batched transpose."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_012 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn((batch, rows * 2, cols * 2), generator=generator, dtype=dtype)
    x = base[:, ::2, 1::2]
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.transpose(x, 1, 2).contiguous()
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
