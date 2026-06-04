# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_016`

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
  "atol": 0.0001,
  "category": "irregular_reduction",
  "description": "Apply layer normalization over the last dimension of a 3D float32 tensor with dynamic and irregular feature sizes.",
  "dtype": "float32",
  "expected_output": "3D contiguous tensor with shape [batch, rows, cols]",
  "input_kind": "batched_matrix_with_feature_vectors",
  "input_names": [
    "x",
    "weight",
    "bias"
  ],
  "name": "irregular_lastdim_layer_norm",
  "notes": "Harder reduction task intended to expose hardcoded last-dimension sizes and boundary mistakes on irregular feature counts.",
  "original_shape": [
    8,
    128,
    512
  ],
  "rtol": 0.0001,
  "task_id": "task_016"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    2,
    256,
    769
  ],
  "larger": [
    12,
    128,
    1024
  ],
  "non_power_of_two": [
    7,
    125,
    1009
  ],
  "odd": [
    5,
    127,
    513
  ],
  "original": [
    8,
    128,
    512
  ],
  "smaller": [
    4,
    64,
    257
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 016: layer normalization over irregular dynamic last dimensions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


EPS = 1e-5


class Model(nn.Module):
    """Reference PyTorch module for last-dimension layer normalization."""

    def forward(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + EPS)
        return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create deterministic 3D input plus last-dimension affine parameters."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_016 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((batch, rows, cols), generator=generator, dtype=dtype)
    weight = torch.randn((cols,), generator=generator, dtype=dtype)
    bias = torch.randn((cols,), generator=generator, dtype=dtype)
    return x.to(device), weight.to(device), bias.to(device)


def reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + EPS)
    return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
