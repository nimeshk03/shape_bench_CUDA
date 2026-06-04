# ShapeBench-CUDA Rendered Prompt

Prompt mode: `shape_aware`
Task id: `task_009`

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
  "atol": 1e-06,
  "category": "non_contiguous",
  "description": "Apply relu(x * scale + bias) to a 2D float32 tensor where x is a non-contiguous column-strided view.",
  "dtype": "float32",
  "expected_output": "2D contiguous tensor with shape [rows, cols]",
  "input_kind": "strided_matrix_with_feature_vectors",
  "input_names": [
    "x",
    "scale",
    "bias"
  ],
  "name": "noncontiguous_affine_relu",
  "notes": "Elementwise task intended to expose generated kernels that assume contiguous input storage.",
  "original_shape": [
    1024,
    1024
  ],
  "rtol": 1e-06,
  "task_id": "task_009"
}
```

## Shape Variants

```json
{
  "batch_variant": [
    256,
    2048
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
"""Task 009: affine ReLU on a non-contiguous matrix view."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for relu(x * scale + bias)."""

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        return torch.relu(x * scale + bias)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a deterministic non-contiguous 2D input and feature vectors."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_009 expects a 2D shape, got {normalized_shape}")
    rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn((rows, cols * 2), generator=generator, dtype=dtype)
    x = base[:, ::2]
    scale = torch.randn((cols,), generator=generator, dtype=dtype)
    bias = torch.randn((cols,), generator=generator, dtype=dtype)
    return x.to(device), scale.to(device), bias.to(device)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.relu(x * scale + bias)
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
