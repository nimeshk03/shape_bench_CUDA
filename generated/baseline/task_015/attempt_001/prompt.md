# ShapeBench-CUDA Rendered Prompt

Prompt mode: `baseline`
Task id: `task_015`

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
  "atol": 1e-06,
  "category": "strong_non_contiguous",
  "description": "Apply relu(x * scale + bias) to a 3D float32 tensor where x is a non-contiguous view with storage offset and irregular row/column strides.",
  "dtype": "float32",
  "expected_output": "3D contiguous tensor with shape [batch, rows, cols]",
  "input_kind": "offset_strided_batched_matrix",
  "input_names": [
    "x",
    "scale",
    "bias"
  ],
  "name": "offset_strided_affine_relu",
  "notes": "Stronger non-contiguous task intended to expose kernels that ignore storage offset or tensor strides.",
  "original_shape": [
    8,
    128,
    192
  ],
  "rtol": 1e-06,
  "task_id": "task_015"
}
```

## Original Shape

Baseline generation should optimize for this original shape.

```json
{
  "original": [
    8,
    128,
    192
  ]
}
```

## PyTorch Reference Model

The generated CUDA implementation should match `Model.forward` and `reference`.
`create_inputs` is included only to show how local tests create inputs.

```python
"""Task 015: affine ReLU over an offset, irregularly strided 3D view."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for stride-sensitive affine ReLU."""

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create deterministic non-contiguous 3D input plus feature vectors."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_015 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn((batch + 1, rows * 2 + 3, cols * 3 + 5), generator=generator, dtype=dtype)
    x = base[1:, 1 : 1 + rows * 2 : 2, 2 : 2 + cols * 3 : 3]
    scale = torch.randn((cols,), generator=generator, dtype=dtype)
    bias = torch.randn((rows,), generator=generator, dtype=dtype)
    return x.to(device), scale.to(device), bias.to(device)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
```

## Generated Code Expectations

- Implement CUDA/C++ extension code that matches the PyTorch reference.
- Keep the generated implementation self-contained for the harness.
- Do not include unrelated prose, training code, or experiment orchestration.
