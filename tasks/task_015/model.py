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
