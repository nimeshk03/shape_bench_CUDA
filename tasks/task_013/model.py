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
