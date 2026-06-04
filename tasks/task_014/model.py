"""Task 014: transpose from tile-aligned original shape to irregular variants."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for transposing batched matrices."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(x, 1, 2).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 3D input for batched transpose."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_014 expects a 3D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.transpose(x, 1, 2).contiguous()
