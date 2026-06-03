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
