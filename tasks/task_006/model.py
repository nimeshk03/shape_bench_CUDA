"""Task 006: row-wise layer normalization without affine parameters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


EPSILON = 1e-5


class Model(nn.Module):
    """Reference PyTorch module for row-wise layer normalization."""

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + EPSILON)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor]:
    """Create deterministic 2D input for row-wise normalization."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_006 expects a 2D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)


def reference(x: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    mean = torch.mean(x, dim=1, keepdim=True)
    variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
    return (x - mean) * torch.rsqrt(variance + EPSILON)
