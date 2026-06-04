"""Task 011: batched matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for batched matrix multiplication."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.bmm(a, b)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic batched matrix multiplication inputs."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 4:
        raise ValueError(f"task_011 expects a 4D shape descriptor, got {normalized_shape}")
    batch, rows, shared, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((batch, rows, shared), generator=generator, dtype=dtype)
    b = torch.randn((batch, shared, cols), generator=generator, dtype=dtype)
    return a.to(device), b.to(device)


def reference(a: Tensor, b: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.bmm(a, b)
