"""Task 004: matrix multiplication."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for C = A @ B."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic matrix-multiply inputs from [M, K, N]."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_004 expects a 3D shape descriptor [M, K, N], got {normalized_shape}")
    m, k, n = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((m, k), generator=generator, dtype=dtype)
    b = torch.randn((k, n), generator=generator, dtype=dtype)
    return a.to(device), b.to(device)


def reference(a: Tensor, b: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.matmul(a, b)
