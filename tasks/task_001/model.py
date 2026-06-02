"""Task 001: elementwise add followed by ReLU."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Model(nn.Module):
    """Reference PyTorch module for relu(x + y)."""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.relu(x + y)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic inputs for local correctness checks."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    normalized_shape = tuple(int(dim) for dim in shape)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    y = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return x.to(device), y.to(device)


def reference(x: Tensor, y: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    return torch.relu(x + y)
