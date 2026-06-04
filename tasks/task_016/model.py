"""Task 016: layer normalization over irregular dynamic last dimensions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


EPS = 1e-5


class Model(nn.Module):
    """Reference PyTorch module for last-dimension layer normalization."""

    def forward(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + EPS)
        return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create deterministic 3D input plus last-dimension affine parameters."""
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_016 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((batch, rows, cols), generator=generator, dtype=dtype)
    weight = torch.randn((cols,), generator=generator, dtype=dtype)
    bias = torch.randn((cols,), generator=generator, dtype=dtype)
    return x.to(device), weight.to(device), bias.to(device)


def reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Functional reference for generated implementations to match."""
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + EPS)
    return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()
