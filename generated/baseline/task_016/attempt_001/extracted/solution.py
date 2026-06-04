# solution.py - harness entry point
import torch
from torch import Tensor

try:
    import layer_norm_cuda
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Fallback pure PyTorch implementation."""
    EPS = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + EPS)
    return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()


def forward(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    if _USE_CUDA and x.is_cuda:
        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous()
        return layer_norm_cuda.forward(x_c, w_c, b_c)
    return reference(x, weight, bias)
