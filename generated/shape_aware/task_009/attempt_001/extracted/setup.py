# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="affine_relu_ext",
            sources=[os.path.join(src_dir, "affine_relu_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn
from collections.abc import Sequence


class Model(nn.Module):
    """CUDA-accelerated relu(x * scale + bias) for non-contiguous inputs."""

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.affine_relu_cuda(x, scale, bias)
        # CPU fallback
        return torch.relu(x * scale + bias)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    return torch.relu(x * scale + bias)


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_009 expects a 2D shape, got {normalized_shape}")
    rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn((rows, cols * 2), generator=generator, dtype=dtype)
    x = base[:, ::2]
    scale = torch.randn((cols,), generator=generator, dtype=dtype)
    bias = torch.randn((cols,), generator=generator, dtype=dtype)
    return x.to(device), scale.to(device), bias.to(device)
