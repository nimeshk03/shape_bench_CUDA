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
            sources=[os.path.join(src_dir, "affine_relu.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated relu(x * scale + bias) for non-contiguous inputs."""

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.affine_relu(x, scale, bias)
        # CPU fallback
        return torch.relu(x * scale + bias)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """Functional reference."""
    return torch.relu(x * scale + bias)
