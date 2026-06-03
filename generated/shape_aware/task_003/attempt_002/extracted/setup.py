# setup.py / build glue
from torch.utils.cpp_extension import load
import os

_ext = None

def get_extension():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="matrix_transpose_ext",
            sources=[os.path.join(src_dir, "matrix_transpose.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated matrix transpose."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = get_extension()
            return ext.matrix_transpose(x)
        # Fallback for CPU
        return torch.transpose(x, 0, 1).contiguous()


def reference(x: Tensor) -> Tensor:
    return torch.transpose(x, 0, 1).contiguous()
