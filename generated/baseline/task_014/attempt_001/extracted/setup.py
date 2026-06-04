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
            name="transpose_ext",
            sources=[os.path.join(src_dir, "transpose_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-optimized batched matrix transpose."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.batched_transpose(x)
        # Fallback for CPU
        return torch.transpose(x, 1, 2).contiguous()
