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
            name="batched_matmul_ext",
            sources=[os.path.join(src_dir, "batched_matmul.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated batched matrix multiplication."""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if a.is_cuda and b.is_cuda and a.dtype == torch.float32:
            ext = _load_ext()
            return ext.batched_matmul(a, b)
        return torch.bmm(a, b)


def reference(a: Tensor, b: Tensor) -> Tensor:
    return torch.bmm(a, b)
