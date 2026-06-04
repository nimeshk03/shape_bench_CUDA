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
            name="rowwise_softmax_ext",
            sources=[os.path.join(src_dir, "softmax_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated row-wise softmax."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.forward(x.contiguous())
        return torch.softmax(x, dim=1)


def reference(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=1)
