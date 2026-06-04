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
            sources=[os.path.join(src_dir, "solution.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        ext = _load_ext()
        return ext.batched_matmul(a, b)


def reference(a: Tensor, b: Tensor) -> Tensor:
    ext = _load_ext()
    return ext.batched_matmul(a, b)
