# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task015_ext",
            sources=[os.path.join(src_dir, "task015.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        ext = get_ext()
        return ext.forward(x, scale, bias)
