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
            name="task015_ext",
            sources=[os.path.join(src_dir, "task015_kernel.cu")],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        ext = _load_ext()
        return ext.affine_relu_cuda(x, scale, bias)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    ext = _load_ext()
    return ext.affine_relu_cuda(x, scale, bias)
