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
            name="add_relu_ext",
            sources=[os.path.join(src_dir, "add_relu.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor


def reference(x: Tensor, y: Tensor) -> Tensor:
    ext = _load_ext()
    if x.is_cuda and y.is_cuda():
        if x.is_contiguous() and y.is_contiguous():
            return ext.add_relu(x, y)
    return torch.relu(x + y)


# Make it work as a drop-in for the harness
class Model(torch.nn.Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return reference(x, y)
