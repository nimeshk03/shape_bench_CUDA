# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def get_extension():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="broadcast_affine_clamp_ext",
            sources=[os.path.join(src_dir, "solution.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        ext = get_extension()
        if x.is_cuda and x.dtype == torch.float32:
            if not x.is_contiguous():
                x = x.contiguous()
            if not scale.is_contiguous():
                scale = scale.contiguous()
            if not bias.is_contiguous():
                bias = bias.contiguous()
            return ext.broadcast_affine_clamp(x, scale, bias)
        return torch.clamp(x * scale + bias, min=-1.0, max=1.0)
