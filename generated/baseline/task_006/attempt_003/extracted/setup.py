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
            name="rowwise_ln_ext",
            sources=[os.path.join(src_dir, "rowwise_ln.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated row-wise layer normalization."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.rowwise_layer_norm(x)
        # Fallback to PyTorch for CPU tensors
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + 1e-5)


def reference(x: Tensor) -> Tensor:
    mean = torch.mean(x, dim=1, keepdim=True)
    variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
    return (x - mean) * torch.rsqrt(variance + 1e-5)
