# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext_dir = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="task016_ext",
    sources=[os.path.join(_ext_dir, "task016_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def reference(x, weight, bias):
    """CUDA implementation matching the PyTorch reference."""
    if not x.is_cuda:
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    return _ext.layer_norm(x, weight, bias)
