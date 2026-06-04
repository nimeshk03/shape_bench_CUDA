# solution.py
"""Task 016: CUDA layer normalization over irregular dynamic last dimensions."""

from __future__ import annotations

import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task_016_ext",
            sources=[os.path.join(src_dir, "task_016_kernel.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    if not x.is_cuda:
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    ext = _get_ext()
    return ext.layer_norm_cuda(x, weight, bias)
