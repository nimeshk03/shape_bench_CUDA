# solution.py  (harness entry-point)
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build / load the extension on first import
from torch.utils.cpp_extension import load

_ext = load(
    name="task010_ext",
    sources=[os.path.join(os.path.dirname(__file__), "task010_kernel.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.sum_squares(x)
