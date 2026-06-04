# solution.py - harness entry point
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load
import os

_ext = load(
    name="task_006_ext",
    sources=[os.path.join(os.path.dirname(__file__), "task_006_kernel.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Row-wise layer normalization using CUDA kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.rowwise_layer_norm(x)
