# solution.py
"""CUDA extension wrapper for task_006: row-wise layer normalization."""

from __future__ import annotations

import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

# Load the CUDA extension inline
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task_006_ext",
            sources=[os.path.join(src_dir, "task_006_kernel.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Row-wise layer normalization using CUDA kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    ext = _get_ext()
    return ext.rowwise_layer_norm(x)
