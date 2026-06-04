# solution.py  (harness entry point)
"""CUDA extension wrapper for task_006 row-wise layer normalization."""

from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="rowwise_layer_norm_ext",
            sources=[os.path.join(src_dir, "rowwise_layer_norm.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Run row-wise layer normalization via CUDA extension."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    ext = _get_ext()
    return ext.rowwise_layer_norm(x)
