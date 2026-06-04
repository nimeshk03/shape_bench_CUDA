# solution.py
"""Task 013: CUDA extension implementation of diagnostic batched transpose."""

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
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task_013_ext",
            sources=[os.path.join(_src_dir, "task_013_kernel.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """CUDA implementation of batched transpose [B, R, C] -> [B, C, R]."""
    if not x.is_cuda:
        x = x.cuda()
    ext = _get_ext()
    return ext.batched_transpose(x)
