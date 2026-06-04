# solution.py
"""CUDA extension solution for task_012: strided batched transpose."""

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
            name="task_012_ext",
            sources=[os.path.join(src_dir, "task_012_kernel.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Forward pass: transpose last two dims of a (possibly non-contiguous) 3D tensor."""
    if x.is_cuda:
        ext = _get_ext()
        return ext.batched_transpose(x)
    else:
        # CPU fallback
        return torch.transpose(x, 1, 2).contiguous()
