# solution.py
"""CUDA extension solution for task_008: batched matrix transpose."""

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
            name="task_008_ext",
            sources=[os.path.join(src_dir, "task_008_ext.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Batched transpose matching torch.transpose(x, 1, 2).contiguous()."""
    if not x.is_cuda:
        x = x.cuda()
    ext = _get_ext()
    return ext.batched_transpose(x)
