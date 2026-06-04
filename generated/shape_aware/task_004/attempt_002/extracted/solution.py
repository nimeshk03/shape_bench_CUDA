# solution.py
"""CUDA extension wrapper matching the PyTorch reference for task_004."""

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
            name="matmul_cuda",
            sources=[os.path.join(src_dir, "matmul_cuda.cu")],
            verbose=False,
        )
    return _ext


def forward(a: Tensor, b: Tensor) -> Tensor:
    """Compute C = A @ B using the CUDA tiled kernel."""
    if not a.is_cuda:
        a = a.cuda()
    if not b.is_cuda:
        b = b.cuda()
    a = a.contiguous()
    b = b.contiguous()
    ext = _get_ext()
    return ext.matmul(a, b)
