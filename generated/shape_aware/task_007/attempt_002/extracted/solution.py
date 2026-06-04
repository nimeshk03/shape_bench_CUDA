# solution.py
"""CUDA extension wrapper for task_007: broadcast_affine_clamp."""

from __future__ import annotations

import os
import torch
from torch import Tensor

def _load_extension():
    try:
        import task_007_ext
        return task_007_ext
    except ImportError:
        pass

    from torch.utils.cpp_extension import load
    src_dir = os.path.dirname(os.path.abspath(__file__))
    ext = load(
        name="task_007_ext",
        sources=[os.path.join(src_dir, "task_007_ext.cu")],
        verbose=False,
    )
    return ext


_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def forward(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """Compute clamp(x * scale + bias, -1, 1) using CUDA kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not scale.is_cuda:
        scale = scale.cuda()
    if not bias.is_cuda:
        bias = bias.cuda()

    x = x.contiguous()
    scale = scale.contiguous()
    bias = bias.contiguous()

    return get_ext().broadcast_affine_clamp(x, scale, bias)
