# solution.py
"""CUDA extension solution for task_001: elementwise_add_relu."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        src_dir = Path(__file__).parent
        _ext = load(
            name="add_relu_cuda_ext",
            sources=[str(src_dir / "add_relu_cuda.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _ext


def forward(x: Tensor, y: Tensor) -> Tensor:
    """Compute relu(x + y) using CUDA extension."""
    if not x.is_cuda:
        x = x.cuda()
    if not y.is_cuda:
        y = y.cuda()
    
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    
    ext = _get_ext()
    return ext.add_relu(x, y)
