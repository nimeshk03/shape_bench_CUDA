# solution.py
"""CUDA extension wrapper for batched matrix multiplication."""

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
            name="batched_matmul_ext",
            sources=[os.path.join(src_dir, "batched_matmul.cu")],
            verbose=False,
        )
    return _ext


def forward(a: Tensor, b: Tensor) -> Tensor:
    """Compute batched matrix multiplication C = A @ B."""
    ext = _get_ext()
    return ext.batched_matmul(a, b)
