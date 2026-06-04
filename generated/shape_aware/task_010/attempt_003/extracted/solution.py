# harness_entry.py  – drop-in forward() used by the evaluation harness
"""
Harness entry point for task_010: dynamic_lastdim_sum_squares.

The harness calls `forward(x)` and compares the result to the PyTorch reference.
"""
from __future__ import annotations

import os
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Build / load the CUDA extension on first import
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as _cpp_ext

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

_ext = _cpp_ext.load(
    name="task010_ext",
    sources=[os.path.join(_SRC_DIR, "task010_kernel.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Compute sum(x * x, dim=-1) via the CUDA extension."""
    return _ext.sum_squares(x)
