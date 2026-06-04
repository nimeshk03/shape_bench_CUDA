# solution.py  – drop-in replacement for the harness
"""Task 015 CUDA solution: affine ReLU over offset/strided 3D view."""

from __future__ import annotations

import os
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Build / load the extension once
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as _ext

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT = _ext.load(
    name="task015_ext",
    sources=[os.path.join(_SRC_DIR, "task015_kernel.cu")],
    verbose=False,
)


# ---------------------------------------------------------------------------
# Public API expected by the harness
# ---------------------------------------------------------------------------

def forward(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """CUDA implementation matching Model.forward / reference."""
    if x.is_cuda:
        return _EXT.affine_relu_cuda(x, scale, bias)
    # Fallback for CPU tensors (shouldn't happen in normal harness usage)
    return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
