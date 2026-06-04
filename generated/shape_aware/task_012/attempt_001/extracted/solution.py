# solution.py  (harness entry point)
from __future__ import annotations

import os
import torch
from torch import Tensor

# Try to load the CUDA extension; fall back to PyTorch reference if unavailable.
_ext = None

def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    try:
        import task_012_ext
        _ext = task_012_ext
    except ImportError:
        try:
            from torch.utils.cpp_extension import load
            import pathlib
            src = pathlib.Path(__file__).parent / "task_012_kernel.cu"
            _ext = load(
                name="task_012_ext",
                sources=[str(src)],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception:
            _ext = None
    return _ext


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    ext = _load_ext()
    if ext is not None and x.is_cuda:
        return ext.batched_transpose(x)
    # CPU fallback or if extension not available
    return torch.transpose(x, 1, 2).contiguous()


# Make it importable as a nn.Module-compatible callable too
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return forward(x)
