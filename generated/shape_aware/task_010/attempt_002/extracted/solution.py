# harness_entry.py  – drop-in forward() wrapper used by the evaluation harness
import torch
from torch import Tensor

# Build / load the extension on first import
import os, subprocess, sys

_ext = None

def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    _ext = load(
        name="task010_ext",
        sources=[os.path.join(os.path.dirname(__file__), "task010_kernel.cu")],
        verbose=False,
    )
    return _ext


def forward(x: Tensor) -> Tensor:
    ext = _load_ext()
    return ext.sum_squares(x.contiguous())
