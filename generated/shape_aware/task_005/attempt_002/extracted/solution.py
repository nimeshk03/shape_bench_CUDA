# solution.py  – harness entry point
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension on first import
def _load_extension():
    import torch.utils.cpp_extension as ext
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return ext.load(
        name="rowwise_softmax_cuda",
        sources=[os.path.join(src_dir, "rowwise_softmax.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    return _get_ext().rowwise_softmax(x)
