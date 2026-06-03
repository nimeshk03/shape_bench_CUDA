# solution.py  (harness entry-point)
import torch
from torch import Tensor
import os
import sys

# Build and load the extension inline so the harness can import this file directly.
from torch.utils.cpp_extension import load
import pathlib

_dir = pathlib.Path(__file__).parent
_ext = load(
    name="rowwise_sum_ext",
    sources=[str(_dir / "rowwise_sum.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.rowwise_sum(x)
