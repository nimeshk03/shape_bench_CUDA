# solution.py  (harness entry point)
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline so the harness can import this file directly.
from torch.utils.cpp_extension import load
import pathlib

_src = pathlib.Path(__file__).parent / "add_relu.cu"

_ext = load(
    name="elementwise_add_relu_ext",
    sources=[str(_src)],
    verbose=False,
)


def forward(x: Tensor, y: Tensor) -> Tensor:
    """Drop-in replacement for relu(x + y)."""
    return _ext.add_relu(x, y)
