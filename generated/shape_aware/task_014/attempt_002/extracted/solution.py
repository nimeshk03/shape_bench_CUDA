"""Fallback generated-attempt entrypoint.

This file was created by ShapeBench-CUDA because the generated attempt did not
provide an extracted solution.py. It loads the extracted CUDA source and exposes
forward(*inputs) for the evaluator contract.
"""

from __future__ import annotations

import pathlib

from torch import Tensor
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        source = pathlib.Path(__file__).parent / "task014_transpose.cu"
        _EXT = load(
            name="shapebench_task_014_shape_aware_attempt_002",
            sources=[str(source)],
            verbose=False,
        )
    return _EXT


def forward(x) -> Tensor:
    return getattr(_load_ext(), "batched_transpose")(x)
