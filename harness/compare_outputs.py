"""Output comparison helpers for correctness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorComparison:
    passed: bool
    max_abs_error: float | None
    mean_abs_error: float | None
    message: str


def compare_tensors(
    reference: Any,
    candidate: Any,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> TensorComparison:
    """Compare tensor-like outputs using NumPy-compatible tolerances."""
    ref = _to_numpy(reference)
    cand = _to_numpy(candidate)

    if ref.shape != cand.shape:
        return TensorComparison(
            passed=False,
            max_abs_error=None,
            mean_abs_error=None,
            message=f"shape mismatch: reference {ref.shape}, candidate {cand.shape}",
        )

    abs_error = np.abs(ref - cand)
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    mean_abs_error = float(np.mean(abs_error)) if abs_error.size else 0.0
    passed = bool(np.allclose(ref, cand, atol=atol, rtol=rtol))
    return TensorComparison(
        passed=passed,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        message="passed" if passed else "values differ beyond tolerance",
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

