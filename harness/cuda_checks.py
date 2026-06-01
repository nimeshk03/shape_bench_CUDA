"""CUDA availability helpers that are safe to import on CPU-only machines."""

from __future__ import annotations

from typing import Any


def is_cuda_available() -> bool:
    """Return True only when PyTorch is installed and CUDA is available."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def get_gpu_info() -> dict[str, Any]:
    """Return CUDA/PyTorch environment information without requiring CUDA."""
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "device_count": 0,
        }

    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def require_cuda_or_skip(message: str = "CUDA is required for this check") -> None:
    """Skip under pytest when CUDA is unavailable, otherwise raise RuntimeError."""
    if is_cuda_available():
        return
    try:
        import pytest
    except ImportError as exc:
        raise RuntimeError(message) from exc
    pytest.skip(message)

