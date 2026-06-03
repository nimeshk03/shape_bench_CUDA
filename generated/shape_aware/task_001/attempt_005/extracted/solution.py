# extension_interface.py  (harness entry point)
import torch
from torch import Tensor

try:
    import elementwise_add_relu_cuda as _ext
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def forward(x: Tensor, y: Tensor) -> Tensor:
    if _USE_CUDA and x.is_cuda and x.is_contiguous() and y.is_contiguous():
        return _ext.elementwise_add_relu(x, y)
    # Fallback to PyTorch reference
    return torch.relu(x + y)
