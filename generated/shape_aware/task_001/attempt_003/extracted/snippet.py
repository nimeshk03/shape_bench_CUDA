# extension_wrapper.py  (harness entry point)
import torch
from torch import Tensor

try:
    import elementwise_add_relu_cuda as _ext
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def elementwise_add_relu(x: Tensor, y: Tensor) -> Tensor:
    if _USE_CUDA and x.is_cuda and y.is_cuda:
        x = x.contiguous()
        y = y.contiguous()
        return _ext.elementwise_add_relu(x, y)
    # CPU fallback
    return torch.relu(x + y)
