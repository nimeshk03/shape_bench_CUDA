```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void broadcast_affine_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        float val = x[row * cols + col] * scale[col] + bias[col];
        val = val < -1.0f ? -1.0f : (val > 1.0f ? 1.0f : val);
        out[row * cols + col] = val;
    }
}

torch::Tensor broadcast_affine_clamp(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty_like(x);

    dim3 block(32, 8);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    broadcast_affine_clamp_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_affine_clamp", &broadcast_affine_clamp,
          "Broadcast affine clamp: clamp(x * scale + bias, -1, 1)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_007_ext",
    ext_modules=[
        CUDAExtension(
            name="task_007_ext",
            sources=["task_007_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py
"""CUDA extension solution for task_007: broadcast_affine_clamp."""

from __future__ import annotations

import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task_007_ext",
            sources=[os.path.join(src_dir, "task_007_kernel.cu")],
            verbose=False,
        )
    return _ext


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """CUDA-accelerated clamp(x * scale + bias, -1, 1)."""
    if x.is_cuda:
        ext = _get_ext()
        return ext.broadcast_affine_clamp(
            x.contiguous(),
            scale.contiguous(),
            bias.contiguous(),
        )
    # CPU fallback
    return torch.clamp(x * scale + bias, min=-1.0, max=1.0)
```
