```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] + y[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor add_relu_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu_cuda, "ReLU(x + y) CUDA implementation");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="elementwise_add_relu_cuda",
    ext_modules=[
        CUDAExtension(
            name="elementwise_add_relu_cuda",
            sources=["add_relu.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py
"""CUDA extension wrapper matching the harness interface."""

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
            name="elementwise_add_relu_cuda",
            sources=[os.path.join(src_dir, "add_relu.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor, y: Tensor) -> Tensor:
    """Compute relu(x + y) using CUDA kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not y.is_cuda:
        y = y.cuda()
    x = x.contiguous()
    y = y.contiguous()
    ext = _get_ext()
    return ext.add_relu(x, y)
```
