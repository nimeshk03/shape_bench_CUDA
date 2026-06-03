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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    
    // Process 4 elements per thread using float4 for better memory throughput
    int64_t idx4 = idx * 4;
    int64_t stride4 = stride * 4;
    
    while (idx4 + 3 < n) {
        float4 xv = reinterpret_cast<const float4*>(x)[idx];
        float4 yv = reinterpret_cast<const float4*>(y)[idx];
        float4 res;
        res.x = fmaxf(0.0f, xv.x + yv.x);
        res.y = fmaxf(0.0f, xv.y + yv.y);
        res.z = fmaxf(0.0f, xv.z + yv.z);
        res.w = fmaxf(0.0f, xv.w + yv.w);
        reinterpret_cast<float4*>(out)[idx] = res;
        idx += stride;
        idx4 += stride4;
    }
    
    // Handle remaining elements
    for (int64_t i = idx4; i < n; i++) {
        out[i] = fmaxf(0.0f, x[i] + y[i]);
    }
}

torch::Tensor add_relu(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    
    const int threads = 256;
    // Each thread handles 4 elements
    int64_t blocks = (n / 4 + threads - 1) / threads;
    blocks = std::min(blocks, (int64_t)65535);
    if (blocks < 1) blocks = 1;
    
    add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu, "Element-wise add followed by ReLU (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add_relu_cuda",
    ext_modules=[
        CUDAExtension(
            name="add_relu_cuda",
            sources=["add_relu_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py - harness entry point
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load
import os

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="add_relu_cuda",
            sources=[os.path.join(src_dir, "add_relu_cuda.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor, y: Tensor) -> Tensor:
    if x.is_cuda and x.is_contiguous() and y.is_contiguous():
        return _get_ext().add_relu(x, y)
    # Fallback
    return torch.relu(x + y)
```
