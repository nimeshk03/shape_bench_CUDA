```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ data_ptr,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ out_ptr,
    int64_t batch,
    int64_t rows,
    int64_t cols,
    int64_t stride_b,
    int64_t stride_r,
    int64_t stride_c,
    int64_t storage_offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * rows * cols;
    if (idx >= total) return;

    int64_t b = idx / (rows * cols);
    int64_t rem = idx % (rows * cols);
    int64_t r = rem / cols;
    int64_t c = rem % cols;

    int64_t src_idx = storage_offset + b * stride_b + r * stride_r + c * stride_c;
    float val = data_ptr[src_idx];
    float result = val * scale_ptr[c] + bias_ptr[r];
    out_ptr[idx] = result > 0.0f ? result : 0.0f;
}

torch::Tensor affine_relu_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    int64_t batch = x.size(0);
    int64_t rows = x.size(1);
    int64_t cols = x.size(2);

    int64_t stride_b = x.stride(0);
    int64_t stride_r = x.stride(1);
    int64_t stride_c = x.stride(2);
    int64_t storage_offset = x.storage_offset();

    auto out = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int64_t total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* data_ptr = x.storage().data_ptr().get<float>();

    affine_relu_kernel<<<blocks, threads>>>(
        data_ptr,
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, rows, cols,
        stride_b, stride_r, stride_c,
        storage_offset
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU with stride-aware CUDA kernel");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task015_ext",
    ext_modules=[
        CUDAExtension(
            name="task015_ext",
            sources=["task015_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py
"""Task 015 CUDA extension solution."""

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
            name="task015_ext",
            sources=[os.path.join(src_dir, "task015_kernel.cu")],
            verbose=False,
        )
    return _ext


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    ext = _get_ext()
    if x.is_cuda():
        return ext.affine_relu_cuda(x, scale, bias)
    else:
        # Fallback for CPU (shouldn't happen in benchmark but just in case)
        return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
```
