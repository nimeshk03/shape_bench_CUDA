```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Kernel: each block handles one row
// Uses shared memory for reduction
__global__ void rowwise_layer_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;
    float* out_ptr = out + row * cols;

    __shared__ float smem[BLOCK_SIZE];

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_ptr[i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduce sum in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = smem[0] / (float)cols;
    __syncthreads();

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_ptr[i] - mean;
        var_sum += diff * diff;
    }
    smem[threadIdx.x] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float variance = smem[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_ptr[i] = (row_ptr[i] - mean) * inv_std;
    }
}

torch::Tensor rowwise_layer_norm(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty_like(x);

    // Choose block size: use 256 threads, enough for cols up to 1024
    int block_size = BLOCK_SIZE;
    // For larger cols, we might want more threads, but 256 is fine for 1024
    if (cols <= 128) block_size = 128;
    else if (cols <= 256) block_size = 256;
    else block_size = 256; // still 256, loop handles larger cols

    dim3 grid(rows);
    dim3 block(block_size);

    rowwise_layer_norm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_layer_norm", &rowwise_layer_norm, "Row-wise layer normalization (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_006_ext",
    ext_modules=[
        CUDAExtension(
            name="task_006_ext",
            sources=["task_006_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py - harness entry point
from __future__ import annotations
import torch
from torch import Tensor
import os
import sys

# Build and load the extension inline
from torch.utils.cpp_extension import load
import os

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="task_006_ext",
            sources=[os.path.join(src_dir, "task_006_kernel.cu")],
            verbose=False,
        )
    return _ext


class Model(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and x.is_contiguous() and x.dtype == torch.float32:
            return _get_ext().rowwise_layer_norm(x)
        # Fallback to reference
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + 1e-5)
```
