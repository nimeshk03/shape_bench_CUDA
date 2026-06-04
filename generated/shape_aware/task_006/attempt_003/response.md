```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Kernel: each block handles one row
// We use shared memory for reduction
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

    // Shared memory for reductions
    extern __shared__ float smem[];
    float* s_sum = smem;           // size: blockDim.x
    float* s_sum2 = smem + blockDim.x; // size: blockDim.x

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: compute sum for mean
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        local_sum += row_ptr[i];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    // Pass 2: compute sum of squared deviations
    float local_sum2 = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float diff = row_ptr[i] - mean;
        local_sum2 += diff * diff;
    }
    s_sum2[tid] = local_sum2;
    __syncthreads();

    // Reduce sum2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum2[tid] += s_sum2[tid + s];
        }
        __syncthreads();
    }
    float variance = s_sum2[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // Normalize
    for (int i = tid; i < cols; i += stride) {
        out_ptr[i] = (row_ptr[i] - mean) * inv_std;
    }
}

torch::Tensor rowwise_layer_norm_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty_like(x);

    // Choose block size: use BLOCK_SIZE threads per block
    // Each block handles one row
    int threads = BLOCK_SIZE;
    // For very short rows, we can use fewer threads
    if (cols < threads) {
        // Round up to next power of 2 for clean reduction
        threads = 1;
        while (threads < cols) threads <<= 1;
        if (threads > BLOCK_SIZE) threads = BLOCK_SIZE;
    }

    dim3 grid(rows);
    dim3 block(threads);
    size_t smem_size = 2 * threads * sizeof(float);

    rowwise_layer_norm_kernel<<<grid, block, smem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_layer_norm", &rowwise_layer_norm_cuda, "Row-wise layer normalization (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_layer_norm_ext",
    ext_modules=[
        CUDAExtension(
            name="rowwise_layer_norm_ext",
            sources=["rowwise_layer_norm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  (harness entry point)
"""CUDA extension wrapper for task_006 row-wise layer normalization."""

from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="rowwise_layer_norm_ext",
            sources=[os.path.join(src_dir, "rowwise_layer_norm.cu")],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Run row-wise layer normalization via CUDA extension."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    ext = _get_ext()
    return ext.rowwise_layer_norm(x)
```
