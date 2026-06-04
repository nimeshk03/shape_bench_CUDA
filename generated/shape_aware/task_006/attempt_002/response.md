```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Each block handles one row
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

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sum2 = smem + blockDim.x;

    // Step 1: Compute sum for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += row_ptr[i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    // Step 2: Compute sum of squared deviations for variance
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_ptr[i] - mean;
        local_sum2 += diff * diff;
    }
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Reduce sum2
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = s_sum2[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // Step 3: Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
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
        // Round up to next power of 2 for cleaner reduction
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
# solution.py
"""CUDA extension wrapper for task_006: row-wise layer normalization."""

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
        _src = os.path.join(os.path.dirname(__file__), "rowwise_layer_norm.cu")
        _ext = load(
            name="rowwise_layer_norm_ext",
            sources=[_src],
            verbose=False,
        )
    return _ext


def forward(x: Tensor) -> Tensor:
    """Run row-wise layer normalization via CUDA extension."""
    ext = _get_ext()
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return ext.rowwise_layer_norm(x)
```
