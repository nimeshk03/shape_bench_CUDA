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
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float smem[BLOCK_SIZE];

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_in[i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = smem[0] / (float)cols;
    __syncthreads();

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    smem[threadIdx.x] = var_sum;
    __syncthreads();

    // Reduction for variance sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = smem[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // Step 3: Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * inv_std;
    }
}

torch::Tensor rowwise_layer_norm_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    // Choose block size: use BLOCK_SIZE threads per block, one block per row
    int threads = BLOCK_SIZE;
    // For very short rows, we could use fewer threads, but 256 is fine
    if (cols < threads) {
        // Round up to next power of 2
        threads = 1;
        while (threads < cols) threads <<= 1;
        if (threads > BLOCK_SIZE) threads = BLOCK_SIZE;
    }

    dim3 grid(rows);
    dim3 block(threads);

    rowwise_layer_norm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_layer_norm", &rowwise_layer_norm_cuda, "Row-wise layer normalization (CUDA)");
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

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load
import os

_ext = load(
    name="task_006_ext",
    sources=[os.path.join(os.path.dirname(__file__), "task_006_kernel.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Row-wise layer normalization using CUDA kernel."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.rowwise_layer_norm(x)
```
