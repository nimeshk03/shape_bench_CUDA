```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Each block handles one row
// Use shared memory for inter-warp reductions
__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows, int cols) {
    extern __shared__ float smem[];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int num_warps = blockSize / 32;
    
    float* warp_max = smem;
    float* warp_sum = smem + num_warps;
    
    // Step 1: Find row max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockSize) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    
    // Warp reduce max
    local_max = warp_reduce_max(local_max);
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final max reduction across warps
    float row_max = -FLT_MAX;
    if (tid < num_warps) {
        row_max = warp_max[tid];
    }
    if (tid < 32) {
        row_max = warp_reduce_max(row_max);
    }
    if (tid == 0) {
        warp_max[0] = row_max;
    }
    __syncthreads();
    row_max = warp_max[0];
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockSize) {
        float val = __expf(row_in[i] - row_max);
        row_out[i] = val;
        local_sum += val;
    }
    
    // Warp reduce sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    float row_sum = 0.0f;
    if (tid < num_warps) {
        row_sum = warp_sum[tid];
    }
    if (tid < 32) {
        row_sum = warp_reduce_sum(row_sum);
    }
    if (tid == 0) {
        warp_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = warp_sum[0];
    
    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockSize) {
        row_out[i] *= inv_sum;
    }
}

torch::Tensor rowwise_softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    
    int rows = x.size(0);
    int cols = x.size(1);
    
    auto output = torch::empty_like(x);
    
    // Choose block size: use 256 threads for cols up to 256, 
    // 512 for larger, capped at 1024
    int block_size;
    if (cols <= 256) block_size = 256;
    else if (cols <= 512) block_size = 512;
    else block_size = 1024;
    
    int num_warps = block_size / 32;
    // Shared memory: 2 * num_warps floats (for max and sum)
    int smem_size = 2 * num_warps * sizeof(float);
    
    softmax_kernel<<<rows, block_size, smem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rowwise_softmax_cuda, "Row-wise softmax (CUDA)");
}
```

```python
# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="rowwise_softmax_ext",
            sources=[os.path.join(src_dir, "softmax_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated row-wise softmax."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.forward(x.contiguous())
        return torch.softmax(x, dim=1)


def reference(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=1)
```

Wait, I need to provide a single self-contained implementation. Let me restructure:

```python
"""Task 005: row-wise softmax - CUDA implementation."""

from __future__ import annotations

import os
import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import load as _load_ext

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows, int cols) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int num_warps = blockSize / 32;

    float* warp_max = smem;
    float* warp_sum = smem + num_warps;

    // Step 1: Find row max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockSize)
        local_max = fmaxf(local_max, row_in[i]);

    local_max = warp_reduce_max(local_max);

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    float row_max = -FLT_MAX;
    if (tid < num_warps) row_max = warp_max[tid];
    if (tid < 32) row_max = warp_reduce_max(row_max);
    if (tid == 0) warp_max[0] = row_max;
    __syncthreads();
    row_max = warp_max[0];

    // Step 2: Compute exp(x - max) and partial sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockSize) {
        float val = __expf(row_in[i] - row_max);
        row_out[i] = val;
        local_sum += val;
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (tid < num_warps) row_sum = warp_sum[tid];
    if (tid < 32) row_sum = warp_reduce_sum(row_sum);
    if (tid == 0) warp_sum[0] = row_sum;
    __syncthreads();
    row_sum = warp_sum[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockSize)
        row_out[i] *= inv_sum;
}

torch::Tensor rowwise_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    int block_size;
    if (cols <= 256) block_size = 256;
    else if (cols <= 512) block_size = 512;
    else block_size = 1024;

    int num_warps = block_size / 32;
    int smem_size = 2 * num_warps * sizeof(float);

    softmax_kernel<<<rows, block_size, smem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rowwise_softmax_forward, "Row-wise softmax (CUDA)");
}
"""

_ext_module = None


def _get_ext():
    global _ext_module
    if _ext_module is None:
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        src_path = os.path.join(tmpdir, "softmax_ext.cu")
        with open(src_path, "w") as f:
            f.write(_CUDA_SRC)
        _ext_module = _load_ext(
            name="rowwise_softmax_ext",
            sources=[src_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext_module


class Model(nn.Module):
    """CUDA-accelerated row-wise softmax."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return _get_ext().forward(x.contiguous())
        return torch.softmax(x, dim=1)


def reference(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=1)


def create_inputs(shape, *, dtype=torch.float32, device="cpu", seed=0):
    normalized_shape = tuple(int(d) for d in shape)
    if len(normalized_shape) != 2:
        raise ValueError(f"task_005 expects a 2D shape, got {normalized_shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(normalized_shape, generator=generator, dtype=dtype)
    return (x.to(device),)
```
