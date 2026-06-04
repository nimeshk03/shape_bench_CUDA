```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int K, int N
) {
    // Each block handles one (batch, tile_row, tile_col) combination
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (b >= batch) return;

    const float* a_batch = A + b * M * K;
    const float* b_batch = B + b * K * N;
    float* c_batch = C + b * M * N;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // Load tile from A
        if (row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = a_batch[row * K + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B
        if (b_row < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = b_batch[b_row * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c_batch[row * N + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 3, "a must be 3D");
    TORCH_CHECK(b.dim() == 3, "b must be 3D");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");

    int batch = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);

    TORCH_CHECK(b.size(0) == batch, "batch size mismatch");
    TORCH_CHECK(b.size(1) == K, "K dimension mismatch");

    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    auto c = torch::zeros({batch, M, N}, a.options());

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch
    );

    batched_matmul_kernel<<<grid, block>>>(
        a_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        c.data_ptr<float>(),
        batch, M, K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul_cuda, "Batched matrix multiplication (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="batched_matmul_ext",
    ext_modules=[
        CUDAExtension(
            name="batched_matmul_ext",
            sources=["batched_matmul.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py
"""CUDA extension wrapper for batched matrix multiplication."""

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
            name="batched_matmul_ext",
            sources=[os.path.join(src_dir, "batched_matmul.cu")],
            verbose=False,
        )
    return _ext


def forward(a: Tensor, b: Tensor) -> Tensor:
    """Compute batched matrix multiplication C = A @ B."""
    ext = _get_ext()
    return ext.batched_matmul(a, b)
```
