```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load tile from A
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiply(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");

    a = a.contiguous();
    b = b.contiguous();

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<gridDim, blockDim>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply", &matrix_multiply, "Matrix multiplication (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_cuda",
    ext_modules=[
        CUDAExtension(
            name="matmul_cuda",
            sources=["matmul_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py
"""CUDA extension wrapper matching the reference interface."""

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
        _src = os.path.join(os.path.dirname(__file__), "matmul_cuda.cu")
        _ext = load(
            name="matmul_cuda",
            sources=[_src],
            verbose=False,
        )
    return _ext


def reference(a: Tensor, b: Tensor) -> Tensor:
    """Drop-in replacement for the PyTorch reference using the CUDA kernel."""
    if a.is_cuda and b.is_cuda:
        return _get_ext().matrix_multiply(a.contiguous(), b.contiguous())
    # Fallback for CPU tensors: move to CUDA if available, else use torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        result = _get_ext().matrix_multiply(
            a.to(device).contiguous(),
            b.to(device).contiguous()
        )
        return result.to(a.device)
    return torch.matmul(a, b)
```
