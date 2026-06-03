```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input (rows x cols) into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < rows && col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * cols + col];
        }
    }

    __syncthreads();

    // Write transposed tile to output (cols x rows)
    // Output dimensions: cols rows, rows cols
    // Block in output: blockIdx.y * TILE_DIM columns, blockIdx.x * TILE_DIM rows
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int out_row = out_y + j;
        int out_col = out_x;
        if (out_row < cols && out_col < rows) {
            output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({cols, rows}, x.options());

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_transpose", &matrix_transpose_cuda, "Matrix transpose (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matrix_transpose_cuda",
    ext_modules=[
        CUDAExtension(
            name="matrix_transpose_cuda",
            sources=["matrix_transpose.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py - harness entry point
import torch
from torch import Tensor

try:
    import matrix_transpose_cuda as _ext
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False

import torch.utils.cpp_extension as _cpp_ext
import os

def _build_and_load():
    global _ext, _USE_CUDA
    src = os.path.join(os.path.dirname(__file__), "matrix_transpose.cu")
    if not os.path.exists(src):
        # Write the CUDA source inline
        cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < rows && col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * cols + col];
        }
    }

    __syncthreads();

    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int out_row = out_y + j;
        int out_col = out_x;
        if (out_row < cols && out_col < rows) {
            output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({cols, rows}, x.options());

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_transpose", &matrix_transpose_cuda, "Matrix transpose (CUDA)");
}
"""
        with open(src, "w") as f:
            f.write(cuda_src)

    _ext = _cpp_ext.load(
        name="matrix_transpose_cuda",
        sources=[src],
        verbose=False,
    )
    _USE_CUDA = True


import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        if not _USE_CUDA:
            _build_and_load()

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and _USE_CUDA:
            return _ext.matrix_transpose(x)
        # Fallback
        return torch.transpose(x, 0, 1).contiguous()


def reference(x: Tensor) -> Tensor:
    return torch.transpose(x, 0, 1).contiguous()
```
