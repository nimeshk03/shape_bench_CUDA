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
    // Output index: output[col, row] where col was x, row was y
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int out_row = out_y + j;
        int out_col = out_x;
        // tile[threadIdx.x][threadIdx.y + j] corresponds to transposed element
        if (out_col < rows && out_row < cols) {
            output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");

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
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension
from torch.utils.cpp_extension import load
import os

_ext = load(
    name="matrix_transpose_cuda",
    sources=[os.path.join(os.path.dirname(__file__), "matrix_transpose.cu")],
    verbose=False,
)


def reference(x: Tensor) -> Tensor:
    """CUDA implementation of matrix transpose."""
    if not x.is_cuda:
        x = x.cuda()
    return _ext.matrix_transpose(x)
```
