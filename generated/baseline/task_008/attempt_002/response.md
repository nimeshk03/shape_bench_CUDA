```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled transpose kernel for batched matrix transpose
// Input:  [batch, rows, cols]
// Output: [batch, cols, rows]

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int b = blockIdx.z;
    if (b >= batch) return;

    // Input tile coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // col index
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row index

    // Load tile from input [b, row, col]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[b * rows * cols + (y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transposed output coordinates
    // Output is [batch, cols, rows]
    // We write tile transposed: output col block -> output row, input row block -> output col
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x; // was row, now col in output
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y; // was col, now row in output

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (out_x < rows && (out_y + j) < cols) {
            output[b * cols * rows + (out_y + j) * rows + out_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    auto output = torch::empty({batch, cols, rows}, x.options());

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM,
        batch
    );

    batched_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose_cuda, "Batched matrix transpose (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_008_ext",
    ext_modules=[
        CUDAExtension(
            name="task_008_ext",
            sources=["task_008_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  – harness entry point
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline
from torch.utils.cpp_extension import load

_ext = load(
    name="task_008_ext",
    sources=[os.path.join(os.path.dirname(__file__), "task_008_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def reference(x: Tensor) -> Tensor:
    """CUDA-accelerated batched transpose matching torch.transpose(x,1,2).contiguous()."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.batched_transpose(x)
```
