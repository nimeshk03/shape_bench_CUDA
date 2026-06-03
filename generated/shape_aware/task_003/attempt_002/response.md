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
from torch.utils.cpp_extension import load
import os

_ext = None

def get_extension():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="matrix_transpose_ext",
            sources=[os.path.join(src_dir, "matrix_transpose.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated matrix transpose."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = get_extension()
            return ext.matrix_transpose(x)
        # Fallback for CPU
        return torch.transpose(x, 0, 1).contiguous()


def reference(x: Tensor) -> Tensor:
    return torch.transpose(x, 0, 1).contiguous()
```
