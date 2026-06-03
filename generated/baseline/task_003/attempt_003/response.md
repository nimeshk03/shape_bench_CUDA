```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
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
    // Output shape: [cols, rows]
    // blockIdx.x -> col dimension, blockIdx.y -> row dimension
    // After transpose: output[col, row] = input[row, col]
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x; // row index in original = col index in output's second dim
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y; // col index in original = row index in output

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int out_row = out_y + j;
        int out_col = out_x;
        if (out_row < cols && out_col < rows) {
            output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose(torch::Tensor x) {
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
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_transpose", &matrix_transpose, "Matrix transpose (CUDA)");
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
            name="matrix_transpose_ext",
            sources=[os.path.join(src_dir, "solution.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext

import torch
from torch import Tensor, nn

class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        ext = _load_ext()
        if x.is_cuda:
            return ext.matrix_transpose(x)
        # fallback for CPU
        return torch.transpose(x, 0, 1).contiguous()
```
