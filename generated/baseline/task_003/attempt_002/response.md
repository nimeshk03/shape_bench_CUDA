```cpp
#include <torch/extension.h>
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
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write transposed tile to output (cols x rows)
    // Output shape: [cols, rows]
    // x_out = blockIdx.y * TILE_DIM + threadIdx.x (column index in output = row index in input)
    // y_out = blockIdx.x * TILE_DIM + threadIdx.y (row index in output = col index in input)
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_out < rows && (y_out + j) < cols) {
            output[(y_out + j) * rows + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

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
    m.def("forward", &matrix_transpose_cuda, "Matrix transpose CUDA");
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
            sources=[os.path.join(src_dir, "matrix_transpose.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext

import torch
from torch import Tensor, nn

class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda:
            ext = _load_ext()
            return ext.forward(x)
        return torch.transpose(x, 0, 1).contiguous()
```
