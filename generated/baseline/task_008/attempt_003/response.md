```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized batched transpose kernel using shared memory tiling
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

    int bx = blockIdx.z; // batch index
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input [batch, rows, cols]
    // Each thread loads TILE_DIM/BLOCK_ROWS elements
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = 
                input[bx * rows * cols + (y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write transposed tile to output [batch, cols, rows]
    // Swap x and y for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[bx * cols * rows + (y + j) * rows + x] = 
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    x = x.contiguous();
    
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
import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="batched_transpose_ext",
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
        return ext.batched_transpose(x)

def reference(x: Tensor) -> Tensor:
    ext = _load_ext()
    return ext.batched_transpose(x)
```
