```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: transpose last two dims of a 3D tensor [B, R, C] -> [B, C, R]
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int R, int C
) {
    // Use shared memory tile for coalesced access
    __shared__ float tile[32][33]; // 33 to avoid bank conflicts

    int b = blockIdx.z;
    int tile_row = blockIdx.y * 32;
    int tile_col = blockIdx.x * 32;

    int r = tile_row + threadIdx.y;
    int c = tile_col + threadIdx.x;

    // Load from input [B, R, C] into shared memory
    if (b < B && r < R && c < C) {
        tile[threadIdx.y][threadIdx.x] = input[b * R * C + r * C + c];
    }

    __syncthreads();

    // Write to output [B, C, R] - transposed
    // output[b, c, r] = input[b, r, c]
    // New indices: out_c = tile_col + threadIdx.y, out_r = tile_row + threadIdx.x
    int out_c = tile_col + threadIdx.y;
    int out_r = tile_row + threadIdx.x;

    if (b < B && out_r < R && out_c < C) {
        output[b * C * R + out_c * R + out_r] = tile[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int B = x.size(0);
    int R = x.size(1);
    int C = x.size(2);

    auto output = torch::empty({B, C, R}, x.options());

    dim3 block(32, 32, 1);
    dim3 grid(
        (C + 31) / 32,
        (R + 31) / 32,
        B
    );

    batched_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, R, C
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched transpose last two dims (CUDA)");
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
            name="task013_ext",
            sources=[os.path.join(src_dir, "solution.cu")],
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
            return ext.batched_transpose(x)
        # Fallback for CPU
        return torch.transpose(x, 1, 2).contiguous()


def reference(x: Tensor) -> Tensor:
    return torch.transpose(x, 1, 2).contiguous()
```
