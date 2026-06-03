```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one row. Threads cooperatively sum elements in the row.
template <int BLOCK_SIZE>
__global__ void rowwise_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = input + row * cols;

    float sum = 0.0f;
    // Each thread accumulates a partial sum over strided elements
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        sum += row_ptr[col];
    }

    // Shared memory reduction
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x < 64)  smem[threadIdx.x] += smem[threadIdx.x + 64];  __syncthreads(); }
    // Warp-level reduction (no sync needed within a warp)
    if (threadIdx.x < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        output[row] = smem[0];
    }
}

torch::Tensor rowwise_sum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({rows}, x.options());

    if (rows == 0) return output;

    // Choose block size based on cols
    const int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    rowwise_sum_kernel<BLOCK_SIZE><<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_sum", &rowwise_sum_cuda, "Row-wise sum (CUDA)");
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
            name="rowwise_sum_ext",
            sources=[os.path.join(src_dir, "rowwise_sum.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-accelerated row-wise sum."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            ext = _load_ext()
            return ext.rowwise_sum(x)
        # Fallback
        return torch.sum(x, dim=1)


def reference(x: Tensor) -> Tensor:
    return torch.sum(x, dim=1)
```
