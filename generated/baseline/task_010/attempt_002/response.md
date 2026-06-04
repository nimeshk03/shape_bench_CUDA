```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one row (batch, row pair)
// Threads in the block cooperatively reduce along the last dimension
template <int BLOCK_SIZE>
__global__ void sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int cols
) {
    // blockIdx.x = linear index into (batch * rows)
    int row_idx = blockIdx.x;
    const float* row_ptr = x + row_idx * cols;
    
    __shared__ float smem[BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Grid-stride loop over columns
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float val = row_ptr[col];
        sum += val * val;
    }
    
    smem[threadIdx.x] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    if (BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (threadIdx.x < 64) smem[threadIdx.x] += smem[threadIdx.x + 64];
        __syncthreads();
    }
    
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
        out[row_idx] = smem[0];
    }
}

torch::Tensor sum_squares_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    
    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    
    auto out = torch::empty({batch, rows}, x.options());
    
    int total_rows = batch * rows;
    
    // Choose block size based on cols
    const int BLOCK_SIZE = 256;
    
    dim3 grid(total_rows);
    dim3 block(BLOCK_SIZE);
    
    sum_squares_kernel<BLOCK_SIZE><<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        cols
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_squares", &sum_squares_cuda, "Sum of squares along last dimension (CUDA)");
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
            name="sum_squares_ext",
            sources=[os.path.join(src_dir, "sum_squares.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-optimized sum(x * x, dim=-1)."""

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and x.is_contiguous() and x.dtype == torch.float32 and x.dim() == 3:
            ext = _load_ext()
            return ext.sum_squares(x)
        # Fallback
        return torch.sum(x * x, dim=-1)


def reference(x: Tensor) -> Tensor:
    """Functional reference."""
    if x.is_cuda and x.is_contiguous() and x.dtype == torch.float32 and x.dim() == 3:
        ext = _load_ext()
        return ext.sum_squares(x)
    return torch.sum(x * x, dim=-1)
```
