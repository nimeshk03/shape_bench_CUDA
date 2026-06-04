```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one (batch, row) pair and reduces over the cols dimension.
// We use shared memory for the partial sums.

__global__ void sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch,
    int rows,
    int cols
) {
    // blockIdx.x -> row index within batch
    // blockIdx.y -> batch index
    int b = blockIdx.y;
    int r = blockIdx.x;

    if (b >= batch || r >= rows) return;

    const float* row_ptr = x + b * rows * cols + r * cols;
    float* out_ptr = out + b * rows + r;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    // Each thread accumulates over strided elements
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = row_ptr[c];
        local_sum += val * val;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_ptr = sdata[0];
    }
}

torch::Tensor sum_squares_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    auto out = torch::zeros({batch, rows}, x.options());

    // Choose block size: next power of two >= min(cols, 1024), capped at 1024
    int block_size = 1;
    while (block_size < cols && block_size < 1024) block_size <<= 1;
    if (block_size > 1024) block_size = 1024;

    dim3 grid(rows, batch);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    sum_squares_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, rows, cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_squares", &sum_squares_cuda, "Sum of squares over last dim (CUDA)");
}
```

```python
# setup.py  (build glue)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task010_ext",
    ext_modules=[
        CUDAExtension(
            name="task010_ext",
            sources=["task010_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  (harness entry-point)
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build / load the extension on first import
from torch.utils.cpp_extension import load

_ext = load(
    name="task010_ext",
    sources=[os.path.join(os.path.dirname(__file__), "task010_kernel.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.sum_squares(x)
```
