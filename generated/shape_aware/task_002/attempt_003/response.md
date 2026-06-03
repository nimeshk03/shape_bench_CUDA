```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one row. Threads cooperatively sum elements of that row.
__global__ void rowwise_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + (long long)row * cols;

    // Use shared memory for reduction
    extern __shared__ float sdata[];

    float sum = 0.0f;
    // Grid-stride loop over columns
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += row_ptr[col];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = sdata[0];
    }
}

torch::Tensor rowwise_sum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty({rows}, x.options());

    if (rows == 0 || cols == 0) {
        if (rows > 0) out.zero_();
        return out;
    }

    // Choose block size: power of two, at most 512, at least 32
    int block_size = 32;
    while (block_size < cols && block_size < 512) {
        block_size <<= 1;
    }
    // block_size is now a power of two >= min(cols, 512)
    // Cap at 512
    if (block_size > 512) block_size = 512;

    dim3 grid(rows);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    rowwise_sum_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_sum", &rowwise_sum_cuda, "Row-wise sum (CUDA)");
}
```

```python
# setup.py  (build glue)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_sum_ext",
    ext_modules=[
        CUDAExtension(
            name="rowwise_sum_ext",
            sources=["rowwise_sum.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  (harness entry-point)
import torch
from torch import Tensor
import os
import sys

# Build and load the extension inline so the harness can import this file directly.
from torch.utils.cpp_extension import load
import pathlib

_dir = pathlib.Path(__file__).parent
_ext = load(
    name="rowwise_sum_ext",
    sources=[str(_dir / "rowwise_sum.cu")],
    verbose=False,
)


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    if not x.is_contiguous():
        x = x.contiguous()
    return _ext.rowwise_sum(x)
```
