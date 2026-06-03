```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one row. Threads cooperatively sum elements in that row.
__global__ void rowwise_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + (long long)row * cols;

    // Shared memory for partial sums
    extern __shared__ float sdata[];

    float partial = 0.0f;
    // Each thread accumulates over its strided portion of the row
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        partial += row_ptr[col];
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();

    // Tree reduction in shared memory
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

    x = x.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty({rows}, x.options());

    if (rows == 0) return out;

    // Choose block size: next power of two >= min(cols, 1024), capped at 1024
    int block_size = 1;
    while (block_size < cols && block_size < 1024) block_size <<= 1;
    if (block_size > 1024) block_size = 1024;
    // Ensure block_size is at least 1
    if (block_size < 1) block_size = 1;

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
# rowwise_sum_impl.py  (harness entry-point)
import torch
from torch import Tensor
import rowwise_sum_ext


def forward(x: Tensor) -> Tensor:
    return rowwise_sum_ext.rowwise_sum(x)
```
