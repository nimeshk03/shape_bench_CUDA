```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one (batch, row) pair.
// Threads in the block cooperatively reduce over the cols dimension.
__global__ void sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols)
{
    // blockIdx.x -> row index within batch
    // blockIdx.y -> batch index
    int batch = blockIdx.y;
    int row   = blockIdx.x;

    const float* row_ptr = x + (batch * rows + row) * cols;
    float* out_ptr       = out + batch * rows + row;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    // Grid-stride loop over cols
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row_ptr[c];
        sum += v * v;
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
        rows,
        cols
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
# harness_entry.py  – drop-in forward() wrapper used by the evaluation harness
import torch
from torch import Tensor

# Build / load the extension on first import
import os, subprocess, sys

_ext = None

def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    _ext = load(
        name="task010_ext",
        sources=[os.path.join(os.path.dirname(__file__), "task010_kernel.cu")],
        verbose=False,
    )
    return _ext


def forward(x: Tensor) -> Tensor:
    ext = _load_ext()
    return ext.sum_squares(x.contiguous())
```
