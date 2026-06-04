```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Kernel: one block per row, shared memory reduction for max and sum
// Handles arbitrary row lengths via a loop over columns

template <int BLOCK_SIZE>
__global__ void rowwise_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Shared memory for reductions
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Step 1: Find row max for numerical stability
    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float val = row_in[col];
        if (val > local_max) local_max = val;
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float val = __expf(row_in[col] - row_max);
        row_out[col] = val;
        local_sum += val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        row_out[col] *= inv_sum;
    }
}

torch::Tensor rowwise_softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");

    x = x.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Choose block size based on column count
    // Use 256 threads as default; for very wide rows use 512 or 1024
    int block_size;
    if (cols <= 128) {
        block_size = 128;
    } else if (cols <= 256) {
        block_size = 256;
    } else if (cols <= 512) {
        block_size = 512;
    } else {
        block_size = 1024;
    }

    dim3 grid(rows);

    if (block_size == 128) {
        rowwise_softmax_kernel<128><<<grid, 128>>>(input_ptr, output_ptr, rows, cols);
    } else if (block_size == 256) {
        rowwise_softmax_kernel<256><<<grid, 256>>>(input_ptr, output_ptr, rows, cols);
    } else if (block_size == 512) {
        rowwise_softmax_kernel<512><<<grid, 512>>>(input_ptr, output_ptr, rows, cols);
    } else {
        rowwise_softmax_kernel<1024><<<grid, 1024>>>(input_ptr, output_ptr, rows, cols);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_softmax", &rowwise_softmax_cuda, "Row-wise softmax (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rowwise_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="rowwise_softmax_cuda",
            sources=["rowwise_softmax.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_70",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  – harness entry point
from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension on first import
def _load_extension():
    import torch.utils.cpp_extension as ext
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return ext.load(
        name="rowwise_softmax_cuda",
        sources=[os.path.join(src_dir, "rowwise_softmax.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def forward(x: Tensor) -> Tensor:
    """Drop-in replacement for Model.forward / reference."""
    if not x.is_cuda:
        x = x.cuda()
    return _get_ext().rowwise_softmax(x)
```
