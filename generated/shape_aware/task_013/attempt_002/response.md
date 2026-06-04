```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for batched transpose: [B, R, C] -> [B, C, R]
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * rows * cols;
    
    if (idx >= total) return;
    
    // Decode flat index into (b, r, c)
    int c = idx % cols;
    int tmp = idx / cols;
    int r = tmp % rows;
    int b = tmp / rows;
    
    // Input index: b * rows * cols + r * cols + c
    // Output index: b * cols * rows + c * rows + r
    int in_idx = b * rows * cols + r * cols + c;
    int out_idx = b * cols * rows + c * rows + r;
    
    output[out_idx] = input[in_idx];
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");
    
    // Make input contiguous
    x = x.contiguous();
    
    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    
    // Output shape: [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options());
    
    int total = batch * rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    batched_transpose_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched matrix transpose (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='batched_transpose_cuda',
    ext_modules=[
        CUDAExtension(
            name='batched_transpose_cuda',
            sources=['batched_transpose.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

```python
# solution.py
"""CUDA extension implementation for task_013: diagnostic batched transpose."""

from __future__ import annotations

import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

# Load the CUDA extension inline
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name='batched_transpose_cuda',
            sources=[os.path.join(_src_dir, 'batched_transpose.cu')],
            verbose=False,
        )
    return _ext


def reference(x: Tensor) -> Tensor:
    """CUDA implementation matching torch.transpose(x, 1, 2).contiguous()"""
    if x.is_cuda:
        ext = _get_ext()
        return ext.batched_transpose(x)
    else:
        # Fallback for CPU tensors: move to CUDA, compute, move back
        device = x.device
        x_cuda = x.cuda()
        ext = _get_ext()
        result = ext.batched_transpose(x_cuda)
        return result.to(device)
```
