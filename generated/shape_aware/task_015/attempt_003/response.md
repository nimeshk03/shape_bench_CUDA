```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ data,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Storage offset
    int64_t storage_offset,
    // Strides for x (in elements)
    int64_t stride_batch,
    int64_t stride_row,
    int64_t stride_col,
    // Output dimensions
    int batch,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (b >= batch || row >= rows || col >= cols) return;

    // Compute input index using strides + storage offset
    int64_t in_idx = storage_offset
                   + (int64_t)b   * stride_batch
                   + (int64_t)row * stride_row
                   + (int64_t)col * stride_col;

    float val = data[in_idx];
    float s   = scale[col];
    float bi  = bias[row];

    float result = val * s + bi;
    if (result < 0.0f) result = 0.0f;

    // Output is contiguous [batch, rows, cols]
    int64_t out_idx = (int64_t)b * rows * cols + (int64_t)row * cols + col;
    output[out_idx] = result;
}

torch::Tensor affine_relu_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    // Get strides (in elements, not bytes)
    int64_t stride_batch = x.stride(0);
    int64_t stride_row   = x.stride(1);
    int64_t stride_col   = x.stride(2);
    int64_t storage_offset = x.storage_offset();

    // Allocate contiguous output
    auto output = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    // Get raw data pointer from storage
    const float* data_ptr = x.storage().data_ptr<float>();

    dim3 block(32, 8, 1);
    dim3 grid(
        (cols  + block.x - 1) / block.x,
        (rows  + block.y - 1) / block.y,
        batch
    );

    affine_relu_kernel<<<grid, block>>>(
        data_ptr,
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        storage_offset,
        stride_batch,
        stride_row,
        stride_col,
        batch,
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU with stride-aware CUDA kernel");
}
```

```python
# setup.py  (build glue)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task015_ext",
    ext_modules=[
        CUDAExtension(
            name="task015_ext",
            sources=["task015_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  – drop-in replacement for the harness
"""Task 015 CUDA solution: affine ReLU over offset/strided 3D view."""

from __future__ import annotations

import os
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Build / load the extension once
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as _ext

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT = _ext.load(
    name="task015_ext",
    sources=[os.path.join(_SRC_DIR, "task015_kernel.cu")],
    verbose=False,
)


# ---------------------------------------------------------------------------
# Public API expected by the harness
# ---------------------------------------------------------------------------

def forward(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """CUDA implementation matching Model.forward / reference."""
    if x.is_cuda:
        return _EXT.affine_relu_cuda(x, scale, bias)
    # Fallback for CPU tensors (shouldn't happen in normal harness usage)
    return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
```
