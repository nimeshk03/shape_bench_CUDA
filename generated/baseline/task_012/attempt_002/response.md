```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: transpose last two dims of a 3D tensor with arbitrary strides
// Input shape: [batch, rows, cols], output shape: [batch, cols, rows]
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int stride_b,   // input stride for batch dim
    int stride_r,   // input stride for row dim
    int stride_c,   // input stride for col dim
    int out_stride_b,  // output stride for batch dim = cols * rows
    int out_stride_r,  // output stride for new row dim (old col) = rows
    int out_stride_c   // output stride for new col dim (old row) = 1
) {
    // Output indexing: [b, c, r] -> input[b, r, c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cols * rows;
    if (idx >= total) return;

    int b = idx / (cols * rows);
    int rem = idx % (cols * rows);
    int c = rem / rows;
    int r = rem % rows;

    // Read from input with arbitrary strides
    float val = input[b * stride_b + r * stride_r + c * stride_c];
    // Write to contiguous output [b, c, r]
    output[b * out_stride_b + c * out_stride_r + r] = val;
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    // Input strides (in elements)
    int stride_b = x.stride(0);
    int stride_r = x.stride(1);
    int stride_c = x.stride(2);

    // Output: contiguous [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int out_stride_b = cols * rows;
    int out_stride_r = rows;

    int total = batch * cols * rows;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    batched_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols,
        stride_b, stride_r, stride_c,
        out_stride_b, out_stride_r, 1
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched transpose of last two dims (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="task_012_ext",
    ext_modules=[
        CUDAExtension(
            name="task_012_ext",
            sources=["task_012_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py  – harness entry point
from __future__ import annotations
from collections.abc import Sequence

import torch
from torch import Tensor, nn

try:
    import task_012_ext as _ext
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if _USE_CUDA and x.is_cuda and x.dtype == torch.float32:
            return _ext.batched_transpose(x)
        # Fallback to PyTorch
        return torch.transpose(x, 1, 2).contiguous()
```
