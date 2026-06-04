```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ storage,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // storage offset
    int64_t storage_offset,
    // strides for x (in elements)
    int64_t stride_batch,
    int64_t stride_row,
    int64_t stride_col,
    // output dimensions
    int batch,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (b >= batch || row >= rows || col >= cols) return;

    // Compute index into storage
    int64_t src_idx = storage_offset
                    + (int64_t)b   * stride_batch
                    + (int64_t)row * stride_row
                    + (int64_t)col * stride_col;

    float val = storage[src_idx];
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
    // x is a non-contiguous 3D tensor
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32);

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    // Get strides (in elements, since float32)
    int64_t stride_batch = x.stride(0);
    int64_t stride_row   = x.stride(1);
    int64_t stride_col   = x.stride(2);

    // Storage offset in elements
    int64_t storage_offset = x.storage_offset();

    // Get raw storage pointer
    const float* storage_ptr = x.storage().data<float>();

    // Ensure scale and bias are contiguous
    auto scale_c = scale.contiguous();
    auto bias_c  = bias.contiguous();

    // Allocate output
    auto output = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    // Launch kernel
    dim3 block(16, 16, 1);
    dim3 grid(
        (cols  + block.x - 1) / block.x,
        (rows  + block.y - 1) / block.y,
        batch
    );

    affine_relu_kernel<<<grid, block>>>(
        storage_ptr,
        scale_c.data_ptr<float>(),
        bias_c.data_ptr<float>(),
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
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU over offset strided 3D tensor (CUDA)");
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
from __future__ import annotations
from collections.abc import Sequence
import os, torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Build / load the CUDA extension once
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as _ext

_SRC = os.path.join(os.path.dirname(__file__), "task015_kernel.cu")

_ext_module = _ext.load(
    name="task015_ext",
    sources=[_SRC],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Public API expected by the harness
# ---------------------------------------------------------------------------

class Model(torch.nn.Module):
    """CUDA-accelerated affine ReLU over an offset, irregularly strided 3D view."""

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        return _ext_module.affine_relu_cuda(x, scale, bias)


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()


def create_inputs(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    normalized_shape = tuple(int(d) for d in shape)
    if len(normalized_shape) != 3:
        raise ValueError(f"task_015 expects a 3D shape, got {normalized_shape}")
    batch, rows, cols = normalized_shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn(
        (batch + 1, rows * 2 + 3, cols * 3 + 5),
        generator=generator, dtype=dtype,
    )
    x = base[1:, 1 : 1 + rows * 2 : 2, 2 : 2 + cols * 3 : 3]
    scale = torch.randn((cols,), generator=generator, dtype=dtype)
    bias  = torch.randn((rows,), generator=generator, dtype=dtype)
    return x.to(device), scale.to(device), bias.to(device)
```
