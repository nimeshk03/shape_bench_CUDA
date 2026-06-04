```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ x_data,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int rows,
    int cols,
    int stride_batch,
    int stride_row,
    int stride_col,
    int storage_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * rows * cols;
    if (idx >= total) return;

    int b = idx / (rows * cols);
    int rem = idx % (rows * cols);
    int r = rem / cols;
    int c = rem % cols;

    int x_idx = storage_offset + b * stride_batch + r * stride_row + c * stride_col;
    float val = x_data[x_idx] * scale[c] + bias[r];
    out[idx] = val > 0.0f ? val : 0.0f;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);

    auto out = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int stride_batch = x.stride(0);
    int stride_row = x.stride(1);
    int stride_col = x.stride(2);
    int storage_offset = x.storage_offset();

    int total = batch * rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    affine_relu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, rows, cols,
        stride_batch, stride_row, stride_col,
        storage_offset
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Affine ReLU with stride-aware CUDA kernel");
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
            name="affine_relu_ext",
            sources=[os.path.join(src_dir, "affine_relu_cuda.cu")],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        ext = _load_ext()
        if x.is_cuda:
            return ext.forward(x, scale, bias)
        # fallback for CPU
        return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()


def reference(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    ext = _load_ext()
    if x.is_cuda:
        return ext.forward(x, scale, bias)
    return torch.relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
```
