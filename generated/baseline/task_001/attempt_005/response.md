```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized kernel using float4 for coalesced memory access
__global__ void add_relu_kernel_vec4(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int n_vec4,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vec4) {
        float4 xv = reinterpret_cast<const float4*>(x)[idx];
        float4 yv = reinterpret_cast<const float4*>(y)[idx];
        float4 res;
        res.x = fmaxf(xv.x + yv.x, 0.0f);
        res.y = fmaxf(xv.y + yv.y, 0.0f);
        res.z = fmaxf(xv.z + yv.z, 0.0f);
        res.w = fmaxf(xv.w + yv.w, 0.0f);
        reinterpret_cast<float4*>(out)[idx] = res;
    }
    
    // Handle remainder elements
    int rem_start = n_vec4 * 4;
    int rem_idx = rem_start + (idx - n_vec4);
    if (idx >= n_vec4 && rem_idx < n_total) {
        out[rem_idx] = fmaxf(x[rem_idx] + y[rem_idx], 0.0f);
    }
}

torch::Tensor add_relu_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    
    auto out = torch::empty_like(x);
    
    int n_total = x.numel();
    int n_vec4 = n_total / 4;
    int n_rem = n_total % 4;
    
    // Total threads needed: n_vec4 for vectorized + n_rem for remainder
    int total_threads = n_vec4 + n_rem;
    
    if (total_threads > 0) {
        const int block_size = 256;
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        add_relu_kernel_vec4<<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            out.data_ptr<float>(),
            n_vec4,
            n_total
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu_cuda, "Fused add + ReLU (CUDA)");
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
            name="add_relu_ext",
            sources=[os.path.join(src_dir, "add_relu.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    return _ext


import torch
from torch import Tensor, nn


class Model(nn.Module):
    """CUDA-optimized relu(x + y)."""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.is_cuda and x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32:
            ext = _load_ext()
            return ext.add_relu(x, y)
        # Fallback
        return torch.relu(x + y)


def reference(x: Tensor, y: Tensor) -> Tensor:
    if x.is_cuda and x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32:
        ext = _load_ext()
        return ext.add_relu(x, y)
    return torch.relu(x + y)
```
