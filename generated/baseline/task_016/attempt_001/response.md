```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define EPS 1e-5f

// Each block handles one row (one (batch, row) pair)
// Threads in the block cooperate to compute mean and variance, then normalize
__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int cols
) {
    // Each block processes one row
    int row_idx = blockIdx.x;
    const float* row_in = x + row_idx * cols;
    float* row_out = out + row_idx * cols;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sum2 = smem + blockDim.x;

    // Step 1: Compute sum and sum of squares
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_in[i];
        local_sum += val;
        local_sum2 += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / cols;
    float mean2 = s_sum2[0] / cols;
    float variance = mean2 - mean * mean;
    float inv_std = rsqrtf(variance + EPS);

    // Step 2: Normalize and apply affine transform
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_in[i];
        float norm = (val - mean) * inv_std;
        row_out[i] = norm * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    int total_rows = batch * rows;

    auto out = torch::empty_like(x);

    // Choose block size: power of 2, at most 512
    int block_size = 1;
    while (block_size < cols && block_size < 512) block_size <<= 1;
    if (block_size > 512) block_size = 512;
    // Make sure block_size is at least 32
    if (block_size < 32) block_size = 32;

    int smem_size = 2 * block_size * sizeof(float);

    layer_norm_kernel<<<total_rows, block_size, smem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_forward, "Layer Norm forward (CUDA)");
}
```

```python
# setup.py / build glue
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="layer_norm_cuda",
    ext_modules=[
        CUDAExtension(
            name="layer_norm_cuda",
            sources=["layer_norm_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# solution.py - harness entry point
import torch
from torch import Tensor

try:
    import layer_norm_cuda
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Fallback pure PyTorch implementation."""
    EPS = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + EPS)
    return (normalized * weight.view(1, 1, -1) + bias.view(1, 1, -1)).contiguous()


def forward(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    if _USE_CUDA and x.is_cuda:
        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous()
        return layer_norm_cuda.forward(x_c, w_c, b_c)
    return reference(x, weight, bias)
```
