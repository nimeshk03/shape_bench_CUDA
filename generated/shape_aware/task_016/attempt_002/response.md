Looking at this task, I need to implement layer normalization over the last dimension of a 3D float32 tensor. The key operations are:
1. Compute mean over last dim
2. Compute variance over last dim  
3. Normalize
4. Apply affine transform (weight, bias)

I'll use a warp/block reduction approach where each block handles one row (one [batch, row] pair), and threads cooperate to compute mean and variance.

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPS 1e-5f
#define BLOCK_SIZE 256

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int cols
) {
    // Each block handles one row (one [batch, row] pair)
    int row = blockIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Shared memory for partial sums
    __shared__ float smem[BLOCK_SIZE];

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        sum += x_row[i];
    }

    // Block reduce
    smem[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = nthreads / 2; stride > 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    // Warp reduce for last 32
    if (tid < 32) {
        float val = smem[tid];
        val = warp_reduce_sum(val);
        smem[tid] = val;
    }
    __syncthreads();

    float mean = smem[0] / (float)cols;
    __syncthreads();

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    smem[tid] = var_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        float val = smem[tid];
        val = warp_reduce_sum(val);
        smem[tid] = val;
    }
    __syncthreads();

    float variance = smem[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPS);
    __syncthreads();

    // Step 3: Normalize and apply affine
    for (int i = tid; i < cols; i += nthreads) {
        float norm = (x_row[i] - mean) * inv_std;
        out_row[i] = norm * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    int total_rows = batch * rows;

    auto out = torch::empty_like(x);

    // Choose block size: use BLOCK_SIZE but cap at cols rounded up to warp
    int block_size = BLOCK_SIZE;
    if (cols < block_size) {
        // Round up to nearest warp
        block_size = ((cols + 31) / 32) * 32;
        if (block_size < 32) block_size = 32;
    }

    dim3 grid(total_rows);
    dim3 block(block_size);

    layer_norm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layer_norm_cuda, "Layer Norm CUDA");
}
```

```python
# setup.py / build glue
import os
import torch
from torch.utils.cpp_extension import load

_ext_dir = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="task016_ext",
    sources=[os.path.join(_ext_dir, "task016_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def reference(x, weight, bias):
    """CUDA implementation matching the PyTorch reference."""
    if not x.is_cuda:
        x = x.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    return _ext.layer_norm(x, weight, bias)
```

The implementation:
- Each CUDA block handles one row (one `[batch, row]` pair)
- Uses shared memory tree reduction + warp shuffle for mean and variance
- Handles arbitrary `cols` sizes (non-power-of-two, odd, etc.) via strided loops with boundary-safe indexing
- Block size adapts to small `cols` values by rounding up to nearest warp
- No hardcoded dimension assumptions
