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
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Warp reduce for last 32
    if (tid < 32) {
        float val = smem[tid];
        if (nthreads >= 64) val += smem[tid + 32];
        val = warp_reduce_sum(val);
        smem[tid] = val;
    }
    __syncthreads();

    float mean = smem[0] / (float)cols;

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    smem[tid] = var_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = smem[tid];
        if (nthreads >= 64) val += smem[tid + 32];
        val = warp_reduce_sum(val);
        smem[tid] = val;
    }
    __syncthreads();

    float variance = smem[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPS);

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
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    int total_rows = batch * rows;

    auto out = torch::empty_like(x);

    // Choose block size: use BLOCK_SIZE but cap at cols rounded up to warp
    int block_size = BLOCK_SIZE;
    if (cols <= 32) block_size = 32;
    else if (cols <= 64) block_size = 64;
    else if (cols <= 128) block_size = 128;
    else block_size = BLOCK_SIZE;

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
