#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define EPS 1e-5f

// Each block handles one row (one feature vector of length cols)
// We use shared memory for reduction
__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int cols
) {
    // blockIdx.x = row index (batch * rows flattened)
    int row = blockIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float smem[];

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += x_row[i];
    }

    // Reduce sum across threads in block
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = smem[0] / (float)cols;
    __syncthreads();

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    smem[threadIdx.x] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = smem[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPS);
    __syncthreads();

    // Step 3: Normalize and apply affine transform
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float normalized = (x_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
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

    // Choose block size: power of 2, at most 1024
    int block_size = 1;
    while (block_size < cols && block_size < 1024) {
        block_size <<= 1;
    }
    if (block_size > 1024) block_size = 1024;

    // Shared memory size
    size_t smem_size = block_size * sizeof(float);

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
