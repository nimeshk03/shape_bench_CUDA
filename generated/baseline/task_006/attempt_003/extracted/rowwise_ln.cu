#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Kernel: each block handles one row
// Uses shared memory for reduction
__global__ void rowwise_layer_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;
    float* out_ptr = out + row * cols;

    __shared__ float smem[BLOCK_SIZE];

    // --- Pass 1: compute mean ---
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_ptr[i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = smem[0] / (float)cols;
    __syncthreads();

    // --- Pass 2: compute variance ---
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_ptr[i] - mean;
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
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // --- Pass 3: normalize ---
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_ptr[i] = (row_ptr[i] - mean) * inv_std;
    }
}

torch::Tensor rowwise_layer_norm_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    x = x.contiguous();
    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty_like(x);

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    rowwise_layer_norm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_layer_norm", &rowwise_layer_norm_cuda, "Row-wise layer norm (CUDA)");
}
