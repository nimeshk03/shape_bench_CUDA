#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Kernel: each block handles one row
// We use shared memory for reduction
__global__ void rowwise_layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sum2 = smem + blockDim.x;

    // Step 1: compute partial sums for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += row_in[i];
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)cols;
    __syncthreads();

    // Step 2: compute partial sums for variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    s_sum2[threadIdx.x] = local_var;
    __syncthreads();

    // Reduce variance sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = s_sum2[0] / (float)cols;
    float inv_std = rsqrtf(variance + EPSILON);
    __syncthreads();

    // Step 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * inv_std;
    }
}

torch::Tensor rowwise_layer_norm_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    // Choose block size: use BLOCK_SIZE threads per block
    // Each block handles one row
    int threads = BLOCK_SIZE;
    // For very short rows, we can use fewer threads
    if (cols < threads) {
        // Round up to next power of 2 for clean reduction
        threads = 1;
        while (threads < cols) threads <<= 1;
        if (threads > BLOCK_SIZE) threads = BLOCK_SIZE;
    }

    dim3 grid(rows);
    dim3 block(threads);
    size_t shared_mem = 2 * threads * sizeof(float);

    rowwise_layer_norm_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_layer_norm", &rowwise_layer_norm_cuda, "Row-wise layer normalization (CUDA)");
}
