#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Kernel: one block per row, shared memory reduction for max and sum
// Handles arbitrary row lengths via a loop over columns

template <int BLOCK_SIZE>
__global__ void rowwise_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float sdata[BLOCK_SIZE];

    // Step 1: find row max for numerical stability
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float val = row_in[col];
        if (val > thread_max) thread_max = val;
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduce max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: compute exp(x - max) and partial sum
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float val = __expf(row_in[col] - row_max);
        row_out[col] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Step 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        row_out[col] *= inv_sum;
    }
}

torch::Tensor rowwise_softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty_like(x);

    const int BLOCK_SIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    rowwise_softmax_kernel<BLOCK_SIZE><<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_softmax", &rowwise_softmax_cuda, "Row-wise softmax (CUDA)");
}
