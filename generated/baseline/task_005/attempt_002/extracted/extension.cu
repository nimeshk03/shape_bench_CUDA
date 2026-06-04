#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Each block handles one row
// Uses shared memory for inter-warp reductions
__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows, int cols) {
    extern __shared__ float smem[];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int num_warps = blockSize / 32;
    
    float* smem_max = smem;
    float* smem_sum = smem + num_warps;
    
    // Step 1: Find row max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockSize) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    
    // Warp reduce max
    local_max = warp_reduce_max(local_max);
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) smem_max[warp_id] = local_max;
    __syncthreads();
    
    // Final reduce across warps
    float row_max = -FLT_MAX;
    if (tid < num_warps) {
        row_max = smem_max[tid];
    }
    if (tid < 32) {
        row_max = warp_reduce_max(row_max);
    }
    if (tid == 0) smem_max[0] = row_max;
    __syncthreads();
    row_max = smem_max[0];
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockSize) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;
        local_sum += val;
    }
    
    // Warp reduce sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) smem_sum[warp_id] = local_sum;
    __syncthreads();
    
    float row_sum = 0.0f;
    if (tid < num_warps) {
        row_sum = smem_sum[tid];
    }
    if (tid < 32) {
        row_sum = warp_reduce_sum(row_sum);
    }
    if (tid == 0) smem_sum[0] = row_sum;
    __syncthreads();
    row_sum = smem_sum[0];
    
    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockSize) {
        row_out[i] *= inv_sum;
    }
}

torch::Tensor rowwise_softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    x = x.contiguous();
    
    int rows = x.size(0);
    int cols = x.size(1);
    
    auto output = torch::empty_like(x);
    
    // Choose block size: use 256 threads for cols <= 1024, else 512
    int block_size;
    if (cols <= 256) block_size = 64;
    else if (cols <= 512) block_size = 128;
    else if (cols <= 1024) block_size = 256;
    else block_size = 512;
    
    // Ensure block_size is multiple of 32
    block_size = ((block_size + 31) / 32) * 32;
    if (block_size > 1024) block_size = 1024;
    
    int num_warps = block_size / 32;
    int smem_size = 2 * num_warps * sizeof(float);
    
    softmax_kernel<<<rows, block_size, smem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rowwise_softmax_cuda, "Row-wise softmax (CUDA)");
}
