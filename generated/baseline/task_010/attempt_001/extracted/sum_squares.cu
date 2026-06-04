#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one row (batch, row pair)
// Threads in the block cooperatively reduce along the last dimension
template <int BLOCK_SIZE>
__global__ void sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int cols
) {
    // blockIdx.x = row index in flattened (batch * rows) space
    int row = blockIdx.x;
    const float* row_ptr = x + row * cols;
    
    __shared__ float smem[BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Grid-stride loop within the block
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = row_ptr[i];
        sum += val * val;
    }
    
    smem[threadIdx.x] = sum;
    __syncthreads();
    
    // Tree reduction
    if (BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (threadIdx.x < 64) smem[threadIdx.x] += smem[threadIdx.x + 64];
        __syncthreads();
    }
    // Warp-level reduction (no sync needed within a warp)
    if (threadIdx.x < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        out[row] = smem[0];
    }
}

torch::Tensor sum_squares_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    
    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    int total_rows = batch * rows;
    
    auto out = torch::empty({batch, rows}, x.options());
    
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    // Choose block size based on cols
    // For cols=512, use 256 threads (each thread handles 2 elements)
    int grid = total_rows;
    
    if (cols <= 64) {
        sum_squares_kernel<64><<<grid, 64>>>(x_ptr, out_ptr, cols);
    } else if (cols <= 128) {
        sum_squares_kernel<128><<<grid, 128>>>(x_ptr, out_ptr, cols);
    } else if (cols <= 256) {
        sum_squares_kernel<256><<<grid, 256>>>(x_ptr, out_ptr, cols);
    } else {
        sum_squares_kernel<256><<<grid, 256>>>(x_ptr, out_ptr, cols);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_squares", &sum_squares_cuda, "Sum of squares along last dimension (CUDA)");
}
