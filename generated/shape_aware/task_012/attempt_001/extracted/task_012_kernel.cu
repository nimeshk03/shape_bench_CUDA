#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that handles non-contiguous input with arbitrary strides
// Input: [batch, rows, cols] with strides (stride_b, stride_r, stride_c)
// Output: [batch, cols, rows] contiguous
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int stride_b,  // input stride for batch dim
    int stride_r,  // input stride for row dim
    int stride_c,  // input stride for col dim
    int out_stride_b,  // output stride for batch dim = cols * rows
    int out_stride_r,  // output stride for row dim (in output) = rows
    int out_stride_c   // output stride for col dim (in output) = 1
) {
    // Each thread handles one element
    // Output shape: [batch, cols, rows]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cols * rows;
    
    if (idx >= total) return;
    
    // Decompose linear index into (b, oc, or) where oc=output col index, or=output row index
    int b = idx / (cols * rows);
    int rem = idx % (cols * rows);
    int oc = rem / rows;  // output col dimension = original row dimension
    int or_ = rem % rows; // output row dimension = original col dimension
    
    // Wait - let me reconsider:
    // Output is [batch, cols, rows]
    // output[b][oc][or_] = input[b][or_][oc]
    // where oc ranges over cols, or_ ranges over rows
    
    // Input element: input[b][or_][oc] with strides
    int in_offset = b * stride_b + or_ * stride_r + oc * stride_c;
    
    // Output element: output[b * cols * rows + oc * rows + or_]
    int out_offset = b * (cols * rows) + oc * rows + or_;
    
    output[out_offset] = input[in_offset];
}

// Tiled version for better memory access patterns
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void batched_transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int stride_b,
    int stride_r,
    int stride_c
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int b = blockIdx.z;
    if (b >= batch) return;
    
    // Tile coordinates in the output space [cols, rows]
    // We tile over (cols, rows) dimensions
    int tile_col = blockIdx.x * TILE_DIM; // output col (= input row)
    int tile_row = blockIdx.y * TILE_DIM; // output row (= input col)
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load tile from input: input[b][tile_col + tx][tile_row + ty]
    // (reading input[b][row][col] where row = tile_col+tx, col = tile_row+ty)
    // This reads input rows contiguously if stride_c is small
    
    // Each thread loads multiple elements
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int in_row = tile_col + tx;      // input row index
        int in_col = tile_row + ty + j;  // input col index
        
        if (in_row < cols && in_col < rows) {
            // input[b][in_row][in_col] -> tile[tx][ty+j]
            int in_offset = b * stride_b + in_row * stride_r + in_col * stride_c;
            tile[tx][ty + j] = input[in_offset];
        } else {
            tile[tx][ty + j] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Write tile to output: output[b][tile_col + tx][tile_row + ty]
    // output shape is [batch, cols, rows], contiguous
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int out_col_idx = tile_col + tx;  // output col dimension
        int out_row_idx = tile_row + ty + j; // output row dimension
        
        if (out_col_idx < cols && out_row_idx < rows) {
            int out_offset = b * (cols * rows) + out_col_idx * rows + out_row_idx;
            output[out_offset] = tile[tx][ty + j];
        }
    }
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    
    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    
    // Get actual strides (handles non-contiguous input)
    int stride_b = x.stride(0);
    int stride_r = x.stride(1);
    int stride_c = x.stride(2);
    
    // Output: [batch, cols, rows] contiguous
    auto output = torch::empty({batch, cols, rows}, 
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(x.device()));
    
    if (batch == 0 || rows == 0 || cols == 0) {
        return output;
    }
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Use tiled kernel for better performance
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM,
        batch
    );
    
    batched_transpose_tiled_kernel<<<grid, block>>>(
        input_ptr, output_ptr,
        batch, rows, cols,
        stride_b, stride_r, stride_c
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// Fallback simple kernel
torch::Tensor batched_transpose_simple_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    
    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);
    
    int stride_b = x.stride(0);
    int stride_r = x.stride(1);
    int stride_c = x.stride(2);
    
    auto output = torch::empty({batch, cols, rows},
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(x.device()));
    
    if (batch == 0 || rows == 0 || cols == 0) {
        return output;
    }
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    batched_transpose_kernel<<<blocks, threads>>>(
        input_ptr, output_ptr,
        batch, rows, cols,
        stride_b, stride_r, stride_c,
        cols * rows, rows, 1
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose_cuda, "Batched transpose (tiled CUDA)");
    m.def("batched_transpose_simple", &batched_transpose_simple_cuda, "Batched transpose (simple CUDA)");
}
