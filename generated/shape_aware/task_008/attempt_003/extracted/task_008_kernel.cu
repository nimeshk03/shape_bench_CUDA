#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched matrix transpose
// Input:  [batch, rows, cols]
// Output: [batch, cols, rows]
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    // Use 2D tile for coalesced memory access
    // Each thread block handles a TILE x TILE tile
    const int TILE = 32;
    
    __shared__ float tile[32][33]; // +1 to avoid bank conflicts
    
    int bx = blockIdx.z; // batch index
    int row_tile = blockIdx.y * TILE;
    int col_tile = blockIdx.x * TILE;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load from input[bx, row_tile+ty, col_tile+tx]
    int in_row = row_tile + ty;
    int in_col = col_tile + tx;
    
    if (bx < batch && in_row < rows && in_col < cols) {
        tile[ty][tx] = input[bx * rows * cols + in_row * cols + in_col];
    } else {
        tile[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Write to output[bx, col_tile+tx, row_tile+ty]
    // After transpose: output shape is [batch, cols, rows]
    // output[bx, col_tile+ty, row_tile+tx] = tile[tx][ty]
    int out_row = col_tile + ty; // this is the new "row" dimension (old col)
    int out_col = row_tile + tx; // this is the new "col" dimension (old row)
    
    if (bx < batch && out_row < cols && out_col < rows) {
        output[bx * cols * rows + out_row * rows + out_col] = tile[tx][ty];
    }
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");
    
    x = x.contiguous();
    
    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);
    
    // Output shape: [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options());
    
    const int TILE = 32;
    
    dim3 block(TILE, TILE, 1);
    dim3 grid(
        (cols + TILE - 1) / TILE,
        (rows + TILE - 1) / TILE,
        batch
    );
    
    batched_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose_cuda, "Batched matrix transpose (CUDA)");
}
