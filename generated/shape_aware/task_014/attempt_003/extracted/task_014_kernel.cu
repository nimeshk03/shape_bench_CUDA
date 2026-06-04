#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for the transpose kernel
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int b = blockIdx.z;
    if (b >= batch) return;

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Input: [batch, rows, cols], output: [batch, cols, rows]
    const float* in = input + b * rows * cols;
    float* out = output + b * cols * rows;

    // Load tile from input (rows x cols) into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < rows && col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = in[row * cols + col];
        }
    }

    __syncthreads();

    // Write transposed tile to output (cols x rows)
    // Now x indexes into cols dimension, y indexes into rows dimension
    // After transpose: output[col, row] = input[row, col]
    // Output shape: [cols, rows]
    // We write: out[col * rows + row]
    // Transposed block: blockIdx.x * TILE_DIM -> col start, blockIdx.y * TILE_DIM -> row start
    int out_col = blockIdx.y * TILE_DIM + threadIdx.x;  // was row, now col in output
    int out_row = blockIdx.x * TILE_DIM + threadIdx.y;  // was col, now row in output

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int oc = out_col;
        int or_ = out_row + j;
        if (oc < rows && or_ < cols) {
            // tile[threadIdx.x][threadIdx.y + j] contains input[blockIdx.y*TILE_DIM + threadIdx.x][blockIdx.x*TILE_DIM + threadIdx.y + j]
            out[or_ * rows + oc] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);

    // Output shape: [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options());

    if (batch == 0 || rows == 0 || cols == 0) {
        return output;
    }

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM,
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
    m.def("batched_transpose", &batched_transpose, "Batched matrix transpose (CUDA)");
}
