#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled transpose kernel for batched matrix transpose
// Input:  [batch, rows, cols]
// Output: [batch, cols, rows]

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int b = blockIdx.z;
    if (b >= batch) return;

    // Input: [batch, rows, cols], stride = rows*cols per batch
    // Output: [batch, cols, rows], stride = cols*rows per batch
    const float* in_batch  = input  + b * rows * cols;
    float*       out_batch = output + b * cols * rows;

    int col_in = blockIdx.x * TILE_DIM + threadIdx.x; // column index in input
    int row_in = blockIdx.y * TILE_DIM + threadIdx.y; // row index in input

    // Load tile from input (coalesced reads along cols)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int r = row_in + i;
        if (r < rows && col_in < cols) {
            tile[threadIdx.y + i][threadIdx.x] = in_batch[r * cols + col_in];
        }
    }

    __syncthreads();

    // Write transposed tile to output
    // In output: [cols, rows], so output[col_in][row_in]
    int col_out = blockIdx.y * TILE_DIM + threadIdx.x; // was row in input
    int row_out = blockIdx.x * TILE_DIM + threadIdx.y; // was col in input

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int r = row_out + i;
        if (r < cols && col_out < rows) {
            out_batch[r * rows + col_out] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    auto output = torch::empty({batch, cols, rows}, x.options());

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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose_cuda, "Batched matrix transpose (CUDA)");
}
