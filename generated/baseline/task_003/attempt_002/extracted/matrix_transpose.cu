#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input (rows x cols) into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write transposed tile to output (cols x rows)
    // Output shape: [cols, rows]
    // x_out = blockIdx.y * TILE_DIM + threadIdx.x (column index in output = row index in input)
    // y_out = blockIdx.x * TILE_DIM + threadIdx.y (row index in output = col index in input)
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_out < rows && (y_out + j) < cols) {
            output[(y_out + j) * rows + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({cols, rows}, x.options());

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matrix_transpose_cuda, "Matrix transpose CUDA");
}
