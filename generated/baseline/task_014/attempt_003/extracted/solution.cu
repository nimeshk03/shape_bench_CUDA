#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled transpose kernel for batched matrices
// Transposes dims 1 and 2 of a 3D tensor [B, M, N] -> [B, N, M]
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int M, int N
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int batch = blockIdx.z;
    if (batch >= B) return;

    // Input: [B, M, N], output: [B, N, M]
    // We read a tile from input at (row_in, col_in) and write transposed to output

    int col_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int row_in = blockIdx.y * TILE_DIM + threadIdx.y;

    const float* in_batch = input + batch * M * N;
    float* out_batch = output + batch * N * M;

    // Load tile from input (row_in, col_in) with bounds check
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int r = row_in + i;
        int c = col_in;
        if (r < M && c < N) {
            tile[threadIdx.y + i][threadIdx.x] = in_batch[r * N + c];
        }
    }

    __syncthreads();

    // Write transposed tile to output
    // Output shape: [N, M]
    // The tile block in output corresponds to:
    //   col_out = blockIdx.y * TILE_DIM + threadIdx.x  (was row in input)
    //   row_out = blockIdx.x * TILE_DIM + threadIdx.y  (was col in input)

    int col_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int row_out = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int r = row_out + i;
        int c = col_out;
        if (r < N && c < M) {
            out_batch[r * M + c] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    x = x.contiguous();

    int B = x.size(0);
    int M = x.size(1);
    int N = x.size(2);

    auto output = torch::empty({B, N, M}, x.options());

    if (B == 0 || M == 0 || N == 0) {
        return output;
    }

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM,
        B
    );

    batched_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, M, N
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched matrix transpose (CUDA)");
}
