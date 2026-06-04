#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int K, int N
) {
    // Each block handles one (batch, tile_row, tile_col) combination
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (b >= batch) return;

    const float* a_batch = A + b * M * K;
    const float* b_batch = B + b * K * N;
    float* c_batch = C + b * M * N;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int k_a = t * TILE_SIZE + threadIdx.x;
        int k_b = t * TILE_SIZE + threadIdx.y;

        // Load tile from A
        if (row < M && k_a < K)
            sA[threadIdx.y][threadIdx.x] = a_batch[row * K + k_a];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (k_b < K && col < N)
            sB[threadIdx.y][threadIdx.x] = b_batch[k_b * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c_batch[row * N + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 3, "a must be 3D");
    TORCH_CHECK(b.dim() == 3, "b must be 3D");
    TORCH_CHECK(a.size(0) == b.size(0), "batch sizes must match");
    TORCH_CHECK(a.size(2) == b.size(1), "inner dimensions must match");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "only float32 supported");

    a = a.contiguous();
    b = b.contiguous();

    int batch = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);

    auto c = torch::empty({batch, M, N}, a.options());

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch
    );

    batched_matmul_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch, M, K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul_cuda, "Batched matrix multiplication (CUDA)");
}
