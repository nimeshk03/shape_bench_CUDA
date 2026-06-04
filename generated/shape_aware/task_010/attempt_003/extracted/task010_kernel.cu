#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each block handles one (batch, row) pair.
// Threads in the block cooperatively reduce over the cols dimension.
__global__ void sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols)
{
    // blockIdx.x -> row index within batch
    // blockIdx.y -> batch index
    int batch_idx = blockIdx.y;
    int row_idx   = blockIdx.x;

    const float* row_ptr = x + (batch_idx * rows + row_idx) * cols;
    float* out_ptr       = out + batch_idx * rows + row_idx;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    // Grid-stride loop over cols
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row_ptr[c];
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_ptr = sdata[0];
    }
}

torch::Tensor sum_squares_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    x = x.contiguous();

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    auto out = torch::zeros({batch, rows}, x.options());

    // Choose block size: power of two, at most 512, at least 1
    int block_size = 1;
    while (block_size * 2 <= cols && block_size < 512) {
        block_size *= 2;
    }
    // Clamp to valid range
    if (block_size > 512) block_size = 512;
    if (block_size < 1)   block_size = 1;

    dim3 grid(rows, batch);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    sum_squares_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_squares", &sum_squares_cuda, "Sum of squares over last dim (CUDA)");
}
