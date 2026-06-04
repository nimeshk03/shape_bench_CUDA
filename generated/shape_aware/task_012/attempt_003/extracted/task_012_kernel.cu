#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that handles non-contiguous input with arbitrary strides
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int stride_batch,  // input stride for batch dimension
    int stride_row,    // input stride for row dimension
    int stride_col,    // input stride for col dimension
    int out_stride_batch,  // output stride for batch dimension (= cols * rows)
    int out_stride_row,    // output stride for row dimension (= rows)
    int out_stride_col     // output stride for col dimension (= 1)
) {
    // Output shape is [batch, cols, rows]
    // We iterate over output indices
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;  // output col index (= input row)
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;  // output row index (= input col)
    int b = blockIdx.z;

    if (b >= batch || out_row >= cols || out_col >= rows) return;

    // Input index: [b, out_col, out_row] in original [batch, rows, cols] layout
    int in_idx = b * stride_batch + out_col * stride_row + out_row * stride_col;
    // Output index: [b, out_row, out_col] in [batch, cols, rows] layout (contiguous)
    int out_idx = b * out_stride_batch + out_row * out_stride_row + out_col * out_stride_col;

    output[out_idx] = input[in_idx];
}

torch::Tensor batched_transpose_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);

    // Get input strides (handles non-contiguous input)
    int stride_batch = x.stride(0);
    int stride_row = x.stride(1);
    int stride_col = x.stride(2);

    // Output is contiguous with shape [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int out_stride_batch = cols * rows;
    int out_stride_row = rows;
    int out_stride_col = 1;

    // Use 16x16 thread blocks
    dim3 block(16, 16, 1);
    dim3 grid(
        (rows + block.x - 1) / block.x,
        (cols + block.y - 1) / block.y,
        batch
    );

    batched_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols,
        stride_batch, stride_row, stride_col,
        out_stride_batch, out_stride_row, out_stride_col
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose_cuda, "Batched transpose of non-contiguous strided tensor (CUDA)");
}
