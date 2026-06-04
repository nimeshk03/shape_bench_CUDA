#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ data,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Storage offset
    int64_t storage_offset,
    // Strides for x (in elements)
    int64_t stride_batch,
    int64_t stride_row,
    int64_t stride_col,
    // Output dimensions
    int batch,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (b >= batch || row >= rows || col >= cols) return;

    // Compute input index using strides + storage offset
    int64_t in_idx = storage_offset
                   + (int64_t)b   * stride_batch
                   + (int64_t)row * stride_row
                   + (int64_t)col * stride_col;

    float val = data[in_idx];
    float s   = scale[col];
    float bi  = bias[row];

    float result = val * s + bi;
    if (result < 0.0f) result = 0.0f;

    // Output is contiguous [batch, rows, cols]
    int64_t out_idx = (int64_t)b * rows * cols + (int64_t)row * cols + col;
    output[out_idx] = result;
}

torch::Tensor affine_relu_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");

    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);

    // Get strides (in elements, not bytes)
    int64_t stride_batch = x.stride(0);
    int64_t stride_row   = x.stride(1);
    int64_t stride_col   = x.stride(2);
    int64_t storage_offset = x.storage_offset();

    // Allocate contiguous output
    auto output = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    // Get raw data pointer from storage
    const float* data_ptr = x.storage().data_ptr<float>();

    dim3 block(32, 8, 1);
    dim3 grid(
        (cols  + block.x - 1) / block.x,
        (rows  + block.y - 1) / block.y,
        batch
    );

    affine_relu_kernel<<<grid, block>>>(
        data_ptr,
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        storage_offset,
        stride_batch,
        stride_row,
        stride_col,
        batch,
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU with stride-aware CUDA kernel");
}
