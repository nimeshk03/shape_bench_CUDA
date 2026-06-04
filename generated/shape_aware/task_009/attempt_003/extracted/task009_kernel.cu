#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    int cols,
    int x_stride_row,  // stride in elements between rows of x
    int x_stride_col   // stride in elements between cols of x
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        float xval = x[row * x_stride_row + col * x_stride_col];
        float val = xval * scale[col] + bias[col];
        out[row * cols + col] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor affine_relu_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    int rows = x.size(0);
    int cols = x.size(1);

    // Get strides in number of elements
    int x_stride_row = x.stride(0);
    int x_stride_col = x.stride(1);

    // Ensure scale and bias are contiguous
    scale = scale.contiguous();
    bias = bias.contiguous();

    // Output is always contiguous
    auto out = torch::empty({rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    // Make sure x data pointer is accessible (it may be non-contiguous but on same device)
    // We need the raw data pointer of the base storage
    const float* x_ptr = x.data_ptr<float>();
    const float* scale_ptr = scale.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    dim3 block(32, 8);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    affine_relu_kernel<<<grid, block>>>(
        x_ptr, scale_ptr, bias_ptr, out_ptr,
        rows, cols,
        x_stride_row, x_stride_col
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU on non-contiguous matrix (CUDA)");
}
