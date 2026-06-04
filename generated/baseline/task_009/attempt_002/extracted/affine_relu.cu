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
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    
    int rows = x.size(0);
    int cols = x.size(1);
    
    // Get strides in elements (not bytes)
    int x_stride_row = x.stride(0);
    int x_stride_col = x.stride(1);
    
    auto out = torch::empty({rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));
    
    // Make sure scale and bias are contiguous
    auto scale_c = scale.contiguous();
    auto bias_c = bias.contiguous();
    
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    affine_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        scale_c.data_ptr<float>(),
        bias_c.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols,
        x_stride_row,
        x_stride_col
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu", &affine_relu_cuda, "Affine ReLU on non-contiguous tensor (CUDA)");
}
