#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void affine_relu_kernel(
    const float* __restrict__ x_data,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch, int rows, int cols,
    int stride_b, int stride_r, int stride_c,
    int storage_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * rows * cols;
    if (idx >= total) return;

    int b = idx / (rows * cols);
    int rem = idx % (rows * cols);
    int r = rem / cols;
    int c = rem % cols;

    int x_idx = storage_offset + b * stride_b + r * stride_r + c * stride_c;
    float val = x_data[x_idx] * scale[c] + bias[r];
    out[idx] = val > 0.0f ? val : 0.0f;
}

torch::Tensor affine_relu_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(scale.device().is_cuda(), "scale must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int batch = x.size(0);
    int rows = x.size(1);
    int cols = x.size(2);

    auto out = torch::empty({batch, rows, cols}, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int stride_b = x.stride(0);
    int stride_r = x.stride(1);
    int stride_c = x.stride(2);
    int storage_offset = x.storage_offset();

    const float* x_data = x.storage().data<float>();

    int total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    affine_relu_kernel<<<blocks, threads>>>(
        x_data,
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, rows, cols,
        stride_b, stride_r, stride_c,
        storage_offset
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine_relu_cuda", &affine_relu_cuda, "Affine ReLU with non-contiguous input (CUDA)");
}
