#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void broadcast_affine_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        float val = x[row * cols + col] * scale[col] + bias[col];
        // clamp to [-1, 1]
        val = val < -1.0f ? -1.0f : (val > 1.0f ? 1.0f : val);
        out[row * cols + col] = val;
    }
}

torch::Tensor broadcast_affine_clamp(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    x = x.contiguous();
    scale = scale.contiguous();
    bias = bias.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    TORCH_CHECK(scale.size(0) == cols, "scale size must match cols");
    TORCH_CHECK(bias.size(0) == cols, "bias size must match cols");

    auto out = torch::empty_like(x);

    dim3 block(32, 8);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    broadcast_affine_clamp_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_affine_clamp", &broadcast_affine_clamp,
          "Broadcast affine clamp: clamp(x * scale + bias, -1, 1)");
}
