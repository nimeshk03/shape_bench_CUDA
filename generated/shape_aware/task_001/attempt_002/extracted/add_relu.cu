#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] + y[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor add_relu_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();

    if (n == 0) return out;

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);

    add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

torch::Tensor add_relu(torch::Tensor x, torch::Tensor y) {
    if (x.is_cuda()) {
        auto x_c = x.contiguous();
        auto y_c = y.contiguous();
        return add_relu_cuda(x_c, y_c);
    }
    // CPU fallback
    return torch::relu(x + y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu, "ReLU(x + y) CUDA implementation");
}
