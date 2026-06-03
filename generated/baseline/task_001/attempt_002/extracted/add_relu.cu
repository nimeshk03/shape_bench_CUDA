#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread using float4
    int idx4 = idx * 4;
    int stride4 = stride * 4;
    
    while (idx4 + 3 < n) {
        float4 xv = reinterpret_cast<const float4*>(x)[idx];
        float4 yv = reinterpret_cast<const float4*>(y)[idx];
        float4 ov;
        ov.x = fmaxf(xv.x + yv.x, 0.0f);
        ov.y = fmaxf(xv.y + yv.y, 0.0f);
        ov.z = fmaxf(xv.z + yv.z, 0.0f);
        ov.w = fmaxf(xv.w + yv.w, 0.0f);
        reinterpret_cast<float4*>(out)[idx] = ov;
        idx += stride;
        idx4 += stride4;
    }
    
    // Handle remaining elements
    while (idx4 < n) {
        out[idx4] = fmaxf(x[idx4] + y[idx4], 0.0f);
        idx4++;
    }
}

torch::Tensor add_relu_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    // Each thread handles 4 elements
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    blocks = min(blocks, 65535);
    
    add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu_cuda, "Add + ReLU (CUDA)");
}
