```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_relu_kernel(
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

torch::Tensor elementwise_add_relu(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    elementwise_add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_relu", &elementwise_add_relu, "ReLU(x + y) CUDA kernel");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="elementwise_add_relu_cuda",
    ext_modules=[
        CUDAExtension(
            name="elementwise_add_relu_cuda",
            sources=["elementwise_add_relu.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```python
# extension_wrapper.py  (harness entry point)
import torch
from torch import Tensor

try:
    import elementwise_add_relu_cuda as _ext
    _USE_CUDA = True
except ImportError:
    _USE_CUDA = False


def elementwise_add_relu(x: Tensor, y: Tensor) -> Tensor:
    if _USE_CUDA and x.is_cuda and y.is_cuda:
        x = x.contiguous()
        y = y.contiguous()
        return _ext.elementwise_add_relu(x, y)
    # CPU fallback
    return torch.relu(x + y)
```
