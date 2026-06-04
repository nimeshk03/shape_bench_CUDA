# solution.py
"""CUDA extension wrapper for task_007: broadcast_affine_clamp."""

from __future__ import annotations

import os
import torch
from torch import Tensor

# Build and load the extension inline if not already compiled
def _load_extension():
    import torch.utils.cpp_extension as cpp_ext
    import tempfile

    cuda_src = r"""
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
        val = val < -1.0f ? -1.0f : (val > 1.0f ? 1.0f : val);
        out[row * cols + col] = val;
    }
}

torch::Tensor broadcast_affine_clamp(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias
) {
    x = x.contiguous();
    scale = scale.contiguous();
    bias = bias.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

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
"""
    build_dir = os.path.join(tempfile.gettempdir(), "task_007_ext_build")
    os.makedirs(build_dir, exist_ok=True)
    src_path = os.path.join(build_dir, "task_007_kernel.cu")
    with open(src_path, "w") as f:
        f.write(cuda_src)

    ext = cpp_ext.load(
        name="task_007_ext",
        sources=[src_path],
        build_directory=build_dir,
        verbose=False,
    )
    return ext


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def forward(x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
    """CUDA implementation of clamp(x * scale + bias, -1, 1)."""
    if x.is_cuda:
        return _get_ext().broadcast_affine_clamp(x, scale, bias)
    # Fallback for CPU (shouldn't be needed in harness but just in case)
    return torch.clamp(x * scale + bias, min=-1.0, max=1.0)
