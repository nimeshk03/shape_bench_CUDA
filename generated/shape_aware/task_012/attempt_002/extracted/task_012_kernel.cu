#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that handles non-contiguous input with arbitrary strides
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    // Input strides (in elements)
    int stride_batch,
    int stride_row,
    int stride_col,
    // Output strides (contiguous: [batch, cols, rows])
    int out_stride_batch,
    int out_stride_col,
    int out_stride_row
) {
    // Each thread handles one element
    // Output shape: [batch, cols, rows]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cols * rows;
    
    if (idx >= total) return;
    
    // Decompose linear index into (b, c, r) in output space
    int r = idx % rows;
    int tmp = idx / rows;
    int c = tmp % cols;
    int b = tmp / cols;
    
    // Read from input at (b, r, c) using input strides
    float val = input[b * stride_batch + r * stride_row + c * stride_col];
    
    // Write to output at (b, c, r) - contiguous output
    output[b * out_stride_batch + c * out_stride_col + r * out_stride_row] = val;
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    
    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);
    
    // Get input strides (in bytes -> convert to elements)
    int stride_batch = x.stride(0);
    int stride_row   = x.stride(1);
    int stride_col   = x.stride(2);
    
    // Output is contiguous with shape [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, 
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(x.device()));
    
    // Output strides (contiguous)
    int out_stride_batch = cols * rows;
    int out_stride_col   = rows;
    int out_stride_row   = 1;
    
    int total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    batched_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols,
        stride_batch, stride_row, stride_col,
        out_stride_batch, out_stride_col, out_stride_row
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched transpose with stride support");
}
