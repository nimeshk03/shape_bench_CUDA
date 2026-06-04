#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: transpose last two dims of a 3D tensor with arbitrary strides
// Input shape: [batch, rows, cols], output shape: [batch, cols, rows]
// Input strides: arbitrary (non-contiguous), output is contiguous

__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int stride_batch,
    int stride_row,
    int stride_col
) {
    // Output is contiguous: [batch, cols, rows]
    // output[b, c, r] = input[b, r, c]
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cols * rows;
    
    if (idx >= total) return;
    
    int r = idx % rows;
    int tmp = idx / rows;
    int c = tmp % cols;
    int b = tmp / cols;
    
    // Read from non-contiguous input
    float val = input[b * stride_batch + r * stride_row + c * stride_col];
    
    // Write to contiguous output: [batch, cols, rows]
    output[b * (cols * rows) + c * rows + r] = val;
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    
    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);
    
    int stride_batch = x.stride(0);
    int stride_row   = x.stride(1);
    int stride_col   = x.stride(2);
    
    // Output: [batch, cols, rows], contiguous
    auto output = torch::empty({batch, cols, rows}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    
    int total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    batched_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols,
        stride_batch, stride_row, stride_col
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched transpose with stride support (CUDA)");
}
