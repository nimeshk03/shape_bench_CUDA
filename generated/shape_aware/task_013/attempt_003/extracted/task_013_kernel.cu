#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for batched transpose: [B, R, C] -> [B, C, R]
__global__ void batched_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols
) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * rows * cols;
    
    if (idx >= total) return;
    
    // Decompose linear index into (b, r, c)
    int c = idx % cols;
    int tmp = idx / cols;
    int r = tmp % rows;
    int b = tmp / rows;
    
    // Input index: b * rows * cols + r * cols + c
    // Output index: b * cols * rows + c * rows + r
    int out_idx = b * (cols * rows) + c * rows + r;
    
    output[out_idx] = input[idx];
}

torch::Tensor batched_transpose(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device");
    
    // Make input contiguous
    x = x.contiguous();
    
    int batch = x.size(0);
    int rows  = x.size(1);
    int cols  = x.size(2);
    
    // Output shape: [batch, cols, rows]
    auto output = torch::empty({batch, cols, rows}, x.options());
    
    int total = batch * rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    batched_transpose_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, rows, cols
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_transpose", &batched_transpose, "Batched matrix transpose (CUDA)");
}
