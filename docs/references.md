# Technical References

This file lists the main technical references for ShapeBench-CUDA.

Use official documentation first when implementing CUDA/PyTorch-related code.

## 1. CUDA References

### NVIDIA CUDA C++ Programming Guide

Official CUDA programming guide.

Use for:

- CUDA execution model
- grids, blocks, and threads
- memory hierarchy
- shared memory
- global memory
- synchronization
- occupancy concepts
- CUDA language features
- performance considerations

Reference:

```text
https://docs.nvidia.com/cuda/cuda-programming-guide/
```

The CUDA C++ Programming Guide is the official NVIDIA resource for the CUDA programming model and writing GPU code. It covers the CUDA platform, programming model, language extensions, hardware-specific features, and technical appendices.

Source: NVIDIA CUDA Programming Guide.

### CUDA C++ Programming Guide PDF

Useful if a PDF version is easier to search or save locally.

```text
https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
```

Source: NVIDIA CUDA C++ Programming Guide PDF.

## 2. PyTorch CUDA/C++ Extension References

### `torch.utils.cpp_extension`

Official PyTorch documentation for building C++ and CUDA extensions.

Use for:

- `CppExtension`
- `CUDAExtension`
- `BuildExtension`
- JIT extension loading
- compiling custom CUDA/C++ operators
- PyTorch extension build configuration

Reference:

```text
https://docs.pytorch.org/docs/stable/cpp_extension.html
```

PyTorch provides `CUDAExtension` as a convenience wrapper for building CUDA/C++ extensions, including CUDA include paths, library paths, and runtime library setup.

### PyTorch Custom C++ and CUDA Operators Tutorial

Use for:

- custom operator examples
- extension structure
- Python bindings
- ahead-of-time extension builds
- integrating C++/CUDA code with PyTorch

Reference:

```text
https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
```

Note: If this exact page moves, search the PyTorch docs for:

```text
PyTorch custom C++ CUDA operators
```

## 3. `torch.compile` References

### `torch.compile` API Documentation

Use for:

- understanding `torch.compile`
- compiler baseline behavior
- dynamic shape behavior
- backend options
- compile modes

Reference:

```text
https://docs.pytorch.org/docs/stable/generated/torch.compile.html
```

The official API docs describe `torch.compile` arguments and behavior, including dynamic recompilation behavior when dynamism is detected.

### Introduction to `torch.compile`

Use for:

- conceptual explanation
- examples
- basic usage
- why `torch.compile` improves PyTorch performance

Reference:

```text
https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
```

The PyTorch tutorial explains that `torch.compile` speeds up PyTorch code by JIT-compiling PyTorch programs into optimized kernels.

### `torch.compiler` User Guide

Use for:

- compiler concepts
- graph capture
- PyTorch compiler workflow
- broader `torch.compile` context

Reference:

```text
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html
```

The PyTorch compiler docs describe `torch.compile` as a PyTorch 2.x function for graph capture and optimized execution.

## 4. Benchmarking References

For Phase 1, keep benchmarking simple.

Recommended principles:

- warm up before timing
- synchronize CUDA before and after timing
- repeat multiple iterations
- report average or median runtime
- log GPU name, PyTorch version, CUDA version, and shape
- compare generated kernels against PyTorch eager
- compare against `torch.compile` when available

Important CUDA timing pattern:

```python
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(iters):
    output = fn(*inputs)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Do not measure CUDA kernels without synchronization.

## 5. Shape-Generalization Rules for Generated CUDA Kernels

When generating or reviewing CUDA code, prefer kernels that:

- use runtime shape values
- avoid hardcoded dimensions
- guard memory accesses with boundary checks
- handle non-power-of-two dimensions
- handle smaller and larger batch sizes
- use contiguous assumptions only when explicitly required and checked
- report clear failure if input assumptions are violated

Avoid kernels that:

- assume fixed shape constants
- assume power-of-two dimensions
- ignore tail elements
- omit boundary checks
- silently produce incorrect output for odd shapes
- only pass the original benchmark input

## 6. CUDA-Agent Reference

The sibling repository is used as reference only:

```text
../CUDA-Agent
```

Do not modify it unless explicitly asked.

Useful files:

```text
../CUDA-Agent/README.md
../CUDA-Agent/agent_workdir/SKILL.md
../CUDA-Agent/agent_workdir/model.py
../CUDA-Agent/agent_workdir/model_new.py
../CUDA-Agent/agent_workdir/utils/
```

The CUDA-Agent paper and repository are useful for understanding:

- agent workdir structure
- CUDA extension layout
- verification/profiling flow
- constraints on generated kernels
- baseline comparison against PyTorch eager and `torch.compile`

## 7. Documentation Use Policy for Codex

When implementing CUDA/PyTorch functionality:

1. Prefer official docs listed above.
2. Do not invent APIs.
3. If unsure about CUDA or PyTorch extension behavior, leave a TODO and implement the safer minimal version.
4. Keep local code CPU-compatible.
5. Do not require CUDA for local tests.
6. Add CUDA-specific checks only behind availability guards.

