# Failure Root-Cause Notes

This note summarizes the first source-level inspection of failed generated CUDA
kernels from the harder layout/reduction batch.

Artifacts checked:

```text
results/experiments/20260604T132953Z  # Vast.ai RTX 4090
results/experiments/20260617T182927Z  # Colab Tesla T4
```

Both backends reproduced the same failure pattern:

```text
task_015 baseline attempt_002:    compilation_failure
task_015 baseline attempt_003:    compilation_failure
task_015 shape_aware attempt_001: compilation_failure
task_015 shape_aware attempt_002: compilation_failure
task_015 shape_aware attempt_003: compilation_failure
task_016 shape_aware attempt_002: original_and_variant_correctness_failure
```

## Summary Table

| Task | Prompt mode | Attempt(s) | Failure class | Likely root cause | Evidence | Benchmark implication |
|---|---|---:|---|---|---|---|
| `task_015` | baseline | 2, 3 | compilation failure | Invalid PyTorch C++ storage pointer API | Generated code calls `x.storage().data<float>()`, which fails to compile in both RTX 4090 and Colab T4 runs. | The task is exposing API-knowledge failures before shape robustness can be measured. |
| `task_015` | shape_aware | 1, 2, 3 | compilation failure | Same invalid storage pointer pattern, with minor variants | Generated code calls `x.storage().data<float>()`, `x.storage().data_ptr().get<float>()`, or `x.storage().data_ptr<float>()`; all fail during extension compilation. | Shape-aware prompting encouraged stride/storage reasoning but did not ensure valid PyTorch extension API usage. |
| `task_016` | shape_aware | 2 | original and variant correctness failure | Shared-memory reduction drops partial sums from threads 32-63 when block size is 256 | The reduction loop stops at `stride > 32`, then warp-reduces only `smem[0..31]`; unlike passing attempts, it never adds `smem[tid + 32]` before warp reduction. | Dynamic reduction tasks can expose subtle reduction bugs even when dimensions are handled dynamically. |

## Details

### `task_015`: Offset Strided Affine ReLU

Intended task stressor:

```text
x is a non-contiguous 3D view with storage offset and irregular strides.
reference: relu(x * scale.view(1, 1, -1) + bias.view(1, -1, 1)).contiguous()
```

The five failed generated attempts all try to recover a raw base-storage pointer
and then manually add `storage_offset` plus tensor strides in the CUDA kernel.
That strategy would be relevant for a true base-storage pointer, but the
generated PyTorch C++ API calls are invalid:

```text
x.storage().data<float>()
x.storage().data_ptr().get<float>()
x.storage().data_ptr<float>()
```

The compiler error is consistent across Vast.ai and Colab:

```text
error: a pointer to a bound function may only be used to call the function
error: type name is not allowed
error: expected an expression
```

One baseline attempt passed. It used:

```text
x.data_ptr<float>()
```

with tensor strides and `storage_offset`. This compiled and passed all current
shapes, but it should not be treated as proof that the offset handling is
correct. PyTorch `Tensor.data_ptr()` points at the tensor's first logical
element on CPU, so adding `storage_offset` to that pointer would normally
double-count the offset.

There is also an experiment-design caveat: `task_015.create_inputs()` currently
slices the CPU tensor first and then transfers the sliced view to the requested
device. The CPU unit test confirms a non-zero storage offset before transfer,
but a device transfer can materialize a new tensor whose storage offset is reset
while preserving non-contiguous strides. That means the current GPU results are
strong evidence for stride/API failure, but they should not yet be used as clean
evidence about non-zero CUDA storage offsets. The next offset-focused batch
should create the base tensor on the target device before slicing, then record
or assert `x.storage_offset() > 0` in the evaluation metadata.

Research interpretation:

```text
task_015 currently reveals a build/API robustness issue more strongly than a
shape-generalization issue. The model often understands that strides and storage
offsets matter conceptually, but it does not reliably know the valid C++
extension API for accessing tensor data. The storage-offset part of the task
needs a follow-up fix before offset-specific conclusions are reported.
```

### `task_016`: Irregular Last-Dimension Layer Norm

Task stressor:

```text
last-dimension layer norm over dynamic feature sizes:
512, 257, 1024, 513, 769, 1009
```

The failed shape-aware attempt compiles, runs, and fails every shape including
the original shape. That means it is not a shape-variant-only failure.

The bug is in the block reduction:

```text
for (int stride = nthreads / 2; stride > 32; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
}
if (tid < 32) {
    float val = smem[tid];
    val = warp_reduce_sum(val);
    smem[tid] = val;
}
```

With `nthreads = 256`, the shared-memory loop reduces from 256 to 64 partial
values, then stops. The warp reduction only consumes `smem[0..31]`, leaving
`smem[32..63]` out of the final sum. Passing shape-aware attempts contain the
missing correction:

```text
if (nthreads >= 64) val += smem[tid + 32];
```

The same reduction pattern is used for both mean and variance, so the output can
be substantially wrong. The observed max absolute error is about `7.24` on both
GPU backends.

Research interpretation:

```text
task_016 exposes algorithmic CUDA reduction errors rather than shape-only
errors. The generated code uses dynamic `cols`, but its reduction implementation
is mathematically incomplete for the chosen block size.
```

## Implications For The Next Batch

To produce true shape-variant-only failures, the next tasks should make the
original shape easier than the variants while keeping the same operation:

```text
1. Use original dimensions that align with common assumptions, such as powers of
   two or block multiples.
2. Use variants that violate those assumptions: odd dimensions, prime sizes,
   non-contiguous views, and verified non-zero CUDA storage offsets.
3. Include tasks where a kernel can pass contiguous/original inputs while
   failing only offset or stride variants.
4. Add prompt language that explicitly states valid PyTorch C++ pointer rules
   for non-contiguous tensors, then compare against the current prompts.
```

Candidate next tasks:

```text
contiguous-original / non-contiguous-variant affine operation
storage-offset copy or transform with verified non-zero CUDA slice offsets
last-dimension reduction where original cols=512 and variants include 257/513/769/1009
batched matmul where original M/N/K are tile-aligned and variants are odd
```

The next experiment should still report compile failures separately from
correctness failures, because API/build failures are currently an important
part of generated-kernel robustness.
